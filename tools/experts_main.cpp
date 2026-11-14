// locus-experts: serves MoE expert calls for expert parallelism (#781).
//
//   locus-experts --model <m.gguf> --listen HOST:PORT --experts LO:HI
//                 [--allow CIDR]... [--backend NAME]

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/pipeline/expert_server.hpp"
#include "locus/pipeline/net.hpp"

namespace {

const char* kUsage =
    "usage: locus-experts --model <m.gguf> --listen HOST:PORT "
    "--experts LO:HI [--allow CIDR]... [--backend NAME]\n"
    "\n"
    "Serves MoE expert calls for expert parallelism: a peer sends a\n"
    "layer, an expert id in [LO, HI) and a hidden state, and gets that\n"
    "expert's output back, the bits its own forward would compute. One\n"
    "connection is served at a time.\n"
    "\n"
    "  --allow CIDR  peers allowed to connect (repeatable); without it\n"
    "                only loopback (127.0.0.0/8, ::1), since the port\n"
    "                has no authentication and runs compute for any\n"
    "                peer it accepts\n"
    "  --backend B   math backend (default: the model's)\n";

/** Parses "LO:HI" with LO < HI. @returns false when malformed. */
bool parse_range(const std::string& s, std::uint32_t& lo,
                 std::uint32_t& hi) {
    const auto c = s.find(':');
    if (c == std::string::npos || c == 0 || c + 1 == s.size()) {
        return false;
    }
    char* end = nullptr;
    const unsigned long a = std::strtoul(s.c_str(), &end, 10);
    if (end != s.c_str() + c) {
        return false;
    }
    const unsigned long b = std::strtoul(s.c_str() + c + 1, &end, 10);
    if (*end != '\0' || a >= b || b > 1u << 20) {
        return false;
    }
    lo = static_cast<std::uint32_t>(a);
    hi = static_cast<std::uint32_t>(b);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, listen, range, backend;
    std::vector<std::string> allow_str;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n%s", a.c_str(),
                             kUsage);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") {
            model_path = next();
        } else if (a == "--listen") {
            listen = next();
        } else if (a == "--experts") {
            range = next();
        } else if (a == "--allow") {
            allow_str.push_back(next());
        } else if (a == "--backend") {
            backend = next();
        } else if (a == "--help" || a == "-h") {
            std::printf("%s", kUsage);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n%s", a.c_str(), kUsage);
            return 2;
        }
    }
    std::string host;
    int port = 0;
    std::uint32_t lo = 0, hi = 0;
    if (model_path.empty() || !locus::pipeline::parse_hostport(listen, host,
                                                               port) ||
        !parse_range(range, lo, hi)) {
        std::fprintf(stderr, "%s", kUsage);
        return 2;
    }
    if (allow_str.empty()) {
        allow_str = {"127.0.0.0/8", "::1"};
    }
    std::vector<locus::pipeline::Cidr> allow;
    for (const auto& s : allow_str) {
        const auto c = locus::pipeline::Cidr::parse(s);
        if (!c) {
            std::fprintf(stderr, "bad --allow %s\n", s.c_str());
            return 2;
        }
        allow.push_back(*c);
    }
    try {
        auto g = locus::gguf::GgufFile::open(model_path);
        auto model = locus::model::TransformerModel::load(g);
        if (!backend.empty()) {
            model.use_backend(locus::backend::resolve_backend(backend));
        }
        if (model.hparams().n_expert == 0 || hi > model.hparams().n_expert) {
            std::fprintf(stderr, "--experts %u:%u: the model has %u\n", lo,
                         hi, model.hparams().n_expert);
            return 2;
        }
        int bound = 0;
        const int lfd = locus::pipeline::listen_on(host, port, &bound);
        if (lfd < 0) {
            std::fprintf(stderr, "listen on %s failed\n", listen.c_str());
            return 1;
        }
        std::fprintf(stderr, "locus-experts: experts [%u,%u) on :%d\n", lo,
                     hi, bound);
        for (;;) {
            std::string peer;
            const int fd = locus::pipeline::accept_one(lfd, &peer);
            if (fd < 0) {
                continue;
            }
            if (!locus::pipeline::ip_allowed(peer, allow)) {
                std::fprintf(stderr, "locus-experts: refused %s\n",
                             peer.c_str());
                ::close(fd);
                continue;
            }
            const auto n =
                locus::pipeline::serve_expert_calls(model, fd, lo, hi);
            std::fprintf(stderr, "locus-experts: %s: %llu calls\n",
                         peer.c_str(), static_cast<unsigned long long>(n));
            ::close(fd);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "locus-experts: %s\n", e.what());
        return 1;
    }
}
