#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "locus/pipeline/net.hpp"
#include "locus/pipeline/stage.hpp"
#include "locus/pipeline/stage_server.hpp"

namespace {

const char* kUsage =
    "usage: locus-stage --model <m.gguf> --layers A:B "
    "--listen HOST:PORT --downstream HOST:PORT [--allow CIDR]...\n"
    "\n"
    "Runs one pipeline stage (multi-server, DESIGN.md R15+): accepts an\n"
    "input connection on --listen, runs the model's layers [A,B), and\n"
    "sends its output to --downstream. --listen HOST may be empty for\n"
    "the wildcard address (\":PORT\"). --allow restricts which peer\n"
    "addresses may connect and is repeatable (e.g. --allow 10.0.0.0/8);\n"
    "with no --allow, any peer may connect.\n";

// Splits "host:port". Accepts a bracketed IPv6 host "[::1]:port"
// (brackets stripped) as well as "host:port" and ":port" (empty host
// == wildcard); a bare IPv6 literal would be ambiguous, so brackets
// are the way to give one.
bool split_hostport(const std::string& s, std::string& host,
                    int& port) {
    std::string p;
    if (!s.empty() && s.front() == '[') {
        const auto close = s.find(']');
        if (close == std::string::npos) {
            return false;
        }
        host = s.substr(1, close - 1);
        std::string rest = s.substr(close + 1);
        if (rest.empty() || rest.front() != ':') {
            return false;
        }
        p = rest.substr(1);
    } else {
        const auto c = s.rfind(':');
        if (c == std::string::npos) {
            return false;
        }
        host = s.substr(0, c);
        p = s.substr(c + 1);
    }
    if (p.empty()) {
        return false;
    }
    char* end = nullptr;
    const long v = std::strtol(p.c_str(), &end, 10);
    if (*end != '\0' || v < 0 || v > 65535) {
        return false;
    }
    port = static_cast<int>(v);
    return true;
}

// Parses "A:B" into a half-open layer range, requiring A < B.
bool parse_layers(const std::string& s, std::uint32_t& a,
                  std::uint32_t& b) {
    const auto c = s.find(':');
    if (c == std::string::npos) {
        return false;
    }
    char* e1 = nullptr;
    char* e2 = nullptr;
    const long la = std::strtol(s.substr(0, c).c_str(), &e1, 10);
    const std::string bs = s.substr(c + 1);
    const long lb = std::strtol(bs.c_str(), &e2, 10);
    if (*e1 != '\0' || *e2 != '\0' || la < 0 || lb < 0 || la >= lb) {
        return false;
    }
    a = static_cast<std::uint32_t>(la);
    b = static_cast<std::uint32_t>(lb);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, layers, listen, downstream;
    std::vector<std::string> allow_str;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") {
            model_path = next("--model");
        } else if (a == "--layers") {
            layers = next("--layers");
        } else if (a == "--listen") {
            listen = next("--listen");
        } else if (a == "--downstream") {
            downstream = next("--downstream");
        } else if (a == "--allow") {
            allow_str.push_back(next("--allow"));
        } else if (a == "-h" || a == "--help") {
            std::printf("%s", kUsage);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n%s", a.c_str(),
                         kUsage);
            return 2;
        }
    }
    if (model_path.empty() || layers.empty() || listen.empty() ||
        downstream.empty()) {
        std::fprintf(stderr, "%s", kUsage);
        return 2;
    }

    std::uint32_t la = 0, lb = 0;
    if (!parse_layers(layers, la, lb)) {
        std::fprintf(stderr, "bad --layers (want A:B with A < B)\n");
        return 2;
    }
    std::string lh, dh;
    int lp = 0, dp = 0;
    if (!split_hostport(listen, lh, lp)) {
        std::fprintf(stderr, "bad --listen (want HOST:PORT)\n");
        return 2;
    }
    if (!split_hostport(downstream, dh, dp)) {
        std::fprintf(stderr, "bad --downstream (want HOST:PORT)\n");
        return 2;
    }
    std::vector<locus::pipeline::Cidr> allow;
    for (const auto& s : allow_str) {
        const auto c = locus::pipeline::Cidr::parse(s);
        if (!c) {
            std::fprintf(stderr, "bad --allow CIDR: %s\n", s.c_str());
            return 2;
        }
        allow.push_back(*c);
    }

    try {
        auto g = locus::gguf::GgufFile::open(model_path);
        auto model = locus::model::LlamaModel::load(g);
        if (lb > model.hparams().n_layers) {
            std::fprintf(stderr,
                         "--layers end %u exceeds model n_layers %u\n",
                         lb, model.hparams().n_layers);
            return 2;
        }
        locus::pipeline::PipelineStage stage(model, la, lb);
        const int lfd = locus::pipeline::listen_on(lh, lp, nullptr);
        if (lfd < 0) {
            std::fprintf(stderr, "listen on %s failed\n",
                         listen.c_str());
            return 1;
        }
        std::fprintf(stderr,
                     "locus-stage: layers [%u,%u) on %s -> %s\n", la,
                     lb, listen.c_str(), downstream.c_str());
        const bool ok =
            locus::pipeline::serve_stage(stage, lfd, allow, dh, dp);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "locus-stage: %s\n", e.what());
        return 1;
    }
}
