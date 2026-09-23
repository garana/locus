#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/pipeline/driver.hpp"
#include "locus/pipeline/net.hpp"
#include "locus/tok/tokenizer.hpp"

namespace {

const char* kUsage =
    "usage: locus-driver --model <m.gguf> --head HOST:PORT "
    "--logits-listen HOST:PORT <prompt> [--allow CIDR]...\n"
    "\n"
    "Entry point of a multi-server pipeline (DESIGN.md R15+): drives\n"
    "generation across a remote stage chain. It connects to the first\n"
    "stage on --head and sends each token there; the last stage sends\n"
    "logits back to --logits-listen (which the last stage's --downstream\n"
    "must point at). The driver loads only the tokenizer, not the model\n"
    "weights: the remote stages hold the layers and their KV caches.\n"
    "\n"
    "  --model M         gguf file (for the tokenizer + eos only)\n"
    "  --head H:P        first stage's input address to connect to\n"
    "  --logits-listen H:P  address to listen on for the last stage's\n"
    "                    logits (empty host == wildcard, \":PORT\")\n"
    "  --allow CIDR      restrict who may connect back (repeatable)\n"
    "  --max-tokens N    cap generated tokens (default 64)\n"
    "  --temp T          sampling temperature (default 0 = greedy)\n"
    "  --seed S          RNG seed for non-greedy sampling (default 0)\n"
    "  --connect-timeout N  --head connect timeout ms (default 5000)\n"
    "  --logits-timeout N   how long to wait for the last stage to\n"
    "                    connect back before giving up, ms (default\n"
    "                    10000; 0 = wait forever)\n";

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, head, logits_listen, prompt;
    bool have_prompt = false;
    std::vector<std::string> allow_str;
    int max_tokens = 64, connect_timeout = 5000, logits_timeout = 10000;
    double temp = 0.0;
    unsigned long long seed = 0;

    const auto as_int = [](const std::string& v, const char* name) -> int {
        char* e = nullptr;
        const long x = std::strtol(v.c_str(), &e, 10);
        if (*e != '\0' || x < 0 || x > 2147483647L) {
            std::fprintf(stderr, "%s must be a non-negative integer\n",
                         name);
            std::exit(2);
        }
        return static_cast<int>(x);
    };
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
        } else if (a == "--head") {
            head = next("--head");
        } else if (a == "--logits-listen") {
            logits_listen = next("--logits-listen");
        } else if (a == "--allow") {
            allow_str.push_back(next("--allow"));
        } else if (a == "--max-tokens") {
            max_tokens = as_int(next("--max-tokens"), "--max-tokens");
        } else if (a == "--temp") {
            temp = std::atof(next("--temp").c_str());
        } else if (a == "--seed") {
            seed = std::strtoull(next("--seed").c_str(), nullptr, 10);
        } else if (a == "--connect-timeout") {
            connect_timeout =
                as_int(next("--connect-timeout"), "--connect-timeout");
        } else if (a == "--logits-timeout") {
            logits_timeout =
                as_int(next("--logits-timeout"), "--logits-timeout");
        } else if (a == "-h" || a == "--help") {
            std::printf("%s", kUsage);
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown arg: %s\n%s", a.c_str(), kUsage);
            return 2;
        } else {
            prompt = a;  // the sole positional argument
            have_prompt = true;
        }
    }
    if (model_path.empty() || head.empty() || logits_listen.empty() ||
        !have_prompt) {
        std::fprintf(stderr, "%s", kUsage);
        return 2;
    }

    std::string hh, lh;
    int hp = 0, lp = 0;
    if (!locus::pipeline::parse_hostport(head, hh, hp)) {
        std::fprintf(stderr, "bad --head (want HOST:PORT)\n");
        return 2;
    }
    if (!locus::pipeline::parse_hostport(logits_listen, lh, lp)) {
        std::fprintf(stderr, "bad --logits-listen (want HOST:PORT)\n");
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
        auto tok_ptr = locus::tok::tokenizer_from_gguf(g);
        auto& tok = *tok_ptr;
        const auto ids = tok.encode(prompt, true);

        // Listen for the tail's logits first, then connect to the head
        // (which triggers the chain to wire up), then accept the tail.
        const int le = locus::pipeline::listen_on(lh, lp, nullptr);
        if (le < 0) {
            std::fprintf(stderr, "listen on %s failed\n",
                         logits_listen.c_str());
            return 1;
        }
        const int head_fd =
            locus::pipeline::connect_to(hh, hp, connect_timeout);
        if (head_fd < 0) {
            std::fprintf(stderr, "connect to head %s failed\n",
                         head.c_str());
            return 1;
        }
        std::string peer;
        const int logits_fd =
            locus::pipeline::accept_one(le, &peer, logits_timeout);
        if (logits_fd < 0) {
            std::fprintf(stderr,
                         "no logits connection within %d ms: the last "
                         "stage never connected back (its --downstream "
                         "should point at %s). Check the stage chain.\n",
                         logits_timeout, logits_listen.c_str());
            return 1;
        }
        ::close(le);
        if (!locus::pipeline::ip_allowed(peer, allow)) {
            std::fprintf(stderr,
                         "logits connection from %s refused (not in "
                         "--allow)\n",
                         peer.c_str());
            return 1;
        }

        locus::pipeline::DriverParams params;
        params.sampling.temperature = static_cast<float>(temp);
        params.max_tokens = static_cast<std::uint32_t>(max_tokens);
        params.seed = seed;
        auto res = locus::pipeline::drive_generation(
            head_fd, logits_fd, ids, tok.eos_id(), params);

        auto gen = std::move(res.tokens);
        if (!gen.empty() && gen.back() == tok.eos_id()) {
            gen.pop_back();  // do not print the stop token
        }
        std::printf("%s%s\n", tok.decode(ids).c_str(),
                    tok.decode(gen).c_str());
        if (!res.complete) {
            std::fprintf(stderr,
                         "locus-driver: generation truncated by a "
                         "transport failure (the stage chain dropped "
                         "mid-stream); output above is partial\n");
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "locus-driver: %s\n", e.what());
        return 1;
    }
    return 0;
}
