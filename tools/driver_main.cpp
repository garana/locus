#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "cli_spec.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/pipeline/driver.hpp"
#include "locus/pipeline/message.hpp"
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
    "                    10000; 0 = wait forever)\n"
    "  --concurrent N    run the prompt as N sessions at once (default\n"
    "                    1), each with its own request id; a logits\n"
    "                    connection is paired with its session by the\n"
    "                    id of its first frame. Prints each text and\n"
    "                    the aggregate tok/s. All sessions use --seed,\n"
    "                    so with --temp > 0 they draw the same text\n";

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, head, logits_listen, prompt;
    bool have_prompt = false;
    std::vector<std::string> allow_str;
    int max_tokens = 64, connect_timeout = 5000, logits_timeout = 10000;
    int concurrent = 1;
    double temp = 0.0;
    unsigned long long seed = 0;

    // Shares the one guarded parse (parse_nonneg_int, cli_spec.hpp) with
    // the stage tool's flags, so an empty value / junk / overflow is
    // rejected here the same way -- the previous inline strtol let ""
    // through as 0 (strtol does no conversion and leaves end at the
    // start, which for "" is also the terminator).
    const auto as_int = [](const std::string& v, const char* name) -> int {
        int x = 0;
        if (!locus_tools::parse_nonneg_int(v, x)) {
            std::fprintf(stderr, "%s must be a non-negative integer\n",
                         name);
            std::exit(2);
        }
        return x;
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
        } else if (a == "--concurrent") {
            concurrent = as_int(next("--concurrent"), "--concurrent");
            if (concurrent < 1) {
                std::fprintf(stderr, "--concurrent must be at least 1\n");
                return 2;
            }
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
        // (which triggers the chain to wire up). Session i (request_id
        // i + 1) sends its first prompt token; the last stage may dial
        // the logits connections in any order (a stage that reuses a
        // pooled downstream dials late), so each accepted connection is
        // paired with its session by the request_id of its first frame.
        const int le = locus::pipeline::listen_on(lh, lp, nullptr);
        if (le < 0) {
            std::fprintf(stderr, "listen on %s failed\n",
                         logits_listen.c_str());
            return 1;
        }
        if (ids.empty()) {
            std::fprintf(stderr, "the prompt encodes to no tokens\n");
            return 2;
        }
        struct Session {
            int head_fd = -1;
            int logits_fd = -1;
            std::vector<float> first_logits;
            locus::pipeline::DriverResult res;
        };
        std::vector<Session> sessions(static_cast<std::size_t>(concurrent));
        for (std::size_t i = 0; i < sessions.size(); ++i) {
            auto& s = sessions[i];
            s.head_fd = locus::pipeline::connect_to(hh, hp, connect_timeout);
            if (s.head_fd < 0 ||
                !locus::pipeline::write_message(
                    s.head_fd, locus::pipeline::make_token(i + 1, 0, ids[0]))) {
                std::fprintf(stderr, "connect to head %s failed\n",
                             head.c_str());
                return 1;
            }
        }
        for (std::size_t k = 0; k < sessions.size(); ++k) {
            std::string peer;
            const int fd =
                locus::pipeline::accept_one(le, &peer, logits_timeout);
            if (fd < 0) {
                std::fprintf(stderr,
                             "no logits connection within %d ms: the last "
                             "stage never connected back (its --downstream "
                             "should point at %s). Check the stage chain.\n",
                             logits_timeout, logits_listen.c_str());
                return 1;
            }
            if (!locus::pipeline::ip_allowed(peer, allow)) {
                std::fprintf(stderr,
                             "logits connection from %s refused (not in "
                             "--allow)\n",
                             peer.c_str());
                return 1;
            }
            // A peer that connects and never sends must not hang us: the
            // first frame gets the same deadline as the accept.
            locus::pipeline::set_recv_timeout(fd, logits_timeout);
            locus::pipeline::Message first;
            const auto got = locus::pipeline::read_message(fd, first);
            locus::pipeline::set_recv_timeout(fd, 0);
            if (got !=
                    locus::pipeline::ReadResult::kOk ||
                first.type != locus::pipeline::MsgType::kLogits ||
                first.position != 0 || first.request_id < 1 ||
                first.request_id > sessions.size() ||
                sessions[first.request_id - 1].logits_fd >= 0) {
                std::fprintf(stderr,
                             "a logits connection's first frame names no "
                             "waiting session (type %d, request %llu, "
                             "position %u)\n",
                             static_cast<int>(first.type),
                             static_cast<unsigned long long>(
                                 first.request_id),
                             first.position);
                return 1;
            }
            auto& s = sessions[first.request_id - 1];
            s.logits_fd = fd;
            s.first_logits = std::move(first.data);
        }
        ::close(le);

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;
        for (std::size_t i = 0; i < sessions.size(); ++i) {
            threads.emplace_back([&, i] {
                locus::pipeline::DriverParams params;
                params.sampling.temperature = static_cast<float>(temp);
                params.max_tokens = static_cast<std::uint32_t>(max_tokens);
                params.seed = seed;
                params.request_id = i + 1;
                // An exception must not escape the thread (that calls
                // std::terminate): the session just ends incomplete.
                try {
                    sessions[i].res =
                        locus::pipeline::drive_generation_continue(
                            sessions[i].head_fd, sessions[i].logits_fd, ids,
                            std::move(sessions[i].first_logits),
                            tok.eos_id(), params);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "locus-driver: session %zu: %s\n",
                                 i + 1, e.what());
                    sessions[i].res.complete = false;
                }
            });
        }
        for (auto& t : threads) {
            t.join();
        }
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          t0)
                .count();

        bool all_complete = true;
        std::size_t n_tokens = 0;
        for (auto& s : sessions) {
            auto gen = std::move(s.res.tokens);
            if (!gen.empty() && gen.back() == tok.eos_id()) {
                gen.pop_back();  // do not print or count the stop token
            }
            n_tokens += gen.size();
            std::printf("%s%s\n", tok.decode(ids).c_str(),
                        tok.decode(gen).c_str());
            all_complete = all_complete && s.res.complete;
            ::close(s.head_fd);
            ::close(s.logits_fd);
        }
        std::fprintf(stderr,
                     "locus-driver: %d session(s), %zu tokens in %.2f s "
                     "(%.2f tok/s in all; the time counts the prompt "
                     "after each session's first token)\n",
                     concurrent, n_tokens, secs,
                     static_cast<double>(n_tokens) / secs);
        if (!all_complete) {
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
