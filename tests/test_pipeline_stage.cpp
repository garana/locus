#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "locus/pipeline/message.hpp"
#include "locus/pipeline/net.hpp"
#include "locus/pipeline/stage.hpp"
#include "locus/pipeline/stage_server.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::pipeline::Message;
using locus::pipeline::MsgType;
using locus::pipeline::PipelineStage;
using locus::pipeline::ReadResult;

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

}  // namespace

// Multi-server (#71) increment 3: chaining PipelineStage workers, each
// owning a layer slice and its own KV cache, connected by sockets, must
// reproduce a single-process generation byte-for-byte. This is the
// in-process proof of the whole stage protocol (kToken in, kActivation
// between stages, kLogits out) and per-stage position advancement,
// before real TCP / separate processes.
TEST_CASE("pipeline stages reproduce single-process generation",
          "[pipeline][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::uint32_t L = model.hparams().n_layers;
    const std::uint32_t V = model.hparams().n_vocab;
    REQUIRE(L >= 2);
    const auto prompt =
        tok.encode("Once upon a time, there was a little", true);
    constexpr int kGen = 12;

    // Single-process reference.
    auto mono = [&]() {
        auto cache = model.make_cache();
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<float> logits(V);
        for (auto t : prompt) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(t, cache, seq, ws, logits);
        }
        std::vector<locus::tok::TokenId> gen;
        for (int i = 0; i < kGen; ++i) {
            const auto n = locus::model::argmax(logits);
            gen.push_back(n);
            if (n == tok.eos_id()) {
                break;
            }
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(n, cache, seq, ws, logits);
        }
        return gen;
    };
    const auto ref = mono();
    REQUIRE(ref.size() >= 1);

    // Runs generation across the stages defined by `bounds` (0..L),
    // each stage on its own thread, connected by socketpairs. Returns
    // the generated tokens (empty on a flow error).
    // connect_pair(writer, reader) wires one stream; swappable so the
    // same chain runs over socketpairs or loopback TCP.
    auto pipeline = [&](const std::vector<std::uint32_t>& bounds,
                        auto&& connect_pair) {
        const std::size_t N = bounds.size() - 1;
        std::vector<std::unique_ptr<PipelineStage>> stages;
        for (std::size_t i = 0; i < N; ++i) {
            stages.push_back(std::make_unique<PipelineStage>(
                model, bounds[i], bounds[i + 1]));
        }
        // Links: entry -> s0 -> ... -> s(N-1) -> entry.
        std::vector<int> in_fd(N), out_fd(N);
        int entry_w = -1, entry_r = -1;
        connect_pair(entry_w, in_fd[0]);
        for (std::size_t i = 0; i + 1 < N; ++i) {
            connect_pair(out_fd[i], in_fd[i + 1]);
        }
        connect_pair(out_fd[N - 1], entry_r);

        std::vector<int> results(N, 0);
        std::vector<std::thread> th;
        for (std::size_t i = 0; i < N; ++i) {
            th.emplace_back([&, i] {
                results[i] =
                    stages[i]->run(in_fd[i], out_fd[i]) ? 1 : 0;
            });
        }

        // Entry: send one token, read one logits, repeat. Strictly
        // serial (one token in flight), which is all functional
        // correctness needs.
        bool flow = true;
        std::uint32_t pos = 0;
        std::vector<float> logits;
        auto round_trip = [&](locus::tok::TokenId t) {
            const Message in = locus::pipeline::make_token(1, pos, t);
            if (!locus::pipeline::write_message(entry_w, in)) {
                flow = false;
                return;
            }
            Message lg;
            if (locus::pipeline::read_message(entry_r, lg) !=
                    ReadResult::kOk ||
                lg.type != MsgType::kLogits) {
                flow = false;
                return;
            }
            logits = std::move(lg.data);
            ++pos;
        };

        std::vector<locus::tok::TokenId> gen;
        for (auto t : prompt) {
            round_trip(t);
            if (!flow) {
                break;
            }
        }
        for (int i = 0; i < kGen && flow; ++i) {
            const auto n = locus::model::argmax(logits);
            gen.push_back(n);
            if (n == tok.eos_id()) {
                break;
            }
            round_trip(n);
        }

        ::close(entry_w);  // EOF cascades down the chain
        for (auto& t : th) {
            t.join();
        }
        ::close(entry_r);

        REQUIRE(flow);
        for (std::size_t i = 0; i < N; ++i) {
            REQUIRE(results[i] == 1);  // every stage exited cleanly
        }
        return gen;
    };

    // A connected local stream: [1] is the send end, [0] the recv end.
    auto sock_pair = [](int& w, int& r) {
        int p[2];
        REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, p) == 0);
        w = p[1];
        r = p[0];
    };
    // The same over a loopback TCP connection (ephemeral port). connect
    // completes into the listen backlog, so accept in the same thread
    // returns without a separate accept thread.
    auto tcp_pair = [](int& w, int& r) {
        int port = 0;
        const int l = locus::pipeline::listen_on("127.0.0.1", 0, &port);
        REQUIRE(l >= 0);
        w = locus::pipeline::connect_to("127.0.0.1", port);
        REQUIRE(w >= 0);
        r = locus::pipeline::accept_one(l);
        REQUIRE(r >= 0);
        ::close(l);
    };

    SECTION("two stages, every split point (socketpair)") {
        for (std::uint32_t k = 1; k < L; ++k) {
            CAPTURE(k);
            REQUIRE(pipeline({0, k, L}, sock_pair) == ref);
        }
    }

    SECTION("three stages, a middle stage relays (socketpair)") {
        if (L >= 3) {
            REQUIRE(pipeline({0, 1, L - 1, L}, sock_pair) == ref);
        }
    }

    SECTION("two stages, every split point (loopback TCP)") {
        for (std::uint32_t k = 1; k < L; ++k) {
            CAPTURE(k);
            REQUIRE(pipeline({0, k, L}, tcp_pair) == ref);
        }
    }

    SECTION("three stages over loopback TCP") {
        if (L >= 3) {
            REQUIRE(pipeline({0, 1, L - 1, L}, tcp_pair) == ref);
        }
    }
}

// Multi-server (#71): stages run as servers via serve_stage (the
// locus-stage CLI's core) -- listen, accept from an allowed peer,
// connect downstream -- chained over loopback TCP, must reproduce
// single-process generation byte-for-byte.
TEST_CASE("pipeline serve_stage chain reproduces generation",
          "[pipeline][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::uint32_t L = model.hparams().n_layers;
    const std::uint32_t V = model.hparams().n_vocab;
    REQUIRE(L >= 2);
    const auto prompt =
        tok.encode("Once upon a time, there was a little", true);
    constexpr int kGen = 12;

    auto mono = [&]() {
        auto cache = model.make_cache();
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<float> logits(V);
        for (auto t : prompt) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(t, cache, seq, ws, logits);
        }
        std::vector<locus::tok::TokenId> gen;
        for (int i = 0; i < kGen; ++i) {
            const auto n = locus::model::argmax(logits);
            gen.push_back(n);
            if (n == tok.eos_id()) {
                break;
            }
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(n, cache, seq, ws, logits);
        }
        return gen;
    };
    const auto ref = mono();

    // Stages [0,k) and [k,L) served over loopback TCP; the entry drives
    // tokens into stage 0 and reads logits back from stage 1. Listeners
    // are created first so downstream ports are known before wiring.
    auto serve_two = [&](std::uint32_t k) {
        int p0 = 0, p1 = 0, pe = 0;
        const int l0 = locus::pipeline::listen_on("127.0.0.1", 0, &p0);
        const int l1 = locus::pipeline::listen_on("127.0.0.1", 0, &p1);
        const int le = locus::pipeline::listen_on("127.0.0.1", 0, &pe);
        REQUIRE(l0 >= 0);
        REQUIRE(l1 >= 0);
        REQUIRE(le >= 0);

        PipelineStage s0(model, 0, k);
        PipelineStage s1(model, k, L);
        // Restrict incoming to loopback (exercises the allowlist path).
        const std::vector<locus::pipeline::Cidr> allow{
            *locus::pipeline::Cidr::parse("127.0.0.0/8")};

        std::atomic<int> r0{-1}, r1{-1};
        std::thread t0([&] {
            r0 = locus::pipeline::serve_stage(s0, l0, allow,
                                              "127.0.0.1", p1)
                     ? 1
                     : 0;
        });
        std::thread t1([&] {
            r1 = locus::pipeline::serve_stage(s1, l1, allow,
                                              "127.0.0.1", pe)
                     ? 1
                     : 0;
        });

        const int entry_w =
            locus::pipeline::connect_to("127.0.0.1", p0);
        REQUIRE(entry_w >= 0);
        const int entry_r = locus::pipeline::accept_one(le, nullptr);
        REQUIRE(entry_r >= 0);
        ::close(le);

        bool flow = true;
        std::uint32_t pos = 0;
        std::vector<float> logits;
        auto round_trip = [&](locus::tok::TokenId t) {
            const Message in = locus::pipeline::make_token(1, pos, t);
            if (!locus::pipeline::write_message(entry_w, in)) {
                flow = false;
                return;
            }
            Message lg;
            if (locus::pipeline::read_message(entry_r, lg) !=
                    ReadResult::kOk ||
                lg.type != MsgType::kLogits) {
                flow = false;
                return;
            }
            logits = std::move(lg.data);
            ++pos;
        };

        std::vector<locus::tok::TokenId> gen;
        for (auto t : prompt) {
            round_trip(t);
            if (!flow) {
                break;
            }
        }
        for (int i = 0; i < kGen && flow; ++i) {
            const auto n = locus::model::argmax(logits);
            gen.push_back(n);
            if (n == tok.eos_id()) {
                break;
            }
            round_trip(n);
        }

        ::close(entry_w);
        t0.join();
        t1.join();
        ::close(entry_r);
        REQUIRE(flow);
        REQUIRE(r0.load() == 1);  // stage 0 exited cleanly
        REQUIRE(r1.load() == 1);  // stage 1 exited cleanly
        return gen;
    };

    for (std::uint32_t k = 1; k < L; ++k) {
        CAPTURE(k);
        REQUIRE(serve_two(k) == ref);
    }
}

// #14: serve_stage retries the downstream connect on a fixed wait (no
// backoff) and gives up after reconnect_attempts when the downstream
// never comes up. The established connection is the health signal.
TEST_CASE("serve_stage reconnect gives up after N attempts",
          "[pipeline][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;
    // First+last stage; only its accept + downstream-connect phase runs
    // here (the downstream never appears, so stage.run is not reached).
    PipelineStage stage(model, 0, L);

    int pin = 0;
    const int lin = locus::pipeline::listen_on("127.0.0.1", 0, &pin);
    REQUIRE(lin >= 0);
    // A downstream port with nothing listening (reserve then release).
    int pdn = 0;
    {
        const int t = locus::pipeline::listen_on("127.0.0.1", 0, &pdn);
        REQUIRE(t >= 0);
        ::close(t);
    }

    locus::pipeline::StageConn conn;
    conn.connect_timeout_ms = 100;
    conn.reconnect_wait_ms = 50;
    conn.reconnect_attempts = 3;  // 2 waits between 3 attempts

    std::atomic<int> result{-1};
    const auto t0 = std::chrono::steady_clock::now();
    std::thread th([&] {
        result = locus::pipeline::serve_stage(stage, lin, {},
                                              "127.0.0.1", pdn, conn)
                     ? 1
                     : 0;
    });
    // Provide the input connection so serve_stage passes its accept
    // phase and reaches the (failing) downstream-connect loop.
    const int cin = locus::pipeline::connect_to("127.0.0.1", pin);
    REQUIRE(cin >= 0);
    th.join();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

    REQUIRE(result.load() == 0);  // gave up: downstream never came up
    REQUIRE(ms >= 90);            // ~2 reconnect waits of 50 ms
    REQUIRE(ms < 4000);
    ::close(cin);  // lin and the accepted input fd are closed by serve_stage
}
