#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
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
        // One session then return (serve_stage otherwise serves
        // forever, re-accepting after each disconnect).
        locus::pipeline::StageConn conn;
        conn.serve_sessions = 1;

        std::atomic<int> r0{-1}, r1{-1};
        std::thread t0([&] {
            r0 = locus::pipeline::serve_stage(s0, l0, allow,
                                              "127.0.0.1", p1, conn)
                     ? 1
                     : 0;
        });
        std::thread t1([&] {
            r1 = locus::pipeline::serve_stage(s1, l1, allow,
                                              "127.0.0.1", pe, conn)
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

// #15: serve_stage loops to re-serve successive sessions (automatic
// reconnect), and PipelineStage::reset() gives each session a clean
// sequence -- so a second client after the first disconnects gets a
// byte-exact result, not one contaminated by the first session's KV.
TEST_CASE("serve_stage re-serves sessions with reset",
          "[pipeline][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::uint32_t L = model.hparams().n_layers;
    const std::uint32_t V = model.hparams().n_vocab;
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

    // One stage covering all layers (first + last): input kToken,
    // output kLogits back to the entry.
    PipelineStage stage(model, 0, L);
    int pin = 0, pe = 0;
    const int lin = locus::pipeline::listen_on("127.0.0.1", 0, &pin);
    const int le = locus::pipeline::listen_on("127.0.0.1", 0, &pe);
    REQUIRE(lin >= 0);
    REQUIRE(le >= 0);

    locus::pipeline::StageConn conn;
    conn.serve_sessions = 2;
    conn.connect_timeout_ms = 1000;
    conn.reconnect_wait_ms = 50;
    std::atomic<int> result{-1};
    std::thread th([&] {
        result = locus::pipeline::serve_stage(stage, lin, {},
                                              "127.0.0.1", pe, conn)
                     ? 1
                     : 0;
    });

    for (int session = 0; session < 2; ++session) {
        const int w = locus::pipeline::connect_to("127.0.0.1", pin);
        REQUIRE(w >= 0);  // stage accepts this as input
        const int r = locus::pipeline::accept_one(le, nullptr);
        REQUIRE(r >= 0);  // stage connected downstream -> entry accepts

        bool flow = true;
        std::uint32_t pos = 0;
        std::vector<float> logits;
        auto round_trip = [&](locus::tok::TokenId t) {
            const Message in = locus::pipeline::make_token(1, pos, t);
            if (!locus::pipeline::write_message(w, in)) {
                flow = false;
                return;
            }
            Message lg;
            if (locus::pipeline::read_message(r, lg) !=
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
        ::close(w);  // EOF ends the session; serve_stage re-accepts
        ::close(r);
        CAPTURE(session);
        REQUIRE(flow);
        REQUIRE(gen == ref);  // reset() -> clean sequence each session
    }

    th.join();
    ::close(le);
    REQUIRE(result.load() == 1);  // served 2 sessions, then returned
}

// #16: serve_stage takes a POOL of interchangeable downstream replicas
// (no load balancer). Each connect sweeps the pool and uses the first
// replica that accepts, so a dead one is skipped (an established
// connection is the health check); and successive sessions round-robin
// across the pool, so work spreads across the live replicas. Both are
// proved end-to-end: the chain still reproduces single-process
// generation byte-for-byte.
TEST_CASE("serve_stage distributes across a downstream pool",
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
    const std::uint32_t k = 1;  // stage 0 = layer 0; downstream = [1, L)

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

    const std::vector<locus::pipeline::Cidr> allow{
        *locus::pipeline::Cidr::parse("127.0.0.0/8")};

    // Drives one session over (w -> stage 0 input, r <- logits out).
    // @returns {flow_ok, generated tokens}.
    auto run_session = [&](int w, int r) {
        bool flow = true;
        std::uint32_t pos = 0;
        std::vector<float> logits;
        auto round_trip = [&](locus::tok::TokenId t) {
            const Message in = locus::pipeline::make_token(1, pos, t);
            if (!locus::pipeline::write_message(w, in)) {
                flow = false;
                return;
            }
            Message lg;
            if (locus::pipeline::read_message(r, lg) !=
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
        return std::make_pair(flow, gen);
    };

    SECTION("skips a dead replica and uses a live one") {
        int p0 = 0, p1 = 0, pe = 0;
        const int l0 = locus::pipeline::listen_on("127.0.0.1", 0, &p0);
        const int l1 = locus::pipeline::listen_on("127.0.0.1", 0, &p1);
        const int le = locus::pipeline::listen_on("127.0.0.1", 0, &pe);
        REQUIRE(l0 >= 0);
        REQUIRE(l1 >= 0);
        REQUIRE(le >= 0);
        // A downstream port with nothing listening (reserve, release).
        int pdead = 0;
        {
            const int t =
                locus::pipeline::listen_on("127.0.0.1", 0, &pdead);
            REQUIRE(t >= 0);
            ::close(t);
        }

        PipelineStage s0(model, 0, k);
        PipelineStage s1(model, k, L);
        locus::pipeline::StageConn conn0;
        conn0.serve_sessions = 1;
        conn0.connect_timeout_ms = 300;  // the dead one fails fast
        conn0.reconnect_wait_ms = 50;
        // Dead first, then the live replica -> the sweep must skip it.
        const std::vector<locus::pipeline::HostPort> pool{
            {"127.0.0.1", pdead}, {"127.0.0.1", p1}};
        locus::pipeline::StageConn conn1;
        conn1.serve_sessions = 1;

        std::atomic<int> r0{-1}, r1{-1};
        std::thread t0([&] {
            r0 = locus::pipeline::serve_stage(s0, l0, allow, pool, conn0)
                     ? 1
                     : 0;
        });
        std::thread t1([&] {
            r1 = locus::pipeline::serve_stage(s1, l1, allow,
                                              "127.0.0.1", pe, conn1)
                     ? 1
                     : 0;
        });

        const int entry_w = locus::pipeline::connect_to("127.0.0.1", p0);
        REQUIRE(entry_w >= 0);
        const int entry_r = locus::pipeline::accept_one(le, nullptr);
        REQUIRE(entry_r >= 0);
        ::close(le);

        const auto [flow, gen] = run_session(entry_w, entry_r);
        ::close(entry_w);
        t0.join();
        t1.join();
        ::close(entry_r);
        REQUIRE(flow);
        REQUIRE(r0.load() == 1);
        REQUIRE(r1.load() == 1);
        REQUIRE(gen == ref);  // dead replica skipped, live one served
    }

    SECTION("round-robins two sessions across two live replicas") {
        int p0 = 0, pa = 0, pb = 0, pe = 0;
        const int l0 = locus::pipeline::listen_on("127.0.0.1", 0, &p0);
        const int la = locus::pipeline::listen_on("127.0.0.1", 0, &pa);
        const int lb = locus::pipeline::listen_on("127.0.0.1", 0, &pb);
        const int le = locus::pipeline::listen_on("127.0.0.1", 0, &pe);
        REQUIRE(l0 >= 0);
        REQUIRE(la >= 0);
        REQUIRE(lb >= 0);
        REQUIRE(le >= 0);

        PipelineStage s0(model, 0, k);
        PipelineStage sa(model, k, L);  // replica A of [k, L)
        PipelineStage sb(model, k, L);  // replica B of [k, L)
        locus::pipeline::StageConn conn0;
        conn0.serve_sessions = 2;
        conn0.connect_timeout_ms = 1000;
        conn0.reconnect_wait_ms = 50;
        const std::vector<locus::pipeline::HostPort> pool{
            {"127.0.0.1", pa}, {"127.0.0.1", pb}};
        locus::pipeline::StageConn conn1;
        conn1.serve_sessions = 1;  // each replica serves exactly one

        std::atomic<int> r0{-1}, ra{-1}, rb{-1};
        std::thread t0([&] {
            r0 = locus::pipeline::serve_stage(s0, l0, allow, pool, conn0)
                     ? 1
                     : 0;
        });
        std::thread ta([&] {
            ra = locus::pipeline::serve_stage(sa, la, allow,
                                              "127.0.0.1", pe, conn1)
                     ? 1
                     : 0;
        });
        std::thread tb([&] {
            rb = locus::pipeline::serve_stage(sb, lb, allow,
                                              "127.0.0.1", pe, conn1)
                     ? 1
                     : 0;
        });

        for (int session = 0; session < 2; ++session) {
            const int w = locus::pipeline::connect_to("127.0.0.1", p0);
            REQUIRE(w >= 0);
            const int r = locus::pipeline::accept_one(le, nullptr);
            REQUIRE(r >= 0);
            const auto [flow, gen] = run_session(w, r);
            ::close(w);
            ::close(r);
            CAPTURE(session);
            REQUIRE(flow);
            REQUIRE(gen == ref);
        }

        t0.join();
        ta.join();
        tb.join();
        ::close(le);
        REQUIRE(r0.load() == 1);  // stage 0 served both sessions cleanly
        // Both replicas returned, so each served exactly one session:
        // if stage 0 had not round-robined, one replica's serve_stage
        // (serve_sessions == 1) would still be blocked in accept.
        REQUIRE(ra.load() == 1);
        REQUIRE(rb.load() == 1);
    }
}

// #18 slice-only loading: a model loaded for only [k, L) wires up just
// those transformer blocks (the rest stay default-constructed and never
// fault in), and running its slice against a slice-sized, base-remapped
// KV cache produces the SAME logits as the full model executing the
// same [k, L) slice. This is the unit-level guard behind the
// multi-process [mp] byte-exactness: it isolates exactly what slice
// loading changes (which layers load + the cache height/base) by
// feeding both models an identical synthetic residual.
TEST_CASE("slice-only load matches full model on its layer range",
          "[pipeline]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto full = locus::model::LlamaModel::load(g);
    const std::uint32_t L = full.hparams().n_layers;
    const std::uint32_t V = full.hparams().n_vocab;
    const std::uint32_t E = full.hparams().n_embd;
    REQUIRE(L >= 2);
    const std::uint32_t k = L / 2 > 0 ? L / 2 : 1;

    auto slice = locus::model::LlamaModel::load(g, k, L);
    // The slice records its range; hparams().n_layers stays the full
    // count; layers_ stays absolute-indexed with only [k, L) wired up.
    REQUIRE(slice.layer_begin() == k);
    REQUIRE(slice.layer_end() == L);
    REQUIRE(slice.hparams().n_layers == L);
    REQUIRE(slice.layers().size() == L);
    REQUIRE(slice.layers()[0].attn_norm.empty());       // not loaded
    REQUIRE(slice.layers()[k].attn_norm.size() == E);    // loaded
    REQUIRE(slice.layers()[L - 1].attn_norm.size() == E);

    // Drive many positions with an identical deterministic residual
    // stream (what the previous stage would hand over). Spanning several
    // 16-token cache blocks exercises the block stride -- the term whose
    // SIZE the slice changes -- not just the per-layer offset, so a
    // remap slip surfaces here rather than only in the [mp] end-to-end.
    constexpr std::uint32_t kPos = 40;  // > 2 blocks at block_tokens 16
    auto run_tail = [&](locus::model::LlamaModel& m) {
        auto cache = m.make_cache();
        auto ws = m.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<std::vector<float>> all;
        for (std::uint32_t p = 0; p < kPos; ++p) {
            std::vector<float> hidden(E);
            for (std::uint32_t i = 0; i < E; ++i) {
                hidden[i] = 0.01f * static_cast<float>(
                                        static_cast<int>((i + p) % 7) - 3);
            }
            REQUIRE(cache.ensure_capacity(seq, 1));
            std::vector<float> logits(V);
            m.forward_layers(0, hidden, k, L, cache, seq, ws, logits);
            all.push_back(std::move(logits));
        }
        return all;
    };
    const auto lf = run_tail(full);   // full cache, base 0
    const auto ls = run_tail(slice);  // sliced cache (base k, height L-k)
    REQUIRE(ls == lf);  // identical across every position and block

    // The slice must refuse a range it did not load (layer 0 is absent).
    auto cache = slice.make_cache();
    auto ws = slice.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> out(V);
    REQUIRE(cache.ensure_capacity(seq, 1));
    REQUIRE_THROWS_AS(
        slice.forward_layers(0, {}, 0, L, cache, seq, ws, out),
        std::invalid_argument);
}
