#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <thread>
#include <vector>

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

// Greedy single-process generation of `n_gen` tokens after `prompt`,
// the reference a multiplexed session must reproduce exactly.
std::vector<locus::tok::TokenId> mono_generate(
    const locus::model::LlamaModel& model,
    std::span<const locus::tok::TokenId> prompt,
    locus::tok::TokenId eos, int n_gen) {
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    for (auto t : prompt) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(t, cache, seq, ws, logits);
    }
    std::vector<locus::tok::TokenId> gen;
    for (int i = 0; i < n_gen; ++i) {
        const auto n = locus::model::argmax(logits);
        gen.push_back(n);
        if (n == eos) {
            break;
        }
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(n, cache, seq, ws, logits);
    }
    return gen;
}

// Drives one multiplexed session one token at a time: writes a kToken
// to the stage and reads the kLogits the stage returns on this session's
// own downstream fd. Position advances per call.
struct Client {
    int to_stage = -1;   // write tokens here (into the stage)
    int from_stage = -1;  // read logits here (session's downstream)
    std::uint64_t request_id = 0;
    std::uint32_t pos = 0;
    std::vector<float> logits;
    bool ok = true;

    void round_trip(locus::tok::TokenId t) {
        if (!ok) {
            return;
        }
        if (!locus::pipeline::write_message(
                to_stage, locus::pipeline::make_token(request_id, pos, t))) {
            ok = false;
            return;
        }
        Message lg;
        if (locus::pipeline::read_message(from_stage, lg) !=
                ReadResult::kOk ||
            lg.type != MsgType::kLogits) {
            ok = false;
            return;
        }
        logits = std::move(lg.data);
        ++pos;
    }
};

}  // namespace

// The event-loop server (serve_stage_mux) must serve two clients at once
// over a single stage/thread, interleaving their traffic, and give each
// the SAME tokens a single-process run would -- proving the per-session
// KV sequences do not cross-contaminate. One all-in-one stage ([0, L))
// takes kToken and returns kLogits, so each client's own downstream fd
// carries only that client's logits (no request_id demux needed).
TEST_CASE("serve_stage_mux serves two concurrent sessions in isolation",
          "[pipeline][mux][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::uint32_t L = model.hparams().n_layers;
    const locus::tok::TokenId eos = tok.eos_id();
    constexpr int kGen = 8;

    // Two distinct prompts, so a crosstalk bug shows as a wrong token
    // rather than two identical streams that happen to agree.
    const auto pa = tok.encode("Once upon a time, there was a little", true);
    const auto pb = tok.encode("The quick brown fox jumped over the", true);
    const auto ref_a = mono_generate(model, pa, eos, kGen);
    const auto ref_b = mono_generate(model, pb, eos, kGen);

    // Logits collector: the stage's downstream. Each accepted session
    // dials it, so it hands back one dedicated fd per client.
    int pc = 0;
    const int lc = locus::pipeline::listen_on("127.0.0.1", 0, &pc);
    REQUIRE(lc >= 0);
    // The stage's own listener, where clients send tokens.
    int ps = 0;
    const int ls = locus::pipeline::listen_on("127.0.0.1", 0, &ps);
    REQUIRE(ls >= 0);

    PipelineStage stage(model, 0, L);
    const std::vector<locus::pipeline::Cidr> allow{
        *locus::pipeline::Cidr::parse("127.0.0.0/8")};
    locus::pipeline::StageConn conn;
    conn.serve_sessions = 2;  // stop after both sessions end

    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(
                   stage, ls, allow, {{"127.0.0.1", pc}}, conn)
                   ? 1
                   : 0;
    });

    // Bring up A fully (connect, then let the stage dial A's downstream)
    // before B, so the collector's two accepts pair deterministically
    // with the two clients.
    Client a;
    a.request_id = 101;
    a.to_stage = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(a.to_stage >= 0);
    a.from_stage = locus::pipeline::accept_one(lc, nullptr);
    REQUIRE(a.from_stage >= 0);

    Client b;
    b.request_id = 202;
    b.to_stage = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(b.to_stage >= 0);
    b.from_stage = locus::pipeline::accept_one(lc, nullptr);
    REQUIRE(b.from_stage >= 0);
    ::close(lc);

    // Feed both prompts interleaved, so both sessions are live at the
    // stage at the same time (the actual concurrency being tested).
    const std::size_t pmax = std::max(pa.size(), pb.size());
    for (std::size_t i = 0; i < pmax; ++i) {
        if (i < pa.size()) {
            a.round_trip(pa[i]);
        }
        if (i < pb.size()) {
            b.round_trip(pb[i]);
        }
    }

    // Generate interleaved, each client greedy over its own logits.
    std::vector<locus::tok::TokenId> gen_a, gen_b;
    bool a_done = false, b_done = false;
    for (int i = 0; i < kGen; ++i) {
        if (!a_done && a.ok) {
            const auto n = locus::model::argmax(a.logits);
            gen_a.push_back(n);
            if (n == eos) {
                a_done = true;
            } else {
                a.round_trip(n);
            }
        }
        if (!b_done && b.ok) {
            const auto n = locus::model::argmax(b.logits);
            gen_b.push_back(n);
            if (n == eos) {
                b_done = true;
            } else {
                b.round_trip(n);
            }
        }
    }

    ::close(a.to_stage);  // EOF ends session A
    ::close(b.to_stage);  // EOF ends session B
    server.join();
    ::close(a.from_stage);
    ::close(b.from_stage);

    REQUIRE(a.ok);
    REQUIRE(b.ok);
    REQUIRE(gen_a == ref_a);  // A's output untouched by B's traffic
    REQUIRE(gen_b == ref_b);  // and vice versa
    REQUIRE(sres.load() == 1);  // both sessions ended cleanly
}

// A frame split across two writes must be reassembled from the
// per-connection buffer, not dropped: this is the decode()-buffer path
// the blocking read_message never exercised. Write a token frame's bytes
// in two halves with a gap, and the stage must still answer.
TEST_CASE("serve_stage_mux reassembles a frame split across reads",
          "[pipeline][mux][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;

    int pc = 0;
    const int lc = locus::pipeline::listen_on("127.0.0.1", 0, &pc);
    REQUIRE(lc >= 0);
    int ps = 0;
    const int ls = locus::pipeline::listen_on("127.0.0.1", 0, &ps);
    REQUIRE(ls >= 0);

    PipelineStage stage(model, 0, L);
    locus::pipeline::StageConn conn;
    conn.serve_sessions = 1;
    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(
                   stage, ls, {}, {{"127.0.0.1", pc}}, conn)
                   ? 1
                   : 0;
    });

    const int to_stage = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(to_stage >= 0);
    const int from_stage = locus::pipeline::accept_one(lc, nullptr);
    REQUIRE(from_stage >= 0);
    ::close(lc);

    // Encode one kToken frame and send it in two halves, with a pause so
    // the stage's recv() returns the first half alone and must buffer it.
    std::string frame;
    locus::pipeline::encode(locus::pipeline::make_token(7, 0, 1), frame);
    REQUIRE(frame.size() > 4);
    const std::size_t half = frame.size() / 2;
    REQUIRE(locus::pipeline::write_all(
        to_stage, std::span<const char>(frame.data(), half)));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    REQUIRE(locus::pipeline::write_all(
        to_stage,
        std::span<const char>(frame.data() + half, frame.size() - half)));

    Message lg;
    REQUIRE(locus::pipeline::read_message(from_stage, lg) ==
            ReadResult::kOk);
    REQUIRE(lg.type == MsgType::kLogits);  // reassembled and answered

    ::close(to_stage);
    server.join();
    ::close(from_stage);
    REQUIRE(sres.load() == 1);
}

// The drop contract's clean-vs-truncated distinction (mirroring
// read_message): a peer that closes at a frame boundary ends the session
// cleanly (serve returns true), while a peer that closes mid-frame leaves
// a half-read buffer and ends unclean (serve returns false).
TEST_CASE("serve_stage_mux: EOF at a boundary is clean, mid-frame is not",
          "[pipeline][mux][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;

    auto run_once = [&](bool truncate) {
        int pc = 0;
        const int lc = locus::pipeline::listen_on("127.0.0.1", 0, &pc);
        REQUIRE(lc >= 0);
        int ps = 0;
        const int ls = locus::pipeline::listen_on("127.0.0.1", 0, &ps);
        REQUIRE(ls >= 0);

        PipelineStage stage(model, 0, L);
        locus::pipeline::StageConn conn;
        conn.serve_sessions = 1;
        std::atomic<int> sres{-1};
        std::thread server([&] {
            sres = locus::pipeline::serve_stage_mux(
                       stage, ls, {}, {{"127.0.0.1", pc}}, conn)
                       ? 1
                       : 0;
        });

        const int to_stage = locus::pipeline::connect_to("127.0.0.1", ps);
        REQUIRE(to_stage >= 0);
        const int from_stage = locus::pipeline::accept_one(lc, nullptr);
        REQUIRE(from_stage >= 0);
        ::close(lc);

        std::string frame;
        locus::pipeline::encode(locus::pipeline::make_token(1, 0, 1),
                                frame);
        if (truncate) {
            // Send only the first few bytes, then close: a partial frame.
            REQUIRE(locus::pipeline::write_all(
                to_stage, std::span<const char>(frame.data(), 4)));
        } else {
            // Send one whole frame, read its reply, then close cleanly at
            // the boundary.
            REQUIRE(locus::pipeline::write_all(
                to_stage,
                std::span<const char>(frame.data(), frame.size())));
            Message lg;
            REQUIRE(locus::pipeline::read_message(from_stage, lg) ==
                    ReadResult::kOk);
        }
        ::close(to_stage);
        server.join();
        ::close(from_stage);
        return sres.load();
    };

    REQUIRE(run_once(/*truncate=*/false) == 1);  // clean boundary EOF
    REQUIRE(run_once(/*truncate=*/true) == 0);    // mid-frame truncation
}

// A peer that RSTs while idle -- no pending write for the stage to fail
// on -- must still be dropped promptly and counted unclean. The drop
// comes only from the unconditional recv() draining to -1/ECONNRESET,
// never from a flag. This is the shape where the two backends diverge
// most (epoll: error=1, readable=0; kqueue: readable=1, error=0), so the
// drop is reached by a different poller state on each while the outcome
// -- serve returns false -- is the same. Covers the read_error half of
// the clean predicate from the peer-RST direction, which the other
// cases do not.
TEST_CASE("serve_stage_mux drops an idle peer that resets the connection",
          "[pipeline][mux][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;

    int pc = 0;
    const int lc = locus::pipeline::listen_on("127.0.0.1", 0, &pc);
    REQUIRE(lc >= 0);
    int ps = 0;
    const int ls = locus::pipeline::listen_on("127.0.0.1", 0, &ps);
    REQUIRE(ls >= 0);

    PipelineStage stage(model, 0, L);
    locus::pipeline::StageConn conn;
    conn.serve_sessions = 1;
    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(
                   stage, ls, {}, {{"127.0.0.1", pc}}, conn)
                   ? 1
                   : 0;
    });

    const int to_stage = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(to_stage >= 0);
    // Abortive close: with SO_LINGER timeout 0, close() sends a RST
    // instead of a FIN, so the stage's input socket errors rather than
    // seeing a clean EOF.
    struct linger lg;
    lg.l_onoff = 1;
    lg.l_linger = 0;
    REQUIRE(::setsockopt(to_stage, SOL_SOCKET, SO_LINGER, &lg,
                         sizeof(lg)) == 0);
    // Let the stage accept and dial back, so the session is fully
    // established and idle before the reset (nothing is ever written to
    // it, so a failed write can never be what notices the drop).
    const int from_stage = locus::pipeline::accept_one(lc, nullptr);
    REQUIRE(from_stage >= 0);
    ::close(lc);

    ::close(to_stage);  // RST to the stage's input fd
    server.join();
    ::close(from_stage);
    REQUIRE(sres.load() == 0);  // dropped, counted unclean
}
