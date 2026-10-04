#include <poll.h>
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
#include "locus/model/transformer.hpp"
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
    const locus::model::TransformerModel& model,
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
    auto model = locus::model::TransformerModel::load(g);
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

// The multi-executor form (i#24 inc 3): two executors, two concurrent
// sessions. The loop pins each session to a different (least-loaded)
// executor, so they run on separate stages with separate KV caches. Each
// client must still get the SAME tokens a single-process run would --
// proving per-executor isolation and that session->executor routing
// (step and release) goes to the right lane.
TEST_CASE("serve_stage_mux spreads sessions across two executors",
          "[pipeline][mux][batch][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::uint32_t L = model.hparams().n_layers;
    const locus::tok::TokenId eos = tok.eos_id();
    constexpr int kGen = 8;

    const auto pa = tok.encode("Once upon a time, there was a little", true);
    const auto pb = tok.encode("The quick brown fox jumped over the", true);
    const auto ref_a = mono_generate(model, pa, eos, kGen);
    const auto ref_b = mono_generate(model, pb, eos, kGen);

    int pc = 0;
    const int lc = locus::pipeline::listen_on("127.0.0.1", 0, &pc);
    REQUIRE(lc >= 0);
    int ps = 0;
    const int ls = locus::pipeline::listen_on("127.0.0.1", 0, &ps);
    REQUIRE(ls >= 0);

    // Two stages -> two executors. The two concurrent sessions should be
    // assigned one each (least-loaded: the second accept sees exec 0 at
    // load 1, so it picks exec 1).
    PipelineStage stage0(model, 0, L);
    PipelineStage stage1(model, 0, L);
    std::vector<PipelineStage*> stages{&stage0, &stage1};
    const std::vector<locus::pipeline::Cidr> allow{
        *locus::pipeline::Cidr::parse("127.0.0.0/8")};
    locus::pipeline::StageConn conn;
    conn.serve_sessions = 2;

    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(
                   stages, ls, allow, {{"127.0.0.1", pc}}, conn)
                   ? 1
                   : 0;
    });

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

    const std::size_t pmax = std::max(pa.size(), pb.size());
    for (std::size_t i = 0; i < pmax; ++i) {
        if (i < pa.size()) {
            a.round_trip(pa[i]);
        }
        if (i < pb.size()) {
            b.round_trip(pb[i]);
        }
    }

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

    ::close(a.to_stage);
    ::close(b.to_stage);
    server.join();
    ::close(a.from_stage);
    ::close(b.from_stage);

    REQUIRE(a.ok);
    REQUIRE(b.ok);
    REQUIRE(gen_a == ref_a);
    REQUIRE(gen_b == ref_b);
    REQUIRE(sres.load() == 1);
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
    auto model = locus::model::TransformerModel::load(g);
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
    auto model = locus::model::TransformerModel::load(g);
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
    auto model = locus::model::TransformerModel::load(g);
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

// A downstream that does not read for a while must not block the loop or
// lose frames: the stage parks the backlog in its per-session outbuf and
// drains it on write-readiness, in order. The collector's receive buffer
// is shrunk and left unread while the client sends a burst far larger
// than any kernel socket buffer, so flush_out is forced to hit EAGAIN
// and watch out_fd for write (the new path); then the collector drains
// and every frame must arrive, in position order, none lost.
TEST_CASE("serve_stage_mux parks a backlog for a slow downstream",
          "[pipeline][mux][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;
    const std::uint32_t V = model.hparams().n_vocab;

    // Burst enough bytes to dwarf any kernel send+recv buffering (~1.5 MB
    // target), so the stage cannot have delivered them all without
    // parking a backlog -- that is what forces the EAGAIN/out-watch path.
    const std::size_t frame_bytes = 28 + static_cast<std::size_t>(V) * 4;
    int n = static_cast<int>((1536u * 1024u) / frame_bytes) + 1;
    n = std::max(400, std::min(n, 1200));
    const std::uint32_t blocks =
        static_cast<std::uint32_t>(n) / 16 + 8;  // KV room for n positions

    int pc = 0;
    const int lc = locus::pipeline::listen_on("127.0.0.1", 0, &pc);
    REQUIRE(lc >= 0);
    // Shrink the receive buffer (inherited by the accepted fd) so the TCP
    // window is small and the stage backpressures early.
    int rcv = 4096;
    ::setsockopt(lc, SOL_SOCKET, SO_RCVBUF, &rcv, sizeof(rcv));
    int ps = 0;
    const int ls = locus::pipeline::listen_on("127.0.0.1", 0, &ps);
    REQUIRE(ls >= 0);

    PipelineStage stage(model, 0, L, blocks);
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

    // Send the whole burst WITHOUT reading any logits back, so the
    // downstream (from_stage) stays unread and the stage's sends fill the
    // tiny window and start parking in outbuf.
    for (int i = 0; i < n; ++i) {
        REQUIRE(locus::pipeline::write_message(
            to_stage,
            locus::pipeline::make_token(1, static_cast<std::uint32_t>(i),
                                        1)));
    }

    // Now drain. Every frame must arrive, as kLogits, in position order
    // 0..n-1 -- proving the parked backlog flushed completely and in
    // order. (The bytes could not have fit in kernel buffers, so the
    // park path necessarily ran.)
    for (int i = 0; i < n; ++i) {
        Message lg;
        REQUIRE(locus::pipeline::read_message(from_stage, lg) ==
                ReadResult::kOk);
        REQUIRE(lg.type == MsgType::kLogits);
        REQUIRE(lg.position == static_cast<std::uint32_t>(i));
    }

    ::close(to_stage);
    server.join();
    ::close(from_stage);
    REQUIRE(sres.load() == 1);  // ended cleanly after the backlog drained
}

// A client that sends its last frame then half-closes its send side
// (shutdown SHUT_WR) -- the "no more input, now give me the answer"
// pattern -- still gets its answer. The stage sees the frame and a
// recv()==0 EOF in the same batch, and because the drop is honored only
// AFTER the batch's frames are processed and flushed, the logits are
// delivered before the session is torn down. This pins the load-bearing
// loop ordering: hoisting the drop above the frame loop would break this
// while leaving every other test green (none half-close).
TEST_CASE("serve_stage_mux answers a client that half-closes after a frame",
          "[pipeline][mux][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
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

    // One token, then half-close the send side: the stage must still
    // answer before honoring the resulting EOF.
    REQUIRE(locus::pipeline::write_message(
        to_stage, locus::pipeline::make_token(1, 0, 1)));
    REQUIRE(::shutdown(to_stage, SHUT_WR) == 0);

    Message lg;
    REQUIRE(locus::pipeline::read_message(from_stage, lg) ==
            ReadResult::kOk);
    REQUIRE(lg.type == MsgType::kLogits);  // answered despite the EOF

    ::close(to_stage);
    server.join();
    ::close(from_stage);
    REQUIRE(sres.load() == 1);  // ended cleanly (EOF at a frame boundary)
}

// ---- issue 39: non-blocking, poll-driven downstream connect ----

namespace {
// A bound-then-closed port: a connect there is refused -- synchronously
// on some stacks, as an async error event on others, both of which the
// dial-failure path must handle.
int refused_port() {
    int p = 0;
    const int l = locus::pipeline::listen_on("127.0.0.1", 0, &p);
    REQUIRE(l >= 0);
    ::close(l);
    return p;
}
}  // namespace

// dial_start returns immediately with the handshake in flight; the fd
// becomes writable and connect_result reports success. Runs for IPv4 and
// IPv6 loopback, because dial_start must build the right sockaddr family
// from the numeric literal (the resolver hands back v6 literals).
TEST_CASE("dial_start connects v4 and v6 loopback, poll-driven",
          "[net][connect]") {
    for (const char* host : {"127.0.0.1", "::1"}) {
        int port = 0;
        const int l = locus::pipeline::listen_on(host, 0, &port);
        if (l < 0) {
            continue;  // this family not available on the host
        }
        const int fd = locus::pipeline::dial_start(host, port);
        REQUIRE(fd >= 0);
        pollfd pfd{fd, POLLOUT, 0};
        REQUIRE(::poll(&pfd, 1, 2000) == 1);
        REQUIRE(locus::pipeline::connect_result(fd) == 0);
        ::close(fd);
        ::close(l);
    }
}

// A refused dial surfaces as a failure one way or the other: either
// dial_start returns -1 synchronously, or the fd goes writable/errored
// and connect_result returns non-zero. Never a silent success (the
// SO_ERROR read-once contract).
TEST_CASE("dial_start to a refused port reports failure, never success",
          "[net][connect]") {
    const int port = refused_port();
    const int fd = locus::pipeline::dial_start("127.0.0.1", port);
    if (fd < 0) {
        SUCCEED("refused synchronously");
        return;
    }
    pollfd pfd{fd, POLLOUT, 0};
    ::poll(&pfd, 1, 2000);
    REQUIRE(locus::pipeline::connect_result(fd) != 0);
    ::close(fd);
}

// A refused pool entry is retried onto a live one WITHIN the dial, with
// no blocking sleep: the session connects to the second entry and
// produces output byte-identical to the single-process run. Proves the
// dial walks the pool past a dead entry.
TEST_CASE("serve_stage_mux dials across the pool past a refused entry",
          "[pipeline][mux][connect][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::uint32_t L = model.hparams().n_layers;
    const locus::tok::TokenId eos = tok.eos_id();
    constexpr int kGen = 6;
    const auto prompt = tok.encode("Once upon a time", true);
    const auto ref = mono_generate(model, prompt, eos, kGen);

    int pc = 0;
    const int lc = locus::pipeline::listen_on("127.0.0.1", 0, &pc);
    REQUIRE(lc >= 0);
    int ps = 0;
    const int ls = locus::pipeline::listen_on("127.0.0.1", 0, &ps);
    REQUIRE(ls >= 0);
    const int dead = refused_port();

    PipelineStage stage(model, 0, L);
    const std::vector<locus::pipeline::Cidr> allow{
        *locus::pipeline::Cidr::parse("127.0.0.0/8")};
    locus::pipeline::StageConn conn;
    conn.serve_sessions = 1;

    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(
                   stage, ls, allow,
                   {{"127.0.0.1", dead}, {"127.0.0.1", pc}}, conn)
                   ? 1
                   : 0;
    });

    Client a;
    a.request_id = 7;
    a.to_stage = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(a.to_stage >= 0);
    // The stage dials `dead` (fails), then `pc` (this accept completes).
    a.from_stage = locus::pipeline::accept_one(lc, nullptr);
    REQUIRE(a.from_stage >= 0);
    ::close(lc);

    for (auto t : prompt) {
        a.round_trip(t);
    }
    std::vector<locus::tok::TokenId> gen;
    for (int i = 0; i < kGen && a.ok; ++i) {
        const auto n = locus::model::argmax(a.logits);
        gen.push_back(n);
        if (n == eos) {
            break;
        }
        a.round_trip(n);
    }
    ::close(a.to_stage);
    server.join();
    ::close(a.from_stage);

    REQUIRE(a.ok);
    REQUIRE(gen == ref);
    REQUIRE(sres.load() == 1);
}

// A session whose entire downstream pool is dead is DROPPED (per-session)
// and the server stays up and returns -- it is not fatal, and the failed
// dial does not busy-spin: a spin would never reach the drop, so
// serve_sessions == 1 would never trip and server.join() would hang. The
// join returning is the no-spin assertion.
TEST_CASE("serve_stage_mux drops a session with a dead downstream pool",
          "[pipeline][mux][connect][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;

    int ps = 0;
    const int ls = locus::pipeline::listen_on("127.0.0.1", 0, &ps);
    REQUIRE(ls >= 0);
    const int dead = refused_port();

    PipelineStage stage(model, 0, L);
    const std::vector<locus::pipeline::Cidr> allow{
        *locus::pipeline::Cidr::parse("127.0.0.0/8")};
    locus::pipeline::StageConn conn;
    conn.serve_sessions = 1;
    conn.reconnect_attempts = 1;  // one sweep, then drop
    conn.reconnect_wait_ms = 10;

    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(
                   stage, ls, allow, {{"127.0.0.1", dead}}, conn)
                   ? 1
                   : 0;
    });

    const int c = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(c >= 0);
    // The stage accepts, dials the dead pool, exhausts it, drops. Server
    // must RETURN (not hang) with an unclean result.
    server.join();
    REQUIRE(sres.load() == 0);
    ::close(c);
}

// Issue 39: a non-numeric downstream with NO resolver can never be dialed
// (the async dial needs a numeric address). The stage must detect that at
// accept and drop the session promptly with a cause-specific message --
// NOT burn reconnect_attempts x reconnect_wait_ms first, and NOT hang.
TEST_CASE("serve_stage_mux drops a hostname downstream with no resolver",
          "[pipeline][mux][connect][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;

    int ps = 0;
    const int ls = locus::pipeline::listen_on("127.0.0.1", 0, &ps);
    REQUIRE(ls >= 0);

    PipelineStage stage(model, 0, L);
    const std::vector<locus::pipeline::Cidr> allow{
        *locus::pipeline::Cidr::parse("127.0.0.0/8")};
    locus::pipeline::StageConn conn;  // conn.resolver stays nullptr
    conn.serve_sessions = 1;
    // Large retry budget so a hang here (treating it as transient) would
    // be obvious: the accept-time permanent-fail path must bypass it.
    conn.reconnect_attempts = 1000;
    conn.reconnect_wait_ms = 1000;

    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(
                   stage, ls, allow, {{"needs-a-resolver.invalid", 9}},
                   conn)
                   ? 1
                   : 0;
    });

    const int c = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(c >= 0);
    // Must return promptly (the big reconnect budget is never spent) with
    // an unclean result.
    server.join();
    REQUIRE(sres.load() == 0);
    ::close(c);
}
