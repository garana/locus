#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
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
#include "locus/pipeline/resolver.hpp"
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

// Issue 76: a SESSION_END frame ends a sequence but KEEPS the connection
// open; the next request_id reuses the SAME inbound fd with a fresh KV
// seq. The reused sequence must produce exactly what it would on a fresh
// connection -- proving the reset released the first sequence's KV and
// did not carry it into the second. A broken reset is silent cross-
// session KV corruption, which is the hazard the i#68 downstream pool
// would introduce without this receive-side seam.
TEST_CASE("serve_stage_mux reuses a connection after SESSION_END, "
          "KV-isolated", "[pipeline][mux][e2e]") {
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

    PipelineStage stage(model, 0, L);
    const std::vector<locus::pipeline::Cidr> allow{
        *locus::pipeline::Cidr::parse("127.0.0.0/8")};
    locus::pipeline::StageConn conn;
    conn.serve_sessions = 1;  // the one reused connection ends once (after B)
    // Isolate the RECEIVE-side reset (i#76): with downstream pooling off
    // (issue 68), reset_session closes A's downstream and dials a fresh
    // one for B, so each sequence is a fresh accept on the collector --
    // the behaviour this test was written against. The dial-side reuse
    // that pooling adds (same downstream fd serves both sequences) is
    // covered by the issue-68 tests below, which drain the SESSION_END
    // frame the stage sends on a pooled connection.
    conn.downstream_idle_max = 0;

    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(
                   stage, ls, allow, {{"127.0.0.1", pc}}, conn)
                   ? 1
                   : 0;
    });

    // ONE upstream connection, reused for both sequences.
    const int to_stage = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(to_stage >= 0);

    // Runs a full sequence (prompt + greedy gen) over the shared
    // connection; each sequence's downstream is the stage's dial, so it is
    // a fresh accept on the collector. @returns the generated tokens.
    auto run_seq = [&](std::uint64_t rid,
                       const std::vector<locus::tok::TokenId>& prompt,
                       int& from_out) {
        Client c;
        c.to_stage = to_stage;  // the shared, reused inbound connection
        c.request_id = rid;
        c.from_stage = locus::pipeline::accept_one(lc, nullptr);
        REQUIRE(c.from_stage >= 0);
        from_out = c.from_stage;
        for (const auto t : prompt) {
            c.round_trip(t);
        }
        std::vector<locus::tok::TokenId> gen;
        for (int i = 0; i < kGen && c.ok; ++i) {
            const auto n = locus::model::argmax(c.logits);
            gen.push_back(n);
            if (n == eos) {
                break;
            }
            c.round_trip(n);
        }
        REQUIRE(c.ok);
        return gen;
    };

    int a_from = -1, b_from = -1;
    const auto gen_a = run_seq(101, pa, a_from);
    REQUIRE(gen_a == ref_a);

    // End sequence A but keep the connection: the stage releases A's KV,
    // closes A's downstream, and re-dials a fresh one for the next.
    REQUIRE(locus::pipeline::write_message(
        to_stage, locus::pipeline::make_session_end(101)));

    // Sequence B on the SAME connection, a new request_id.
    const auto gen_b = run_seq(202, pb, b_from);
    REQUIRE(gen_b == ref_b);  // fresh KV: B did not inherit A's cache

    ::close(to_stage);  // ends the reused connection -> server stops
    server.join();
    if (a_from >= 0) {
        ::close(a_from);
    }
    if (b_from >= 0) {
        ::close(b_from);
    }
    ::close(lc);
    REQUIRE(sres.load() == 1);  // the connection's final session ended clean
}

// ---- issue 68: dial-side downstream connection pool ----

// With the pool ON (the default), a SESSION_END that ends a sequence
// returns the stage's downstream fd to the idle pool and the next
// sequence on the SAME inbound connection REUSES it rather than dialing
// fresh. The reused sequence must stay KV-isolated and byte-identical to
// a fresh dial, and the collector must see exactly one downstream accept
// across both sequences (no second dial). On the wire the reuse shows as
// a SESSION_END frame -- echoing the forwarded request_id -- arriving on
// the same downstream fd between A's logits and B's.
TEST_CASE("serve_stage_mux reuses a pooled downstream (issue 68)",
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

    PipelineStage stage(model, 0, L);
    const std::vector<locus::pipeline::Cidr> allow{
        *locus::pipeline::Cidr::parse("127.0.0.0/8")};
    locus::pipeline::StageConn conn;
    conn.serve_sessions = 1;       // one inbound connection, reused for B
    conn.downstream_idle_max = 8;  // pooling ON (also the default)

    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(
                   stage, ls, allow, {{"127.0.0.1", pc}}, conn)
                   ? 1
                   : 0;
    });

    const int to_stage = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(to_stage >= 0);

    // The stage dials its downstream once, for sequence A. That one fd
    // carries A's logits, then the pool SESSION_END, then B's logits.
    const int from_stage = locus::pipeline::accept_one(lc, nullptr, 5000);
    REQUIRE(from_stage >= 0);
    locus::pipeline::set_recv_timeout(from_stage, 5000);

    Client a;
    a.to_stage = to_stage;
    a.from_stage = from_stage;
    a.request_id = 101;
    for (const auto t : pa) {
        a.round_trip(t);
    }
    std::vector<locus::tok::TokenId> gen_a;
    for (int i = 0; i < kGen && a.ok; ++i) {
        const auto n = locus::model::argmax(a.logits);
        gen_a.push_back(n);
        if (n == eos) {
            break;
        }
        a.round_trip(n);
    }
    REQUIRE(a.ok);
    REQUIRE(gen_a == ref_a);

    // End sequence A. The stage pools (does not close) its downstream and
    // reuses it for B, so a SESSION_END echoing A's request_id arrives on
    // the SAME downstream fd before any of B's logits.
    REQUIRE(locus::pipeline::write_message(
        to_stage, locus::pipeline::make_session_end(101)));
    Message se;
    REQUIRE(locus::pipeline::read_message(from_stage, se) ==
            ReadResult::kOk);
    REQUIRE(se.type == MsgType::kSessionEnd);
    REQUIRE(se.request_id == 101);  // the forwarded id, echoed back

    // No fresh dial happened: the collector gets no second accept.
    REQUIRE(locus::pipeline::accept_one(lc, nullptr, 500) < 0);

    // Sequence B over the reused downstream fd: byte-identical to a fresh
    // dial and KV-isolated from A (a leaked A seq would corrupt these).
    Client b;
    b.to_stage = to_stage;
    b.from_stage = from_stage;  // the SAME fd -- the reused downstream
    b.request_id = 202;
    for (const auto t : pb) {
        b.round_trip(t);
    }
    std::vector<locus::tok::TokenId> gen_b;
    for (int i = 0; i < kGen && b.ok; ++i) {
        const auto n = locus::model::argmax(b.logits);
        gen_b.push_back(n);
        if (n == eos) {
            break;
        }
        b.round_trip(n);
    }
    REQUIRE(b.ok);
    REQUIRE(gen_b == ref_b);

    ::close(to_stage);  // ends the reused connection -> server stops
    server.join();
    ::close(from_stage);
    ::close(lc);
    REQUIRE(sres.load() == 1);
}

// A downstream pooled by one client and found DEAD when the next client
// checks it out (the peer closed while it sat idle) must degrade to a
// fresh dial, never a failure. Client 1 ends (its downstream is pooled);
// the collector then closes that downstream; client 2 dials, the stage's
// checkout probe sees the dead fd, discards it, and dials fresh, so the
// collector gets a SECOND accept and client 2 is served normally. This is
// the cross-client reuse path (free_session), and the "stale pooled fd
// always degrades to a fresh dial" invariant.
TEST_CASE("serve_stage_mux replaces a dead pooled downstream (issue 68)",
          "[pipeline][mux][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::uint32_t L = model.hparams().n_layers;
    const locus::tok::TokenId eos = tok.eos_id();
    constexpr int kGen = 6;

    const auto prompt = tok.encode("Once upon a time, there was a", true);
    const auto ref = mono_generate(model, prompt, eos, kGen);

    int pc = 0;
    const int lc = locus::pipeline::listen_on("127.0.0.1", 0, &pc);
    REQUIRE(lc >= 0);
    int ps = 0;
    const int ls = locus::pipeline::listen_on("127.0.0.1", 0, &ps);
    REQUIRE(ls >= 0);

    PipelineStage stage(model, 0, L);
    const std::vector<locus::pipeline::Cidr> allow{
        *locus::pipeline::Cidr::parse("127.0.0.0/8")};
    locus::pipeline::StageConn conn;
    conn.serve_sessions = 2;       // two separate inbound connections
    conn.downstream_idle_max = 8;  // pooling ON

    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(
                   stage, ls, allow, {{"127.0.0.1", pc}}, conn)
                   ? 1
                   : 0;
    });

    // Runs one client to completion over its own inbound connection and
    // downstream accept; returns the collector-side downstream fd (left
    // open) so the caller can inspect/kill it.
    auto run_client = [&](std::uint64_t rid, int& from_out) {
        const int to = locus::pipeline::connect_to("127.0.0.1", ps);
        REQUIRE(to >= 0);
        const int from = locus::pipeline::accept_one(lc, nullptr, 5000);
        REQUIRE(from >= 0);
        locus::pipeline::set_recv_timeout(from, 5000);
        Client c;
        c.to_stage = to;
        c.from_stage = from;
        c.request_id = rid;
        for (const auto t : prompt) {
            c.round_trip(t);
        }
        std::vector<locus::tok::TokenId> gen;
        for (int i = 0; i < kGen && c.ok; ++i) {
            const auto n = locus::model::argmax(c.logits);
            gen.push_back(n);
            if (n == eos) {
                break;
            }
            c.round_trip(n);
        }
        REQUIRE(c.ok);
        from_out = from;
        return std::make_pair(to, gen);
    };

    // Client 1: run it, then end its inbound connection so its downstream
    // is pooled. free_session sends a SESSION_END on that downstream as it
    // pools it; reading that frame confirms the pool actually holds the fd
    // BEFORE we kill it (so client 2 is guaranteed to probe a dead entry,
    // not an empty pool).
    int c1_from = -1;
    auto [c1_to, gen1] = run_client(101, c1_from);
    REQUIRE(gen1 == ref);
    ::close(c1_to);  // clean EOF -> free_session pools the downstream
    Message se;
    REQUIRE(locus::pipeline::read_message(c1_from, se) == ReadResult::kOk);
    REQUIRE(se.type == MsgType::kSessionEnd);  // pooled, confirmed
    ::close(c1_from);  // the pooled peer dies while idle

    // Client 2: the stage checks the pooled fd out, probes it, finds it
    // dead, closes it, and dials fresh -- so a SECOND accept arrives and
    // the generation still matches the reference.
    int c2_from = -1;
    auto [c2_to, gen2] = run_client(202, c2_from);
    REQUIRE(gen2 == ref);  // fresh dial, correct output

    ::close(c2_to);
    server.join();
    ::close(c2_from);
    ::close(lc);
    REQUIRE(sres.load() == 1);  // both inbound connections ended clean
}

// Downstreams still parked in the idle pool when the server stops are
// closed at teardown (no fd leak). One client runs a sequence and
// disconnects, so its downstream is pooled; when the server returns, the
// collector sees that connection close (kEof) right after the SESSION_END
// the pool handshake sent -- proving the teardown path closes idle fds.
TEST_CASE("serve_stage_mux closes pooled downstreams at teardown (i#68)",
          "[pipeline][mux][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const std::uint32_t L = model.hparams().n_layers;
    const locus::tok::TokenId eos = tok.eos_id();
    constexpr int kGen = 6;

    const auto prompt = tok.encode("The quick brown fox jumped", true);
    const auto ref = mono_generate(model, prompt, eos, kGen);

    int pc = 0;
    const int lc = locus::pipeline::listen_on("127.0.0.1", 0, &pc);
    REQUIRE(lc >= 0);
    int ps = 0;
    const int ls = locus::pipeline::listen_on("127.0.0.1", 0, &ps);
    REQUIRE(ls >= 0);

    PipelineStage stage(model, 0, L);
    const std::vector<locus::pipeline::Cidr> allow{
        *locus::pipeline::Cidr::parse("127.0.0.0/8")};
    locus::pipeline::StageConn conn;
    conn.serve_sessions = 1;
    conn.downstream_idle_max = 8;  // pooling ON

    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(
                   stage, ls, allow, {{"127.0.0.1", pc}}, conn)
                   ? 1
                   : 0;
    });

    const int to_stage = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(to_stage >= 0);
    const int from_stage = locus::pipeline::accept_one(lc, nullptr, 5000);
    REQUIRE(from_stage >= 0);
    locus::pipeline::set_recv_timeout(from_stage, 5000);

    Client c;
    c.to_stage = to_stage;
    c.from_stage = from_stage;
    c.request_id = 303;
    for (const auto t : prompt) {
        c.round_trip(t);
    }
    std::vector<locus::tok::TokenId> gen;
    for (int i = 0; i < kGen && c.ok; ++i) {
        const auto n = locus::model::argmax(c.logits);
        gen.push_back(n);
        if (n == eos) {
            break;
        }
        c.round_trip(n);
    }
    REQUIRE(c.ok);
    REQUIRE(gen == ref);

    ::close(to_stage);  // clean EOF -> free_session pools the downstream
    server.join();      // serve_sessions reached -> teardown runs

    // On the pooled downstream: the SESSION_END from the pool handshake,
    // then a clean EOF as teardown closes the idle fd.
    Message se;
    REQUIRE(locus::pipeline::read_message(from_stage, se) ==
            ReadResult::kOk);
    REQUIRE(se.type == MsgType::kSessionEnd);
    REQUIRE(locus::pipeline::read_message(from_stage, se) ==
            ReadResult::kEof);  // teardown closed the pooled fd
    ::close(from_stage);
    ::close(lc);
    REQUIRE(sres.load() == 1);
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

// Issue 39: a reload can replace the pool wholesale, and build_runtime
// accepts a config with no downstream directives, so pool_live can be
// empty at accept even though the initial pool was not. The accept path
// must drop the session rather than reach `% pool_live.size()`: on
// x86-64 that raises SIGFPE, and on arm64 (where UDIV by zero yields 0)
// it falls into try_dial with an empty pool and, with the default
// reconnect_attempts == 0, retries forever -- the server never returns.
// No signal is needed to drive this: the reload flag is a plain variable
// and apply() a callback (same harness as test_pipeline_stage's reload).
TEST_CASE("serve_stage_mux drops a session when a reload empties the pool",
          "[pipeline][mux][connect][e2e]") {
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
    // A resolver MUST be set for this to reach the guard: with no
    // resolver the no-numeric-entry check already catches an empty pool
    // (vacuously, since there is no numeric entry), so only the
    // resolver-configured path -- what the CLI always uses -- falls
    // through to the `% pool_live.size()`.
    locus::pipeline::Resolver resolver(
        locus::pipeline::Resolver::Options{},
        [](const std::string&) { return std::vector<std::string>{}; }, {},
        /*start_thread=*/false);
    conn.resolver = &resolver;

    // Reload already pending: the loop applies it (emptying the pool) on
    // the wakeup that the client's connect causes, before accepting.
    volatile std::sig_atomic_t flag = 1;
    locus::pipeline::StageReload reload;
    reload.flag = &flag;
    reload.apply = [](std::vector<locus::pipeline::Cidr>&,
                      std::vector<locus::pipeline::HostPort>& p,
                      locus::pipeline::StageConn&) { p.clear(); };

    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(stage, ls, {},
                                                {{"127.0.0.1", pc}}, conn,
                                                reload)
                   ? 1
                   : 0;
    });

    const int c = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(c >= 0);
    server.join();  // must return promptly: dropped, not hung, not crashed
    ::close(c);
    ::close(lc);
    REQUIRE(sres.load() == 0);  // ended unclean (nothing to dial)
}

// Issue 76 regression: the SESSION_END reset path has its OWN
// round-robin modulo (cursor % pool_live.size()) and so needs the same
// empty-pool guard the accept path carries. A reload that empties the
// pool while a connection is live, then a SESSION_END on it, must DROP
// the session -- not divide by zero (SIGFPE on x86-64, silent reconnect-
// forever livelock on arm64). Deterministic: the reload's apply() signals
// `applied`, and the test sends SESSION_END only after the pool is
// confirmed emptied, so the reset is guaranteed to see an empty pool.
TEST_CASE("serve_stage_mux: SESSION_END with an emptied pool drops, no "
          "divide-by-zero", "[pipeline][mux][connect][e2e]") {
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

    // A wake pipe so a mid-session reload is applied promptly. The read
    // end must be non-blocking: the loop drains it with while(read>0),
    // which blocks on a blocking fd (the real SIGHUP self-pipe is
    // non-blocking for the same reason).
    int wfds[2];
    REQUIRE(::pipe(wfds) == 0);
    ::fcntl(wfds[0], F_SETFL, ::fcntl(wfds[0], F_GETFL, 0) | O_NONBLOCK);

    PipelineStage stage(model, 0, L);
    locus::pipeline::StageConn conn;
    conn.serve_sessions = 1;

    std::atomic<bool> applied{false};
    volatile std::sig_atomic_t flag = 0;
    locus::pipeline::StageReload reload;
    reload.flag = &flag;
    reload.wake_fd = wfds[0];
    reload.apply = [&applied](std::vector<locus::pipeline::Cidr>&,
                              std::vector<locus::pipeline::HostPort>& p,
                              locus::pipeline::StageConn&) {
        p.clear();  // empty the pool
        applied.store(true);
    };

    std::atomic<int> sres{-1};
    std::thread server([&] {
        sres = locus::pipeline::serve_stage_mux(stage, ls, {},
                                                {{"127.0.0.1", pc}}, conn,
                                                reload)
                   ? 1
                   : 0;
    });

    // Bring a session up on the (non-empty) pool and run one token.
    Client a;
    a.request_id = 101;
    a.to_stage = locus::pipeline::connect_to("127.0.0.1", ps);
    REQUIRE(a.to_stage >= 0);
    a.from_stage = locus::pipeline::accept_one(lc, nullptr);
    REQUIRE(a.from_stage >= 0);
    a.round_trip(1);
    REQUIRE(a.ok);

    // Empty the pool via reload, and WAIT until it is applied, so the
    // SESSION_END below is guaranteed to reset against an empty pool.
    flag = 1;
    const char b = 1;
    REQUIRE(::write(wfds[1], &b, 1) == 1);
    for (int i = 0; i < 1000 && !applied.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(applied.load());

    // SESSION_END on the live connection: the reset hits the empty pool
    // and must drop, not hang/crash.
    REQUIRE(locus::pipeline::write_message(
        a.to_stage, locus::pipeline::make_session_end(101)));

    server.join();  // returns promptly iff the guard dropped the session
    ::close(a.to_stage);
    ::close(a.from_stage);
    ::close(lc);
    ::close(wfds[0]);
    ::close(wfds[1]);
    REQUIRE(sres.load() == 0);  // ended unclean: nothing left to dial
}
