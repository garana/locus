#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/engine/engine.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/server/engine_loop.hpp"
#include "locus/tok/tokenizer.hpp"

namespace {
std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}
}  // namespace

// R1 regression (#59): the worker must release mu_ between step()s
// during active generation so client threads see progress within one
// iteration -- the "block for at most one iteration" contract in the
// EngineLoop class doc. The pre-fix bug held mu_ across the WHOLE
// generation, releasing it only on the idle park, so a client blocked
// in wait_progress could not re-acquire mu_ to observe a new token
// until the request was already kDone: exactly ONE observation, i.e.
// SSE arrives all-at-once at the end. The fix yields per-token
// streaming (many observations).
//
// (view()/timing can't test this: view() can race in during the
// worker's wake-up window before it re-acquires mu_, returning fast
// regardless of the bug. wait_progress genuinely blocks -- it waits on
// progress_cv_ and must re-lock mu_ to see each new token.)
TEST_CASE("EngineLoop streams progress incrementally, not all-at-once",
          "[e2e][engine_loop]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    locus::engine::Engine::Config cfg;
    locus::server::EngineLoop loop(model, tok.eos_id(), cfg);

    const auto prompt = tok.encode("Once upon a time", true);
    // Warm up so weights are paged in (a cold first step is slow and
    // would let the observer lag); then a long generation gives a wide
    // window to observe streaming.
    loop.wait_done(loop.submit(prompt, 4));

    const std::uint32_t max_new = 256;
    const std::uint64_t id = loop.submit(prompt, max_new);

    // Streaming observer: block for each next token, count distinct
    // progress points before the terminal state.
    std::size_t seen = 0;
    int observations = 0;
    for (;;) {
        const auto v = loop.wait_progress(id, seen);
        ++observations;
        seen = v.generated.size();
        if (v.status == locus::engine::Status::kDone ||
            v.status == locus::engine::Status::kFailed) {
            break;
        }
    }
    INFO("observations=" << observations << " tokens=" << seen);
    // Pre-fix: exactly 1 (only observable once done). Post-fix: many.
    REQUIRE(observations > 1);
}

// i#81: a non-streaming request whose client disconnects mid-generation
// must stop decoding rather than run to max_tokens. wait_done_or_cancel
// polls a liveness probe between steps; once the probe reports the
// client gone it cancels the request, which returns kCancelled having
// generated far fewer than max_new tokens. Without the fix the handler
// blocked in wait_done until the full generation completed, pinning a
// core on an abandoned request.
TEST_CASE("EngineLoop cancels a request when its client disconnects",
          "[e2e][engine_loop]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    locus::engine::Engine::Config cfg;
    // Use an EOS id the model can never emit (vocab size is one past the
    // last valid token) so generation always runs the FULL max_new and
    // cannot finish early -- otherwise the worker could complete a short
    // EOS-terminated story before the reader's first poll, racing the
    // cancel. With no early stop, cancel deterministically lands while
    // the request is still running.
    const locus::tok::TokenId never_eos = model.hparams().n_vocab;
    locus::server::EngineLoop loop(model, never_eos, cfg);

    const auto prompt = tok.encode("Once upon a time", true);
    loop.wait_done(loop.submit(prompt, 4));  // warm the weights in

    const std::uint32_t max_new = 512;
    const std::uint64_t id = loop.submit(prompt, max_new);

    // Stay "connected" until the request is a couple tokens into
    // generation, then drop: this exercises a genuine MID-FLIGHT cancel
    // through wait_done_or_cancel (probe -> cancel -> kCancelled), not
    // just a pre-start one. The never-emitted EOS keeps the request
    // running (512 tokens) so it cannot self-complete and race the
    // cancel. NOTE: this is an integration test of the wiring, not a
    // deterministic guard of the EngineLoop turnstile -- a polling loop
    // cannot be one (the view() poll itself registers a read-waiter, so
    // even a non-turnstile cancel can win the relock race). The
    // turnstile is correct by inspection of run()'s contract, and
    // engine_.cancel() is covered deterministically by the engine test.
    // view() inside the probe is safe: it runs outside the wait's lock.
    std::function<bool()> alive = [&loop, id] {
        return loop.view(id).generated.size() < 2;
    };
    const auto v = loop.wait_done_or_cancel(id, alive);

    REQUIRE(v.status == locus::engine::Status::kCancelled);
    REQUIRE(v.generated.size() < max_new);  // stopped, not run to max

    // The engine reclaimed the slot: a fresh request still completes.
    const auto v2 = loop.wait_done(loop.submit(prompt, 4));
    REQUIRE(v2.status == locus::engine::Status::kDone);
}

// i#81 turnstile regression (deterministic): EngineLoop::cancel must
// join the read_waiters_ turnstile, or it cannot preempt a running
// worker -- run() holds mu_ across step() and yields it ONLY when a
// read-waiter is pending, so a cancel that does not register one blocks
// until the generation finishes and then no-ops on an already-terminal
// request. The test above cannot guard this (its view() poll registers
// a waiter and opens the gate anyway); this one can, by cancelling with
// NO prior poll: read_waiters_ stays 0, the worker holds mu_ across
// every step, so only a turnstile-registering cancel gets in mid-flight.
// Discriminator is preemption (kCancelled, far short of max_new), not a
// clock: with the turnstile cancel lands in ~1 ms a few dozen tokens in;
// without it, it blocks to completion -> kDone at max_new. A huge
// max_new with a never-emitted EOS keeps generation long on any host so
// the fixed head start is always a small fraction of it.
TEST_CASE("EngineLoop cancel preempts a running worker (turnstile)",
          "[e2e][engine_loop]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    locus::engine::Engine::Config cfg;
    const locus::tok::TokenId never_eos = model.hparams().n_vocab;
    locus::server::EngineLoop loop(model, never_eos, cfg);

    const auto prompt = tok.encode("Once upon a time", true);
    loop.wait_done(loop.submit(prompt, 4));  // warm the weights in

    const std::uint32_t max_new = 4096;
    const std::uint64_t id = loop.submit(prompt, max_new);

    // Let the worker get into generation WITHOUT polling -- no view() or
    // wait_* here, so read_waiters_ stays 0 and run() never yields mu_
    // between steps. That is the exact condition a non-turnstile cancel
    // could not break into.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    loop.cancel(id);
    const auto v = loop.view(id);

    REQUIRE(v.status == locus::engine::Status::kCancelled);  // preempted
    REQUIRE(v.generated.size() < max_new);  // not run to completion
}

// i#928: waiters sleep until the worker publishes progress. A request
// the pool can never admit parks the worker (no step publishes again),
// so cancel() must publish itself: a waiter on it wakes with kCancelled
// instead of sleeping to its deadline. A wait with nothing to report
// sleeps to its deadline and no further.
TEST_CASE("EngineLoop wakes a waiter when a parked request is cancelled",
          "[e2e][engine_loop]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    locus::engine::Engine::Config cfg;
    cfg.n_blocks = 2;  // 32 tokens, the 16 of decode headroom included
    locus::server::EngineLoop loop(model, tok.eos_id(), cfg);

    std::vector<locus::tok::TokenId> big;
    for (int i = 0; i < 4; ++i) {
        const auto t = tok.encode("Once upon a time there was a dog", i == 0);
        big.insert(big.end(), t.begin(), t.end());
    }
    REQUIRE_FALSE(loop.admit_error(
                          static_cast<std::uint32_t>(big.size()))
                      .empty());
    // submit() does not check admission: the request parks at the front.
    const std::uint64_t id = loop.submit(big, 4);

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    const auto idle =
        loop.wait_progress(id, 0, std::chrono::milliseconds(300));
    const auto waited = Clock::now() - t0;
    CHECK(idle.status == locus::engine::Status::kWaiting);
    CHECK(waited >= std::chrono::milliseconds(300));

    std::thread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        loop.cancel(id);
    });
    const auto t1 = Clock::now();
    const auto v = loop.wait_progress(id, 0, std::chrono::seconds(10));
    const auto woke = Clock::now() - t1;
    canceller.join();
    CHECK(v.status == locus::engine::Status::kCancelled);
    CHECK(woke < std::chrono::seconds(5));

    // The queue is free again: a short request completes.
    const auto prompt = tok.encode("Once upon a time", true);
    CHECK(loop.wait_done(loop.submit(prompt, 4)).status ==
          locus::engine::Status::kDone);
}

namespace {
/** Counts the progress hook's calls and wakes a waiter on each. */
struct HookWake {
    std::mutex mu;
    std::condition_variable cv;
    int calls = 0;

    std::function<void()> hook() {
        return [this] {
            {
                std::lock_guard<std::mutex> lk(mu);
                ++calls;
            }
            cv.notify_all();
        };
    }
    /** Sleeps until the hook has run more than `seen` times (or 1 s). */
    int wait(int seen) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait_for(lk, std::chrono::seconds(1),
                    [&] { return calls > seen; });
        return calls;
    }
};
}  // namespace

// i#989: a followed request's progress arrives through take(), woken by
// the progress hook, and adds up to the snapshot view() gives at the
// end: the same tokens, logprobs, status and prefix counts.
TEST_CASE("EngineLoop follow: take() adds up to the final view",
          "[e2e][engine_loop]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    locus::engine::Engine::Config cfg;
    const locus::tok::TokenId never_eos = model.hparams().n_vocab;
    locus::server::EngineLoop loop(model, never_eos, cfg);
    HookWake wake;
    loop.set_progress_hook(wake.hook());

    const auto prompt = tok.encode("Once upon a time", true);
    loop.wait_done(loop.submit(prompt, 4));  // warm the weights in

    locus::engine::LogprobsOpt lp;
    lp.enabled = true;
    lp.top = 2;
    const std::uint64_t id = loop.submit(prompt, 64, {}, 0, nullptr, lp);
    loop.follow(id);
    locus::server::EngineLoop::View acc;
    int takes = 0;
    int calls = 0;
    for (int round = 0; round < 10000; ++round) {
        if (loop.take(id, acc)) ++takes;
        if (locus::engine::is_terminal(acc.status)) break;
        calls = wake.wait(calls);
    }
    REQUIRE(acc.status == locus::engine::Status::kDone);
    CHECK(takes > 1);  // incremental, not all at the end
    CHECK(wake.calls > 0);
    // Ended at the terminal take: nothing more to take.
    CHECK_FALSE(loop.take(id, acc));

    const auto v = loop.view(id);
    CHECK(acc.generated == v.generated);
    REQUIRE(acc.logprobs.size() == v.logprobs.size());
    for (std::size_t i = 0; i < v.logprobs.size(); ++i) {
        CHECK(acc.logprobs[i].token == v.logprobs[i].token);
        CHECK(acc.logprobs[i].logprob == v.logprobs[i].logprob);
        CHECK(acc.logprobs[i].top.size() == v.logprobs[i].top.size());
    }
    CHECK(acc.reused_prefix_tokens == v.reused_prefix_tokens);
    CHECK(acc.cached_prefix_tokens == v.cached_prefix_tokens);
    CHECK(acc.generated.size() == 64);

    // An id nobody follows: nothing to take, acc untouched.
    locus::server::EngineLoop::View other;
    CHECK_FALSE(loop.take(id + 100, other));
    CHECK(other.generated.empty());
}

// i#989: cancelling a followed request queues kCancelled (cancel
// publishes, so a parked worker does not hide it), and unfollow drops
// the queue.
TEST_CASE("EngineLoop follow: a cancel reaches take()",
          "[e2e][engine_loop]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("stories260K model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    locus::engine::Engine::Config cfg;
    const locus::tok::TokenId never_eos = model.hparams().n_vocab;
    locus::server::EngineLoop loop(model, never_eos, cfg);
    HookWake wake;
    loop.set_progress_hook(wake.hook());

    const auto prompt = tok.encode("Once upon a time", true);
    const std::uint64_t id = loop.submit(prompt, 4096);
    loop.follow(id);
    loop.follow(id);  // idempotent
    locus::server::EngineLoop::View acc;
    CHECK(loop.take(id, acc));  // the first take reports the state
    loop.cancel(id);
    // cancel() queued the terminal state before it returned.
    REQUIRE(loop.take(id, acc));
    CHECK(acc.status == locus::engine::Status::kCancelled);
    CHECK(acc.generated.size() < 4096);
    CHECK(loop.release(id));

    const std::uint64_t id2 = loop.submit(prompt, 4096);
    loop.follow(id2);
    loop.unfollow(id2);
    locus::server::EngineLoop::View acc2;
    CHECK_FALSE(loop.take(id2, acc2));
    loop.cancel(id2);
    loop.release(id2);
}
