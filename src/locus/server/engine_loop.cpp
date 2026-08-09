#include "locus/server/engine_loop.hpp"

#include <chrono>
#include <thread>

namespace locus::server {

namespace {
// Backoff between reader polls of the engine state. Small enough that
// SSE latency stays imperceptible, large enough that an idle waiter
// does not spin. The reader is serviced within one step() of a new
// token via the read_waiters_ turnstile regardless.
constexpr auto kPollBackoff = std::chrono::microseconds(200);
}  // namespace

EngineLoop::EngineLoop(const model::LlamaModel& m,
                       tok::TokenId eos,
                       engine::Engine::Config cfg)
    : engine_(m, eos, cfg), worker_([this] { run(); }) {}

EngineLoop::~EngineLoop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    work_cv_.notify_all();
    worker_.join();
}

std::string EngineLoop::admit_error(std::uint32_t n_prompt) {
    std::lock_guard<std::mutex> lk(mu_);
    return engine_.admit_error(n_prompt);
}

std::uint64_t EngineLoop::submit(
    std::vector<tok::TokenId> prompt, std::uint32_t max_new_tokens,
    model::SamplingParams sampling, std::uint64_t seed,
    std::unique_ptr<model::TokenConstraint> constraint,
    engine::LogprobsOpt logprobs) {
    std::uint64_t id;
    {
        std::lock_guard<std::mutex> lk(mu_);
        id = engine_.submit(std::move(prompt), max_new_tokens,
                            sampling, seed, std::move(constraint),
                            logprobs);
    }
    work_cv_.notify_all();
    return id;
}

void EngineLoop::run() {
    std::unique_lock<std::mutex> lk(mu_);
    while (!stop_) {
        engine_.step();
        // Park unless there is work step() can make progress on now.
        // Gating on has_runnable_work (not just idle) keeps the
        // worker from busy-spinning while HOLDING mu_ on a backlog it
        // cannot admit -- the failure that wedged the whole server.
        if (!engine_.has_runnable_work()) {
            work_cv_.wait(lk, [this] {
                return stop_ || engine_.has_runnable_work();
            });
        } else if (read_waiters_.load() > 0) {
            // Active generation with a client reading: release mu_ and
            // let every pending reader acquire it before the next
            // step, so wait_progress/view observe each token (SSE
            // streams) instead of blocking behind the whole
            // generation. Spinning until read_waiters_ drains
            // guarantees the reader wins the mu_ race on an unfair
            // std::mutex -- the "at most one step()" contract.
            // Unwatched generation skips this and runs full speed.
            lk.unlock();
            while (read_waiters_.load() > 0) {
                std::this_thread::yield();
            }
            lk.lock();
        }
    }
}

EngineLoop::View EngineLoop::snapshot_locked(
    std::uint64_t id) const {
    View v;
    if (const engine::Request* r = engine_.get(id)) {
        v.status = r->status;
        v.generated = r->generated;
        v.error = r->error;
        v.logprobs = r->logprobs;
    } else {
        v.status = engine::Status::kFailed;
        v.error = "unknown request id";
    }
    return v;
}

EngineLoop::View EngineLoop::view(std::uint64_t id) {
    read_waiters_.fetch_add(1);
    std::unique_lock<std::mutex> lk(mu_);
    read_waiters_.fetch_sub(1);
    return snapshot_locked(id);
}

EngineLoop::View EngineLoop::wait_progress(std::uint64_t id,
                                           std::size_t n_seen) {
    // Poll under the read_waiters_ turnstile: announce intent, take
    // mu_ (the worker yields it between steps), snapshot, and either
    // return or back off. Polling (vs a condvar) keeps the re-acquire
    // fair -- a condvar wait would race the worker's relock and starve
    // the reader for the whole generation.
    for (;;) {
        read_waiters_.fetch_add(1);
        {
            std::unique_lock<std::mutex> lk(mu_);
            read_waiters_.fetch_sub(1);
            const View v = snapshot_locked(id);
            if (v.generated.size() > n_seen ||
                v.status == engine::Status::kDone ||
                v.status == engine::Status::kFailed) {
                return v;
            }
        }
        std::this_thread::sleep_for(kPollBackoff);
    }
}

EngineLoop::Stats EngineLoop::stats() {
    read_waiters_.fetch_add(1);
    std::unique_lock<std::mutex> lk(mu_);
    read_waiters_.fetch_sub(1);
    Stats s;
    s.free_blocks = engine_.free_blocks();
    s.total_blocks = engine_.total_blocks();
    s.prefix_reused_tokens = engine_.prefix_reused_tokens();
    s.spec_accepted_tokens = engine_.spec_accepted_tokens();
    s.spec_steps = engine_.spec_steps();
    return s;
}

EngineLoop::View EngineLoop::wait_done(std::uint64_t id) {
    for (;;) {
        read_waiters_.fetch_add(1);
        {
            std::unique_lock<std::mutex> lk(mu_);
            read_waiters_.fetch_sub(1);
            const View v = snapshot_locked(id);
            if (v.status == engine::Status::kDone ||
                v.status == engine::Status::kFailed) {
                return v;
            }
        }
        std::this_thread::sleep_for(kPollBackoff);
    }
}

void EngineLoop::release(std::uint64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    engine_.release(id);
}

}  // namespace locus::server
