#include "locus/server/engine_loop.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <limits>
#include <thread>

#include "locus/sys/perf_counters.hpp"

namespace locus::server {

namespace {
// How often wait_done_or_cancel probes its client between progress
// wakes: the bound on noticing a client that left while its request
// makes no progress (queued, or a long prefill).
constexpr auto kAliveProbe = std::chrono::milliseconds(20);
using Clock = std::chrono::steady_clock;
}  // namespace

/** Registers a waiter on a request's progress for its scope. */
class EngineLoop::WatchScope {
  public:
    WatchScope(EngineLoop& l, std::uint64_t id) : l_(l), id_(id) {
        std::lock_guard<std::mutex> pk(l_.pmu_);
        ++l_.watched_[id_].waiters;
    }
    ~WatchScope() {
        std::lock_guard<std::mutex> pk(l_.pmu_);
        const auto it = l_.watched_.find(id_);
        if (it != l_.watched_.end() && --it->second.waiters == 0) {
            l_.watched_.erase(it);
        }
    }
    WatchScope(const WatchScope&) = delete;
    WatchScope& operator=(const WatchScope&) = delete;

  private:
    EngineLoop& l_;
    std::uint64_t id_;
};

EngineLoop::EngineLoop(const model::TransformerModel& m,
                       tok::TokenId eos,
                       engine::Engine::Config cfg)
    : engine_(m, eos, cfg),
      kv_type_(cfg.kv_type),
      worker_([this] { run(); }) {}

EngineLoop::~EngineLoop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    work_cv_.notify_all();
    worker_.join();
}

std::unique_lock<std::mutex> EngineLoop::lock_turn() {
    read_waiters_.fetch_add(1);
    std::unique_lock<std::mutex> lk(mu_);
    read_waiters_.fetch_sub(1);
    return lk;
}

void EngineLoop::publish_locked() {
    bool changed = false;
    {
        std::lock_guard<std::mutex> pk(pmu_);
        for (auto& [id, w] : watched_) {
            const engine::Request* r = engine_.get(id);
            const std::size_t n = r != nullptr ? r->generated.size() : 0;
            const bool done = r == nullptr || engine::is_terminal(r->status);
            if (n != w.generated || done != w.terminal) {
                w.generated = n;
                w.terminal = done;
                changed = true;
            }
        }
    }
    if (changed) {
        progress_cv_.notify_all();
    }
}

void EngineLoop::wait_watch(std::uint64_t id, std::size_t n_seen,
                            Clock::time_point until) {
    std::unique_lock<std::mutex> pk(pmu_);
    const auto ready = [&] {
        const auto it = watched_.find(id);
        return it == watched_.end() || it->second.terminal ||
               it->second.generated > n_seen;
    };
    if (until == Clock::time_point::max()) {
        progress_cv_.wait(pk, ready);
    } else {
        progress_cv_.wait_until(pk, until, ready);
    }
}

std::string EngineLoop::admit_error(std::uint32_t n_prompt) {
    auto lk = lock_turn();
    return engine_.admit_error(n_prompt);
}

std::uint64_t EngineLoop::submit(
    std::vector<tok::TokenId> prompt, std::uint32_t max_new_tokens,
    model::SamplingParams sampling, std::uint64_t seed,
    std::unique_ptr<model::TokenConstraint> constraint,
    engine::LogprobsOpt logprobs) {
    std::uint64_t id;
    {
        auto lk = lock_turn();
        id = engine_.submit(std::move(prompt), max_new_tokens,
                            sampling, seed, std::move(constraint),
                            logprobs);
    }
    work_cv_.notify_all();
    return id;
}

void EngineLoop::run() {
    // The engine worker drives the forward pass and runs the inline
    // matvec slice, so register its cache counters for /metrics too (the
    // pool workers cover the fan-out). No-op unless perf is available.
    locus::sys::perf::ThreadCounters perf_counters;
    std::unique_lock<std::mutex> lk(mu_);
    while (!stop_) {
        try {
            engine_.step();
        } catch (const std::exception& e) {
            // A forward that throws (a lost GPU device, i#589) fails the
            // requests it was stepping, with the reason; the server
            // stays up for the others and for /health.
            const std::size_t n = engine_.fail_running(e.what());
            std::fprintf(stderr,
                         "locus-server: a step failed (%s); %zu request(s) "
                         "failed\n",
                         e.what(), n);
        }
        publish_locked();
        // Park unless there is work step() can make progress on now.
        // Gating on has_runnable_work (not just idle) keeps the
        // worker from busy-spinning while HOLDING mu_ on a backlog it
        // cannot admit -- the failure that wedged the whole server.
        if (!engine_.has_runnable_work() && engine_.loading()) {
            // A disk-tier read is in flight and its end does not
            // notify work_cv_: look again shortly instead of parking.
            work_cv_.wait_for(lk, std::chrono::milliseconds(10), [this] {
                return stop_ || engine_.has_runnable_work();
            });
        } else if (!engine_.has_runnable_work()) {
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
        v.reused_prefix_tokens = r->reused_prefix_tokens;
        v.cached_prefix_tokens = r->cached_prefix_tokens;
    } else {
        v.status = engine::Status::kFailed;
        v.error = "unknown request id";
    }
    return v;
}

EngineLoop::View EngineLoop::view(std::uint64_t id) {
    auto lk = lock_turn();
    return snapshot_locked(id);
}

EngineLoop::View EngineLoop::wait_progress(
    std::uint64_t id, std::size_t n_seen, std::chrono::milliseconds max_wait) {
    // Watch first, then snapshot: a step that lands after the snapshot
    // is published to the watch, so the sleep below cannot miss it. The
    // snapshot itself goes through the turnstile (at most one step).
    WatchScope watch(*this, id);
    const auto until = max_wait.count() > 0 ? Clock::now() + max_wait
                                            : Clock::time_point::max();
    for (;;) {
        const View v = view(id);
        if (v.generated.size() > n_seen || engine::is_terminal(v.status) ||
            Clock::now() >= until) {
            return v;
        }
        wait_watch(id, n_seen, until);
    }
}

EngineLoop::Stats EngineLoop::stats() {
    auto lk = lock_turn();
    Stats s;
    s.free_blocks = engine_.free_blocks();
    s.total_blocks = engine_.total_blocks();
    s.prefix_reused_tokens = engine_.prefix_reused_tokens();
    s.spec_accepted_tokens = engine_.spec_accepted_tokens();
    s.spec_steps = engine_.spec_steps();
    s.spec_drafted_tokens = engine_.spec_drafted_tokens();
    s.spec_draft_counts = engine_.spec_draft_counts();
    s.disk_hits = engine_.disk_prefix_hits();
    s.disk_loaded_tokens = engine_.disk_prefix_loaded_tokens();
    s.disk_swapped_tokens = engine_.disk_prefix_swapped_tokens();
    s.disk_refused_tokens = engine_.disk_prefix_refused_tokens();
    return s;
}

EngineLoop::View EngineLoop::wait_done(std::uint64_t id) {
    WatchScope watch(*this, id);
    for (;;) {
        const View v = view(id);
        if (engine::is_terminal(v.status)) {
            return v;
        }
        wait_watch(id, std::numeric_limits<std::size_t>::max(),
                   Clock::time_point::max());
    }
}

EngineLoop::View EngineLoop::wait_done_or_cancel(
    std::uint64_t id, const std::function<bool()>& alive) {
    WatchScope watch(*this, id);
    for (;;) {
        const View v = view(id);
        if (engine::is_terminal(v.status)) {
            return v;
        }
        // Liveness syscall runs OUTSIDE mu_ so probing the socket never
        // blocks the worker's next step(). A gone client -> cancel and
        // return the now-terminal (kCancelled) snapshot.
        if (alive && !alive()) {
            cancel(id);
            return view(id);
        }
        wait_watch(id, std::numeric_limits<std::size_t>::max(),
                   alive ? Clock::now() + kAliveProbe
                         : Clock::time_point::max());
    }
}

void EngineLoop::release(std::uint64_t id) {
    auto lk = lock_turn();
    engine_.release(id);
}

void EngineLoop::cancel(std::uint64_t id) {
    // Join the read_waiters_ turnstile, exactly like view()/wait_*: the
    // worker does NOT release mu_ between steps while no reader is
    // pending, so a plain lock here would block until the whole
    // generation finished -- cancellation would never preempt a running
    // request. Announcing intent makes the worker yield mu_ after its
    // current step so cancel lands mid-generation.
    {
        auto lk = lock_turn();
        engine_.cancel(id);
        // A parked worker may not step again (the cancelled request was
        // its only work), so publish here: its waiters see kCancelled.
        publish_locked();
    }
    // Wake the worker so it re-evaluates has_runnable_work and parks if
    // the cancelled request was the only work (harmless if it was not).
    work_cv_.notify_all();
}

}  // namespace locus::server
