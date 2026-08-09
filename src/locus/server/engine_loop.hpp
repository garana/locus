#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "locus/engine/engine.hpp"

namespace locus::server {

/**
 * Thread-safe facade over the single-threaded Engine: one worker
 * thread drives step() while HTTP handler threads submit requests
 * and wait on progress. The engine mutex is held across each
 * step() iteration, so submitters block for at most one iteration.
 */
class EngineLoop {
  public:
    /** Terminal-or-progress snapshot of a request. */
    struct View {
        engine::Status status = engine::Status::kWaiting;
        std::vector<tok::TokenId> generated;
        std::string error;
        /** Per-token logprobs (empty unless requested at submit). */
        std::vector<engine::LogprobEntry> logprobs;
    };

    EngineLoop(const model::LlamaModel& m, tok::TokenId eos,
               engine::Engine::Config cfg);
    ~EngineLoop();

    EngineLoop(const EngineLoop&) = delete;
    EngineLoop& operator=(const EngineLoop&) = delete;

    /**
     * Reason a prompt of n_prompt tokens can never be served by the
     * KV pool (empty = admissible); callers reject at the HTTP edge
     * so an oversized request never enters the wait queue.
     */
    std::string admit_error(std::uint32_t n_prompt);

    /** Enqueues a request and wakes the worker. */
    std::uint64_t submit(
        std::vector<tok::TokenId> prompt,
        std::uint32_t max_new_tokens,
        model::SamplingParams sampling = {}, std::uint64_t seed = 0,
        std::unique_ptr<model::TokenConstraint> constraint = nullptr,
        engine::LogprobsOpt logprobs = {});

    /** @returns A snapshot of the request's current state. */
    View view(std::uint64_t id);

    /**
     * Blocks until the request has more than n_seen generated
     * tokens or reaches a terminal state.
     */
    View wait_progress(std::uint64_t id, std::size_t n_seen);

    /** Blocks until the request is kDone or kFailed. */
    View wait_done(std::uint64_t id);

    /** Engine gauges for /metrics (snapshotted under the mutex). */
    struct Stats {
        std::uint32_t free_blocks = 0;
        std::uint32_t total_blocks = 0;
        std::uint64_t prefix_reused_tokens = 0;
        std::uint64_t spec_accepted_tokens = 0;
        std::uint64_t spec_steps = 0;
    };
    Stats stats();

  private:
    void run();
    View snapshot_locked(std::uint64_t id) const;

    std::mutex mu_;
    std::condition_variable work_cv_;
    /**
     * Count of client threads currently trying to acquire mu_ for a
     * read (view/wait_progress/wait_done/stats). The worker checks it
     * between step()s and yields mu_ until pending readers have been
     * serviced, so a reader is never starved behind a whole generation
     * on the unfair std::mutex -- it waits at most one step().
     */
    std::atomic<int> read_waiters_{0};
    engine::Engine engine_;
    bool stop_ = false;
    std::thread worker_;
};

}  // namespace locus::server
