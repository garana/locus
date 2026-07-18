#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cppllm/engine/engine.hpp"

namespace cppllm::server {

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
    };

    EngineLoop(const model::LlamaModel& m, tok::TokenId eos,
               engine::Engine::Config cfg);
    ~EngineLoop();

    EngineLoop(const EngineLoop&) = delete;
    EngineLoop& operator=(const EngineLoop&) = delete;

    /** Enqueues a request and wakes the worker. */
    std::uint64_t submit(std::vector<tok::TokenId> prompt,
                         std::uint32_t max_new_tokens);

    /** @returns A snapshot of the request's current state. */
    View view(std::uint64_t id);

    /**
     * Blocks until the request has more than n_seen generated
     * tokens or reaches a terminal state.
     */
    View wait_progress(std::uint64_t id, std::size_t n_seen);

    /** Blocks until the request is kDone or kFailed. */
    View wait_done(std::uint64_t id);

  private:
    void run();
    View snapshot_locked(std::uint64_t id) const;

    std::mutex mu_;
    std::condition_variable work_cv_;
    std::condition_variable progress_cv_;
    engine::Engine engine_;
    bool stop_ = false;
    std::thread worker_;
};

}  // namespace cppllm::server
