#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "locus/engine/engine.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::server {

/** How an EngineLoop's worker thread runs (i#929), for one of several
 *  engines over one model. The default is one engine on the process's
 *  device and thread pool. At namespace scope, not nested: GCC refuses
 *  a nested struct with member initializers as a default argument of
 *  its enclosing class (CWG 1397). */
struct EngineLoopSetup {
    /** Runs first on the worker thread (e.g. binds it to the Vulkan
     *  device the engine's KV pool was made on). */
    std::function<void()> thread_init;
    /** > 0: the worker's forwards fan out to a pool of its own with
     *  this many threads, the worker included; 0: the process pool. */
    std::size_t threads = 0;
};

/**
 * Thread-safe facade over the single-threaded Engine: one worker
 * thread drives step() while HTTP handler threads submit requests
 * and wait on progress. The engine mutex is held across each
 * step() iteration, so callers block for at most one iteration.
 * Waiters sleep until the worker publishes progress for their request
 * (no polling, i#928).
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
        /** Prompt tokens served from a cached KV prefix (cache_read)
         * and newly written to the cache (cache_creation). */
        std::uint32_t reused_prefix_tokens = 0;
        std::uint32_t cached_prefix_tokens = 0;
    };

    using Setup = EngineLoopSetup;

    /**
     * Makes the engine (and its KV pool) on the calling thread, then
     * starts the worker.
     */
    EngineLoop(const model::TransformerModel& m, tok::TokenId eos,
               engine::Engine::Config cfg, Setup setup = {});
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
     * tokens or reaches a terminal state, or, with a nonzero
     * max_wait, until that much time has passed (i#619).
     * @returns The latest snapshot; after a timeout it may hold no new
     *     tokens.
     */
    View wait_progress(std::uint64_t id, std::size_t n_seen,
                       std::chrono::milliseconds max_wait = {});

    /** Blocks until the request reaches a terminal state. */
    View wait_done(std::uint64_t id);

    /**
     * Like wait_done, but between polls it checks `alive()` (a client
     * liveness probe); when that returns false the request is
     * cancel()ed and the cancelled snapshot returned, so a request
     * whose client has disconnected stops decoding instead of running
     * to max_tokens. An empty `alive` means "always alive" (behaves
     * exactly like wait_done). The probe is called OUTSIDE the engine
     * mutex so its syscall never stalls the worker.
     * @param id Request id.
     * @param alive Returns true while the client is still connected.
     * @returns The terminal snapshot (kCancelled if the client left).
     */
    View wait_done_or_cancel(std::uint64_t id,
                             const std::function<bool()>& alive);

    /** Stops a non-terminal request early (client disconnected): the
     * engine drops it and frees its KV. No-op if unknown/terminal. */
    void cancel(std::uint64_t id);

    /** Drops a finished request's record once its result has been
     * delivered to the client (evict-on-consume); no-op if it is not
     * terminal. Call once, after the last view/wait_done for the id.
     * @returns Whether the request is gone now (terminal, or unknown):
     *     false when it was still running and was kept. */
    bool release(std::uint64_t id);

    /** Engine gauges for /metrics (snapshotted under the mutex). */
    struct Stats {
        std::uint32_t free_blocks = 0;
        std::uint32_t total_blocks = 0;
        std::uint64_t prefix_reused_tokens = 0;
        std::uint64_t spec_accepted_tokens = 0;
        std::uint64_t spec_steps = 0;
        std::uint64_t spec_drafted_tokens = 0;
        /** Verifies by draft length (Engine::spec_draft_counts). */
        std::vector<std::uint64_t> spec_draft_counts;
        /** The disk prefix tier's counters (Engine::disk_prefix_*). */
        std::uint64_t disk_hits = 0, disk_loaded_tokens = 0,
                      disk_swapped_tokens = 0, disk_refused_tokens = 0;
    };
    Stats stats();
    /** The engine's timing metrics; read without the lock, so
     * /metrics never waits for a step. */
    const engine::EngineMetrics& metrics() const {
        return engine_.metrics();
    }
    /** The engine's KV cache precision (fixed at construction). */
    kv::KvType kv_type() const { return kv_type_; }

  private:
    /** What the worker last published for a watched request. */
    struct Watch {
        std::size_t generated = 0;
        bool terminal = false;
        int waiters = 0;
    };
    /** Registers a waiter on a request's progress for its scope. */
    class WatchScope;

    void run();
    View snapshot_locked(std::uint64_t id) const;
    /** Takes mu_ through the read_waiters_ turnstile, so the worker
     *  hands it over after at most one step. */
    std::unique_lock<std::mutex> lock_turn();
    /** Publishes the watched requests' progress (call with mu_ held)
     *  and wakes their waiters when any changed. */
    void publish_locked();
    /**
     * Sleeps until the worker has published more than n_seen tokens or
     * a terminal state for id, or until `until` (time_point::max()
     * waits without limit). The caller holds a WatchScope for id.
     */
    void wait_watch(std::uint64_t id, std::size_t n_seen,
                    std::chrono::steady_clock::time_point until);

    std::mutex mu_;
    std::condition_variable work_cv_;
    /**
     * Progress publication, apart from mu_ so a waiter never sleeps on
     * the engine mutex (whose relock races the worker's). Lock order:
     * mu_ before pmu_; nothing holds pmu_ while taking mu_.
     */
    std::mutex pmu_;
    std::condition_variable progress_cv_;
    std::unordered_map<std::uint64_t, Watch> watched_;
    /**
     * Count of client threads currently trying to acquire mu_ for a
     * read (view/wait_progress/wait_done/stats). The worker checks it
     * between step()s and yields mu_ until pending readers have been
     * serviced, so a reader is never starved behind a whole generation
     * on the unfair std::mutex -- it waits at most one step().
     */
    std::atomic<int> read_waiters_{0};
    engine::Engine engine_;
    kv::KvType kv_type_;
    bool stop_ = false;
    // Before worker_, which reads them as it starts.
    std::function<void()> thread_init_;
    std::unique_ptr<sys::ThreadPool> pool_;
    std::thread worker_;
};

}  // namespace locus::server
