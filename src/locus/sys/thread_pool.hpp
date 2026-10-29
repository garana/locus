#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace locus::sys {

/**
 * Persistent worker pool for data-parallel loops (R9). One loop
 * runs at a time; concurrent callers are serialized internally.
 * Workers live for the lifetime of the process-wide instance.
 *
 * Decode issues ~5 loops per layer, so the hand-off latency matters
 * (#206). After a loop, up to one worker per performance core spins
 * for the next one for a short budget before sleeping on a condition
 * variable, and indices are claimed with one atomic, so a loop starts
 * in a few microseconds instead of the ~18 us of a condition-variable
 * wake. The spinner count matches the performance cores so a 10-slice
 * matvec has 10 runnable threads; nothing pins them, the OS places
 * them (and keeps busy threads on the performance cores).
 */
class ThreadPool {
  public:
    /**
     * @returns The process-wide pool, created on first use with
     *     hardware_concurrency() - 1 workers.
     */
    static ThreadPool& instance();

    /** @returns Max useful fan-out (workers + calling thread). */
    std::size_t parallelism() const { return n_workers_ + 1; }

    /**
     * Invokes fn(i) for every i in [0, n) across the workers and
     * the calling thread; returns when all calls completed. The
     * first exception thrown by fn is rethrown on the caller. Jobs
     * of 2^32 - 1 items or more (past the claim word's index field)
     * run on the caller alone.
     */
    void parallel_for(std::size_t n,
                      const std::function<void(std::size_t)>& fn);

    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

  private:
    explicit ThreadPool(std::size_t n_workers);
    void worker(std::size_t index);
    /** Claims and runs indices of job generation `gen` until its
     * index space is exhausted (or a newer job replaced it). */
    void drain(std::uint64_t gen);
    /** @returns Whether a job newer than `seen` is published. */
    bool job_after(std::uint64_t seen) const;

    std::mutex jobs_m_;  // serializes parallel_for callers

    /**
     * The current job: generation << 32 | next index. The index is
     * kLocked while the caller writes fn_ and n_, so a claim (one
     * compare-exchange that keeps the generation) can only take an
     * index of the job whose fields it read.
     */
    std::atomic<std::uint64_t> word_{0};
    std::atomic<const std::function<void(std::size_t)>*> fn_{nullptr};
    std::atomic<std::size_t> n_{0};
    std::atomic<std::size_t> done_{0};
    std::atomic<bool> stop_{false};
    /** Workers spinning for the next job (they need no wake). */
    std::atomic<std::size_t> spinning_{0};

    std::mutex m_;  // sleeping workers and a sleeping caller
    std::condition_variable cv_work_, cv_done_;

    std::mutex error_m_;
    std::exception_ptr error_;

    std::size_t n_workers_ = 0;
    /** Workers allowed to spin (index below this). */
    std::size_t spin_workers_ = 0;
    std::vector<std::thread> workers_;
};

}  // namespace locus::sys
