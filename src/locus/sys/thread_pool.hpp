#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace locus::sys {

/**
 * Persistent worker pool for data-parallel loops (R9). One loop
 * runs at a time; concurrent callers are serialized internally.
 * Workers live for the lifetime of the process-wide instance.
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
     * first exception thrown by fn is rethrown on the caller.
     */
    void parallel_for(std::size_t n,
                      const std::function<void(std::size_t)>& fn);

    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

  private:
    explicit ThreadPool(std::size_t n_workers);
    void worker();
    /** Claims and runs indices of the current job; returns when
     * the job's index space is exhausted. */
    void drain();

    std::mutex jobs_m_;  // serializes parallel_for callers

    std::mutex m_;
    std::condition_variable cv_work_, cv_done_;
    const std::function<void(std::size_t)>* fn_ = nullptr;
    std::size_t n_ = 0;
    std::size_t next_ = 0;
    std::size_t done_ = 0;
    std::uint64_t epoch_ = 0;
    std::exception_ptr error_;
    bool stop_ = false;

    std::size_t n_workers_ = 0;
    std::vector<std::thread> workers_;
};

}  // namespace locus::sys
