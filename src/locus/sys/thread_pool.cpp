#include "locus/sys/thread_pool.hpp"

#include <algorithm>

#include "locus/sys/perf_counters.hpp"

namespace locus::sys {

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool(
        std::max(1u, std::thread::hardware_concurrency()) - 1);
    return pool;
}

ThreadPool::ThreadPool(std::size_t n_workers)
    : n_workers_(n_workers) {
    workers_.reserve(n_workers_);
    for (std::size_t i = 0; i < n_workers_; ++i) {
        workers_.emplace_back([this] { worker(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    cv_work_.notify_all();
    for (auto& t : workers_) {
        t.join();
    }
}

void ThreadPool::drain() {
    for (;;) {
        std::size_t i;
        {
            std::lock_guard<std::mutex> lk(m_);
            if (next_ >= n_) {
                return;
            }
            i = next_++;
        }
        try {
            (*fn_)(i);
        } catch (...) {
            std::lock_guard<std::mutex> lk(m_);
            if (!error_) {
                error_ = std::current_exception();
            }
        }
        std::lock_guard<std::mutex> lk(m_);
        if (++done_ == n_) {
            cv_done_.notify_all();
        }
    }
}

void ThreadPool::worker() {
    // Register this worker's cache counters for /metrics: the matvec
    // fan-out runs here, so this is where weight-read cache traffic
    // lives. No-op unless perf counters are available (Linux, permitted).
    perf::ThreadCounters perf_counters;
    std::uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_work_.wait(lk, [&] {
                return stop_ || (epoch_ != seen && next_ < n_);
            });
            if (stop_) {
                return;
            }
            seen = epoch_;
        }
        drain();
    }
}

void ThreadPool::parallel_for(
    std::size_t n, const std::function<void(std::size_t)>& fn) {
    if (n == 0) {
        return;
    }
    if (n == 1 || n_workers_ == 0) {
        for (std::size_t i = 0; i < n; ++i) {
            fn(i);
        }
        return;
    }
    std::lock_guard<std::mutex> jobs_lk(jobs_m_);
    {
        std::lock_guard<std::mutex> lk(m_);
        fn_ = &fn;
        n_ = n;
        next_ = 0;
        done_ = 0;
        error_ = nullptr;
        ++epoch_;
    }
    // Wake only as many workers as there are items beyond the caller's
    // own: a 10-slice matvec on a 14-thread pool then has 10 runnable
    // threads, which the OS keeps on the 10 performance cores, instead
    // of 14, some of which land on efficiency cores and finish last.
    // Jobs with many small items (the conv kernels) still wake all.
    for (std::size_t i = 0, k = std::min(n - 1, n_workers_); i < k; ++i) {
        cv_work_.notify_one();
    }
    drain();
    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [&] { return done_ == n_; });
    fn_ = nullptr;
    n_ = 0;
    if (error_) {
        std::exception_ptr e = error_;
        error_ = nullptr;
        std::rethrow_exception(e);
    }
}

}  // namespace locus::sys
