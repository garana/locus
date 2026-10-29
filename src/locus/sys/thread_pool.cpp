#include "locus/sys/thread_pool.hpp"

#include <algorithm>
#include <chrono>

#include "locus/sys/features.hpp"
#include "locus/sys/perf_counters.hpp"

namespace locus::sys {

namespace {

constexpr std::uint64_t kIdx = 0xffffffffu;
constexpr std::uint64_t kLocked = kIdx;

/** How long an idle worker spins for the next job before sleeping:
 *  covers the serial work between a token's loops and short gaps
 *  between tokens, not idle time between requests. */
constexpr auto kWorkerSpin = std::chrono::microseconds(500);
/** How long the caller spins for stragglers before sleeping. */
constexpr auto kCallerSpin = std::chrono::microseconds(2000);

void cpu_relax() {
#if defined(__aarch64__)
    asm volatile("yield");
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}

}  // namespace

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool(
        std::max(1u, std::thread::hardware_concurrency()) - 1);
    return pool;
}

ThreadPool::ThreadPool(std::size_t n_workers)
    : n_workers_(n_workers) {
    // Up to one spinner per performance core besides the caller
    // (a count, not a placement: the OS places the threads); every
    // worker where the platform reports no tiers.
    const unsigned pcores = performance_cores();
    spin_workers_ =
        pcores > 1 ? std::min<std::size_t>(n_workers_, pcores - 1)
                   : n_workers_;
    workers_.reserve(n_workers_);
    for (std::size_t i = 0; i < n_workers_; ++i) {
        workers_.emplace_back([this, i] { worker(i); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_.store(true);
    }
    cv_work_.notify_all();
    for (auto& t : workers_) {
        t.join();
    }
}

bool ThreadPool::job_after(std::uint64_t seen) const {
    const std::uint64_t w = word_.load();
    return (w >> 32) != seen && (w & kIdx) != kLocked;
}

void ThreadPool::drain(std::uint64_t gen) {
    for (;;) {
        std::uint64_t w = word_.load();
        if ((w >> 32) != gen) {
            return;
        }
        if ((w & kIdx) == kLocked) {
            cpu_relax();  // the caller is still writing this job
            continue;
        }
        // n_ read here belongs to `gen` whenever the claim below
        // succeeds: the caller changes the word (to kLocked) before it
        // writes the next job's fields.
        const std::size_t n = n_.load();
        const std::uint64_t i = w & kIdx;
        if (i >= n) {
            return;
        }
        if (!word_.compare_exchange_weak(w, w + 1)) {
            continue;
        }
        // Item i of `gen` is now ours and not done, so the caller is
        // still waiting on this job and fn_ is still its function.
        try {
            (*fn_.load())(static_cast<std::size_t>(i));
        } catch (...) {
            std::lock_guard<std::mutex> lk(error_m_);
            if (!error_) {
                error_ = std::current_exception();
            }
        }
        if (done_.fetch_add(1) + 1 == n) {
            std::lock_guard<std::mutex> lk(m_);
            cv_done_.notify_all();
        }
    }
}

void ThreadPool::worker(std::size_t index) {
    // Register this worker's cache counters for /metrics: the matvec
    // fan-out runs here, so this is where weight-read cache traffic
    // lives. No-op unless perf counters are available (Linux, permitted).
    perf::ThreadCounters perf_counters;
    std::uint64_t seen = word_.load() >> 32;
    for (;;) {
        bool got = false;
        if (index < spin_workers_) {
            spinning_.fetch_add(1);
            const auto until =
                std::chrono::steady_clock::now() + kWorkerSpin;
            for (unsigned k = 0;; ++k) {
                if (stop_.load(std::memory_order_relaxed)) {
                    spinning_.fetch_sub(1);
                    return;
                }
                if (job_after(seen)) {
                    got = true;
                    break;
                }
                if ((k & 255u) == 255u &&
                    std::chrono::steady_clock::now() >= until) {
                    break;
                }
                cpu_relax();
            }
            spinning_.fetch_sub(1);
        }
        if (!got) {
            std::unique_lock<std::mutex> lk(m_);
            cv_work_.wait(lk,
                          [&] { return stop_.load() || job_after(seen); });
            if (stop_.load()) {
                return;
            }
        }
        seen = word_.load() >> 32;
        drain(seen);
    }
}

void ThreadPool::parallel_for(
    std::size_t n, const std::function<void(std::size_t)>& fn) {
    if (n == 0) {
        return;
    }
    // The claim word holds a 32-bit index, kLocked reserved.
    if (n == 1 || n_workers_ == 0 || n >= kLocked) {
        for (std::size_t i = 0; i < n; ++i) {
            fn(i);
        }
        return;
    }
    std::lock_guard<std::mutex> jobs_lk(jobs_m_);
    const std::uint64_t gen = (word_.load() >> 32) + 1;
    word_.store(gen << 32 | kLocked);
    fn_.store(&fn);
    n_.store(n);
    done_.store(0);
    {
        std::lock_guard<std::mutex> lk(error_m_);
        error_ = nullptr;
    }
    word_.store(gen << 32);  // publish
    // Spinning workers pick it up on their own. Wake sleepers only for
    // the items beyond them (the caller drains one share itself), so a
    // 10-slice matvec runs on the spinners, not on extra threads.
    const std::size_t want = std::min(n - 1, n_workers_);
    const std::size_t spin = spinning_.load();
    if (want > spin) {
        std::lock_guard<std::mutex> lk(m_);
        for (std::size_t i = spin; i < want; ++i) {
            cv_work_.notify_one();
        }
    }
    drain(gen);
    if (done_.load() != n) {
        const auto until = std::chrono::steady_clock::now() + kCallerSpin;
        for (unsigned k = 0; done_.load() != n; ++k) {
            if ((k & 255u) == 255u &&
                std::chrono::steady_clock::now() >= until) {
                std::unique_lock<std::mutex> lk(m_);
                cv_done_.wait(lk, [&] { return done_.load() == n; });
                break;
            }
            cpu_relax();
        }
    }
    std::exception_ptr e;
    {
        std::lock_guard<std::mutex> lk(error_m_);
        e = error_;
        error_ = nullptr;
    }
    if (e) {
        std::rethrow_exception(e);
    }
}

}  // namespace locus::sys
