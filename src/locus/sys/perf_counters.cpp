#include "locus/sys/perf_counters.hpp"

#ifdef __linux__

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <mutex>
#include <vector>

namespace locus::sys::perf {
namespace {

// One thread's three cache counters. fd < 0 for any that failed to open
// (a thread may get some and not others); read_totals just skips those.
struct Entry {
    int fd_ref = -1;
    int fd_miss = -1;
    int fd_llc = -1;
};

std::mutex g_mu;
std::vector<Entry*> g_entries;  // live registered compute threads

// Open a per-THREAD (pid 0), any-CPU (cpu -1) counter, counting now.
// exclude_kernel=1 keeps it to user space, which is what the common
// perf_event_paranoid=2 permits unprivileged and is where the matvec's
// cache traffic lives. @returns an fd, or -1 if the kernel refuses.
int perf_open(std::uint32_t type, std::uint64_t config) {
    perf_event_attr attr{};
    attr.size = sizeof(attr);
    attr.type = type;
    attr.config = config;
    attr.disabled = 0;        // free-running from open
    attr.exclude_kernel = 1;  // user space only (paranoid=2 friendly)
    attr.exclude_hv = 1;
    const long fd = ::syscall(__NR_perf_event_open, &attr, /*pid=*/0,
                              /*cpu=*/-1, /*group_fd=*/-1, /*flags=*/0UL);
    return static_cast<int>(fd);
}

std::uint64_t read_one(int fd) {
    if (fd < 0) {
        return 0;
    }
    std::uint64_t v = 0;
    const ssize_t n = ::read(fd, &v, sizeof(v));
    return n == static_cast<ssize_t>(sizeof(v)) ? v : 0;
}

}  // namespace

bool available() {
    // Probe once: if the kernel lets us open a cache-references counter
    // on this thread, per-thread counters work for the process.
    static const bool ok = [] {
        const int fd =
            perf_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
        if (fd < 0) {
            return false;
        }
        ::close(fd);
        return true;
    }();
    return ok;
}

Totals read_totals() {
    Totals t;
    if (!available()) {
        return t;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    for (const Entry* e : g_entries) {
        t.cache_references += read_one(e->fd_ref);
        t.cache_misses += read_one(e->fd_miss);
        t.llc_load_misses += read_one(e->fd_llc);
    }
    return t;
}

ThreadCounters::ThreadCounters() {
    if (!available()) {
        return;
    }
    auto* e = new Entry;
    e->fd_ref = perf_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    e->fd_miss = perf_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    e->fd_llc = perf_open(
        PERF_TYPE_HW_CACHE,
        static_cast<std::uint64_t>(PERF_COUNT_HW_CACHE_LL) |
            (static_cast<std::uint64_t>(PERF_COUNT_HW_CACHE_OP_READ) << 8) |
            (static_cast<std::uint64_t>(PERF_COUNT_HW_CACHE_RESULT_MISS)
             << 16));
    if (e->fd_ref < 0 && e->fd_miss < 0 && e->fd_llc < 0) {
        delete e;  // nothing opened; stay unregistered
        return;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    g_entries.push_back(e);
    entry_ = e;
}

ThreadCounters::~ThreadCounters() {
    if (entry_ == nullptr) {
        return;
    }
    auto* e = static_cast<Entry*>(entry_);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_entries.erase(std::remove(g_entries.begin(), g_entries.end(), e),
                        g_entries.end());
    }
    if (e->fd_ref >= 0) {
        ::close(e->fd_ref);
    }
    if (e->fd_miss >= 0) {
        ::close(e->fd_miss);
    }
    if (e->fd_llc >= 0) {
        ::close(e->fd_llc);
    }
    delete e;
}

}  // namespace locus::sys::perf

#else  // !__linux__

namespace locus::sys::perf {

bool available() { return false; }
Totals read_totals() { return {}; }
ThreadCounters::ThreadCounters() = default;
ThreadCounters::~ThreadCounters() = default;

}  // namespace locus::sys::perf

#endif  // __linux__
