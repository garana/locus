#pragma once

#include <cstdint>

// Per-thread CPU cache performance counters, summed across the inference
// compute threads and surfaced on /metrics. Linux only (raw
// perf_event_open, no dependency); a cleanly absent no-op elsewhere and
// whenever the kernel denies the counters (perf_event_paranoid /
// CAP_PERFMON). The point is to watch the cache-miss rate over time, the
// signal that tells you CPU decode is memory-bandwidth-bound.
namespace locus::sys::perf {

/**
 * @returns true when perf_event_open is PERMITTED on this host (Linux,
 * not blocked by perf_event_paranoid / CAP_PERFMON). Probed once via a
 * software event and cached. It does NOT imply any specific cache event
 * is implemented by the PMU -- per-event support is in Totals::have_*
 * (a host may permit perf yet lack e.g. the LLC event). When false,
 * ThreadCounters is a no-op and read_totals() is all-zero/all-invalid,
 * and callers omit the metrics entirely.
 */
bool available();

/** Cache counters summed over all live registered compute threads.
 *  All-zero / all-invalid when !available(). Counts are free-running
 *  and monotonic for the life of each thread; a thread's contribution
 *  disappears when it exits (so these are process-uptime counters, not
 *  all-time). Each have_* flag is false when that specific event is not
 *  implemented by this host's PMU (common for the LLC event on ARM):
 *  the caller must OMIT a metric whose event is invalid rather than
 *  report a misleading zero. have_* reflect the one-time probe
 *  (implementable on this host), not per-scrape liveness. */
struct Totals {
    std::uint64_t cache_references = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t llc_load_misses = 0;
    bool have_references = false;
    bool have_misses = false;
    bool have_llc = false;
};

/**
 * Sums read() over every registered thread's counters. Cheap -- a
 * handful of fds read under a short lock -- so it is safe on the
 * /metrics accept path (see i#85). Returns zeros when !available().
 */
Totals read_totals();

/**
 * RAII guard: construct once at the top of a compute thread's body. On
 * Linux with counters available it opens this thread's cache counters
 * and registers them; it closes and unregisters them on destruction. A
 * no-op when !available() or on non-Linux, so it is safe to place
 * unconditionally.
 *
 * A registrant MUST outlive the /metrics scrape window: read_totals()
 * sums only live registrants, so a thread that registers, counts, and
 * exits makes the aggregate DROP -- which a Prometheus _total reads as
 * a reset. Today every registrant is process-lifetime (the ThreadPool
 * workers, the EngineLoop worker); keep it that way for anything new.
 */
class ThreadCounters {
  public:
    ThreadCounters();
    ~ThreadCounters();
    ThreadCounters(const ThreadCounters&) = delete;
    ThreadCounters& operator=(const ThreadCounters&) = delete;

  private:
    // Opaque registry handle (null = unregistered). maybe_unused: the
    // non-Linux build's ctor/dtor are no-ops that never touch it.
    [[maybe_unused]] void* entry_ = nullptr;
};

}  // namespace locus::sys::perf
