#include <cstdint>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/sys/perf_counters.hpp"

using locus::sys::perf::available;
using locus::sys::perf::read_totals;
using locus::sys::perf::ThreadCounters;
using Totals = locus::sys::perf::Totals;

// i#86: per-thread CPU cache counters, summed for /metrics. On a host
// without perf (non-Linux, or perf_event_paranoid denies it) the guard
// is a no-op and the totals are zero -- so the metric is absent, not a
// misleading zero. On a permitted Linux host a registered thread that
// touches memory moves the aggregate, and its contribution disappears
// when it exits.
TEST_CASE("perf counters: register, read, unregister", "[perf]") {
    if (!available()) {
        // Unavailable: the guard must be a harmless no-op and totals
        // must read zero without crashing.
        ThreadCounters guard;
        const Totals t = read_totals();
        REQUIRE(t.cache_references == 0);
        REQUIRE(t.cache_misses == 0);
        REQUIRE(t.llc_load_misses == 0);
        SUCCEED("perf counters unavailable here; no-op path verified");
        return;
    }

    // Available: a registered thread doing real memory traffic must
    // RAISE the aggregate while it runs, and its (large) contribution
    // must DROP OUT when it exits and unregisters. Assertions are
    // baseline-relative, not absolute: the matvec ThreadPool is a
    // singleton whose workers register at first use and never exit, so
    // other registered threads may already be counting; only the
    // spawned worker's own delta is a reliable signal.
    const Totals base = read_totals();  // whatever is already live
    if (!base.have_references) {
        SUCCEED("cache-references event not implemented on this PMU");
        return;
    }
    Totals inside;
    std::thread worker([&] {
        ThreadCounters guard;  // registers THIS thread's counters
        std::vector<std::uint64_t> buf(1u << 20, 1);  // 8 MiB
        volatile std::uint64_t sum = 0;
        for (int pass = 0; pass < 50; ++pass) {
            for (std::uint64_t v : buf) {
                sum = sum + v;  // not sum += v: compound assignment to a
                                // volatile is deprecated in C++20 (GCC
                                // -Wvolatile); plain assignment still
                                // defeats the dead-loop elision.
            }
        }
        (void)sum;
        inside = read_totals();  // base + this worker's own traffic
    });
    worker.join();

    const Totals after = read_totals();  // worker unregistered on exit
    REQUIRE(inside.cache_references > base.cache_references);
    REQUIRE(after.cache_references < inside.cache_references);
}
