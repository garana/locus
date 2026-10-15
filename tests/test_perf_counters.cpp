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

    // Available: a registered thread doing real memory traffic must push
    // the aggregate cache-references above zero, and after it exits (its
    // ThreadCounters destructs and unregisters) the aggregate returns to
    // zero.
    std::thread worker([] {
        ThreadCounters guard;  // registers this thread's counters
        std::vector<std::uint64_t> buf(1u << 20, 1);  // 8 MiB
        volatile std::uint64_t sum = 0;
        for (int pass = 0; pass < 50; ++pass) {
            for (std::uint64_t v : buf) {
                sum += v;
            }
        }
        (void)sum;
        const Totals t = read_totals();  // still registered here
        REQUIRE(t.cache_references > 0);
    });
    worker.join();

    const Totals after = read_totals();  // worker unregistered on exit
    REQUIRE(after.cache_references == 0);
}
