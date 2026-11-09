#include <cstdint>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/model/expert_cache.hpp"

using locus::model::ExpertCache;

namespace {

/** A plain list version of step WLRU (moe-trace-sim.py, wlru-step). */
struct Reference {
    std::uint32_t cap, step;
    std::vector<std::uint32_t> order;

    ExpertCache::Touch touch(std::uint32_t key, std::uint32_t n_expert) {
        ExpertCache::Touch t;
        std::size_t i = 0;
        while (i < order.size() && order[i] != key) ++i;
        if (i == order.size()) {
            order.push_back(key);
            t.promoted = i < cap;
            return t;
        }
        t.hot = i < cap;
        const std::size_t j = i > step ? i - step : 0;
        if (j == i) return t;
        if (i >= cap && cap > j) {
            t.promoted = t.evicted = true;
            t.evicted_layer = order[cap - 1] / n_expert;
            t.evicted_expert = order[cap - 1] % n_expert;
        }
        order.erase(order.begin() + static_cast<std::ptrdiff_t>(i));
        order.insert(order.begin() + static_cast<std::ptrdiff_t>(j), key);
        return t;
    }
};

}  // namespace

TEST_CASE("step WLRU promotes into the zone and evicts the one it "
          "passes",
          "[moe][expert_cache]") {
    ExpertCache c(1, 4, 2, 1);
    auto t = c.touch(0, 0);  // A joins rank 0, inside the zone
    REQUIRE((!t.hot && t.promoted && !t.evicted));
    t = c.touch(0, 1);  // B joins rank 1
    REQUIRE((!t.hot && t.promoted));
    t = c.touch(0, 2);  // C joins rank 2, outside: CPU
    REQUIRE((!t.hot && !t.promoted));
    t = c.touch(0, 2);  // C steps to rank 1: B leaves the zone
    REQUIRE((!t.hot && t.promoted && t.evicted));
    REQUIRE(t.evicted_expert == 1);
    REQUIRE(c.resident(0, 2));
    REQUIRE_FALSE(c.resident(0, 1));
    t = c.touch(0, 2);  // C hot now, steps to rank 0
    REQUIRE((t.hot && !t.promoted));
    t = c.touch(0, 1);  // B back in at rank 1: A (rank 1) leaves
    REQUIRE((!t.hot && t.promoted && t.evicted));
    REQUIRE(t.evicted_expert == 0);
}

TEST_CASE("step WLRU matches the list reference", "[moe][expert_cache]") {
    constexpr std::uint32_t kLayers = 3, kExpert = 16;
    for (const std::uint32_t step : {1u, 4u, 16u}) {
        for (const std::uint32_t cap : {0u, 5u, 12u}) {
            ExpertCache c(kLayers, kExpert, cap, step);
            Reference ref{cap, step, {}};
            std::uint32_t s = 12345u + step * 31u + cap;
            for (int n = 0; n < 4000; ++n) {
                s = s * 1664525u + 1013904223u;
                // Skewed picks: low experts come up more often.
                const std::uint32_t r = (s >> 8) % 1000;
                const std::uint32_t e = r * r / (1000 * 1000 / kExpert);
                const std::uint32_t l = (s >> 20) % kLayers;
                const auto got = c.touch(l, e);
                const auto want = ref.touch(l * kExpert + e, kExpert);
                REQUIRE(got.hot == want.hot);
                REQUIRE(got.promoted == want.promoted);
                REQUIRE(got.evicted == want.evicted);
                if (want.evicted) {
                    REQUIRE(got.evicted_layer == want.evicted_layer);
                    REQUIRE(got.evicted_expert == want.evicted_expert);
                }
            }
        }
    }
}

TEST_CASE("step WLRU ignores keys out of range", "[moe][expert_cache]") {
    ExpertCache c(2, 4, 3, 2);
    const auto t = c.touch(2, 0);
    REQUIRE((!t.hot && !t.promoted && !t.evicted));
    REQUIRE_FALSE(c.resident(2, 0));
}
