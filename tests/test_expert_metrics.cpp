#include <cstdint>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/model/expert_metrics.hpp"

using locus::model::ExpertCalls;
using locus::model::expert_api;

namespace {

/** @returns The calls of one (layer, expert, api), 0 when absent. */
std::uint64_t calls_of(const std::vector<ExpertCalls::Series>& s,
                       std::uint32_t layer, std::uint32_t expert,
                       std::string_view api) {
    for (const auto& c : s) {
        if (c.layer == layer && c.expert == expert && c.api == api) {
            return c.calls;
        }
    }
    return 0;
}

}  // namespace

TEST_CASE("expert calls count per layer, expert and api", "[metrics]") {
    ExpertCalls::reset();
    REQUIRE(ExpertCalls::snapshot().empty());

    ExpertCalls::record(1, 3, 4, 8, true, "cuda");
    ExpertCalls::record(1, 3, 4, 8, true, "cuda");
    ExpertCalls::record(2, 7, 4, 8, false, "sse4");
    // Past the table's size: dropped, not a crash.
    ExpertCalls::record(4, 0, 4, 8, true, "cuda");
    ExpertCalls::record(0, 8, 4, 8, true, "cuda");

    const auto s = ExpertCalls::snapshot();
    REQUIRE(s.size() == 2);
    REQUIRE(calls_of(s, 1, 3, "cuda") == 2);
    REQUIRE(calls_of(s, 2, 7, "sse4") == 1);
    for (const auto& c : s) {
        REQUIRE(std::string(c.device) == (c.api == "cuda" ? "gpu" : "cpu"));
    }

    ExpertCalls::reset();
    REQUIRE(ExpertCalls::snapshot().empty());
}

TEST_CASE("expert api label follows the backend", "[metrics]") {
    bool gpu = true;
    REQUIRE(expert_api("sse4", gpu) == "sse4");
    REQUIRE_FALSE(gpu);
    REQUIRE(expert_api("cuda", gpu) == "cuda");
    REQUIRE(gpu);
    gpu = false;
    const std::string_view vk = expert_api("vulkan", gpu);
    REQUIRE(gpu);
    REQUIRE((vk == "vulkan" || vk == "moltenvk"));
}
