#include "catch_amalgamated.hpp"
#include "locus/backend/vulkan/mem_fit.hpp"

using locus::backend::vk::memory_available_from;
using locus::backend::vk::moe_experts_fit;

namespace {

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;

}  // namespace

TEST_CASE("memory_available_from uses the budget when present",
          "[vulkan]") {
    REQUIRE(memory_available_from(true, 27 * kGiB, 2 * kGiB) == 25 * kGiB);
    // Usage at or over the budget leaves nothing.
    REQUIRE(memory_available_from(true, 8 * kGiB, 8 * kGiB) == 0);
    REQUIRE(memory_available_from(true, 8 * kGiB, 9 * kGiB) == 0);
}

TEST_CASE("memory_available_from falls back to 3/4 of the heap",
          "[vulkan]") {
    REQUIRE(memory_available_from(false, 32 * kGiB, 0) == 24 * kGiB);
    REQUIRE(memory_available_from(false, 32 * kGiB, 4 * kGiB) ==
            20 * kGiB);
    REQUIRE(memory_available_from(false, 16 * kGiB, 13 * kGiB) == 0);
}

TEST_CASE("moe_experts_fit allows up to 80% of available memory",
          "[vulkan]") {
    const std::uint64_t avail = 20 * kGiB;
    REQUIRE(moe_experts_fit(15 * kGiB, avail));       // fits
    REQUIRE(moe_experts_fit(16 * kGiB, avail));       // exactly 80%
    REQUIRE_FALSE(moe_experts_fit(16 * kGiB + 1, avail));  // just over
    REQUIRE_FALSE(moe_experts_fit(1, 0));             // nothing left
    // deepseek-coder-v2-lite Q8_0 (~15 GB of experts): GPU-routed on
    // a 36 GB Mac (~28 GB budget), host-routed on a 16 GB box with
    // the no-extension fallback (12 GB heap headroom).
    REQUIRE(moe_experts_fit(15 * kGiB,
                            memory_available_from(true, 28 * kGiB, 0)));
    REQUIRE_FALSE(moe_experts_fit(
        15 * kGiB, memory_available_from(false, 16 * kGiB, 0)));
}

// i#589: on an M2 Pro (25.0 GiB budget) Qwen3-Coder-30B-A3B Q4_K_M's
// 16.35 GiB of imported experts fit next to a 4.5 GiB KV pool
// (--ctx 24576: 20.4 GiB left), not next to a 7.5 GiB one (--ctx 40960:
// 17.5 GiB left), which lost the device.
TEST_CASE("moe_experts_fit counts imported experts on Apple at 90%",
          "[vulkan]") {
    const std::uint64_t mib = 1024ull * 1024;
    const std::uint64_t experts = 16740 * mib;
    REQUIRE(moe_experts_fit(0, experts, 20900 * mib));
    REQUIRE_FALSE(moe_experts_fit(0, experts, 17877 * mib));
    // No imported experts: the 80% rule alone.
    REQUIRE(moe_experts_fit(16 * kGiB, 0, 20 * kGiB));
    REQUIRE_FALSE(moe_experts_fit(16 * kGiB + 1, 0, 20 * kGiB));
    // Copied experts keep the 80% rule even when the sum would fit.
    REQUIRE_FALSE(moe_experts_fit(17 * kGiB, 0, 20 * kGiB));
    REQUIRE(moe_experts_fit(16 * kGiB, 2 * kGiB, 20 * kGiB));
    REQUIRE_FALSE(moe_experts_fit(16 * kGiB, 2 * kGiB + 1, 20 * kGiB));
}
