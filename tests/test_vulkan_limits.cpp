#include "catch_amalgamated.hpp"
#include "locus/backend/vulkan/limits.hpp"

using locus::backend::vk::attn_width_fits;
using locus::backend::vk::gpu_alloc_bytes;
using locus::backend::vk::kAttnPartMax;

TEST_CASE("attention head widths are bounded by the shared stage",
          "[vulkan]") {
    REQUIRE(attn_width_fits(64));
    REQUIRE(attn_width_fits(128));
    // deepseek-v2/v3 MLA latent rank: exactly at the limit.
    REQUIRE(attn_width_fits(512));
    REQUIRE(attn_width_fits(kAttnPartMax));
    REQUIRE_FALSE(attn_width_fits(kAttnPartMax + 1));
    REQUIRE_FALSE(attn_width_fits(1024));
}

TEST_CASE("GPU weight allocations round up to 16 bytes", "[vulkan]") {
    REQUIRE(gpu_alloc_bytes(0) == 0);
    REQUIRE(gpu_alloc_bytes(1) == 16);
    REQUIRE(gpu_alloc_bytes(16) == 16);
    // 3 rows x 1 Q8_0 block: 102 bytes, the last word ends past 102.
    REQUIRE(gpu_alloc_bytes(102) == 112);
    REQUIRE(gpu_alloc_bytes(714) == 720);  // 7 x 3 blocks
}
