#include "catch_amalgamated.hpp"
#include "locus/backend/vulkan/limits.hpp"

using locus::backend::vk::attn_decode_fits;
using locus::backend::vk::attn_width_fits;
using locus::backend::vk::gpu_alloc_bytes;
using locus::backend::vk::attn_grp_dv_fits;
using locus::backend::vk::kAttnGrpDvMax;
using locus::backend::vk::kAttnPartMax;
using locus::backend::vk::seg_rope_fits;

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

TEST_CASE("decode attention keys are bounded by attn_decode's stage",
          "[vulkan]") {
    REQUIRE(attn_decode_fits(64));     // GQA head_dim
    REQUIRE(attn_decode_fits(576));    // MLA rank 512 + rope 64
    REQUIRE(attn_decode_fits(1024));
    REQUIRE_FALSE(attn_decode_fits(1025));
}

TEST_CASE("grouped decode V widths are bounded by its accumulators",
          "[vulkan]") {
    REQUIRE(attn_grp_dv_fits(64));    // GQA head_dim
    REQUIRE(attn_grp_dv_fits(128));
    REQUIRE(attn_grp_dv_fits(512));   // MLA latent rank: at the limit
    REQUIRE(attn_grp_dv_fits(kAttnGrpDvMax));
    REQUIRE_FALSE(attn_grp_dv_fits(kAttnGrpDvMax + 4));
    REQUIRE_FALSE(attn_grp_dv_fits(66));  // vec4 V loads need dv % 4
}

TEST_CASE("segmented matvec rope specs keep the epilogue's pairing",
          "[vulkan]") {
    REQUIRE(seg_rope_fits(4096, 128, 0, 128));  // GQA q: whole heads
    REQUIRE(seg_rope_fits(3072, 192, 128, 64));  // MLA q_pe tails
    REQUIRE(seg_rope_fits(576, 576, 512, 64));   // MLA k_pe (one head)
    REQUIRE(seg_rope_fits(7, 1, 0, 0));          // no rope: anything
    REQUIRE_FALSE(seg_rope_fits(4100, 128, 0, 128));  // partial head
    REQUIRE_FALSE(seg_rope_fits(4096, 128, 1, 64));   // odd offset
    REQUIRE_FALSE(seg_rope_fits(4096, 128, 0, 63));   // odd width
    REQUIRE_FALSE(seg_rope_fits(4096, 128, 96, 64));  // past the head
    REQUIRE_FALSE(seg_rope_fits(4095, 65, 0, 64));    // odd head rows
}
