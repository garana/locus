#pragma once

#include <cstddef>
#include <cstdint>

namespace locus::backend::vk {

/**
 * Size, in floats, of the attention kernels' shared `part` stage
 * (attn_paged, attn_paged_q, attn_mla: ATTN_PART_MAX). They index
 * it up to head width * np with np = max(1, 128 / width), i.e. up to
 * max(128, width), so any head width or MLA rank above this must not
 * reach them. CMake reads this line and passes the same value to the
 * shaders as -DATTN_PART_MAX, so host and GLSL cannot drift.
 */
inline constexpr std::uint32_t kAttnPartMax = 512;

static_assert(kAttnPartMax >= 128,
              "attention kernels need room for their 128-thread "
              "workgroup's slices");

/**
 * @returns Whether a head width (head_dim; for MLA also the latent
 * rank) fits the attention kernels' shared stage.
 */
constexpr bool attn_width_fits(std::uint32_t width) {
    return width <= kAttnPartMax;
}

/**
 * Bytes allocated for a GPU weight copy of `bytes`: rounded up to 16
 * so a kernel reading whole words (Q8_0's 34-byte blocks end
 * mid-word) never reads past the buffer. The padding is zeroed; the
 * host copies only `bytes` from the tensor.
 */
constexpr std::size_t gpu_alloc_bytes(std::size_t bytes) {
    return (bytes + 15) & ~std::size_t{15};
}

}  // namespace locus::backend::vk
