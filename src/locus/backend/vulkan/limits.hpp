#pragma once

#include <cstddef>
#include <cstdint>

namespace locus::backend::vk {

/**
 * Size, in floats, of the attention kernels' shared `part` stage
 * (attn_paged, attn_paged_q, attn_mla: ATTN_PART_MAX). They index
 * it up to head width * np with np = max(1, 128 / width), i.e. up to
 * max(128, width), so any head width or MLA rank above this must not
 * reach them. CMake reads this line (it must be the only one) and
 * passes the value to the shaders as -DATTN_PART_MAX and to the
 * locus target, where context.cpp static_asserts the two agree.
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
 * Size, in floats, of attn_decode.comp's shared query stage
 * (ATTN_DECODE_Q_MAX): one head's whole key, d1 + d2 floats (GQA
 * head_dim; MLA kv_lora_rank + qk_rope_dim). Wider keys take the
 * matmul attention path. CMake passes it to the shader like
 * kAttnPartMax.
 */
inline constexpr std::uint32_t kAttnDecodeQMax = 1024;

/** @returns Whether a decode attention key width fits attn_decode. */
constexpr bool attn_decode_fits(std::uint32_t key_width) {
    return key_width <= kAttnDecodeQMax;
}

/**
 * Widest V (dv floats, a multiple of 4) attn_decode_grp.comp handles
 * (ATTN_GRP_DV_MAX). Its weighted-V stage gives each thread hps =
 * ceil(HG * (dv / 4) / WG) heads of vec4 accumulators, at most HACC;
 * past that every thread idles and the kernel writes nothing. So it
 * needs HG * (dv / 4) <= HACC * WG: 16 * 128 <= 8 * 256 at 512, with
 * no margin. CMake passes it to the shader, which fails to compile if
 * HG / HACC / WG no longer cover it, and to the locus target, where
 * context.cpp static_asserts the two agree.
 */
inline constexpr std::uint32_t kAttnGrpDvMax = 512;

/** @returns Whether a V width (dv floats) fits attn_decode_grp. */
constexpr bool attn_grp_dv_fits(std::uint32_t dv) {
    return dv % 4 == 0 && dv <= kAttnGrpDvMax;
}

/**
 * Whether kv_scatter_f16 can write rows of `width` floats with these
 * destination terms: it packs pairs of floats into one pool word at
 * (dst + i) >> 1, so width must be even (an odd one reads one float
 * past the row and writes it into the pair's high half), and so must
 * every term of dst (block_stride, row_w, dst_off, dst_plane), or the
 * write lands half a word off and interleaves K and V halves (i#131).
 * Every real shape is even: kv_dim = n_kv_heads * an even head_dim,
 * the MLA row rank + rope_d with both even, and layer_stride / 2 =
 * kv_dim.
 *
 * @returns Whether all five are even.
 */
constexpr bool kv_scatter_f16_fits(std::uint32_t width,
                                   std::uint32_t block_stride,
                                   std::uint32_t row_w,
                                   std::uint32_t dst_off,
                                   std::uint32_t dst_plane) {
    return (width | block_stride | row_w | dst_off | dst_plane) % 2 == 0;
}

/**
 * Whether a row segment of matvec_q8_0_seg can be roped in its
 * epilogue: the segment is whole heads (n_rows % hs == 0) of hs rows
 * whose rd rows from ro on rotate, with hs, ro and rd even and the
 * span inside the head. The epilogue pairs row r with r ^ 1 inside
 * its 4-row workgroup, so pairs must start on even rows -- the host
 * also keeps the segment's first row even -- and the sibling of a
 * roped row is then always in range. rd 0 (no rope) always fits.
 *
 * @returns Whether (n_rows, hs, ro, rd) meets those invariants.
 */
constexpr bool seg_rope_fits(std::uint32_t n_rows, std::uint32_t hs,
                             std::uint32_t ro, std::uint32_t rd) {
    return rd == 0 || (hs != 0 && hs % 2 == 0 && ro % 2 == 0 &&
                       rd % 2 == 0 && ro + rd <= hs && n_rows % hs == 0);
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
