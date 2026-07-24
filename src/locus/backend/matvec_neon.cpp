#include "locus/backend/variants.hpp"

#if defined(__aarch64__)

#include <arm_neon.h>

#include <cstring>

#include "iq_grids.h"

namespace locus::backend {

namespace {

std::uint16_t load_u16(const std::byte* p) {
    std::uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

float dot_f32_neon(const float* w, const float* x,
                   std::size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vfmaq_f32(acc, vld1q_f32(w + i), vld1q_f32(x + i));
    }
    float s = vaddvq_f32(acc);
    for (; i < n; ++i) {
        s += w[i] * x[i];
    }
    return s;
}

/** One Q8_0 block: f16 scale + 32 int8 quants (34 bytes). */
float dot_q8_0_block_neon(const std::byte* blk, const float* x) {
    const float d = f16_to_f32(load_u16(blk));
    const auto* q = reinterpret_cast<const std::int8_t*>(blk + 2);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (std::size_t g = 0; g < 32; g += 8) {
        const int16x8_t w16 = vmovl_s8(vld1_s8(q + g));
        const float32x4_t lo =
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
        const float32x4_t hi =
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
        acc = vfmaq_f32(acc, lo, vld1q_f32(x + g));
        acc = vfmaq_f32(acc, hi, vld1q_f32(x + g + 4));
    }
    return d * vaddvq_f32(acc);
}

/** Unpacks the 6-bit scale/min of K-quant sub-block j (ggml's
 * get_scale_min_k4). Matches scale_min_k4 in cpu_ops.cpp. */
void scale_min_k4_neon(int j, const std::uint8_t* q,
                       std::uint8_t& d, std::uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = static_cast<std::uint8_t>((q[j + 4] & 0x0f) |
                                      ((q[j - 4] >> 6) << 4));
        m = static_cast<std::uint8_t>((q[j + 4] >> 4) |
                                      ((q[j] >> 6) << 4));
    }
}

/**
 * Accumulates 16 dequantized nibbles into acc: for each 4-bit
 * value v, adds (d*v - m) * x. `vals` holds the 16 nibble values
 * already masked/shifted to 0..15; `xp` points at 16 floats.
 */
float32x4_t accum16_q4k(float32x4_t acc, uint8x16_t vals, float d,
                        float m, const float* xp) {
    const uint16x8_t vlo = vmovl_u8(vget_low_u8(vals));
    const uint16x8_t vhi = vmovl_u8(vget_high_u8(vals));
    float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vlo)));
    float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vlo)));
    float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vhi)));
    float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vhi)));
    const float32x4_t dv = vdupq_n_f32(d);
    const float32x4_t mv = vdupq_n_f32(m);
    f0 = vsubq_f32(vmulq_f32(f0, dv), mv);
    f1 = vsubq_f32(vmulq_f32(f1, dv), mv);
    f2 = vsubq_f32(vmulq_f32(f2, dv), mv);
    f3 = vsubq_f32(vmulq_f32(f3, dv), mv);
    acc = vfmaq_f32(acc, f0, vld1q_f32(xp));
    acc = vfmaq_f32(acc, f1, vld1q_f32(xp + 4));
    acc = vfmaq_f32(acc, f2, vld1q_f32(xp + 8));
    acc = vfmaq_f32(acc, f3, vld1q_f32(xp + 12));
    return acc;
}

/**
 * One Q4_K super-block (256 elements, 144 bytes): fused dequant +
 * dot against x. Follows dequant_block_q4_k's layout -- eight
 * 32-element sub-blocks paired into 64-element groups, low nibbles
 * then high nibbles, each with its own (d*sc, dmin*mn) scale/min.
 */
float dot_q4_k_block_neon(const std::byte* blk, const float* x) {
    const float d = f16_to_f32(load_u16(blk));
    const float dmin = f16_to_f32(load_u16(blk + 2));
    const auto* scales =
        reinterpret_cast<const std::uint8_t*>(blk + 4);
    const auto* q =
        reinterpret_cast<const std::uint8_t*>(blk + 16);
    const uint8x16_t mask = vdupq_n_u8(0x0f);
    float32x4_t acc = vdupq_n_f32(0.0f);
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        std::uint8_t sc, mn;
        scale_min_k4_neon(is + 0, scales, sc, mn);
        const float d1 = d * sc, m1 = dmin * mn;
        scale_min_k4_neon(is + 1, scales, sc, mn);
        const float d2 = d * sc, m2 = dmin * mn;
        const uint8x16_t qb0 = vld1q_u8(q);
        const uint8x16_t qb1 = vld1q_u8(q + 16);
        acc = accum16_q4k(acc, vandq_u8(qb0, mask), d1, m1, x + j);
        acc = accum16_q4k(acc, vandq_u8(qb1, mask), d1, m1,
                          x + j + 16);
        acc = accum16_q4k(acc, vshrq_n_u8(qb0, 4), d2, m2,
                          x + j + 32);
        acc = accum16_q4k(acc, vshrq_n_u8(qb1, 4), d2, m2,
                          x + j + 48);
        q += 32;
        is += 2;
    }
    return vaddvq_f32(acc);
}

/**
 * Accumulates 16 Q6_K values into acc: reconstructs each 6-bit
 * quant from the low/high nibble of ql (HIGH picks the high
 * nibble) and 2 bits of qh at QH_SHIFT, subtracts the 32 bias,
 * scales by dsc = d*sc, and multiplies by x.
 */
template <bool HIGH, int QH_SHIFT>
float32x4_t accum16_q6k(float32x4_t acc, const std::uint8_t* ql16,
                        const std::uint8_t* qh16, float dsc,
                        const float* xp) {
    const uint8x16_t qlb = vld1q_u8(ql16);
    const uint8x16_t qhb = vld1q_u8(qh16);
    const uint8x16_t low =
        HIGH ? vshrq_n_u8(qlb, 4)
             : vandq_u8(qlb, vdupq_n_u8(0x0f));
    uint8x16_t qh_sh;
    if constexpr (QH_SHIFT == 0) {
        qh_sh = qhb;
    } else {
        qh_sh = vshrq_n_u8(qhb, QH_SHIFT == 0 ? 1 : QH_SHIFT);
    }
    const uint8x16_t high =
        vshlq_n_u8(vandq_u8(qh_sh, vdupq_n_u8(3)), 4);
    const int8x16_t v = vsubq_s8(
        vreinterpretq_s8_u8(vorrq_u8(low, high)), vdupq_n_s8(32));
    const int16x8_t vlo = vmovl_s8(vget_low_s8(v));
    const int16x8_t vhi = vmovl_s8(vget_high_s8(v));
    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vlo)));
    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vlo)));
    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vhi)));
    float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vhi)));
    const float32x4_t s = vdupq_n_f32(dsc);
    acc = vfmaq_f32(acc, vmulq_f32(f0, s), vld1q_f32(xp));
    acc = vfmaq_f32(acc, vmulq_f32(f1, s), vld1q_f32(xp + 4));
    acc = vfmaq_f32(acc, vmulq_f32(f2, s), vld1q_f32(xp + 8));
    acc = vfmaq_f32(acc, vmulq_f32(f3, s), vld1q_f32(xp + 12));
    return acc;
}

/**
 * One Q6_K super-block (256 elements, 210 bytes): fused dequant +
 * dot. Follows dequant_block_q6_k -- two 128-element halves, each
 * with four 32-value streams (ql low/high nibble x qh 2-bit pair)
 * scaled by one of eight int8 scales (two per stream, split at
 * lane 16).
 */
float dot_q6_k_block_neon(const std::byte* blk, const float* x) {
    const auto* ql = reinterpret_cast<const std::uint8_t*>(blk);
    const auto* qh =
        reinterpret_cast<const std::uint8_t*>(blk + 128);
    const auto* sc =
        reinterpret_cast<const std::int8_t*>(blk + 192);
    const float d = f16_to_f32(load_u16(blk + 208));
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int n = 0; n < 256; n += 128) {
        acc = accum16_q6k<false, 0>(acc, ql, qh, d * sc[0],
                                    x + n);
        acc = accum16_q6k<false, 0>(acc, ql + 16, qh + 16,
                                    d * sc[1], x + n + 16);
        acc = accum16_q6k<false, 2>(acc, ql + 32, qh, d * sc[2],
                                    x + n + 32);
        acc = accum16_q6k<false, 2>(acc, ql + 48, qh + 16,
                                    d * sc[3], x + n + 48);
        acc = accum16_q6k<true, 4>(acc, ql, qh, d * sc[4],
                                   x + n + 64);
        acc = accum16_q6k<true, 4>(acc, ql + 16, qh + 16,
                                   d * sc[5], x + n + 80);
        acc = accum16_q6k<true, 6>(acc, ql + 32, qh, d * sc[6],
                                   x + n + 96);
        acc = accum16_q6k<true, 6>(acc, ql + 48, qh + 16,
                                   d * sc[7], x + n + 112);
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return vaddvq_f32(acc);
}

/**
 * One IQ1_S super-block (256 elements, 50 bytes): the codebook
 * lookup stays scalar (a gather), but the per-8 dequant+dot is
 * vectorized. Follows dequant_block_iq1_s: eight sub-blocks, each
 * with a 3-bit scale and a sign delta, four 8-wide grid entries.
 */
float dot_iq1_s_block_neon(const std::byte* blk, const float* x) {
    constexpr float kIq1sDelta = 0.125f;
    const float d = f16_to_f32(load_u16(blk));
    const auto* qs =
        reinterpret_cast<const std::uint8_t*>(blk + 2);
    float32x4_t acc = vdupq_n_f32(0.0f);
    const float* xp = x;
    for (int ib = 0; ib < 8; ++ib) {
        std::uint16_t qh;
        std::memcpy(&qh, blk + 34 + 2 * ib, 2);
        const float dl = d * (2 * ((qh >> 12) & 7) + 1);
        const float delta =
            (qh & 0x8000) ? -kIq1sDelta : kIq1sDelta;
        const float32x4_t dlv = vdupq_n_f32(dl);
        const float32x4_t dv = vdupq_n_f32(delta);
        for (int l = 0; l < 4; ++l) {
            const auto* grid =
                reinterpret_cast<const std::int8_t*>(
                    iq1s_grid +
                    (qs[l] | (((qh >> (3 * l)) & 7) << 8)));
            const int16x8_t g16 = vmovl_s8(vld1_s8(grid));
            float32x4_t glo =
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(g16)));
            float32x4_t ghi =
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(g16)));
            // dl * (grid + delta)
            glo = vmulq_f32(vaddq_f32(glo, dv), dlv);
            ghi = vmulq_f32(vaddq_f32(ghi, dv), dlv);
            acc = vfmaq_f32(acc, glo, vld1q_f32(xp));
            acc = vfmaq_f32(acc, ghi, vld1q_f32(xp + 4));
            xp += 8;
        }
        qs += 4;
    }
    return vaddvq_f32(acc);
}

}  // namespace

void matvec_neon(const Mat& w, std::span<const float> x,
                 std::span<float> out) {
    if (w.type == gguf::TensorType::kF32) {
        const auto* rows =
            reinterpret_cast<const float*>(w.data);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out[r] = dot_f32_neon(
                rows + static_cast<std::size_t>(r) * w.cols,
                x.data(), w.cols);
        }
        return;
    }
    if (w.type == gguf::TensorType::kQ8_0) {
        const std::size_t row_bytes = w.cols / 32ull * 34ull;
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::byte* row = w.data + r * row_bytes;
            float acc = 0.0f;
            for (std::uint32_t b = 0; b < w.cols / 32; ++b) {
                acc += dot_q8_0_block_neon(row + b * 34,
                                           x.data() + b * 32);
            }
            out[r] = acc;
        }
        return;
    }
    if (w.type == gguf::TensorType::kQ4_K) {
        const std::size_t row_bytes = w.cols / 256ull * 144ull;
        const std::size_t nblk = w.cols / 256;
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::byte* row = w.data + r * row_bytes;
            float acc = 0.0f;
            for (std::size_t b = 0; b < nblk; ++b) {
                acc += dot_q4_k_block_neon(
                    row + b * 144, x.data() + b * 256);
            }
            out[r] = acc;
        }
        return;
    }
    if (w.type == gguf::TensorType::kQ6_K) {
        const std::size_t row_bytes = w.cols / 256ull * 210ull;
        const std::size_t nblk = w.cols / 256;
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::byte* row = w.data + r * row_bytes;
            float acc = 0.0f;
            for (std::size_t b = 0; b < nblk; ++b) {
                acc += dot_q6_k_block_neon(
                    row + b * 210, x.data() + b * 256);
            }
            out[r] = acc;
        }
        return;
    }
    if (w.type == gguf::TensorType::kIQ1_S) {
        const std::size_t row_bytes = w.cols / 256ull * 50ull;
        const std::size_t nblk = w.cols / 256;
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::byte* row = w.data + r * row_bytes;
            float acc = 0.0f;
            for (std::size_t b = 0; b < nblk; ++b) {
                acc += dot_iq1_s_block_neon(
                    row + b * 50, x.data() + b * 256);
            }
            out[r] = acc;
        }
        return;
    }
    matvec(w, x, out);  // other types: scalar reference
}

}  // namespace locus::backend

#endif  // __aarch64__
