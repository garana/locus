#include "locus/backend/variants.hpp"

#if defined(__aarch64__)

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "iq_grids.h"
#include "locus/backend/cpu_ops.hpp"

namespace locus::backend {
// matvec_neon_dotprod.cpp compiles this file again with the dot-product
// extension enabled, into neon_dotprod:: (variants.hpp).
#if defined(LOCUS_NEON_DOTPROD_VARIANT)
namespace neon_dotprod {
namespace self = ::locus::backend::neon_dotprod;
#else
namespace self = ::locus::backend;
#endif

namespace {
/** f16 bits -> f32 through the hardware convert: exact for every
 *  half (the conversion only widens), unlike the out-of-line portable
 *  f16_to_f32, which the per-block scale loads were spending a third
 *  of the kernel time in. */
inline float half_to_float(std::uint16_t h) {
    return vget_lane_f32(
        vget_low_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h)))), 0);
}


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

/** Widens BF16 exactly: a bf16 is the top half of an f32, so a 16-bit
 *  left shift is the conversion. */
struct WidenBf16 {
    static float32x4_t lo(uint16x8_t v) {
        return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16));
    }
    static float32x4_t hi(uint16x8_t v) {
        return vreinterpretq_f32_u32(vshll_high_n_u16(v, 16));
    }
    static float one(std::uint16_t h) {
        const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
        float f;
        std::memcpy(&f, &u, sizeof f);
        return f;
    }
};

/** Widens F16 exactly through FCVTL (base AArch64 SIMD, no +fp16). */
struct WidenF16 {
    static float32x4_t lo(uint16x8_t v) {
        return vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(v)));
    }
    static float32x4_t hi(uint16x8_t v) {
        return vcvt_high_f32_f16(vreinterpretq_f16_u16(v));
    }
    static float one(std::uint16_t h) { return half_to_float(h); }
};

/**
 * 16-bit float row . f32 x (W: WidenBf16 or WidenF16). The products go
 * into two 4-lane FMA chains (8 weights per step), summed at the end,
 * so the order differs from the scalar dot's single chain (agrees to
 * float rounding).
 */
template <class W>
float dot_h16_neon(const std::uint16_t* w, const float* x, std::size_t n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t v = vld1q_u16(w + i);
        a0 = vfmaq_f32(a0, W::lo(v), vld1q_f32(x + i));
        a1 = vfmaq_f32(a1, W::hi(v), vld1q_f32(x + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < n; ++i) {
        s += W::one(w[i]) * x[i];
    }
    return s;
}

float dot_bf16_neon(const std::uint16_t* w, const float* x,
                    std::size_t n) {
    return dot_h16_neon<WidenBf16>(w, x, n);
}

/**
 * R 16-bit float rows . T tokens, each dot in dot_h16_neon's exact
 * order (two 4-lane chains, summed at the end, then the scalar tail),
 * so each result is byte-identical to it. Each 8-weight step is
 * widened once and used for all T tokens; each x load is used for all
 * R rows.
 *
 * @param w The first row; row k starts at w + k * cols.
 * @param x The first token; token j starts at x + j * cols.
 * @param out out[k * stride + j] receives row k . token j.
 */
template <class W, int R, int T>
void h16_tile_neon(const std::uint16_t* w, const float* x,
                   std::size_t cols, float* out, std::size_t stride) {
    float32x4_t a0[R][T], a1[R][T];
    for (int k = 0; k < R; ++k) {
        for (int j = 0; j < T; ++j) {
            a0[k][j] = vdupq_n_f32(0.0f);
            a1[k][j] = vdupq_n_f32(0.0f);
        }
    }
    std::size_t i = 0;
    for (; i + 8 <= cols; i += 8) {
        float32x4_t lo[R], hi[R];
        for (int k = 0; k < R; ++k) {
            const uint16x8_t v = vld1q_u16(w + k * cols + i);
            lo[k] = W::lo(v);
            hi[k] = W::hi(v);
        }
        for (int j = 0; j < T; ++j) {
            const float32x4_t x0 = vld1q_f32(x + j * cols + i);
            const float32x4_t x1 = vld1q_f32(x + j * cols + i + 4);
            for (int k = 0; k < R; ++k) {
                a0[k][j] = vfmaq_f32(a0[k][j], lo[k], x0);
                a1[k][j] = vfmaq_f32(a1[k][j], hi[k], x1);
            }
        }
    }
    for (int k = 0; k < R; ++k) {
        for (int j = 0; j < T; ++j) {
            float s = vaddvq_f32(vaddq_f32(a0[k][j], a1[k][j]));
            for (std::size_t e = i; e < cols; ++e) {
                s += W::one(w[k * cols + e]) * x[j * cols + e];
            }
            out[k * stride + j] = s;
        }
    }
}

/** One Q8_0 block: f16 scale + 32 int8 quants (34 bytes). */
float dot_q8_0_block_neon(const std::byte* blk, const float* x) {
    const float d = half_to_float(load_u16(blk));
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
 * Dot of four weight vectors f[0..3] with 16 consecutive x floats at
 * xp, in a fresh accumulator. Shared by the single-token K-quant
 * dots and the R11 batched kernel so both use the identical
 * accumulation order (byte-exactness).
 */
float32x4_t dot16(const float32x4_t f[4], const float* xp) {
    float32x4_t acc = vmulq_f32(f[0], vld1q_f32(xp));
    acc = vfmaq_f32(acc, f[1], vld1q_f32(xp + 4));
    acc = vfmaq_f32(acc, f[2], vld1q_f32(xp + 8));
    acc = vfmaq_f32(acc, f[3], vld1q_f32(xp + 12));
    return acc;
}

/**
 * Widens 16 Q4_K nibbles (already masked/shifted to 0..15) to f32 in
 * f[0..3], unscaled. Weight-only, so the R11 batched kernel converts
 * once and reuses it across all tokens.
 */
void cvt16_q4k(uint8x16_t vals, float32x4_t f[4]) {
    const uint16x8_t vlo = vmovl_u8(vget_low_u8(vals));
    const uint16x8_t vhi = vmovl_u8(vget_high_u8(vals));
    f[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vlo)));
    f[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vlo)));
    f[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vhi)));
    f[3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vhi)));
}

/**
 * Sums x over each 32-element sub-block into xs (cols / 32 floats):
 * the Q4_K min term dmin*mn*sum(x) needs it once per sub-block, and
 * it is shared by every row of the matvec.
 */
void subblock_sums(const float* x, std::size_t cols, float* xs) {
    for (std::size_t s = 0; s < cols / 32; ++s) {
        float32x4_t a = vld1q_f32(x + s * 32);
        for (int k = 4; k < 32; k += 4) {
            a = vaddq_f32(a, vld1q_f32(x + s * 32 + k));
        }
        xs[s] = vaddvq_f32(a);
    }
}

/** The Q4_K scales of one 64-element group (sub-blocks is, is+1). */
struct Q4kGroup {
    float d1, d2;  // d * sc
    float mins;    // dmin*mn0*xs[is] + dmin*mn1*xs[is+1]
};

Q4kGroup q4k_group(const std::uint8_t* scales, int is, float d,
                   float dmin, const float* xs) {
    std::uint8_t sc0, mn0, sc1, mn1;
    scale_min_k4_neon(is, scales, sc0, mn0);
    scale_min_k4_neon(is + 1, scales, sc1, mn1);
    return {d * sc0, d * sc1,
            dmin * mn0 * xs[is] + dmin * mn1 * xs[is + 1]};
}

/**
 * One Q4_K super-block (256 elements, 144 bytes) against x. Follows
 * dequant_block_q4_k's layout -- eight 32-element sub-blocks paired
 * into 64-element groups, low nibbles then high nibbles -- factored
 * per sub-block s as d*sc_s * sum(q * x) - dmin*mn_s * xs[s], with
 * xs = subblock_sums(x) for this block. Each 16-value quant dot runs
 * in a fresh accumulator, so they pipeline instead of chaining.
 */
float dot_q4_k_block_neon(const std::byte* blk, const float* x,
                          const float* xs) {
    const float d = half_to_float(load_u16(blk));
    const float dmin = half_to_float(load_u16(blk + 2));
    const auto* scales = reinterpret_cast<const std::uint8_t*>(blk + 4);
    const auto* q = reinterpret_cast<const std::uint8_t*>(blk + 16);
    const uint8x16_t mask = vdupq_n_u8(0x0f);
    float32x4_t acc = vdupq_n_f32(0.0f);
    float mins = 0.0f;
    for (int j = 0, is = 0; j < 256; j += 64, is += 2, q += 32) {
        const Q4kGroup g = q4k_group(scales, is, d, dmin, xs);
        const uint8x16_t qb0 = vld1q_u8(q);
        const uint8x16_t qb1 = vld1q_u8(q + 16);
        float32x4_t f0[4], f1[4], f2[4], f3[4];
        cvt16_q4k(vandq_u8(qb0, mask), f0);
        cvt16_q4k(vandq_u8(qb1, mask), f1);
        cvt16_q4k(vshrq_n_u8(qb0, 4), f2);
        cvt16_q4k(vshrq_n_u8(qb1, 4), f3);
        acc = vfmaq_n_f32(
            acc, vaddq_f32(dot16(f0, x + j), dot16(f1, x + j + 16)), g.d1);
        acc = vfmaq_n_f32(
            acc, vaddq_f32(dot16(f2, x + j + 32), dot16(f3, x + j + 48)),
            g.d2);
        mins += g.mins;
    }
    return vaddvq_f32(acc) - mins;
}

/**
 * Widens one 16-value Q6_K stream to f32 in g[0..3], unscaled:
 * reconstructs each 6-bit quant from the low/high nibble of ql (HIGH
 * picks the high nibble) and 2 bits of qh at QH_SHIFT, minus the 32
 * bias. The d*sc scale is applied once per stream by the caller.
 */
template <bool HIGH, int QH_SHIFT>
void cvt16_q6k(const std::uint8_t* ql16, const std::uint8_t* qh16,
               float32x4_t g[4]) {
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
    g[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vlo)));
    g[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vlo)));
    g[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vhi)));
    g[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vhi)));
}

/** acc += d*sc * dot(stream, x): one Q6_K stream of 16 values. */
template <bool HIGH, int QH_SHIFT>
float32x4_t stream_q6k(float32x4_t acc, const std::uint8_t* ql16,
                       const std::uint8_t* qh16, float dsc,
                       const float* xp) {
    float32x4_t g[4];
    cvt16_q6k<HIGH, QH_SHIFT>(ql16, qh16, g);
    return vfmaq_n_f32(acc, dot16(g, xp), dsc);
}

/**
 * One Q6_K super-block (256 elements, 210 bytes): fused dequant +
 * dot. Follows dequant_block_q6_k -- two 128-element halves, each
 * with four 32-value streams (ql low/high nibble x qh 2-bit pair)
 * scaled by one of eight int8 scales (two per stream, split at
 * lane 16). Each 16-value stream is dotted raw in a fresh
 * accumulator and scaled once by d*sc.
 */
float dot_q6_k_block_neon(const std::byte* blk, const float* x) {
    const auto* ql = reinterpret_cast<const std::uint8_t*>(blk);
    const auto* qh =
        reinterpret_cast<const std::uint8_t*>(blk + 128);
    const auto* sc =
        reinterpret_cast<const std::int8_t*>(blk + 192);
    const float d = half_to_float(load_u16(blk + 208));
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int n = 0; n < 256; n += 128) {
        acc = stream_q6k<false, 0>(acc, ql, qh, d * sc[0], x + n);
        acc = stream_q6k<false, 0>(acc, ql + 16, qh + 16, d * sc[1],
                                   x + n + 16);
        acc = stream_q6k<false, 2>(acc, ql + 32, qh, d * sc[2],
                                   x + n + 32);
        acc = stream_q6k<false, 2>(acc, ql + 48, qh + 16, d * sc[3],
                                   x + n + 48);
        acc = stream_q6k<true, 4>(acc, ql, qh, d * sc[4], x + n + 64);
        acc = stream_q6k<true, 4>(acc, ql + 16, qh + 16, d * sc[5],
                                  x + n + 80);
        acc = stream_q6k<true, 6>(acc, ql + 32, qh, d * sc[6],
                                  x + n + 96);
        acc = stream_q6k<true, 6>(acc, ql + 48, qh + 16, d * sc[7],
                                  x + n + 112);
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
    const float d = half_to_float(load_u16(blk));
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

/** The 2-bit qh field at SH, moved to bits 4..5 of each byte. */
template <int SH>
uint8x16_t q6_high(uint8x16_t h) {
    const uint8x16_t m3 = vdupq_n_u8(3);
    if constexpr (SH == 0) {
        return vshlq_n_u8(vandq_u8(m3, h), 4);
    } else {
        return vshlq_n_u8(vandq_u8(m3, vshrq_n_u8(h, SH)), 4);
    }
}

/** 16 signed Q6_K codes (low | high bits, minus 32). */
inline int8x16_t q6_codes(uint8x16_t low, uint8x16_t high) {
    return vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(low, high)),
                    vdupq_n_s8(32));
}

/** One block-32 weight block as 2 x 16 int8 values (low, high
 *  halves) and its f16 scale. */
inline float unpack_block32_neon(gguf::TensorType type, const std::byte* blk,
                                 int8x16_t& lo, int8x16_t& hi) {
    const float d = half_to_float(load_u16(blk));
    const auto* q = reinterpret_cast<const std::uint8_t*>(blk + 2);
    if (type == gguf::TensorType::kQ8_0) {
        lo = vld1q_s8(reinterpret_cast<const std::int8_t*>(q));
        hi = vld1q_s8(reinterpret_cast<const std::int8_t*>(q + 16));
        return d;
    }
    const uint8x16_t v = vld1q_u8(q);
    const uint8x16_t nlo = vandq_u8(v, vdupq_n_u8(0x0f));
    const uint8x16_t nhi = vshrq_n_u8(v, 4);
    if (type == gguf::TensorType::kQ4_0) {
        lo = vsubq_s8(vreinterpretq_s8_u8(nlo), vdupq_n_s8(8));
        hi = vsubq_s8(vreinterpretq_s8_u8(nhi), vdupq_n_s8(8));
    } else {  // kIQ4_NL
        const int8x16_t tbl = vld1q_s8(kvalues_iq4nl);
        lo = vqtbl1q_s8(tbl, nlo);
        hi = vqtbl1q_s8(tbl, nhi);
    }
    return d;
}

/** Lane l = sum over quants 4l..4l+3 of both halves (dotprod, or the
 *  same exact integers through widening multiplies without it). */
inline int32x4_t lanes32(int8x16_t qlo, int8x16_t qhi, const std::int8_t* x) {
    const int8x16_t xlo = vld1q_s8(x), xhi = vld1q_s8(x + 16);
#if defined(__ARM_FEATURE_DOTPROD) || defined(LOCUS_NEON_DOTPROD_VARIANT)
    return vdotq_s32(vdotq_s32(vdupq_n_s32(0), qlo, xlo), qhi, xhi);
#else
    auto d4 = [](int8x16_t a, int8x16_t b) {
        const int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
        return vpaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
    };
    return vaddq_s32(d4(qlo, xlo), d4(qhi, xhi));
#endif
}

/**
 * dot(block-32 weight row, Q8_0 activation), bit-identical to the
 * scalar vec_dot_block32_q8_0 (ggml's non-repacked NEON
 * vec_dot_q8_0_q8_0 order): two blocks per step into four float lanes
 * each (a mul, then an add), the pairwise lane reduction, a scalar
 * tail for an odd last block.
 */
float row_block32_q8_0_neon(gguf::TensorType type, const std::byte* row,
                            std::size_t bb, std::size_t nb, const float* yd,
                            const std::int8_t* yq) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    std::size_t b = 0;
    int8x16_t lo, hi;
    for (; b + 1 < nb; b += 2) {
        const float d0 = unpack_block32_neon(type, row + b * bb, lo, hi) *
                         yd[b];
        const int32x4_t p0 = lanes32(lo, hi, yq + b * 32);
        const float d1 =
            unpack_block32_neon(type, row + (b + 1) * bb, lo, hi) *
            yd[b + 1];
        const int32x4_t p1 = lanes32(lo, hi, yq + (b + 1) * 32);
        s0 = vaddq_f32(s0, vmulq_n_f32(vcvtq_f32_s32(p0), d0));
        s1 = vaddq_f32(s1, vmulq_n_f32(vcvtq_f32_s32(p1), d1));
    }
    float sum = vaddvq_f32(s0) + vaddvq_f32(s1);
    for (; b < nb; ++b) {
        const float d =
            unpack_block32_neon(type, row + b * bb, lo, hi) * yd[b];
        const int p = vaddvq_s32(lanes32(lo, hi, yq + b * 32));
        sum += static_cast<float>(p) * d;
    }
    return sum;
}


/**
 * row_block32_q8_0_neon for T tokens: each weight block is unpacked
 * once and dotted against every token, each token with its own lanes
 * and float steps in the same order, so out[t] is bit-identical to
 * row_block32_q8_0_neon on token t (scales at yd + t * nb, quants at
 * yq + t * qs).
 */
template <int T>
void rows_block32_q8_0_xt(gguf::TensorType type, const std::byte* row,
                          std::size_t bb, std::size_t nb, const float* yd,
                          const std::int8_t* yq, std::size_t qs, float* out) {
    float32x4_t s0[T], s1[T];
    for (int t = 0; t < T; ++t) {
        s0[t] = vdupq_n_f32(0.0f);
        s1[t] = vdupq_n_f32(0.0f);
    }
    std::size_t b = 0;
    int8x16_t lo0, hi0, lo1, hi1;
    for (; b + 1 < nb; b += 2) {
        const float w0 = unpack_block32_neon(type, row + b * bb, lo0, hi0);
        const float w1 =
            unpack_block32_neon(type, row + (b + 1) * bb, lo1, hi1);
        for (int t = 0; t < T; ++t) {
            const float* d = yd + t * nb;
            const std::int8_t* q = yq + t * qs;
            const float d0 = w0 * d[b];
            const int32x4_t p0 = lanes32(lo0, hi0, q + b * 32);
            const float d1 = w1 * d[b + 1];
            const int32x4_t p1 = lanes32(lo1, hi1, q + (b + 1) * 32);
            s0[t] = vaddq_f32(s0[t], vmulq_n_f32(vcvtq_f32_s32(p0), d0));
            s1[t] = vaddq_f32(s1[t], vmulq_n_f32(vcvtq_f32_s32(p1), d1));
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = vaddvq_f32(s0[t]) + vaddvq_f32(s1[t]);
    }
    for (; b < nb; ++b) {
        const float w0 = unpack_block32_neon(type, row + b * bb, lo0, hi0);
        for (int t = 0; t < T; ++t) {
            const float d = w0 * yd[t * nb + b];
            const int p = vaddvq_s32(lanes32(lo0, hi0, yq + t * qs + b * 32));
            out[t] += static_cast<float>(p) * d;
        }
    }
}

/** One sdot of 16 int8 pairs into four int32 lanes (widening
 *  multiplies without dotprod; the same exact integers). */
inline int32x4_t dot16(int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD) || defined(LOCUS_NEON_DOTPROD_VARIANT)
    return vdotq_s32(vdupq_n_s32(0), a, b);
#else
    const int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vpaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
#endif
}

/**
 * One Q4_K row x T Q8_K tokens (batched prefill): each block's nibbles
 * are unpacked once and dotted against every token. Per token and
 * block, the integer sum over the 8 scaled sub-blocks is exact; the
 * float steps are one multiply-add per block, so the sum is
 * deterministic but not ggml's scalar order (scalar matvec_q8k).
 * Token t's quants are at yq + t * qs, its scales at yd + t * ds and
 * its 16-sums at ys + t * ss.
 */
template <int T>
void rows_q4_k_q8k_xt(const std::byte* row, std::size_t nb, const float* yd,
                      std::size_t ds, const std::int8_t* yq, std::size_t qs,
                      const std::int16_t* ys, std::size_t ss, float* out) {
    constexpr std::uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f,
                            kmask3 = 0x03030303;
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 144;
        const float bd = half_to_float(load_u16(blk));
        const float bdmin = half_to_float(load_u16(blk + 2));
        std::uint32_t utmp[4];
        std::memcpy(utmp, blk + 4, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) |
                  (((utmp[1] >> 6) & kmask3) << 4);
        const std::uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const auto* scales = reinterpret_cast<const std::uint8_t*>(utmp);
        const auto* mins = scales + 8;
        const auto* q4 = reinterpret_cast<const std::uint8_t*>(blk + 16);
        // Sub-block 2j: low nibbles of bytes 32j..32j+31; 2j+1: high.
        int8x16_t w[16];
        for (int j = 0; j < 4; ++j) {
            const uint8x16_t b0 = vld1q_u8(q4 + 32 * j);
            const uint8x16_t b1 = vld1q_u8(q4 + 32 * j + 16);
            w[4 * j] = vreinterpretq_s8_u8(vandq_u8(b0, m4b));
            w[4 * j + 1] = vreinterpretq_s8_u8(vandq_u8(b1, m4b));
            w[4 * j + 2] = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4));
            w[4 * j + 3] = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
        }
        for (int t = 0; t < T; ++t) {
            const std::int8_t* q8 = yq + t * qs + i * 256;
            const std::int16_t* bs = ys + t * ss + i * 16;
            // Sub-block s's activation sum is bs[2s] + bs[2s + 1].
            const int16x8_t bsum = vpaddq_s16(vld1q_s16(bs), vld1q_s16(bs + 8));
            const int16x8_t mv = vreinterpretq_s16_u16(
                vmovl_u8(vld1_u8(mins)));
            const int minsum =
                vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(bsum),
                                               vget_low_s16(mv)),
                                     vmull_high_s16(bsum, mv)));
            int32x4_t acc = vdupq_n_s32(0);
            for (int s = 0; s < 8; ++s) {
                const int32x4_t p =
                    vaddq_s32(dot16(w[2 * s], vld1q_s8(q8 + 32 * s)),
                              dot16(w[2 * s + 1], vld1q_s8(q8 + 32 * s + 16)));
                acc = vmlaq_n_s32(acc, p, scales[s]);
            }
            const float dy = yd[t * ds + i];
            sumf[t] += bd * dy * static_cast<float>(vaddvq_s32(acc)) -
                       bdmin * dy * static_cast<float>(minsum);
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = sumf[t];
    }
}

/**
 * One Q6_K row x T Q8_K tokens, as rows_q4_k_q8k_xt: the 6-bit codes
 * are unpacked once per block, each token's 16 scaled group dots sum
 * exactly, one float multiply-add per block.
 */
template <int T>
void rows_q6_k_q8k_xt(const std::byte* row, std::size_t nb, const float* yd,
                      std::size_t ds, const std::int8_t* yq, std::size_t qs,
                      float* out) {
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 210;
        const auto* ql = reinterpret_cast<const std::uint8_t*>(blk);
        const auto* qh = reinterpret_cast<const std::uint8_t*>(blk + 128);
        const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
        const float bd = half_to_float(load_u16(blk + 208));
        // Groups 0..15 in q8 order: half n holds groups 8n..8n+7.
        int8x16_t c[16];
        for (int n = 0; n < 2; ++n) {
            const uint8x16_t h0 = vld1q_u8(qh + 32 * n);
            const uint8x16_t h1 = vld1q_u8(qh + 32 * n + 16);
            const uint8x16_t l0 = vld1q_u8(ql + 64 * n);
            const uint8x16_t l1 = vld1q_u8(ql + 64 * n + 16);
            const uint8x16_t l2 = vld1q_u8(ql + 64 * n + 32);
            const uint8x16_t l3 = vld1q_u8(ql + 64 * n + 48);
            int8x16_t* g = c + 8 * n;
            g[0] = q6_codes(vandq_u8(l0, m4b), q6_high<0>(h0));
            g[1] = q6_codes(vandq_u8(l1, m4b), q6_high<0>(h1));
            g[2] = q6_codes(vandq_u8(l2, m4b), q6_high<2>(h0));
            g[3] = q6_codes(vandq_u8(l3, m4b), q6_high<2>(h1));
            g[4] = q6_codes(vshrq_n_u8(l0, 4), q6_high<4>(h0));
            g[5] = q6_codes(vshrq_n_u8(l1, 4), q6_high<4>(h1));
            g[6] = q6_codes(vshrq_n_u8(l2, 4), q6_high<6>(h0));
            g[7] = q6_codes(vshrq_n_u8(l3, 4), q6_high<6>(h1));
        }
        for (int t = 0; t < T; ++t) {
            const std::int8_t* q8 = yq + t * qs + i * 256;
            int32x4_t acc = vdupq_n_s32(0);
            for (int g = 0; g < 16; ++g) {
                acc = vmlaq_n_s32(acc, dot16(c[g], vld1q_s8(q8 + 16 * g)),
                                  sc[g]);
            }
            sumf[t] += bd * yd[t * ds + i] *
                       static_cast<float>(vaddvq_s32(acc));
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = sumf[t];
    }
}

/*
 * Q8_K dots for IQ3_XXS, IQ3_S, IQ1_M, IQ4_XS and Q2_K (i#744). Each
 * one is the scalar vec_dot_*_q8_k of cpu_ops.cpp with the inner
 * integer sums done by dot16: those sums are exact, so their order is
 * free, and the float steps are the scalar ones in the same order. So
 * a row's result is the scalar kernel's bits. yq is a token's plain
 * Q8_K quants, yd its block scales and ys its 16-sums.
 */

/** 0xff in byte j (0..7) where bit j of s0 is set, and in byte 8 + j
 *  where bit j of s1 is. */
inline uint8x16_t sign_mask(std::uint8_t s0, std::uint8_t s1) {
    static constexpr std::uint8_t kBits[16] = {1, 2, 4,  8,  16, 32, 64, 128,
                                               1, 2, 4,  8,  16, 32, 64, 128};
    return vtstq_u8(vcombine_u8(vdup_n_u8(s0), vdup_n_u8(s1)),
                    vld1q_u8(kBits));
}

/** g (bytes <= 127) with the bytes under mask m negated: (g ^ m) - m. */
inline int8x16_t signed_grid(uint8x16_t g, uint8x16_t m) {
    return vreinterpretq_s8_u8(vsubq_u8(veorq_u8(g, m), m));
}

/** Four 4-byte grid entries as 16 bytes, in entry order. */
inline uint8x16_t grid16(std::uint32_t a, std::uint32_t b, std::uint32_t c,
                         std::uint32_t d) {
    const std::uint32_t g[4] = {a, b, c, d};
    return vld1q_u8(reinterpret_cast<const std::uint8_t*>(g));
}

/*
 * Each kernel below runs one weight row against T tokens: a block is
 * decoded once (its 256 weights as 16 int8 vectors, w[2 g] and w[2 g +
 * 1] for 32-weight group g, plus its scales) and dotted against every
 * token. Token t's block scales are at yd + t * ds, its quants at yq +
 * t * qs and its 16-sums at ys + t * ss; out[t] gets its result. A
 * token's integer sums and float steps do not depend on T, so T = 1
 * (matvec_neon_q8k) and T = 4 (matvec_batch_neon_q8k) give each token
 * the same bits, the scalar kernel's.
 */

/** As vec_dot_iq3_xxs_q8_k (98-byte blocks). */
template <int T>
void rows_iq3_xxs_xt(const std::byte* row, std::size_t nb, const float* yd,
                     std::size_t ds, const std::int8_t* yq, std::size_t qs,
                     const std::int16_t*, std::size_t, float* out) {
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const auto* u = reinterpret_cast<const std::uint8_t*>(row + i * 98);
        const float dw = half_to_float(load_u16(row + i * 98));
        int8x16_t w[16];
        std::int32_t ls[8];
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint8_t* q3 = u + 2 + 8 * ib32;
            std::uint32_t aux32;
            std::memcpy(&aux32, u + 66 + 4 * ib32, 4);
            for (int h = 0; h < 2; ++h) {
                const std::uint8_t* q = q3 + 4 * h;
                const uint8x16_t g =
                    grid16(iq3xxs_grid[q[0]], iq3xxs_grid[q[1]],
                           iq3xxs_grid[q[2]], iq3xxs_grid[q[3]]);
                const uint8x16_t m = sign_mask(
                    ksigns_iq2xs[(aux32 >> (14 * h)) & 127],
                    ksigns_iq2xs[(aux32 >> (14 * h + 7)) & 127]);
                w[2 * ib32 + h] = signed_grid(g, m);
            }
            ls[ib32] = static_cast<std::int32_t>(2 * (aux32 >> 28) + 1);
        }
        for (int t = 0; t < T; ++t) {
            const float d = dw * yd[t * ds + i];
            const std::int8_t* q8 = yq + t * qs + i * 256;
            std::int32_t bsum = 0;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const int32x4_t acc = vaddq_s32(
                    dot16(w[2 * ib32], vld1q_s8(q8 + 32 * ib32)),
                    dot16(w[2 * ib32 + 1], vld1q_s8(q8 + 32 * ib32 + 16)));
                bsum += vaddvq_s32(acc) * ls[ib32];
            }
            sumf[t] += d * static_cast<float>(bsum);
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = 0.25f * sumf[t];
    }
}

/** As vec_dot_iq3_s_q8_k (110-byte blocks). */
template <int T>
void rows_iq3_s_xt(const std::byte* row, std::size_t nb, const float* yd,
                   std::size_t ds, const std::int8_t* yq, std::size_t qs,
                   const std::int16_t*, std::size_t, float* out) {
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const auto* u = reinterpret_cast<const std::uint8_t*>(row + i * 110);
        const float dw = half_to_float(load_u16(row + i * 110));
        int8x16_t w[16];
        std::int32_t ls[8];
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint8_t* qsb = u + 2 + 8 * ib32;
            const std::uint32_t qh = u[66 + ib32];
            const std::uint8_t* signs = u + 74 + 4 * ib32;
            // The 9th index bit: bit 2l of qh for entry 2l, 2l + 1
            // for entry 2l + 1.
            const auto at = [&](int k) {
                const int l = k / 2;
                const std::uint32_t hb =
                    k % 2 == 0 ? (qh << (8 - 2 * l)) & 256
                               : (qh << (7 - 2 * l)) & 256;
                return iq3s_grid[qsb[k] | hb];
            };
            for (int h = 0; h < 2; ++h) {
                const uint8x16_t g = grid16(at(4 * h), at(4 * h + 1),
                                            at(4 * h + 2), at(4 * h + 3));
                const uint8x16_t m = sign_mask(signs[2 * h], signs[2 * h + 1]);
                w[2 * ib32 + h] = signed_grid(g, m);
            }
            const std::uint32_t sc = u[106 + ib32 / 2];
            ls[ib32] = static_cast<std::int32_t>(
                2 * (ib32 % 2 == 0 ? sc & 0xf : sc >> 4) + 1);
        }
        for (int t = 0; t < T; ++t) {
            const float d = dw * yd[t * ds + i];
            const std::int8_t* q8 = yq + t * qs + i * 256;
            std::int32_t bsum = 0;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const int32x4_t acc = vaddq_s32(
                    dot16(w[2 * ib32], vld1q_s8(q8 + 32 * ib32)),
                    dot16(w[2 * ib32 + 1], vld1q_s8(q8 + 32 * ib32 + 16)));
                bsum += vaddvq_s32(acc) * ls[ib32];
            }
            sumf[t] += d * static_cast<float>(bsum);
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = sumf[t];
    }
}

/** As vec_dot_iq1_m_q8_k (56-byte blocks). */
template <int T>
void rows_iq1_m_xt(const std::byte* row, std::size_t nb, const float* yd,
                   std::size_t ds, const std::int8_t* yq, std::size_t qs,
                   const std::int16_t*, std::size_t, float* out) {
    constexpr float kDelta = 0.125f;  // IQ1M_DELTA
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const auto* u = reinterpret_cast<const std::uint8_t*>(row + i * 56);
        std::uint16_t sc[4];
        std::memcpy(sc, u + 48, 8);
        const std::uint16_t scale_u16 =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
            ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        if constexpr (T == 1) {
            // One token: decode and dot in one pass (keeping all 16
            // vectors cost about 45% here on a Cortex-A76); the same
            // integers and float steps as the T-token form below.
            const std::int8_t* q8 = yq + i * 256;
            std::int32_t sumi1 = 0, sumi2 = 0;
            for (int ib = 0; ib < 8; ++ib) {
                const std::uint8_t* qsb = u + 4 * ib;
                const std::uint8_t* qh = u + 32 + 2 * ib;
                for (int h = 0; h < 2; ++h) {
                    std::uint64_t g[2];
                    for (int e = 0; e < 2; ++e) {
                        const int l = 2 * h + e;
                        g[e] = iq1s_grid[qsb[l] |
                                         ((static_cast<std::uint32_t>(qh[h])
                                           << (8 - 4 * (l % 2))) &
                                          0x700)];
                    }
                    const int ls =
                        2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3 * h)) & 0x7) +
                        1;
                    const int8x16_t q = vld1q_s8(q8 + 32 * ib + 16 * h);
                    const std::int32_t sum1 = vaddvq_s32(dot16(
                        vld1q_s8(reinterpret_cast<const std::int8_t*>(g)), q));
                    const std::int32_t sum2 =
                        vaddlv_s8(vget_low_s8(q)) *
                            ((qh[h] & 0x08) ? -1 : 1) +
                        vaddlv_s8(vget_high_s8(q)) * ((qh[h] & 0x80) ? -1 : 1);
                    sumi1 += sum1 * ls;
                    sumi2 += sum2 * ls;
                }
            }
            sumf[0] += half_to_float(scale_u16) * yd[i] *
                       (static_cast<float>(sumi1) +
                        kDelta * static_cast<float>(sumi2));
            continue;
        }
        // Per 16 weights v = 2 ib + h: entries 2h, 2h + 1 as int8, the
        // scale, and the deltas of its low and high 8 weights.
        int8x16_t w[16];
        int ls[16], dlo[16], dhi[16];
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* qsb = u + 4 * ib;
            const std::uint8_t* qh = u + 32 + 2 * ib;
            for (int h = 0; h < 2; ++h) {
                std::uint64_t g[2];
                for (int e = 0; e < 2; ++e) {
                    const int l = 2 * h + e;
                    g[e] = iq1s_grid[qsb[l] |
                                     ((static_cast<std::uint32_t>(qh[h])
                                       << (8 - 4 * (l % 2))) &
                                      0x700)];
                }
                const int v = 2 * ib + h;
                w[v] = vld1q_s8(reinterpret_cast<const std::int8_t*>(g));
                ls[v] = 2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3 * h)) & 0x7) + 1;
                // delta of entry 2h + e: bit 0x08 (e = 0) or 0x80 of qh[h].
                dlo[v] = (qh[h] & 0x08) ? -1 : 1;
                dhi[v] = (qh[h] & 0x80) ? -1 : 1;
            }
        }
        const float dw = half_to_float(scale_u16);
        for (int t = 0; t < T; ++t) {
            const std::int8_t* q8 = yq + t * qs + i * 256;
            std::int32_t sumi1 = 0, sumi2 = 0;
            for (int v = 0; v < 16; ++v) {
                const int8x16_t q = vld1q_s8(q8 + 16 * v);
                const std::int32_t sum1 = vaddvq_s32(dot16(w[v], q));
                const std::int32_t sum2 =
                    vaddlv_s8(vget_low_s8(q)) * dlo[v] +
                    vaddlv_s8(vget_high_s8(q)) * dhi[v];
                sumi1 += sum1 * ls[v];
                sumi2 += sum2 * ls[v];
            }
            sumf[t] += dw * yd[t * ds + i] *
                       (static_cast<float>(sumi1) +
                        kDelta * static_cast<float>(sumi2));
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = sumf[t];
    }
}

/** As vec_dot_iq4_xs_q8_k (136-byte blocks). */
template <int T>
void rows_iq4_xs_xt(const std::byte* row, std::size_t nb, const float* yd,
                    std::size_t ds, const std::int8_t* yq, std::size_t qs,
                    const std::int16_t*, std::size_t, float* out) {
    const int8x16_t table = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 136;
        const float dw = half_to_float(load_u16(blk));
        const std::uint32_t h = load_u16(blk + 2);
        const auto* sl = reinterpret_cast<const std::uint8_t*>(blk + 4);
        const auto* qsb = reinterpret_cast<const std::uint8_t*>(blk + 8);
        int8x16_t w[16];
        int lsm[8];  // ls - 32
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint32_t ls =
                (ib % 2 == 0 ? sl[ib / 2] & 0xf : sl[ib / 2] >> 4) |
                (((h >> (2 * ib)) & 3) << 4);
            lsm[ib] = static_cast<int>(ls) - 32;
            const uint8x16_t b = vld1q_u8(qsb + 16 * ib);
            w[2 * ib] = vqtbl1q_s8(table, vandq_u8(b, m4b));
            w[2 * ib + 1] = vqtbl1q_s8(table, vshrq_n_u8(b, 4));
        }
        for (int t = 0; t < T; ++t) {
            const float d4d8 = dw * yd[t * ds + i];
            const std::int8_t* q8 = yq + t * qs + i * 256;
            for (int ib = 0; ib < 8; ++ib) {
                const int32x4_t s = vaddq_s32(
                    dot16(w[2 * ib], vld1q_s8(q8 + 32 * ib)),
                    dot16(w[2 * ib + 1], vld1q_s8(q8 + 32 * ib + 16)));
                const float d1 = d4d8 * static_cast<float>(lsm[ib]);
                sumf[t] += d1 * static_cast<float>(vaddvq_s32(s));
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = sumf[t];
    }
}

/** As vec_dot_q2_k_q8_k (84-byte blocks). */
template <int T>
void rows_q2_k_xt(const std::byte* row, std::size_t nb, const float* yd,
                  std::size_t ds, const std::int8_t* yq, std::size_t qs,
                  const std::int16_t* ys, std::size_t ss, float* out) {
    const uint8x16_t m3 = vdupq_n_u8(3);
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 84;
        const auto* sc = reinterpret_cast<const std::uint8_t*>(blk);
        const auto* q2 = reinterpret_cast<const std::uint8_t*>(blk + 16);
        const float bd = half_to_float(load_u16(blk + 80));
        const float bdmin = half_to_float(load_u16(blk + 82));
        // Sub-block g = 8k + 2j (+ 1): the 2-bit field j of q2 bytes
        // 32k .. 32k + 15 (+ 16 .. 31), against q8[16 g ..].
        int8x16_t w[16];
        for (int k = 0; k < 2; ++k) {
            const uint8x16_t a = vld1q_u8(q2 + 32 * k);
            const uint8x16_t b = vld1q_u8(q2 + 32 * k + 16);
            for (int j = 0; j < 4; ++j) {
                const int8x16_t sh =
                    vdupq_n_s8(static_cast<std::int8_t>(-2 * j));
                w[8 * k + 2 * j] =
                    vreinterpretq_s8_u8(vandq_u8(vshlq_u8(a, sh), m3));
                w[8 * k + 2 * j + 1] =
                    vreinterpretq_s8_u8(vandq_u8(vshlq_u8(b, sh), m3));
            }
        }
        for (int t = 0; t < T; ++t) {
            const std::int16_t* bs = ys + t * ss + i * 16;
            int summs = 0;
            for (int j = 0; j < 16; ++j) {
                summs += bs[j] * (sc[j] >> 4);
            }
            const float dall = yd[t * ds + i] * bd;
            const float dmin = yd[t * ds + i] * bdmin;
            const std::int8_t* q8 = yq + t * qs + i * 256;
            int isum = 0;
            for (int g = 0; g < 16; ++g) {
                isum += (sc[g] & 0xf) *
                        vaddvq_s32(dot16(w[g], vld1q_s8(q8 + 16 * g)));
            }
            sumf[t] += dall * static_cast<float>(isum) -
                       dmin * static_cast<float>(summs);
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = sumf[t];
    }
}

/** As vec_dot_iq2_xs_q8_k (74-byte blocks). */
template <int T>
void rows_iq2_xs_xt(const std::byte* row, std::size_t nb, const float* yd,
                    std::size_t ds, const std::int8_t* yq, std::size_t qs,
                    const std::int16_t*, std::size_t, float* out) {
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 74;
        const float dw = half_to_float(load_u16(blk));
        const auto* sc = reinterpret_cast<const std::uint8_t*>(blk + 66);
        // Per 16 weights v = 2 ib32 + h: entries 2h, 2h + 1 (a 9-bit
        // grid index and a 7-bit ksigns index each), signed.
        int8x16_t w[16];
        for (int v = 0; v < 16; ++v) {
            std::uint64_t g[2];
            std::uint8_t s[2];
            for (int e = 0; e < 2; ++e) {
                const std::uint16_t q = load_u16(blk + 2 + 2 * (2 * v + e));
                g[e] = iq2xs_grid[q & 511];
                s[e] = ksigns_iq2xs[q >> 9];
            }
            w[v] = signed_grid(
                vld1q_u8(reinterpret_cast<const std::uint8_t*>(g)),
                sign_mask(s[0], s[1]));
        }
        for (int t = 0; t < T; ++t) {
            const float d = dw * yd[t * ds + i];
            const std::int8_t* q8 = yq + t * qs + i * 256;
            std::int32_t bsum = 0;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const std::int32_t sumi1 = vaddvq_s32(
                    dot16(w[2 * ib32], vld1q_s8(q8 + 32 * ib32)));
                const std::int32_t sumi2 = vaddvq_s32(
                    dot16(w[2 * ib32 + 1], vld1q_s8(q8 + 32 * ib32 + 16)));
                bsum += sumi1 * static_cast<std::int32_t>(
                                    2 * (sc[ib32] & 0xf) + 1);
                bsum += sumi2 * static_cast<std::int32_t>(
                                    2 * (sc[ib32] >> 4) + 1);
            }
            sumf[t] += d * static_cast<float>(bsum);
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = 0.125f * sumf[t];
    }
}

/** As vec_dot_iq1_s_q8_k (50-byte blocks). */
template <int T>
void rows_iq1_s_xt(const std::byte* row, std::size_t nb, const float* yd,
                   std::size_t ds, const std::int8_t* yq, std::size_t qs,
                   const std::int16_t* ys, std::size_t ss, float* out) {
    constexpr float kDelta = 0.125f;  // IQ1S_DELTA
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 50;
        const float dw = half_to_float(load_u16(blk));
        const auto* qsb = reinterpret_cast<const std::uint8_t*>(blk + 2);
        if constexpr (T == 1) {
            // One token: decode and dot in one pass, as IQ1_M (holding
            // all 16 vectors costs more than it saves here); the same
            // integers and float steps as the T-token form below.
            const std::int8_t* q8 = yq + i * 256;
            const std::int16_t* bs = ys + i * 16;
            std::int32_t sumi = 0, sumi1 = 0;
            for (int ib = 0; ib < 8; ++ib) {
                const std::uint32_t qh = load_u16(blk + 34 + 2 * ib);
                std::uint64_t g[4];
                for (int l = 0; l < 4; ++l) {
                    g[l] = iq1s_grid[qsb[4 * ib + l] |
                                     (((qh >> (3 * l)) & 7) << 8)];
                }
                const std::int32_t lsum = vaddvq_s32(vaddq_s32(
                    dot16(vld1q_s8(reinterpret_cast<const std::int8_t*>(g)),
                          vld1q_s8(q8 + 32 * ib)),
                    dot16(vld1q_s8(
                              reinterpret_cast<const std::int8_t*>(g + 2)),
                          vld1q_s8(q8 + 32 * ib + 16))));
                const int ls = static_cast<int>(2 * ((qh >> 12) & 7) + 1);
                sumi += ls * lsum;
                sumi1 += ls * ((qh & 0x8000) ? -1 : 1) *
                         (bs[2 * ib] + bs[2 * ib + 1]);
            }
            const float d = dw * yd[i];
            sumf[0] += d * (static_cast<float>(sumi) +
                            kDelta * static_cast<float>(sumi1));
            continue;
        }
        // Per 32 weights ib: entries 0..3 as int8 (two vectors), the
        // scale and the delta's sign.
        int8x16_t w[16];
        int ls[8], dl[8];
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint32_t qh = load_u16(blk + 34 + 2 * ib);
            std::uint64_t g[4];
            for (int l = 0; l < 4; ++l) {
                g[l] = iq1s_grid[qsb[4 * ib + l] |
                                 (((qh >> (3 * l)) & 7) << 8)];
            }
            w[2 * ib] = vld1q_s8(reinterpret_cast<const std::int8_t*>(g));
            w[2 * ib + 1] =
                vld1q_s8(reinterpret_cast<const std::int8_t*>(g + 2));
            ls[ib] = static_cast<int>(2 * ((qh >> 12) & 7) + 1);
            dl[ib] = (qh & 0x8000) ? -1 : 1;
        }
        for (int t = 0; t < T; ++t) {
            const std::int8_t* q8 = yq + t * qs + i * 256;
            const std::int16_t* bs = ys + t * ss + i * 16;
            std::int32_t sumi = 0, sumi1 = 0;
            for (int ib = 0; ib < 8; ++ib) {
                const std::int32_t lsum = vaddvq_s32(vaddq_s32(
                    dot16(w[2 * ib], vld1q_s8(q8 + 32 * ib)),
                    dot16(w[2 * ib + 1], vld1q_s8(q8 + 32 * ib + 16))));
                sumi += ls[ib] * lsum;
                sumi1 += ls[ib] * dl[ib] * (bs[2 * ib] + bs[2 * ib + 1]);
            }
            const float d = dw * yd[t * ds + i];
            sumf[t] += d * (static_cast<float>(sumi) +
                            kDelta * static_cast<float>(sumi1));
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = sumf[t];
    }
}

/**
 * As vec_dot_tq1_0_q8_k (54-byte blocks): base-3 digits c in 0..2 from
 * q = (byte * 3^l) mod 256 as (q * 3) >> 8, each weight c - 1.
 */
template <int T>
void rows_tq1_0_xt(const std::byte* row, std::size_t nb, const float* yd,
                   std::size_t ds, const std::int8_t* yq, std::size_t qs,
                   const std::int16_t*, std::size_t, float* out) {
    // The (byte * m mod 256) * 3 >> 8 digit of 16 bytes, minus 1.
    const auto digits = [](uint8x16_t b, uint8x16_t m) {
        const uint8x16_t q = vmulq_u8(b, m);  // mod 256, as the scalar
        const uint8x8_t lo = vshrn_n_u16(vmull_u8(vget_low_u8(q),
                                                  vdup_n_u8(3)), 8);
        const uint8x8_t hi = vshrn_n_u16(vmull_u8(vget_high_u8(q),
                                                  vdup_n_u8(3)), 8);
        return vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(lo, hi)),
                        vdupq_n_s8(1));
    };
    static constexpr std::uint8_t kPow3[5] = {1, 3, 9, 27, 81};
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 54;
        const auto* qsb = reinterpret_cast<const std::uint8_t*>(blk);
        const float bd = half_to_float(load_u16(blk + 52));
        // Weights in q8 order: qs[0..31] x l at 32 l + m (two vectors
        // each), qs[32..47] x l at 160 + 16 l + m, qh x l at 240 + 4 l
        // + j (one vector: qh repeated, multipliers 1, 3, 9, 27).
        int8x16_t w[16];
        const uint8x16_t b0 = vld1q_u8(qsb), b1 = vld1q_u8(qsb + 16);
        const uint8x16_t b2 = vld1q_u8(qsb + 32);
        for (int l = 0; l < 5; ++l) {
            const uint8x16_t m = vdupq_n_u8(kPow3[l]);
            w[2 * l] = digits(b0, m);
            w[2 * l + 1] = digits(b1, m);
            w[10 + l] = digits(b2, m);
        }
        std::uint32_t qh;
        std::memcpy(&qh, qsb + 48, 4);
        static constexpr std::uint8_t kHm[16] = {1, 1, 1, 1, 3,  3,  3,  3,
                                                 9, 9, 9, 9, 27, 27, 27, 27};
        w[15] = digits(vreinterpretq_u8_u32(vdupq_n_u32(qh)), vld1q_u8(kHm));
        for (int t = 0; t < T; ++t) {
            const std::int8_t* q8 = yq + t * qs + i * 256;
            int32x4_t acc = vdupq_n_s32(0);
            for (int v = 0; v < 16; ++v) {
                acc = vaddq_s32(acc, dot16(w[v], vld1q_s8(q8 + 16 * v)));
            }
            sumf[t] += static_cast<float>(vaddvq_s32(acc)) *
                       (bd * yd[t * ds + i]);
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = sumf[t];
    }
}

/** As vec_dot_tq2_0_q8_k (66-byte blocks): 2-bit c, each weight c - 1. */
template <int T>
void rows_tq2_0_xt(const std::byte* row, std::size_t nb, const float* yd,
                   std::size_t ds, const std::int8_t* yq, std::size_t qs,
                   const std::int16_t*, std::size_t, float* out) {
    const uint8x16_t m3 = vdupq_n_u8(3);
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 66;
        const auto* qsb = reinterpret_cast<const std::uint8_t*>(blk);
        const float bd = half_to_float(load_u16(blk + 64));
        // Weights in q8 order: (qs[32 h + k] >> 2 l) & 3, minus 1, at
        // 128 h + 32 l + k (two vectors per (h, l)).
        int8x16_t w[16];
        for (int h = 0; h < 2; ++h) {
            const uint8x16_t a = vld1q_u8(qsb + 32 * h);
            const uint8x16_t b = vld1q_u8(qsb + 32 * h + 16);
            for (int l = 0; l < 4; ++l) {
                const int8x16_t sh =
                    vdupq_n_s8(static_cast<std::int8_t>(-2 * l));
                w[8 * h + 2 * l] = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshlq_u8(a, sh), m3)),
                    vdupq_n_s8(1));
                w[8 * h + 2 * l + 1] = vsubq_s8(
                    vreinterpretq_s8_u8(vandq_u8(vshlq_u8(b, sh), m3)),
                    vdupq_n_s8(1));
            }
        }
        for (int t = 0; t < T; ++t) {
            const std::int8_t* q8 = yq + t * qs + i * 256;
            int32x4_t acc = vdupq_n_s32(0);
            for (int v = 0; v < 16; ++v) {
                acc = vaddq_s32(acc, dot16(w[v], vld1q_s8(q8 + 16 * v)));
            }
            sumf[t] += static_cast<float>(vaddvq_s32(acc)) *
                       (yd[t * ds + i] * bd);
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = sumf[t];
    }
}

/** @returns true for the types rows_iq_xt runs (i#744 and the rest of
 *  the QK_K set NEON had no dot for). */
bool iq_type(gguf::TensorType type) {
    using TT = gguf::TensorType;
    return type == TT::kIQ3_XXS || type == TT::kIQ3_S ||
           type == TT::kIQ1_M || type == TT::kIQ4_XS || type == TT::kQ2_K ||
           type == TT::kIQ2_XS || type == TT::kIQ1_S ||
           type == TT::kTQ1_0 || type == TT::kTQ2_0;
}

/** One row of an iq_type() type against T tokens (see the kernels). */
template <int T>
void rows_iq_xt(gguf::TensorType type, const std::byte* row, std::size_t nb,
                const float* yd, std::size_t ds, const std::int8_t* yq,
                std::size_t qs, const std::int16_t* ys, std::size_t ss,
                float* out) {
    using TT = gguf::TensorType;
    switch (type) {
        case TT::kIQ3_XXS:
            rows_iq3_xxs_xt<T>(row, nb, yd, ds, yq, qs, ys, ss, out);
            break;
        case TT::kIQ3_S:
            rows_iq3_s_xt<T>(row, nb, yd, ds, yq, qs, ys, ss, out);
            break;
        case TT::kIQ1_M:
            rows_iq1_m_xt<T>(row, nb, yd, ds, yq, qs, ys, ss, out);
            break;
        case TT::kIQ4_XS:
            rows_iq4_xs_xt<T>(row, nb, yd, ds, yq, qs, ys, ss, out);
            break;
        case TT::kIQ2_XS:
            rows_iq2_xs_xt<T>(row, nb, yd, ds, yq, qs, ys, ss, out);
            break;
        case TT::kIQ1_S:
            rows_iq1_s_xt<T>(row, nb, yd, ds, yq, qs, ys, ss, out);
            break;
        case TT::kTQ1_0:
            rows_tq1_0_xt<T>(row, nb, yd, ds, yq, qs, ys, ss, out);
            break;
        case TT::kTQ2_0:
            rows_tq2_0_xt<T>(row, nb, yd, ds, yq, qs, ys, ss, out);
            break;
        default:  // kQ2_K
            rows_q2_k_xt<T>(row, nb, yd, ds, yq, qs, ys, ss, out);
            break;
    }
}

}  // namespace

void quantize_activation_q8k_neon(std::span<const float> x, float* d,
                                  std::int8_t* qs, std::int16_t* bsums) {
    const std::size_t nb = x.size() / 256;
    // ggml's nearest_int: adding 1.5 * 2^23 rounds to an integer in
    // the low mantissa bits (half to even), as the scalar does.
    const float32x4_t magic = vdupq_n_f32(12582912.0f);
    const int32x4_t mant = vdupq_n_s32(0x007fffff);
    const int32x4_t bias = vdupq_n_s32(0x00400000);
    const int32x4_t q127 = vdupq_n_s32(127);
    for (std::size_t i = 0; i < nb; ++i) {
        const float* xb = x.data() + i * 256;
        // maxnm skips a NaN as the scalar's `ax > amax` does (an
        // all-NaN block gives 0, as there).
        float32x4_t m = vdupq_n_f32(0.0f);
        for (int j = 0; j < 256; j += 4) {
            m = vmaxnmq_f32(m, vabsq_f32(vld1q_f32(xb + j)));
        }
        const float amax = vmaxnmvq_f32(m);
        std::int8_t* q = qs + i * 256;
        std::int16_t* bs = bsums + i * 16;
        if (amax == 0.0f) {
            d[i] = 0.0f;
            std::memset(q, 0, 256);
            std::memset(bs, 0, 16 * sizeof(std::int16_t));
            continue;
        }
        // The first element of that magnitude, as the scalar's strict
        // > scan picks it.
        int first = 0;
        while (first < 255 && std::fabs(xb[first]) != amax) {
            ++first;
        }
        const float iscale = -127.0f / xb[first];
        const float32x4_t is = vdupq_n_f32(iscale);
        for (int g = 0; g < 16; ++g) {
            int16x8_t lo, hi;
            int32x4_t v[4];
            for (int k = 0; k < 4; ++k) {
                const float32x4_t p =
                    vmulq_f32(is, vld1q_f32(xb + g * 16 + k * 4));
                const int32x4_t bits =
                    vreinterpretq_s32_f32(vaddq_f32(p, magic));
                // A NaN lane (p != p) quantizes to 0, as the scalar.
                const int32x4_t keep =
                    vreinterpretq_s32_u32(vceqq_f32(p, p));
                v[k] = vandq_s32(
                    vminq_s32(vsubq_s32(vandq_s32(bits, mant), bias), q127),
                    keep);
            }
            lo = vcombine_s16(vmovn_s32(v[0]), vmovn_s32(v[1]));
            hi = vcombine_s16(vmovn_s32(v[2]), vmovn_s32(v[3]));
            const int8x16_t q16 = vcombine_s8(vmovn_s16(lo), vmovn_s16(hi));
            vst1q_s8(q + g * 16, q16);
            // The sum of the stored quants, as the scalar sums them.
            bs[g] = static_cast<std::int16_t>(vaddlvq_s8(q16));
        }
        d[i] = 1.0f / iscale;
    }
}

void quantize_activation_q8_0_neon(std::span<const float> x, float* d,
                                   std::int8_t* qs) {
    const std::size_t nb = x.size() / 32;
    for (std::size_t b = 0; b < nb; ++b) {
        const float* xb = x.data() + b * 32;
        float32x4_t v[8];
        float32x4_t m = vdupq_n_f32(0.0f);
        for (int k = 0; k < 8; ++k) {
            v[k] = vld1q_f32(xb + k * 4);
            // NaN skipped, as std::max does.
            m = vmaxnmq_f32(m, vabsq_f32(v[k]));
        }
        const float amax = vmaxnmvq_f32(m);
        const float dd = amax / 127.0f;
        const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
        d[b] = f16_to_f32(f32_to_f16(dd));
        // nearbyint under the default rounding mode: half to even; a NaN
        // converts to 0 (FCVTNS), as the scalar maps it (i#473).
        int16x8_t h[4];
        for (int k = 0; k < 4; ++k) {
            const int32x4_t a = vcvtnq_s32_f32(vmulq_n_f32(v[2 * k], id));
            const int32x4_t c =
                vcvtnq_s32_f32(vmulq_n_f32(v[2 * k + 1], id));
            h[k] = vcombine_s16(vmovn_s32(a), vmovn_s32(c));
        }
        vst1q_s8(qs + b * 32, vcombine_s8(vmovn_s16(h[0]), vmovn_s16(h[1])));
        vst1q_s8(qs + b * 32 + 16,
                 vcombine_s8(vmovn_s16(h[2]), vmovn_s16(h[3])));
    }
}

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
    if (w.type == gguf::TensorType::kBF16 ||
        w.type == gguf::TensorType::kF16) {
        const auto* rows = reinterpret_cast<const std::uint16_t*>(w.data);
        const bool bf = w.type == gguf::TensorType::kBF16;
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::uint16_t* row =
                rows + static_cast<std::size_t>(r) * w.cols;
            out[r] = bf ? dot_bf16_neon(row, x.data(), w.cols)
                        : dot_h16_neon<WidenF16>(row, x.data(), w.cols);
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
        thread_local std::vector<float> xs;
        xs.resize(w.cols / 32);
        subblock_sums(x.data(), w.cols, xs.data());
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::byte* row = w.data + r * row_bytes;
            float acc = 0.0f;
            for (std::size_t b = 0; b < nblk; ++b) {
                acc += dot_q4_k_block_neon(row + b * 144,
                                           x.data() + b * 256,
                                           xs.data() + b * 8);
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

/** 256 sign masks: byte j of entry s is 0xff when bit j of s is set. */
struct SignMasks {
    std::uint64_t m[256];
    constexpr SignMasks() : m() {
        for (int s = 0; s < 256; ++s) {
            std::uint64_t v = 0;
            for (int j = 0; j < 8; ++j) {
                if ((s >> j) & 1) {
                    v |= std::uint64_t{0xff} << (8 * j);
                }
            }
            m[s] = v;
        }
    }
};
inline constexpr SignMasks kSignMasks{};

/** Two 8-byte grid entries, each negated where its sign byte says. */
inline int8x16_t signed_grid2(std::uint64_t g0, std::uint8_t s0,
                              std::uint64_t g1, std::uint8_t s1) {
    const int8x16_t g = vreinterpretq_s8_u64(
        vcombine_u64(vcreate_u64(g0), vcreate_u64(g1)));
    const int8x16_t m = vreinterpretq_s8_u64(vcombine_u64(
        vcreate_u64(kSignMasks.m[s0]), vcreate_u64(kSignMasks.m[s1])));
    return vsubq_s8(veorq_s8(g, m), m);  // (g ^ m) - m: -g where m = -1
}

/**
 * IQ2_XXS (66B) row x T Q8_K tokens (i#733, i#739): each block's
 * sign-masked grid rows are built once and dotted against every token;
 * a token's integer sums are exact and its float steps are
 * vec_dot_iq2_xxs_q8_k's, so each token gets the scalar bits whatever
 * T is. Token t's scales are at yd + t * ds, its quants at yq + t * qs.
 */
template <int T>
void rows_iq2_xxs_q8k_xt(const std::byte* row, std::size_t nb,
                         const float* yd, std::size_t ds,
                         const std::int8_t* yq, std::size_t qs, float* out) {
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 66;
        const float dw = f16_to_f32(load_u16(blk));
        int8x16_t w[16];
        std::int32_t ls[8];
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const auto* a =
                reinterpret_cast<const std::uint8_t*>(blk) + 2 + 8 * ib32;
            std::uint32_t aux1;
            std::memcpy(&aux1, a + 4, 4);
            ls[ib32] = static_cast<std::int32_t>(2 * (aux1 >> 28) + 1);
            w[2 * ib32] = signed_grid2(
                iq2xxs_grid[a[0]], ksigns_iq2xs[aux1 & 127],
                iq2xxs_grid[a[1]], ksigns_iq2xs[(aux1 >> 7) & 127]);
            w[2 * ib32 + 1] = signed_grid2(
                iq2xxs_grid[a[2]], ksigns_iq2xs[(aux1 >> 14) & 127],
                iq2xxs_grid[a[3]], ksigns_iq2xs[(aux1 >> 21) & 127]);
        }
        for (int t = 0; t < T; ++t) {
            const float d = dw * yd[t * ds + i];
            const std::int8_t* q8 = yq + t * qs + i * 256;
            // ls * sumi per sub-block into four lanes, one horizontal
            // add per block: the same exact integer bsum.
            int32x4_t acc = vdupq_n_s32(0);
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const int32x4_t p = vaddq_s32(
                    dot16(w[2 * ib32], vld1q_s8(q8 + 32 * ib32)),
                    dot16(w[2 * ib32 + 1], vld1q_s8(q8 + 32 * ib32 + 16)));
                acc = vmlaq_n_s32(acc, p, ls[ib32]);
            }
            const std::int32_t bsum = vaddvq_s32(acc);
            sumf[t] += d * static_cast<float>(bsum);
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = 0.125f * sumf[t];
    }
}

/**
 * IQ2_S (82B) row x T Q8_K tokens (i#733, i#739): as
 * rows_iq2_xxs_q8k_xt, the float steps vec_dot_iq2_s_q8_k's.
 */
template <int T>
void rows_iq2_s_q8k_xt(const std::byte* row, std::size_t nb,
                       const float* yd, std::size_t ds,
                       const std::int8_t* yq, std::size_t qs, float* out) {
    float sumf[T] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 82;
        const float dw = f16_to_f32(load_u16(blk));
        const auto* qsb = reinterpret_cast<const std::uint8_t*>(blk + 2);
        const auto* qh = reinterpret_cast<const std::uint8_t*>(blk + 66);
        const auto* sc = reinterpret_cast<const std::uint8_t*>(blk + 74);
        int8x16_t w[16];
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint8_t* qsp = qsb + 4 * ib32;
            const std::uint8_t* sg = qsb + 32 + 4 * ib32;
            const std::uint32_t h = qh[ib32];
            auto idx = [&](int l) {
                return qsp[l] | ((h << (8 - 2 * l)) & 0x300);
            };
            w[2 * ib32] = signed_grid2(iq2s_grid[idx(0)], sg[0],
                                       iq2s_grid[idx(1)], sg[1]);
            w[2 * ib32 + 1] = signed_grid2(iq2s_grid[idx(2)], sg[2],
                                           iq2s_grid[idx(3)], sg[3]);
        }
        for (int t = 0; t < T; ++t) {
            const float d = dw * yd[t * ds + i];
            const std::int8_t* q8 = yq + t * qs + i * 256;
            // ls1 * sumi1 + ls2 * sumi2 per sub-block into four lanes,
            // one horizontal add per block: the same exact bsum.
            int32x4_t acc = vdupq_n_s32(0);
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                acc = vmlaq_n_s32(acc,
                                  dot16(w[2 * ib32], vld1q_s8(q8 + 32 * ib32)),
                                  1 + 2 * (sc[ib32] & 0xf));
                acc = vmlaq_n_s32(
                    acc, dot16(w[2 * ib32 + 1], vld1q_s8(q8 + 32 * ib32 + 16)),
                    1 + 2 * (sc[ib32] >> 4));
            }
            const std::int32_t bsum = vaddvq_s32(acc);
            sumf[t] += d * static_cast<float>(bsum);
        }
    }
    for (int t = 0; t < T; ++t) {
        out[t] = 0.125f * sumf[t];
    }
}

/**
 * 16 weights a (int8) times 16 activations q, each product scaled by
 * `scale` and added to lane e % 8 of acc (acc[0] lanes 0..3, acc[1]
 * 4..7): the per-lane integer sums of the scalar K-quant dots.
 */
inline void lanes8_madd(int8x16_t a, const std::int8_t* q,
                        std::int16_t scale, int32x4_t acc[2]) {
    const int8x16_t qv = vld1q_s8(q);
    // Elements e and e + 8 share lane e % 8: their two products added
    // in int16 (|a q| <= 31 * 128, so the pair fits), then scaled.
    const int16x8_t p =
        vmlal_s8(vmull_s8(vget_low_s8(a), vget_low_s8(qv)), vget_high_s8(a),
                 vget_high_s8(qv));
    acc[0] = vmlal_n_s16(acc[0], vget_low_s16(p), scale);
    acc[1] = vmlal_n_s16(acc[1], vget_high_s16(p), scale);
}

/**
 * Q3_K (110B) row x T Q8_K tokens (i#733, i#739): each block's quants
 * are unpacked once; per token the 8 per-lane integer sums of
 * vec_dot_q3_k_q8_k (exact) and its float steps.
 */
template <int T>
void rows_q3_k_q8k_xt(const std::byte* row, std::size_t nb,
                      const float* yd, std::size_t ds,
                      const std::int8_t* yq, std::size_t qs, float* out) {
    constexpr std::uint32_t kmask1 = 0x03030303;
    constexpr std::uint32_t kmask2 = 0x0f0f0f0f;
    float sums[T][8] = {};
    std::int32_t aux32[8];
    std::uint32_t auxs[4];
    const auto* scales = reinterpret_cast<const std::int8_t*>(auxs);
    const uint8x16_t three = vdupq_n_u8(3), four = vdupq_n_u8(4);
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 110;
        const auto* hm = reinterpret_cast<const std::uint8_t*>(blk);
        const auto* q3 = reinterpret_cast<const std::uint8_t*>(blk + 32);
        const float bd = f16_to_f32(load_u16(blk + 108));
        std::memcpy(auxs, blk + 96, 12);
        const std::uint32_t tmp = auxs[2];
        auxs[2] =
            ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] =
            ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const uint8x16_t h0 = vld1q_u8(hm), h1 = vld1q_u8(hm + 16);
        // Element run r (0..7) of 32: half r / 4, 2-bit field r % 4;
        // a[2r], a[2r + 1] its two 16-element scale groups.
        int8x16_t a[16];
        for (int r = 0; r < 8; ++r) {
            const int half = r / 4, sh = r % 4;
            const uint8x16_t b0 = vld1q_u8(q3 + 32 * half);
            const uint8x16_t b1 = vld1q_u8(q3 + 32 * half + 16);
            const int8x16_t shift =
                vdupq_n_s8(static_cast<std::int8_t>(-2 * sh));
            const uint8x16_t m = vdupq_n_u8(static_cast<std::uint8_t>(1u << r));
            // (q >> 2sh) & 3, minus 4 where the hmask bit is clear.
            a[2 * r] = vreinterpretq_s8_u8(vsubq_u8(
                vandq_u8(vshlq_u8(b0, shift), three),
                vbicq_u8(four, vtstq_u8(h0, m))));
            a[2 * r + 1] = vreinterpretq_s8_u8(vsubq_u8(
                vandq_u8(vshlq_u8(b1, shift), three),
                vbicq_u8(four, vtstq_u8(h1, m))));
        }
        for (int t = 0; t < T; ++t) {
            const std::int8_t* q8 = yq + t * qs + i * 256;
            int32x4_t acc[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};
            for (int j = 0; j < 16; ++j) {
                lanes8_madd(a[j], q8 + 16 * j,
                            static_cast<std::int16_t>(scales[j] - 32), acc);
            }
            vst1q_s32(aux32, acc[0]);
            vst1q_s32(aux32 + 4, acc[1]);
            const float d = bd * yd[t * ds + i];
            for (int l = 0; l < 8; ++l) {
                sums[t][l] += d * static_cast<float>(aux32[l]);
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        float sumf = 0.0f;
        for (int l = 0; l < 8; ++l) {
            sumf += sums[t][l];
        }
        out[t] = sumf;
    }
}

/**
 * Q5_K (176B) row x T Q8_K tokens (i#733, i#739): each block's quants
 * are unpacked once; per token the 8 per-lane integer sums of
 * vec_dot_q5_k_q8_k (exact), its mins and float steps. Token t's
 * 16-sums are at ys + t * ss.
 */
template <int T>
void rows_q5_k_q8k_xt(const std::byte* row, std::size_t nb,
                      const float* yd, std::size_t ds,
                      const std::int8_t* yq, std::size_t qs,
                      const std::int16_t* ys, std::size_t ss, float* out) {
    constexpr std::uint32_t kmask1 = 0x3f3f3f3f;
    constexpr std::uint32_t kmask2 = 0x0f0f0f0f;
    constexpr std::uint32_t kmask3 = 0x03030303;
    std::uint32_t utmp[4];
    const auto* scales = reinterpret_cast<const std::uint8_t*>(&utmp[0]);
    const auto* mins = reinterpret_cast<const std::uint8_t*>(&utmp[2]);
    float sums[T][8] = {};
    float sumf[T] = {};
    std::int32_t aux32[8];
    const uint8x16_t lo4 = vdupq_n_u8(0x0f), sixteen = vdupq_n_u8(16);
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * 176;
        const float bd = f16_to_f32(load_u16(blk));
        const float bdmin = f16_to_f32(load_u16(blk + 2));
        const auto* hm = reinterpret_cast<const std::uint8_t*>(blk + 16);
        const auto* q4 = reinterpret_cast<const std::uint8_t*>(blk + 48);
        std::memcpy(utmp, blk + 4, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const std::uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const uint8x16_t h0 = vld1q_u8(hm), h1 = vld1q_u8(hm + 16);
        int8x16_t a[16];
        for (int sb = 0; sb < 8; ++sb) {
            const std::uint8_t* qb = q4 + 32 * (sb / 2);
            uint8x16_t n0 = vld1q_u8(qb), n1 = vld1q_u8(qb + 16);
            if (sb & 1) {
                n0 = vshrq_n_u8(n0, 4);
                n1 = vshrq_n_u8(n1, 4);
            } else {
                n0 = vandq_u8(n0, lo4);
                n1 = vandq_u8(n1, lo4);
            }
            const uint8x16_t m =
                vdupq_n_u8(static_cast<std::uint8_t>(1u << sb));
            a[2 * sb] = vreinterpretq_s8_u8(
                vaddq_u8(n0, vandq_u8(vtstq_u8(h0, m), sixteen)));
            a[2 * sb + 1] = vreinterpretq_s8_u8(
                vaddq_u8(n1, vandq_u8(vtstq_u8(h1, m), sixteen)));
        }
        for (int t = 0; t < T; ++t) {
            int sumi = 0;
            const std::int16_t* bsums = ys + t * ss + i * 16;
            for (int j = 0; j < 16; ++j) {
                sumi += bsums[j] * mins[j / 2];
            }
            const std::int8_t* q8 = yq + t * qs + i * 256;
            int32x4_t acc[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};
            for (int sb = 0; sb < 8; ++sb) {
                const auto sc = static_cast<std::int16_t>(scales[sb]);
                lanes8_madd(a[2 * sb], q8 + 32 * sb, sc, acc);
                lanes8_madd(a[2 * sb + 1], q8 + 32 * sb + 16, sc, acc);
            }
            vst1q_s32(aux32, acc[0]);
            vst1q_s32(aux32 + 4, acc[1]);
            const float dy = yd[t * ds + i];
            const float d = bd * dy;
            for (int l = 0; l < 8; ++l) {
                sums[t][l] += d * static_cast<float>(aux32[l]);
            }
            sumf[t] -= bdmin * dy * static_cast<float>(sumi);
        }
    }
    for (int t = 0; t < T; ++t) {
        for (int l = 0; l < 8; ++l) {
            sumf[t] += sums[t][l];
        }
        out[t] = sumf[t];
    }
}

/** The i#733 types: their T-token NEON kernels, each token the scalar
 *  bits. */
bool i733_type(gguf::TensorType t) {
    using T = gguf::TensorType;
    return t == T::kIQ2_XXS || t == T::kIQ2_S || t == T::kQ3_K ||
           t == T::kQ5_K;
}

std::size_t i733_block(gguf::TensorType t) {
    using T = gguf::TensorType;
    return t == T::kIQ2_XXS ? 66 : t == T::kIQ2_S ? 82
           : t == T::kQ3_K  ? 110 : 176;
}

/** Row `row` of type `type` against T tokens (see the kernels). */
template <int T>
void rows_i733_xt(gguf::TensorType type, const std::byte* row,
                  std::size_t nb, const float* yd, std::size_t ds,
                  const std::int8_t* yq, std::size_t qs,
                  const std::int16_t* ys, std::size_t ss, float* out) {
    switch (type) {
        case gguf::TensorType::kIQ2_XXS:
            rows_iq2_xxs_q8k_xt<T>(row, nb, yd, ds, yq, qs, out);
            break;
        case gguf::TensorType::kIQ2_S:
            rows_iq2_s_q8k_xt<T>(row, nb, yd, ds, yq, qs, out);
            break;
        case gguf::TensorType::kQ3_K:
            rows_q3_k_q8k_xt<T>(row, nb, yd, ds, yq, qs, out);
            break;
        default:
            rows_q5_k_q8k_xt<T>(row, nb, yd, ds, yq, qs, ys, ss, out);
            break;
    }
}

void matvec_neon_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out) {
    if (uses_quantized_activation(w.type) && !is_q8k_type(w.type) &&
        x.size() % 32 == 0) {
        // Block-32 weights x a Q8_0 activation (llama.cpp's vec_dot_type).
        const std::size_t nb = x.size() / 32;
        thread_local std::vector<float> d8;
        thread_local std::vector<std::int8_t> q8;
        d8.resize(nb);
        q8.resize(x.size());
        quantize_activation_q8_0_neon(x, d8.data(), q8.data());
        const std::size_t bb = w.type == gguf::TensorType::kQ8_0 ? 34 : 18;
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out[r] = row_block32_q8_0_neon(w.type, w.data + r * nb * bb, bb,
                                           nb, d8.data(), q8.data());
        }
        return;
    }
    const bool q4k = w.type == gguf::TensorType::kQ4_K;
    const bool q6k = w.type == gguf::TensorType::kQ6_K;
    if (i733_type(w.type) && x.size() % 256 == 0) {
        // The same bits as the scalar matvec_q8k (i#733).
        const std::size_t nb = x.size() / 256;
        thread_local std::vector<float> yd;
        thread_local std::vector<std::int8_t> yq;
        thread_local std::vector<std::int16_t> ys;
        yd.resize(nb);
        yq.resize(x.size());
        ys.resize(nb * 16);
        quantize_activation_q8k_neon(x, yd.data(), yq.data(), ys.data());
        const std::size_t bb = i733_block(w.type);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            rows_i733_xt<1>(w.type, w.data + r * nb * bb, nb, yd.data(), nb,
                            yq.data(), x.size(), ys.data(), nb * 16, &out[r]);
        }
        return;
    }
    const bool iq = iq_type(w.type);
    if ((!q4k && !q6k && !iq) || x.size() % 256 != 0) {
        matvec_q8k(w, x, out);  // other types: scalar Q8_K
        return;
    }
    const std::size_t nb = x.size() / 256;
    thread_local std::vector<float> yd;
    thread_local std::vector<std::int8_t> yq;
    thread_local std::vector<std::int16_t> ys;
    yd.resize(nb);
    yq.resize(x.size());
    ys.resize(nb * 16);
    quantize_activation_q8k_neon(x, yd.data(), yq.data(), ys.data());
    if (iq) {
        const std::size_t rb = mat_row_bytes(w);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            rows_iq_xt<1>(w.type, w.data + r * rb, nb, yd.data(), nb,
                          yq.data(), x.size(), ys.data(), nb * 16, &out[r]);
        }
        return;
    }
    const std::size_t bb = q4k ? 144 : 210;
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const std::byte* row = w.data + r * nb * bb;
        if (q4k) {
            rows_q4_k_q8k_xt<1>(row, nb, yd.data(), nb, yq.data(), x.size(),
                                ys.data(), nb * 16, &out[r]);
        } else {
            rows_q6_k_q8k_xt<1>(row, nb, yd.data(), nb, yq.data(), x.size(),
                                &out[r]);
        }
    }
}

/**
 * Tokens per tile of matvec_batch_neon_q8k: as many quantized
 * activation rows (cols bytes each) as fit about 64 KiB, so a tile
 * stays in L1 while every weight row of the slice streams past it.
 */
static std::uint32_t q8_token_tile(std::size_t cols) {
    return static_cast<std::uint32_t>(
        std::clamp<std::size_t>((std::size_t{64} << 10) / cols, 4, 64));
}

void matvec_batch_neon_q8k(const Mat& w, std::span<const float> x_batch,
                           std::span<float> out_batch, std::uint32_t n) {
    const std::size_t cols = w.cols;
    const bool block32 = uses_quantized_activation(w.type) &&
                         !is_q8k_type(w.type) && cols % 32 == 0;
    const bool q4k = w.type == gguf::TensorType::kQ4_K;
    const bool q6k = w.type == gguf::TensorType::kQ6_K;
    if (i733_type(w.type) && cols % 256 == 0) {
        // IQ2_XXS / IQ2_S / Q3_K / Q5_K (i#739): four tokens per weight
        // unpack, then the tail one by one; each token takes the
        // single-token steps, so the scalar bits.
        const std::size_t nb = cols / 256;
        thread_local std::vector<float> yd;
        thread_local std::vector<std::int8_t> yq;
        thread_local std::vector<std::int16_t> ys;
        yd.resize(nb * n);
        yq.resize(cols * n);
        ys.resize(nb * 16 * n);
        for (std::uint32_t t = 0; t < n; ++t) {
            quantize_activation_q8k_neon(x_batch.subspan(t * cols, cols),
                                         yd.data() + t * nb,
                                         yq.data() + t * cols,
                                         ys.data() + t * nb * 16);
        }
        const std::size_t bb = i733_block(w.type);
        const std::uint32_t tile = q8_token_tile(cols);
        constexpr int kT = 8;
        float o4[kT];
        for (std::uint32_t t0 = 0; t0 < n; t0 += tile) {
            const std::uint32_t t1 = std::min(n, t0 + tile);
            for (std::uint32_t r = 0; r < w.rows; ++r) {
                const std::byte* row = w.data + r * nb * bb;
                float* o = out_batch.data() + std::size_t{r} * n;
                std::uint32_t t = t0;
                for (; t + kT <= t1; t += kT) {
                    rows_i733_xt<kT>(w.type, row, nb, yd.data() + t * nb, nb,
                                     yq.data() + t * cols, cols,
                                     ys.data() + t * nb * 16, nb * 16, o4);
                    std::copy(o4, o4 + kT, o + t);
                }
                for (; t < t1; ++t) {
                    rows_i733_xt<1>(w.type, row, nb, yd.data() + t * nb, nb,
                                    yq.data() + t * cols, cols,
                                    ys.data() + t * nb * 16, nb * 16, o + t);
                }
            }
        }
        return;
    }
    if (iq_type(w.type) && cols % 256 == 0) {
        // The i#744 types: each token quantized once; then per row and
        // token tile, four tokens per decoded block, then the tail one
        // by one (a token's result does not depend on the batch).
        const std::size_t nb = cols / 256;
        thread_local std::vector<float> yd;
        thread_local std::vector<std::int8_t> yq;
        thread_local std::vector<std::int16_t> ys;
        yd.resize(nb * n);
        yq.resize(cols * n);
        ys.resize(nb * 16 * n);
        for (std::uint32_t t = 0; t < n; ++t) {
            quantize_activation_q8k_neon(x_batch.subspan(t * cols, cols),
                                         yd.data() + t * nb,
                                         yq.data() + t * cols,
                                         ys.data() + t * nb * 16);
        }
        const std::size_t rb = mat_row_bytes(w);
        const std::uint32_t tile = q8_token_tile(cols);
        constexpr int kT = 4;
        for (std::uint32_t t0 = 0; t0 < n; t0 += tile) {
            const std::uint32_t t1 = std::min(n, t0 + tile);
            for (std::uint32_t r = 0; r < w.rows; ++r) {
                const std::byte* row = w.data + r * rb;
                float* o = out_batch.data() + std::size_t{r} * n;
                std::uint32_t t = t0;
                for (; t + kT <= t1; t += kT) {
                    rows_iq_xt<kT>(w.type, row, nb, yd.data() + t * nb, nb,
                                   yq.data() + t * cols, cols,
                                   ys.data() + t * nb * 16, nb * 16, o + t);
                }
                for (; t < t1; ++t) {
                    rows_iq_xt<1>(w.type, row, nb, yd.data() + t * nb, nb,
                                  yq.data() + t * cols, cols,
                                  ys.data() + t * nb * 16, nb * 16, o + t);
                }
            }
        }
        return;
    }
    if (!block32 && ((!q4k && !q6k) || cols % 256 != 0)) {
        // Other types: n per-token calls, scattered row-major.
        thread_local std::vector<float> tmp;
        tmp.resize(w.rows);
        for (std::uint32_t t = 0; t < n; ++t) {
            self::matvec_neon_q8k(w, x_batch.subspan(t * cols, cols), tmp);
            for (std::uint32_t r = 0; r < w.rows; ++r) {
                out_batch[std::size_t{r} * n + t] = tmp[r];
            }
        }
        return;
    }
    if (block32) {
        // Block-32 weights x Q8_0 activations, as matvec_neon_q8k.
        const std::size_t nb = cols / 32;
        thread_local std::vector<float> d8;
        thread_local std::vector<std::int8_t> q8;
        d8.resize(nb * n);
        q8.resize(cols * n);
        for (std::uint32_t t = 0; t < n; ++t) {
            quantize_activation_q8_0_neon(x_batch.subspan(t * cols, cols),
                                     d8.data() + t * nb,
                                     q8.data() + t * cols);
        }
        const std::size_t bb = w.type == gguf::TensorType::kQ8_0 ? 34 : 18;
        const std::uint32_t tile = q8_token_tile(cols);
        for (std::uint32_t t0 = 0; t0 < n; t0 += tile) {
            const std::uint32_t t1 = std::min(n, t0 + tile);
            for (std::uint32_t r = 0; r < w.rows; ++r) {
                const std::byte* row = w.data + r * nb * bb;
                float* o = out_batch.data() + std::size_t{r} * n;
                std::uint32_t t = t0;
                for (; t + 4 <= t1; t += 4) {
                    rows_block32_q8_0_xt<4>(w.type, row, bb, nb,
                                            d8.data() + t * nb,
                                            q8.data() + t * cols, cols,
                                            o + t);
                }
                for (; t < t1; ++t) {
                    o[t] = row_block32_q8_0_neon(w.type, row, bb, nb,
                                                 d8.data() + t * nb,
                                                 q8.data() + t * cols);
                }
            }
        }
        return;
    }
    const std::size_t nb = cols / 256;
    thread_local std::vector<float> yd;
    thread_local std::vector<std::int8_t> yq;
    thread_local std::vector<std::int16_t> ys;
    yd.resize(nb * n);
    yq.resize(cols * n);
    ys.resize(nb * 16 * n);
    for (std::uint32_t t = 0; t < n; ++t) {
        quantize_activation_q8k_neon(x_batch.subspan(t * cols, cols),
                                yd.data() + t * nb, yq.data() + t * cols,
                                ys.data() + t * nb * 16);
    }
    const std::size_t bb = q4k ? 144 : 210;
    const std::uint32_t tile = q8_token_tile(cols);
    // Four tokens per row pass (the kernels' T), then the tail one by
    // one through the same kernels (so the result for a token does not
    // depend on where it falls in the batch).
    constexpr int kT = 4;
    float o4[kT];
    for (std::uint32_t t0 = 0; t0 < n; t0 += tile) {
        const std::uint32_t t1 = std::min(n, t0 + tile);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::byte* row = w.data + r * nb * bb;
            float* o = out_batch.data() + std::size_t{r} * n;
            std::uint32_t t = t0;
            for (; t + kT <= t1; t += kT) {
                const float* d = yd.data() + t * nb;
                const std::int8_t* q = yq.data() + t * cols;
                if (q4k) {
                    rows_q4_k_q8k_xt<kT>(row, nb, d, nb, q, cols,
                                         ys.data() + t * nb * 16, nb * 16,
                                         o4);
                } else {
                    rows_q6_k_q8k_xt<kT>(row, nb, d, nb, q, cols, o4);
                }
                std::copy(o4, o4 + kT, o + t);
            }
            for (; t < t1; ++t) {
                const float* d = yd.data() + t * nb;
                const std::int8_t* q = yq.data() + t * cols;
                if (q4k) {
                    rows_q4_k_q8k_xt<1>(row, nb, d, nb, q, cols,
                                        ys.data() + t * nb * 16, nb * 16,
                                        o + t);
                } else {
                    rows_q6_k_q8k_xt<1>(row, nb, d, nb, q, cols, o + t);
                }
            }
        }
    }
}

void matvec_batch_neon(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch,
                       std::uint32_t n) {
    const std::size_t cols = w.cols;
    // Register-blocked Q4_K: read+widen each weight block once, dot
    // it into all n per-token accumulators. Weight DRAM traffic
    // ~n-fold lower than n matvec() calls. Byte-identical to
    // dot_q4_k_block_neon summed by matvec_neon: the same dot16 /
    // q4k_group steps in the same order, each block's vector
    // accumulator reduced (vaddvq) and added to a per-token scalar --
    // NOT one reduction at the end (float addition is not
    // associative).
    if (w.type == gguf::TensorType::kQ4_K) {
        const std::size_t nblk = cols / 256;
        const std::size_t rb = nblk * 144;
        const std::size_t nsub = cols / 32;
        const uint8x16_t mask = vdupq_n_u8(0x0f);
        thread_local std::vector<float32x4_t> vacc;
        thread_local std::vector<float> sacc, mins, xs;
        vacc.resize(n);
        sacc.resize(n);
        mins.resize(n);
        xs.resize(n * nsub);
        for (std::uint32_t t = 0; t < n; ++t) {
            subblock_sums(x_batch.data() + t * cols, cols,
                          xs.data() + t * nsub);
        }
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::byte* row = w.data + r * rb;
            for (std::uint32_t t = 0; t < n; ++t) {
                sacc[t] = 0.0f;
            }
            for (std::size_t b = 0; b < nblk; ++b) {
                const std::byte* blk = row + b * 144;
                const float d = half_to_float(load_u16(blk));
                const float dmin = half_to_float(load_u16(blk + 2));
                const auto* scales =
                    reinterpret_cast<const std::uint8_t*>(blk + 4);
                const auto* q =
                    reinterpret_cast<const std::uint8_t*>(blk + 16);
                const std::size_t xb = b * 256;
                for (std::uint32_t t = 0; t < n; ++t) {
                    vacc[t] = vdupq_n_f32(0.0f);
                    mins[t] = 0.0f;
                }
                for (int j = 0, is = 0; j < 256;
                     j += 64, is += 2, q += 32) {
                    const uint8x16_t qb0 = vld1q_u8(q);
                    const uint8x16_t qb1 = vld1q_u8(q + 16);
                    float32x4_t f0[4], f1[4], f2[4], f3[4];
                    cvt16_q4k(vandq_u8(qb0, mask), f0);
                    cvt16_q4k(vandq_u8(qb1, mask), f1);
                    cvt16_q4k(vshrq_n_u8(qb0, 4), f2);
                    cvt16_q4k(vshrq_n_u8(qb1, 4), f3);
                    for (std::uint32_t t = 0; t < n; ++t) {
                        const float* xt =
                            x_batch.data() + t * cols + xb + j;
                        const Q4kGroup g =
                            q4k_group(scales, is, d, dmin,
                                      xs.data() + t * nsub + b * 8);
                        vacc[t] = vfmaq_n_f32(
                            vacc[t],
                            vaddq_f32(dot16(f0, xt),
                                      dot16(f1, xt + 16)),
                            g.d1);
                        vacc[t] = vfmaq_n_f32(
                            vacc[t],
                            vaddq_f32(dot16(f2, xt + 32),
                                      dot16(f3, xt + 48)),
                            g.d2);
                        mins[t] += g.mins;
                    }
                }
                for (std::uint32_t t = 0; t < n; ++t) {
                    sacc[t] += vaddvq_f32(vacc[t]) - mins[t];
                }
            }
            float* orow = out_batch.data() +
                          static_cast<std::size_t>(r) * n;
            for (std::uint32_t t = 0; t < n; ++t) {
                orow[t] = sacc[t];
            }
        }
        return;
    }
    // Register-blocked Q6_K: same amortization + per-block reduction,
    // mirroring dot_q6_k_block_neon's eight-stream-per-128 order.
    if (w.type == gguf::TensorType::kQ6_K) {
        const std::size_t nblk = cols / 256;
        const std::size_t rb = nblk * 210;
        thread_local std::vector<float32x4_t> vacc;
        thread_local std::vector<float> sacc;
        vacc.resize(n);
        sacc.resize(n);
        auto stream = [&](const std::uint8_t* ql16,
                          const std::uint8_t* qh16, float dsc,
                          std::size_t xoff, auto high,
                          auto qh_shift) {
            float32x4_t g[4];
            cvt16_q6k<decltype(high)::value,
                      decltype(qh_shift)::value>(ql16, qh16, g);
            for (std::uint32_t t = 0; t < n; ++t) {
                vacc[t] = vfmaq_n_f32(
                    vacc[t], dot16(g, x_batch.data() + t * cols + xoff),
                    dsc);
            }
        };
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::byte* row = w.data + r * rb;
            for (std::uint32_t t = 0; t < n; ++t) {
                sacc[t] = 0.0f;
            }
            for (std::size_t b = 0; b < nblk; ++b) {
                const std::byte* blk = row + b * 210;
                const auto* ql =
                    reinterpret_cast<const std::uint8_t*>(blk);
                const auto* qh =
                    reinterpret_cast<const std::uint8_t*>(blk + 128);
                const auto* sc =
                    reinterpret_cast<const std::int8_t*>(blk + 192);
                const float d = half_to_float(load_u16(blk + 208));
                for (std::uint32_t t = 0; t < n; ++t) {
                    vacc[t] = vdupq_n_f32(0.0f);
                }
                for (int half = 0; half < 256; half += 128) {
                    const std::size_t xh = b * 256 + half;
                    std::integral_constant<bool, false> lo;
                    std::integral_constant<bool, true> hi;
                    stream(ql, qh, d * sc[0], xh + 0, lo,
                           std::integral_constant<int, 0>{});
                    stream(ql + 16, qh + 16, d * sc[1], xh + 16, lo,
                           std::integral_constant<int, 0>{});
                    stream(ql + 32, qh, d * sc[2], xh + 32, lo,
                           std::integral_constant<int, 2>{});
                    stream(ql + 48, qh + 16, d * sc[3], xh + 48, lo,
                           std::integral_constant<int, 2>{});
                    stream(ql, qh, d * sc[4], xh + 64, hi,
                           std::integral_constant<int, 4>{});
                    stream(ql + 16, qh + 16, d * sc[5], xh + 80, hi,
                           std::integral_constant<int, 4>{});
                    stream(ql + 32, qh, d * sc[6], xh + 96, hi,
                           std::integral_constant<int, 6>{});
                    stream(ql + 48, qh + 16, d * sc[7], xh + 112, hi,
                           std::integral_constant<int, 6>{});
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
                for (std::uint32_t t = 0; t < n; ++t) {
                    sacc[t] += vaddvq_f32(vacc[t]);
                }
            }
            float* orow = out_batch.data() +
                          static_cast<std::size_t>(r) * n;
            for (std::uint32_t t = 0; t < n; ++t) {
                orow[t] = sacc[t];
            }
        }
        return;
    }
    // Register-blocked BF16 and F16: 2-row x 4-token tiles
    // (h16_tile_neon), byte-identical to dot_h16_neon per token. Token
    // blocks are the outer loop, so a block's 4 x rows stay in L1 while
    // the weight rows stream past them.
    const auto h16 = [&]<class W>(W) {
        const auto* rows = reinterpret_cast<const std::uint16_t*>(w.data);
        float* out = out_batch.data();
        std::uint32_t t = 0;
        for (; t + 4 <= n; t += 4) {
            const float* xt = x_batch.data() + t * cols;
            std::uint32_t r = 0;
            for (; r + 2 <= w.rows; r += 2) {
                h16_tile_neon<W, 2, 4>(rows + r * cols, xt, cols,
                                       out + std::size_t{r} * n + t, n);
            }
            if (r < w.rows) {
                h16_tile_neon<W, 1, 4>(rows + r * cols, xt, cols,
                                       out + std::size_t{r} * n + t, n);
            }
        }
        for (; t < n; ++t) {
            const float* xt = x_batch.data() + t * cols;
            for (std::uint32_t r = 0; r < w.rows; ++r) {
                out[std::size_t{r} * n + t] =
                    dot_h16_neon<W>(rows + r * cols, xt, cols);
            }
        }
    };
    if (w.type == gguf::TensorType::kBF16) {
        h16(WidenBf16{});
        return;
    }
    if (w.type == gguf::TensorType::kF16) {
        h16(WidenF16{});
        return;
    }
    // Other types: per-token NEON, scattered row-major. Byte-
    // identical to the per-token path (same matvec_neon dots).
    thread_local std::vector<float> col;
    col.resize(w.rows);
    for (std::uint32_t t = 0; t < n; ++t) {
        self::matvec_neon(
            w, x_batch.subspan(static_cast<std::size_t>(t) * cols,
                               cols),
            col);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out_batch[static_cast<std::size_t>(r) * n + t] = col[r];
        }
    }
}

#if defined(LOCUS_NEON_DOTPROD_VARIANT)
}  // namespace neon_dotprod
#endif
}  // namespace locus::backend

#endif  // __aarch64__
