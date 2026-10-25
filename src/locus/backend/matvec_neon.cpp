#include "locus/backend/variants.hpp"

#if defined(__aarch64__)

#include <arm_neon.h>

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

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
    const float d = f16_to_f32(load_u16(blk));
    const float dmin = f16_to_f32(load_u16(blk + 2));
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
    const float d = f16_to_f32(load_u16(blk + 208));
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
                const float d = f16_to_f32(load_u16(blk));
                const float dmin = f16_to_f32(load_u16(blk + 2));
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
                const float d = f16_to_f32(load_u16(blk + 208));
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
    // Other types: per-token NEON, scattered row-major. Byte-
    // identical to the per-token path (same matvec_neon dots).
    thread_local std::vector<float> col;
    col.resize(w.rows);
    for (std::uint32_t t = 0; t < n; ++t) {
        matvec_neon(
            w, x_batch.subspan(static_cast<std::size_t>(t) * cols,
                               cols),
            col);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out_batch[static_cast<std::size_t>(r) * n + t] = col[r];
        }
    }
}

}  // namespace locus::backend

#endif  // __aarch64__
