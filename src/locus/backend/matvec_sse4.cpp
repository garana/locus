#include "locus/backend/variants.hpp"

#if defined(__x86_64__) && defined(__SSE4_1__)

#include <smmintrin.h>  // SSE4.1
#if defined(LOCUS_SSE4_AVX_VARIANT)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cstring>
#include <vector>

#include "iq_grids.h"

namespace locus::backend {
// matvec_sse4_avx.cpp compiles this file again with AVX enabled, into
// sse4_avx:: (variants.hpp).
#if defined(LOCUS_SSE4_AVX_VARIANT)
namespace sse4_avx {
namespace self = ::locus::backend::sse4_avx;
#else
namespace self = ::locus::backend;
#endif

namespace {

std::uint16_t load_u16(const std::byte* p) {
    std::uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

float hsum(__m128 v) {
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

float dot_f32_sse4(const float* w, const float* x,
                   std::size_t n) {
    __m128 acc = _mm_setzero_ps();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = _mm_add_ps(
            _mm_mul_ps(_mm_loadu_ps(w + i), _mm_loadu_ps(x + i)),
            acc);
    }
    float s = hsum(acc);
    for (; i < n; ++i) {
        s += w[i] * x[i];
    }
    return s;
}

/** One Q8_0 block: f16 scale + 32 int8 quants (34 bytes). */
float dot_q8_0_block_sse4(const std::byte* blk, const float* x) {
    const float d = f16_to_f32(load_u16(blk));
    const auto* q = reinterpret_cast<const std::int8_t*>(blk + 2);
    __m128 acc = _mm_setzero_ps();
    for (int g = 0; g < 32; g += 4) {
        std::int32_t packed;
        std::memcpy(&packed, q + g, sizeof(packed));
        const __m128 wf = _mm_cvtepi32_ps(
            _mm_cvtepi8_epi32(_mm_cvtsi32_si128(packed)));
        acc = _mm_add_ps(_mm_mul_ps(wf, _mm_loadu_ps(x + g)), acc);
    }
    return d * hsum(acc);
}

/** get_scale_min_k4; matches scale_min_k4 in cpu_ops.cpp. */
void scale_min_k4_sse(int j, const std::uint8_t* q,
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

/** One step of accum16_q4k_sse: values K..K+3. K is a template
 * argument because _mm_srli_si128 takes an 8-bit immediate; a loop
 * variable there only compiled where -O3 unrolled the loop. */
template <int K>
__m128 accum4_q4k_sse(__m128 acc, __m128i vals, __m128 dv, __m128 mv,
                      const float* xp) {
    const __m128i v4 = _mm_srli_si128(vals, K);
    __m128 f = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(v4));
    f = _mm_sub_ps(_mm_mul_ps(f, dv), mv);
    return _mm_add_ps(acc, _mm_mul_ps(f, _mm_loadu_ps(xp + K)));
}

/** Accumulates 16 unsigned nibble values (each byte 0..15) as
 * (d*v - m)*x into acc. */
__m128 accum16_q4k_sse(__m128 acc, __m128i vals, float d, float m,
                       const float* xp) {
    const __m128 dv = _mm_set1_ps(d), mv = _mm_set1_ps(m);
    acc = accum4_q4k_sse<0>(acc, vals, dv, mv, xp);
    acc = accum4_q4k_sse<4>(acc, vals, dv, mv, xp);
    acc = accum4_q4k_sse<8>(acc, vals, dv, mv, xp);
    return accum4_q4k_sse<12>(acc, vals, dv, mv, xp);
}

/** One Q4_K super-block (256 elems, 144 bytes): fused dequant +
 * dot, mirroring dequant_block_q4_k. */
float dot_q4_k_block_sse4(const std::byte* blk, const float* x) {
    const float d = f16_to_f32(load_u16(blk));
    const float dmin = f16_to_f32(load_u16(blk + 2));
    const auto* scales =
        reinterpret_cast<const std::uint8_t*>(blk + 4);
    const auto* q =
        reinterpret_cast<const std::uint8_t*>(blk + 16);
    const __m128i mask = _mm_set1_epi8(0x0f);
    __m128 acc = _mm_setzero_ps();
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        std::uint8_t sc, mn;
        scale_min_k4_sse(is + 0, scales, sc, mn);
        const float d1 = d * sc, m1 = dmin * mn;
        scale_min_k4_sse(is + 1, scales, sc, mn);
        const float d2 = d * sc, m2 = dmin * mn;
        const __m128i qb0 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
        const __m128i qb1 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(q + 16));
        acc = accum16_q4k_sse(acc, _mm_and_si128(qb0, mask), d1,
                              m1, x + j);
        acc = accum16_q4k_sse(acc, _mm_and_si128(qb1, mask), d1,
                              m1, x + j + 16);
        acc = accum16_q4k_sse(
            acc, _mm_and_si128(_mm_srli_epi16(qb0, 4), mask), d2,
            m2, x + j + 32);
        acc = accum16_q4k_sse(
            acc, _mm_and_si128(_mm_srli_epi16(qb1, 4), mask), d2,
            m2, x + j + 48);
        q += 32;
        is += 2;
    }
    return hsum(acc);
}

/** One step of accum16_q6k_sse: values K..K+3 (K an immediate, as in
 * accum4_q4k_sse). */
template <int K>
__m128 accum4_q6k_sse(__m128 acc, __m128i sv, __m128 s,
                      const float* xp) {
    const __m128i v4 = _mm_srli_si128(sv, K);
    const __m128 f = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(v4));
    return _mm_add_ps(
        acc, _mm_mul_ps(_mm_mul_ps(f, s), _mm_loadu_ps(xp + K)));
}

/** Accumulates 16 Q6_K combined values (each byte 0..63) as
 * (v - 32) * dsc * x into acc. */
__m128 accum16_q6k_sse(__m128 acc, __m128i vals, float dsc,
                       const float* xp) {
    const __m128i sv = _mm_sub_epi8(vals, _mm_set1_epi8(32));
    const __m128 s = _mm_set1_ps(dsc);
    acc = accum4_q6k_sse<0>(acc, sv, s, xp);
    acc = accum4_q6k_sse<4>(acc, sv, s, xp);
    acc = accum4_q6k_sse<8>(acc, sv, s, xp);
    return accum4_q6k_sse<12>(acc, sv, s, xp);
}

/** Reconstructs 16 Q6_K combined values (0..63) from ql (high
 * nibble if HIGH) and qh 2 bits at qh_shift, shifted into place. */
__m128i q6k_vals_sse(__m128i qlb, __m128i qhb, bool high,
                     int qh_shift) {
    const __m128i mask4 = _mm_set1_epi8(0x0f);
    const __m128i low =
        high ? _mm_and_si128(_mm_srli_epi16(qlb, 4), mask4)
             : _mm_and_si128(qlb, mask4);
    const __m128i hbits = _mm_slli_epi16(
        _mm_and_si128(_mm_srli_epi16(qhb, qh_shift),
                      _mm_set1_epi8(3)),
        4);
    return _mm_or_si128(low, hbits);
}

/** One Q6_K super-block (256 elems, 210 bytes): fused dequant +
 * dot, mirroring dequant_block_q6_k. */
float dot_q6_k_block_sse4(const std::byte* blk, const float* x) {
    const auto* ql = reinterpret_cast<const std::uint8_t*>(blk);
    const auto* qh =
        reinterpret_cast<const std::uint8_t*>(blk + 128);
    const auto* sc =
        reinterpret_cast<const std::int8_t*>(blk + 192);
    const float d = f16_to_f32(load_u16(blk + 208));
    __m128 acc = _mm_setzero_ps();
    for (int n = 0; n < 256; n += 128) {
        const __m128i ql0 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(ql));
        const __m128i ql1 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(ql + 16));
        const __m128i ql2 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(ql + 32));
        const __m128i ql3 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(ql + 48));
        const __m128i qh0 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(qh));
        const __m128i qh1 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(qh + 16));
        acc = accum16_q6k_sse(
            acc, q6k_vals_sse(ql0, qh0, false, 0), d * sc[0],
            x + n);
        acc = accum16_q6k_sse(
            acc, q6k_vals_sse(ql1, qh1, false, 0), d * sc[1],
            x + n + 16);
        acc = accum16_q6k_sse(
            acc, q6k_vals_sse(ql2, qh0, false, 2), d * sc[2],
            x + n + 32);
        acc = accum16_q6k_sse(
            acc, q6k_vals_sse(ql3, qh1, false, 2), d * sc[3],
            x + n + 48);
        acc = accum16_q6k_sse(
            acc, q6k_vals_sse(ql0, qh0, true, 4), d * sc[4],
            x + n + 64);
        acc = accum16_q6k_sse(
            acc, q6k_vals_sse(ql1, qh1, true, 4), d * sc[5],
            x + n + 80);
        acc = accum16_q6k_sse(
            acc, q6k_vals_sse(ql2, qh0, true, 6), d * sc[6],
            x + n + 96);
        acc = accum16_q6k_sse(
            acc, q6k_vals_sse(ql3, qh1, true, 6), d * sc[7],
            x + n + 112);
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return hsum(acc);
}

// R11 batched matvec: block-outer / token-inner so each weight block
// is streamed from DRAM once and reused across all n tokens (n-fold
// less weight traffic), with the dequant precomputed once per block.
// Each token's accumulator runs the identical accum16 sequence + hsum
// as the per-token kernel, so the output is byte-identical to n
// matvec_sse4() calls. out is row-major: out[r*n + t].

void batch_q4_k_sse(const Mat& w, const float* xb, float* ob,
                    std::uint32_t n) {
    const std::size_t rb = w.cols / 256ull * 144ull;
    const std::uint32_t nblk = w.cols / 256;
    const __m128i mask = _mm_set1_epi8(0x0f);
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const std::byte* row = w.data + r * rb;
        float* o = ob + static_cast<std::size_t>(r) * n;
        for (std::uint32_t t = 0; t < n; ++t) {
            o[t] = 0.0f;
        }
        for (std::uint32_t b = 0; b < nblk; ++b) {
            const std::byte* blk = row + b * 144;
            const float d = f16_to_f32(load_u16(blk));
            const float dmin = f16_to_f32(load_u16(blk + 2));
            const auto* scales =
                reinterpret_cast<const std::uint8_t*>(blk + 4);
            const auto* q =
                reinterpret_cast<const std::uint8_t*>(blk + 16);
            __m128i gv[16];
            float gd[16], gm[16];
            std::uint32_t goff[16];
            int gi = 0, is = 0;
            for (int j = 0; j < 256; j += 64) {
                std::uint8_t sc, mn;
                scale_min_k4_sse(is + 0, scales, sc, mn);
                const float d1 = d * sc, m1 = dmin * mn;
                scale_min_k4_sse(is + 1, scales, sc, mn);
                const float d2 = d * sc, m2 = dmin * mn;
                const __m128i qb0 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(q));
                const __m128i qb1 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(q + 16));
                const std::uint32_t xb0 = b * 256 + j;
                gv[gi] = _mm_and_si128(qb0, mask);
                gd[gi] = d1, gm[gi] = m1, goff[gi] = xb0, ++gi;
                gv[gi] = _mm_and_si128(qb1, mask);
                gd[gi] = d1, gm[gi] = m1, goff[gi] = xb0 + 16, ++gi;
                gv[gi] = _mm_and_si128(_mm_srli_epi16(qb0, 4), mask);
                gd[gi] = d2, gm[gi] = m2, goff[gi] = xb0 + 32, ++gi;
                gv[gi] = _mm_and_si128(_mm_srli_epi16(qb1, 4), mask);
                gd[gi] = d2, gm[gi] = m2, goff[gi] = xb0 + 48, ++gi;
                q += 32;
                is += 2;
            }
            for (std::uint32_t t = 0; t < n; ++t) {
                const float* xt =
                    xb + static_cast<std::size_t>(t) * w.cols;
                __m128 bacc = _mm_setzero_ps();
                for (int i = 0; i < 16; ++i) {
                    bacc = accum16_q4k_sse(bacc, gv[i], gd[i], gm[i],
                                           xt + goff[i]);
                }
                o[t] += hsum(bacc);
            }
        }
    }
}

void batch_q6_k_sse(const Mat& w, const float* xb, float* ob,
                    std::uint32_t n) {
    const std::size_t rb = w.cols / 256ull * 210ull;
    const std::uint32_t nsb = w.cols / 256;
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const std::byte* row = w.data + r * rb;
        float* o = ob + static_cast<std::size_t>(r) * n;
        for (std::uint32_t t = 0; t < n; ++t) {
            o[t] = 0.0f;
        }
        for (std::uint32_t b = 0; b < nsb; ++b) {
            const std::byte* blk = row + b * 210;
            const auto* qlp =
                reinterpret_cast<const std::uint8_t*>(blk);
            const auto* qhp =
                reinterpret_cast<const std::uint8_t*>(blk + 128);
            const auto* scp =
                reinterpret_cast<const std::int8_t*>(blk + 192);
            const float d = f16_to_f32(load_u16(blk + 208));
            __m128i gv[16];
            float gdsc[16];
            std::uint32_t goff[16];
            int gi = 0;
            for (int nn = 0; nn < 256; nn += 128) {
                const __m128i q0 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(qlp));
                const __m128i q1 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(qlp + 16));
                const __m128i q2 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(qlp + 32));
                const __m128i q3 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(qlp + 48));
                const __m128i h0 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(qhp));
                const __m128i h1 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(qhp + 16));
                const std::uint32_t base = b * 256 + nn;
                gv[gi] = q6k_vals_sse(q0, h0, false, 0);
                gdsc[gi] = d * scp[0], goff[gi] = base, ++gi;
                gv[gi] = q6k_vals_sse(q1, h1, false, 0);
                gdsc[gi] = d * scp[1], goff[gi] = base + 16, ++gi;
                gv[gi] = q6k_vals_sse(q2, h0, false, 2);
                gdsc[gi] = d * scp[2], goff[gi] = base + 32, ++gi;
                gv[gi] = q6k_vals_sse(q3, h1, false, 2);
                gdsc[gi] = d * scp[3], goff[gi] = base + 48, ++gi;
                gv[gi] = q6k_vals_sse(q0, h0, true, 4);
                gdsc[gi] = d * scp[4], goff[gi] = base + 64, ++gi;
                gv[gi] = q6k_vals_sse(q1, h1, true, 4);
                gdsc[gi] = d * scp[5], goff[gi] = base + 80, ++gi;
                gv[gi] = q6k_vals_sse(q2, h0, true, 6);
                gdsc[gi] = d * scp[6], goff[gi] = base + 96, ++gi;
                gv[gi] = q6k_vals_sse(q3, h1, true, 6);
                gdsc[gi] = d * scp[7], goff[gi] = base + 112, ++gi;
                qlp += 64;
                qhp += 32;
                scp += 8;
            }
            for (std::uint32_t t = 0; t < n; ++t) {
                const float* xt =
                    xb + static_cast<std::size_t>(t) * w.cols;
                __m128 bacc = _mm_setzero_ps();
                for (int i = 0; i < 16; ++i) {
                    bacc = accum16_q6k_sse(bacc, gv[i], gdsc[i],
                                           xt + goff[i]);
                }
                o[t] += hsum(bacc);
            }
        }
    }
}

}  // namespace

void matvec_sse4(const Mat& w, std::span<const float> x,
                 std::span<float> out) {
    if (w.type == gguf::TensorType::kF32) {
        const auto* rows = reinterpret_cast<const float*>(w.data);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out[r] = dot_f32_sse4(
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
                acc += dot_q8_0_block_sse4(row + b * 34,
                                           x.data() + b * 32);
            }
            out[r] = acc;
        }
        return;
    }
    if (w.type == gguf::TensorType::kQ4_K) {
        const std::size_t row_bytes = w.cols / 256ull * 144ull;
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::byte* row = w.data + r * row_bytes;
            float acc = 0.0f;
            for (std::uint32_t b = 0; b < w.cols / 256; ++b) {
                acc += dot_q4_k_block_sse4(row + b * 144,
                                           x.data() + b * 256);
            }
            out[r] = acc;
        }
        return;
    }
    if (w.type == gguf::TensorType::kQ6_K) {
        const std::size_t row_bytes = w.cols / 256ull * 210ull;
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::byte* row = w.data + r * row_bytes;
            float acc = 0.0f;
            for (std::uint32_t b = 0; b < w.cols / 256; ++b) {
                acc += dot_q6_k_block_sse4(row + b * 210,
                                           x.data() + b * 256);
            }
            out[r] = acc;
        }
        return;
    }
    matvec(w, x, out);  // other types: scalar reference
}

namespace {

/** SSE4 Q3_K x Q8_K aux32[8]: 2-bit + hmask 3rd-bit unpack (-4 offset),
 *  6-bit scales unpacked from the packed 12 bytes, 16 groups x 2 with
 *  scale=(scales[j]-32); no min term. Bit-identical to scalar
 *  vec_dot_q3_k_q8_k. */
void q3_k_q8k_aux32_sse4(const std::byte* blk, const std::int8_t* q8,
                         std::int32_t aux32[8]) {
    constexpr std::uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const auto* hm = reinterpret_cast<const std::uint8_t*>(blk);
    const auto* q3 = reinterpret_cast<const std::uint8_t*>(blk + 32);
    std::int8_t aux8[256];
    std::int8_t* a = aux8;
    std::uint8_t m = 1;
    const std::uint8_t* q3p = q3;
    for (int j = 0; j < 256; j += 128) {
        for (int sh = 0; sh < 8; sh += 2) {
            for (int l = 0; l < 32; ++l) {
                a[l] = static_cast<std::int8_t>((q3p[l] >> sh) & 3);
            }
            for (int l = 0; l < 32; ++l) {
                a[l] = static_cast<std::int8_t>(
                    a[l] - ((hm[l] & m) ? 0 : 4));
            }
            a += 32;
            m <<= 1;
        }
        q3p += 32;
    }
    std::uint32_t auxs[4];
    std::memcpy(auxs, blk + 96, 12);
    const std::uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const auto* scales = reinterpret_cast<const std::int8_t*>(auxs);
    __m128i acc_lo = _mm_setzero_si128(), acc_hi = _mm_setzero_si128();
    const std::int8_t* ap = aux8;
    const std::int8_t* q8p = q8;
    for (int j = 0; j < 16; ++j) {
        const __m128i scv = _mm_set1_epi32(scales[j] - 32);
        for (int k = 0; k < 2; ++k) {
            const __m128i qv = _mm_cvtepi8_epi16(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q8p)));
            const __m128i av = _mm_cvtepi8_epi16(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ap)));
            const __m128i p = _mm_mullo_epi16(qv, av);
            acc_lo = _mm_add_epi32(
                acc_lo, _mm_mullo_epi32(_mm_cvtepi16_epi32(p), scv));
            acc_hi = _mm_add_epi32(
                acc_hi, _mm_mullo_epi32(
                            _mm_cvtepi16_epi32(_mm_srli_si128(p, 8)),
                            scv));
            q8p += 8;
            ap += 8;
        }
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32), acc_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32 + 4), acc_hi);
}

/** @returns The sum of the four int32 lanes of v. */
int hsum_epi32_sse4(__m128i v) {
    const __m128i hi = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    const __m128i lo = _mm_add_epi32(hi, _mm_srli_si128(hi, 4));
    return _mm_cvtsi128_si32(lo);
}

/** SSE4 Q2_K x Q8_K: computes isum (the scaled 2-bit dot) and summs
 *  (the bsums min term) for one super-block. isuml over 16 values is a
 *  total, so maddubs pair-collapse is exact here (no overflow: 3*127
 *  per product). Bit-identical to scalar vec_dot_q2_k_q8_k. */
void q2_k_q8k_block_sse4(const std::byte* blk, const std::int8_t* q8,
                         const std::int16_t* bsums, int& isum,
                         int& summs) {
    const auto* sc = reinterpret_cast<const std::uint8_t*>(blk);
    const auto* q2 = reinterpret_cast<const std::uint8_t*>(blk + 16);
    summs = 0;
    for (int j = 0; j < 16; ++j) {
        summs += bsums[j] * (sc[j] >> 4);
    }
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i m3 = _mm_set1_epi8(3);
    isum = 0;
    int is = 0;
    for (int k = 0; k < 2; ++k) {
        const __m128i q2lo = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(q2 + k * 32));
        const __m128i q2hi = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(q2 + k * 32 + 16));
        for (int j = 0; j < 4; ++j) {
            const int shift = 2 * j;
            const __m128i v0 = _mm_and_si128(
                _mm_srli_epi16(q2lo, shift), m3);  // 16 x 2-bit
            const __m128i v1 =
                _mm_and_si128(_mm_srli_epi16(q2hi, shift), m3);
            const __m128i qa = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(q8));
            const __m128i qb = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(q8 + 16));
            const int isum0 = hsum_epi32_sse4(
                _mm_madd_epi16(_mm_maddubs_epi16(v0, qa), ones));
            const int isum1 = hsum_epi32_sse4(
                _mm_madd_epi16(_mm_maddubs_epi16(v1, qb), ones));
            isum += (sc[is++] & 0xF) * isum0;
            isum += (sc[is++] & 0xF) * isum1;
            q8 += 32;
        }
    }
}

/** Byte order (0,8,1,9,...,7,15) inside each 16-byte chunk. Applied to
 *  both weights and q8, it makes each _mm_maddubs_epi16 pair-sum add
 *  elements e and e+8, which share the scalar lane e % 8. */
__m128i lane_pairs_mask() {
    return _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14,
                         7, 15);
}

/** Adds scale * s (8 int16 per-lane sums) to the int32 lanes lo (0..3)
 *  and hi (4..7). */
void add_scaled_lanes(__m128i s, int scale, __m128i& lo, __m128i& hi) {
    // madd on s interleaved with zeros: each int32 is s_e * scale + 0 *
    // (scale's high half), the product exactly, one fast instruction
    // where mullo_epi32 is two slow ones (i#719). Signed scales (Q6_K's
    // int8) work too: the zero cancels the sign-extended high half.
    const __m128i scv = _mm_set1_epi32(scale);
    const __m128i z = _mm_setzero_si128();
    lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(s, z), scv));
    hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(s, z), scv));
}

/** Adds scale * (m0 + m1) to the int32 lanes lo (0..3) and hi (4..7),
 *  m0 and m1 being two int16 pair sums per lane (maddubs results): the
 *  two interleaved, one madd by the scale adds both, exactly. */
void add_scaled_pairs(__m128i m0, __m128i m1, int scale, __m128i& lo,
                      __m128i& hi) {
    const __m128i scv = _mm_set1_epi16(static_cast<short>(scale));
    lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(m0, m1), scv));
    hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(m0, m1), scv));
}

/** The Q4_K / Q5_K min term: bsums[j] * mins[j / 2] over j = 0..15, the
 *  mins being the bytes of utmp[2..3] after the scale unpack; one madd
 *  per 8 bsums (|bsum| <= 32 * 127, mins < 64: exact in int32). */
int k_min_term(const std::uint32_t utmp[4], const std::int16_t* bsums) {
    const auto mins16 = [](std::uint32_t m) {
        const __m128i v = _mm_cvtsi32_si128(static_cast<int>(m));
        return _mm_cvtepu8_epi16(_mm_unpacklo_epi8(v, v));
    };
    return hsum_epi32_sse4(_mm_add_epi32(
        _mm_madd_epi16(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(bsums)),
            mins16(utmp[2])),
        _mm_madd_epi16(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(bsums + 8)),
            mins16(utmp[3]))));
}

/** SSE4 Q4_K x Q8_K: fills aux32[8] and sumi (the bsums min term)
 *  exactly as scalar vec_dot_q4_k_q8_k does, so the caller's f32 sums
 *  match bit-for-bit. q8il is the block's q8 in lane_pairs_mask order;
 *  the nibbles get the same shuffle, so _mm_maddubs_epi16 (unsigned
 *  nibble x signed q8) sums two products of one lane. Integer sums are
 *  exact, so the add order inside a lane does not matter. */
void q4_k_q8k_aux32_sse4(const std::byte* blk, const std::int8_t* q8il,
                         const std::int16_t* bsums, std::int32_t aux32[8],
                         int& sumi) {
    constexpr std::uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f,
                            kmask3 = 0x03030303;
    const auto* q4 = reinterpret_cast<const std::uint8_t*>(blk + 16);
    std::uint32_t utmp[4];
    std::memcpy(utmp, blk + 4, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const std::uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const auto* scales = reinterpret_cast<const std::uint8_t*>(utmp);
    sumi = k_min_term(utmp, bsums);
    const __m128i lo4 = _mm_set1_epi8(0x0f);
    const __m128i il = lane_pairs_mask();
    __m128i acc_lo = _mm_setzero_si128(), acc_hi = _mm_setzero_si128();
    // Sub-blocks 2k (low nibbles) and 2k + 1 (high nibbles) share one
    // 32-byte chunk: load and shuffle it once for both.
    for (int k = 0; k < 4; ++k) {
        const std::uint8_t* q4b = q4 + k * 32;
        const __m128i r0 = _mm_shuffle_epi8(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4b)), il);
        const __m128i r1 = _mm_shuffle_epi8(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4b + 16)),
            il);
        for (int h = 0; h < 2; ++h) {
            const int j = 2 * k + h;
            const std::int8_t* q8g = q8il + j * 32;
            const __m128i n0 =
                h == 0 ? _mm_and_si128(r0, lo4)
                       : _mm_and_si128(_mm_srli_epi16(r0, 4), lo4);
            const __m128i n1 =
                h == 0 ? _mm_and_si128(r1, lo4)
                       : _mm_and_si128(_mm_srli_epi16(r1, 4), lo4);
            const __m128i a0 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8g));
            const __m128i a1 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(q8g + 16));
            // |pair sum| <= 2*15*127 per maddubs lane
            add_scaled_pairs(_mm_maddubs_epi16(n0, a0),
                             _mm_maddubs_epi16(n1, a1), scales[j], acc_lo,
                             acc_hi);
        }
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32), acc_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32 + 4), acc_hi);
}

/** SSE4 Q5_K x Q8_K: like q4_k_q8k_aux32_sse4, plus the 5th bit from
 *  hm (shuffled the same way) added in-register; the value stays
 *  unsigned [0,31], so maddubs still applies. Bit-identical to scalar
 *  vec_dot_q5_k_q8_k. */
void q5_k_q8k_aux32_sse4(const std::byte* blk, const std::int8_t* q8il,
                         const std::int16_t* bsums, std::int32_t aux32[8],
                         int& sumi) {
    constexpr std::uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f,
                            kmask3 = 0x03030303;
    const auto* hm = reinterpret_cast<const std::uint8_t*>(blk + 16);
    const auto* q4 = reinterpret_cast<const std::uint8_t*>(blk + 48);
    std::uint32_t utmp[4];
    std::memcpy(utmp, blk + 4, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const std::uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const auto* scales = reinterpret_cast<const std::uint8_t*>(utmp);
    sumi = k_min_term(utmp, bsums);
    const __m128i lo4 = _mm_set1_epi8(0x0f);
    const __m128i s16 = _mm_set1_epi8(16);
    const __m128i il = lane_pairs_mask();
    const __m128i hm0 = _mm_shuffle_epi8(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(hm)), il);
    const __m128i hm1 = _mm_shuffle_epi8(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(hm + 16)), il);
    __m128i acc_lo = _mm_setzero_si128(), acc_hi = _mm_setzero_si128();
    // As q4_k_q8k_aux32_sse4: one load and shuffle per nibble chunk for
    // sub-blocks 2k and 2k + 1, one madd per sub-block.
    for (int k = 0; k < 4; ++k) {
        const std::uint8_t* q4b = q4 + k * 32;
        const __m128i r0 = _mm_shuffle_epi8(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4b)), il);
        const __m128i r1 = _mm_shuffle_epi8(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4b + 16)),
            il);
        for (int h = 0; h < 2; ++h) {
            const int j = 2 * k + h;
            const std::int8_t* q8g = q8il + j * 32;
            __m128i n0 = h == 0
                             ? _mm_and_si128(r0, lo4)
                             : _mm_and_si128(_mm_srli_epi16(r0, 4), lo4);
            __m128i n1 = h == 0
                             ? _mm_and_si128(r1, lo4)
                             : _mm_and_si128(_mm_srli_epi16(r1, 4), lo4);
            // add 16 where the j-th hm bit is set (5th weight bit)
            const __m128i bit = _mm_set1_epi8(static_cast<char>(1 << j));
            n0 = _mm_add_epi8(
                n0, _mm_and_si128(
                        _mm_cmpeq_epi8(_mm_and_si128(hm0, bit), bit), s16));
            n1 = _mm_add_epi8(
                n1, _mm_and_si128(
                        _mm_cmpeq_epi8(_mm_and_si128(hm1, bit), bit), s16));
            const __m128i a0 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8g));
            const __m128i a1 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(q8g + 16));
            add_scaled_pairs(_mm_maddubs_epi16(n0, a0),
                             _mm_maddubs_epi16(n1, a1), scales[j], acc_lo,
                             acc_hi);
        }
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32), acc_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32 + 4), acc_hi);
}

/** A Q6_K block's 256 raw 6-bit codes into aux, in scalar's layout. */
void q6_k_raw6_sse4(const std::byte* blk, std::uint8_t aux[256]) {
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i lo4 = _mm_set1_epi8(0x0f);
    const auto* ql = reinterpret_cast<const std::uint8_t*>(blk);
    const auto* qh = reinterpret_cast<const std::uint8_t*>(blk + 128);
    std::uint8_t* a = aux;
    for (int n = 0; n < 256; n += 128) {
        for (int h = 0; h < 32; h += 16) {
            const __m128i qlA = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(ql + h));
            const __m128i qlB = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(ql + 32 + h));
            const __m128i qhv = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(qh + h));
            const __m128i hb0 = _mm_slli_epi16(
                _mm_and_si128(qhv, m3), 4);
            const __m128i hb2 = _mm_slli_epi16(
                _mm_and_si128(_mm_srli_epi16(qhv, 2), m3), 4);
            const __m128i hb4 = _mm_slli_epi16(
                _mm_and_si128(_mm_srli_epi16(qhv, 4), m3), 4);
            const __m128i hb6 = _mm_slli_epi16(
                _mm_and_si128(_mm_srli_epi16(qhv, 6), m3), 4);
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(a + h),
                _mm_or_si128(_mm_and_si128(qlA, lo4), hb0));
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(a + 32 + h),
                _mm_or_si128(_mm_and_si128(qlB, lo4), hb2));
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(a + 64 + h),
                _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(qlA, 4), lo4), hb4));
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(a + 96 + h),
                _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(qlB, 4), lo4), hb6));
        }
        ql += 64;
        qh += 32;
        a += 128;
    }
}

/** SSE4 Q6_K x Q8_K: fills aux32[8] exactly as scalar
 *  vec_dot_q6_k_q8_k does (no min term). Q6_K values are signed
 *  (raw6-32), so maddubs (unsigned x signed) needs the offset trick
 *  per lane: sum((raw6-32)*q8) = sum(raw6*q8) - 32*sum(q8). raw6 (the
 *  ql-nibble | qh-2bit code, [0,63]) is reconstructed in-register,
 *  then shuffled to lane_pairs_mask order to match q8il. */
void q6_k_q8k_aux32_sse4(const std::byte* blk, const std::int8_t* q8il,
                         std::int32_t aux32[8]) {
    const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
    const auto* ql = reinterpret_cast<const std::uint8_t*>(blk);
    const auto* qh = reinterpret_cast<const std::uint8_t*>(blk + 128);
    const __m128i il = lane_pairs_mask();
    const __m128i u1 = _mm_set1_epi8(1);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i lo4 = _mm_set1_epi8(0x0f);
    __m128i acc_lo = _mm_setzero_si128(), acc_hi = _mm_setzero_si128();
    // Group j's 16 raw codes, as q6_k_raw6_sse4 lays them out, built in
    // a register: in each 128-weight half n, ql[h..h+15] low nibbles
    // with qh bits 0-1 are group 8n + h / 16, ql[32 + h..] low nibbles
    // with bits 2-3 group + 2, the high nibbles of the same with bits
    // 4-5 and 6-7 groups + 4 and + 6.
    const auto group = [&](__m128i rv, int j) {
        rv = _mm_shuffle_epi8(rv, il);
        const __m128i qv = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(q8il + j * 16));
        // per lane: |raw6*q8 pair| <= 2*63*127, 32*|q8 pair| <= 32*254
        const __m128i s = _mm_sub_epi16(
            _mm_maddubs_epi16(rv, qv),
            _mm_slli_epi16(_mm_maddubs_epi16(u1, qv), 5));
        add_scaled_lanes(s, sc[j], acc_lo, acc_hi);
    };
    for (int n = 0; n < 2; ++n) {
        for (int h = 0; h < 32; h += 16) {
            const __m128i qlA = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(ql + n * 64 + h));
            const __m128i qlB = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(ql + n * 64 + 32 + h));
            const __m128i qhv = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(qh + n * 32 + h));
            const int j = n * 8 + h / 16;
            group(_mm_or_si128(_mm_and_si128(qlA, lo4),
                               _mm_slli_epi16(_mm_and_si128(qhv, m3), 4)),
                  j);
            group(_mm_or_si128(
                      _mm_and_si128(qlB, lo4),
                      _mm_slli_epi16(
                          _mm_and_si128(_mm_srli_epi16(qhv, 2), m3), 4)),
                  j + 2);
            group(_mm_or_si128(
                      _mm_and_si128(_mm_srli_epi16(qlA, 4), lo4),
                      _mm_slli_epi16(
                          _mm_and_si128(_mm_srli_epi16(qhv, 4), m3), 4)),
                  j + 4);
            group(_mm_or_si128(
                      _mm_and_si128(_mm_srli_epi16(qlB, 4), lo4),
                      _mm_slli_epi16(
                          _mm_and_si128(_mm_srli_epi16(qhv, 6), m3), 4)),
                  j + 6);
        }
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32), acc_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32 + 4), acc_hi);
}

/**
 * The four lane sums of 16 weights a against 16 activations b: lane l
 * is a[4l..4l+3] . b[4l..4l+3]. maddubs takes |a| (unsigned) times b
 * with a's sign, two products per int16: at most 2 * 128 * 127 =
 * 32512, so no saturation, and b (Q8_0 activations) is never -128, so
 * negating it cannot overflow. madd with ones adds the pairs.
 */
__m128i lanes16_sse4(__m128i abs_a, __m128i a, __m128i b) {
    const __m128i p = _mm_maddubs_epi16(abs_a, _mm_sign_epi8(b, a));
    return _mm_madd_epi16(p, _mm_set1_epi16(1));
}

/**
 * The lane sums of one vec_dot_block32_q8_0 block: lane l is quants
 * 4l..4l+3 of both halves, q (weights) times x (activations). Integer
 * sums are exact, so the SIMD add order matches the scalar loop.
 */
__m128i block32_lanes_sse4(const std::int8_t* q, const std::int8_t* x) {
    const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
    const __m128i q1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 16));
    const __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x));
    const __m128i x1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + 16));
    return _mm_add_epi32(lanes16_sse4(_mm_abs_epi8(q0), q0, x0),
                         lanes16_sse4(_mm_abs_epi8(q1), q1, x1));
}

/**
 * One Q8_0 weight row dotted with a Q8_0-quantized activation, in the
 * float order of scalar vec_dot_block32_q8_0 (two blocks per step,
 * four float lanes each, the pairwise lane reduction, then a scalar
 * tail block), so the result is bit-identical to it.
 */
float row_q8_0_q8_0_sse4(const std::byte* row, const float* xd,
                         const std::int8_t* xq, std::size_t nb) {
    constexpr std::size_t kBlk = 34;  // f16 scale + 32 int8
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    std::size_t b = 0;
    for (; b + 1 < nb; b += 2) {
        const std::byte* b0 = row + b * kBlk;
        const std::byte* b1 = b0 + kBlk;
        const float d0 = f16_to_f32(load_u16(b0)) * xd[b];
        const float d1 = f16_to_f32(load_u16(b1)) * xd[b + 1];
        const __m128i p0 = block32_lanes_sse4(
            reinterpret_cast<const std::int8_t*>(b0 + 2), xq + b * 32);
        const __m128i p1 = block32_lanes_sse4(
            reinterpret_cast<const std::int8_t*>(b1 + 2), xq + (b + 1) * 32);
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(p0), _mm_set1_ps(d0)));
        s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_cvtepi32_ps(p1), _mm_set1_ps(d1)));
    }
    alignas(16) float l0[4], l1[4];
    _mm_store_ps(l0, s0);
    _mm_store_ps(l1, s1);
    float sum = ((l0[0] + l0[1]) + (l0[2] + l0[3])) +
                ((l1[0] + l1[1]) + (l1[2] + l1[3]));
    for (; b < nb; ++b) {
        const std::byte* blk = row + b * kBlk;
        const float d = f16_to_f32(load_u16(blk)) * xd[b];
        alignas(16) std::int32_t p[4];
        _mm_store_si128(
            reinterpret_cast<__m128i*>(p),
            block32_lanes_sse4(reinterpret_cast<const std::int8_t*>(blk + 2),
                               xq + b * 32));
        sum += static_cast<float>(p[0] + p[1] + p[2] + p[3]) * d;
    }
    return sum;
}

/**
 * row_q8_0_q8_0_sse4 for kT tokens: each weight block and its absolute
 * values are loaded once and dotted with every token's quants; each token's
 * float lanes and reduction keep the one-token order, so out[t] is the
 * bits row_q8_0_q8_0_sse4 gives for token t.
 */
template <int kT>
void row_q8_0_q8_0_sse4_xt(const std::byte* row, const float* const* xd,
                           const std::int8_t* const* xq, std::size_t nb,
                           float* out) {
    constexpr std::size_t kBlk = 34;
    // A block's 32 weights and their absolute values, taken once.
    struct W16 {
        __m128i q[2], abs_q[2];
    };
    const auto widen = [](const std::byte* blk) {
        const auto* q = reinterpret_cast<const __m128i*>(blk + 2);
        const __m128i a = _mm_loadu_si128(q);
        const __m128i b = _mm_loadu_si128(q + 1);
        return W16{{a, b}, {_mm_abs_epi8(a), _mm_abs_epi8(b)}};
    };
    // The lane sums block32_lanes_sse4 gives, from the loaded weights.
    const auto lanes = [](const W16& w, const std::int8_t* x) {
        const __m128i xa =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(x));
        const __m128i xb =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + 16));
        return _mm_add_epi32(lanes16_sse4(w.abs_q[0], w.q[0], xa),
                             lanes16_sse4(w.abs_q[1], w.q[1], xb));
    };
    __m128 s0[kT], s1[kT];
    std::size_t b = 0;
#if defined(LOCUS_SSE4_AVX_VARIANT)
    // One 256-bit sum per token: block b in lanes 0-3 and b + 1 in lanes
    // 4-7, the same adds as s0 and s1 below, so the same bits.
    __m256 s8[kT];
    for (int t = 0; t < kT; ++t) {
        s8[t] = _mm256_setzero_ps();
    }
    for (; b + 1 < nb; b += 2) {
        const std::byte* b0 = row + b * kBlk;
        const std::byte* b1 = b0 + kBlk;
        const float w0 = f16_to_f32(load_u16(b0));
        const float w1 = f16_to_f32(load_u16(b1));
        const W16 v0 = widen(b0), v1 = widen(b1);
        for (int t = 0; t < kT; ++t) {
            const __m256 d = _mm256_set_m128(_mm_set1_ps(w1 * xd[t][b + 1]),
                                             _mm_set1_ps(w0 * xd[t][b]));
            const __m256i p = _mm256_set_m128i(
                lanes(v1, xq[t] + (b + 1) * 32), lanes(v0, xq[t] + b * 32));
            s8[t] = _mm256_add_ps(s8[t],
                                  _mm256_mul_ps(_mm256_cvtepi32_ps(p), d));
        }
    }
    for (int t = 0; t < kT; ++t) {
        s0[t] = _mm256_castps256_ps128(s8[t]);
        s1[t] = _mm256_extractf128_ps(s8[t], 1);
    }
#else
    for (int t = 0; t < kT; ++t) {
        s0[t] = _mm_setzero_ps();
        s1[t] = _mm_setzero_ps();
    }
    for (; b + 1 < nb; b += 2) {
        const std::byte* b0 = row + b * kBlk;
        const std::byte* b1 = b0 + kBlk;
        const float w0 = f16_to_f32(load_u16(b0));
        const float w1 = f16_to_f32(load_u16(b1));
        const W16 v0 = widen(b0), v1 = widen(b1);
        for (int t = 0; t < kT; ++t) {
            const float d0 = w0 * xd[t][b];
            const float d1 = w1 * xd[t][b + 1];
            const __m128i p0 = lanes(v0, xq[t] + b * 32);
            const __m128i p1 = lanes(v1, xq[t] + (b + 1) * 32);
            s0[t] = _mm_add_ps(
                s0[t], _mm_mul_ps(_mm_cvtepi32_ps(p0), _mm_set1_ps(d0)));
            s1[t] = _mm_add_ps(
                s1[t], _mm_mul_ps(_mm_cvtepi32_ps(p1), _mm_set1_ps(d1)));
        }
    }
#endif
    for (int t = 0; t < kT; ++t) {
        alignas(16) float l0[4], l1[4];
        _mm_store_ps(l0, s0[t]);
        _mm_store_ps(l1, s1[t]);
        float sum = ((l0[0] + l0[1]) + (l0[2] + l0[3])) +
                    ((l1[0] + l1[1]) + (l1[2] + l1[3]));
        for (std::size_t bt = b; bt < nb; ++bt) {
            const std::byte* blk = row + bt * kBlk;
            const float d = f16_to_f32(load_u16(blk)) * xd[t][bt];
            alignas(16) std::int32_t p[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(p),
                            lanes(widen(blk), xq[t] + bt * 32));
            sum += static_cast<float>(p[0] + p[1] + p[2] + p[3]) * d;
        }
        out[t] = sum;
    }
}

/** A row's Q8_K-quantized activation: one token's d, qs and bsums. */
struct Q8kAct {
    const float* d;
    const std::int8_t* qs;
    const std::int16_t* bsums;
};

/** One Q2_K row against a Q8_K activation, as matvec_sse4_q8k's rows. */
float row_q2_k_q8k_sse4(const std::byte* row, const Q8kAct& a,
                        std::size_t nb) {
    float sumf = 0.0f;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* blk = row + b * 84;
        const float bd = f16_to_f32(load_u16(blk + 80));
        const float bdmin = f16_to_f32(load_u16(blk + 82));
        int isum = 0, summs = 0;
        q2_k_q8k_block_sse4(blk, a.qs + b * 256, a.bsums + b * 16, isum,
                            summs);
        sumf += a.d[b] * bd * static_cast<float>(isum) -
                a.d[b] * bdmin * static_cast<float>(summs);
    }
    return sumf;
}

/**
 * One Q3_K..Q6_K row against a Q8_K activation (qs in lane_pairs_mask
 * order except for Q3_K), in scalar matvec_q8k's f32 order: 8 lane
 * sums, the min term subtracted from sumf per block, the lanes added
 * last.
 */
float row_k_q8k_sse4(gguf::TensorType type, const std::byte* row,
                     const Q8kAct& a, std::size_t nb) {
    const bool q4k = type == gguf::TensorType::kQ4_K;
    const bool q5k = type == gguf::TensorType::kQ5_K;
    const bool q3k = type == gguf::TensorType::kQ3_K;
    const std::size_t blk_bytes =
        q4k ? 144 : (q5k ? 176 : (q3k ? 110 : 210));
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float sumf = 0.0f;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* blk = row + b * blk_bytes;
        const std::int8_t* q8 = a.qs + b * 256;
        std::int32_t aux32[8];
        float bd = 0.0f;
        if (q4k || q5k) {  // bsums min term
            bd = f16_to_f32(load_u16(blk));
            const float bdmin = f16_to_f32(load_u16(blk + 2));
            int sumi = 0;
            if (q4k) {
                q4_k_q8k_aux32_sse4(blk, q8, a.bsums + b * 16, aux32, sumi);
            } else {
                q5_k_q8k_aux32_sse4(blk, q8, a.bsums + b * 16, aux32, sumi);
            }
            sumf -= bdmin * a.d[b] * static_cast<float>(sumi);
        } else if (q3k) {  // no min
            bd = f16_to_f32(load_u16(blk + 108));
            q3_k_q8k_aux32_sse4(blk, q8, aux32);
        } else {  // Q6_K, no min
            bd = f16_to_f32(load_u16(blk + 208));
            q6_k_q8k_aux32_sse4(blk, q8, aux32);
        }
        const float d = bd * a.d[b];
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * static_cast<float>(aux32[l]);
        }
    }
    for (int l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    return sumf;
}

/**
 * q4_k_q8k_aux32_sse4 for kT tokens at once: the block's nibbles are
 * decoded once and dotted with each token's q8, so each token's aux32
 * and sumi are the same integers the one-token kernel gives.
 */
template <int kT>
void q4_k_q8k_aux32_sse4_xt(const std::byte* blk, const Q8kAct* a,
                            std::size_t b, std::int32_t (*aux32)[8],
                            int* sumi) {
    constexpr std::uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f,
                            kmask3 = 0x03030303;
    const auto* q4 = reinterpret_cast<const std::uint8_t*>(blk + 16);
    std::uint32_t utmp[4];
    std::memcpy(utmp, blk + 4, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const std::uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const auto* scales = reinterpret_cast<const std::uint8_t*>(utmp);
    for (int t = 0; t < kT; ++t) {
        sumi[t] = k_min_term(utmp, a[t].bsums + b * 16);
    }
    const __m128i lo4 = _mm_set1_epi8(0x0f);
    const __m128i il = lane_pairs_mask();
    __m128i acc_lo[kT], acc_hi[kT];
    for (int t = 0; t < kT; ++t) {
        acc_lo[t] = _mm_setzero_si128();
        acc_hi[t] = _mm_setzero_si128();
    }
    for (int k = 0; k < 4; ++k) {
        const std::uint8_t* q4b = q4 + k * 32;
        const __m128i r0 = _mm_shuffle_epi8(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4b)), il);
        const __m128i r1 = _mm_shuffle_epi8(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4b + 16)),
            il);
        for (int h = 0; h < 2; ++h) {
            const int j = 2 * k + h;
            const __m128i n0 =
                h == 0 ? _mm_and_si128(r0, lo4)
                       : _mm_and_si128(_mm_srli_epi16(r0, 4), lo4);
            const __m128i n1 =
                h == 0 ? _mm_and_si128(r1, lo4)
                       : _mm_and_si128(_mm_srli_epi16(r1, 4), lo4);
            for (int t = 0; t < kT; ++t) {
                const std::int8_t* q8g = a[t].qs + b * 256 + j * 32;
                const __m128i a0 =
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8g));
                const __m128i a1 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(q8g + 16));
                add_scaled_pairs(_mm_maddubs_epi16(n0, a0),
                                 _mm_maddubs_epi16(n1, a1), scales[j],
                                 acc_lo[t], acc_hi[t]);
            }
        }
    }
    for (int t = 0; t < kT; ++t) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32[t]), acc_lo[t]);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32[t] + 4),
                         acc_hi[t]);
    }
}

/**
 * row_k_q8k_sse4 for a Q4_K row and kT tokens: each token's f32 sums in
 * the one-token order (the min term per block, 8 lanes, lanes last),
 * the block decoded once for all of them.
 */
template <int kT>
void row_q4_k_q8k_sse4_xt(const std::byte* row, const Q8kAct* a,
                          std::size_t nb, float* out) {
    float sums[kT][8] = {};
    float sumf[kT] = {};
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* blk = row + b * 144;
        const float bd = f16_to_f32(load_u16(blk));
        const float bdmin = f16_to_f32(load_u16(blk + 2));
        std::int32_t aux32[kT][8];
        int sumi[kT];
        q4_k_q8k_aux32_sse4_xt<kT>(blk, a, b, aux32, sumi);
        for (int t = 0; t < kT; ++t) {
            sumf[t] -= bdmin * a[t].d[b] * static_cast<float>(sumi[t]);
            const float d = bd * a[t].d[b];
            for (int l = 0; l < 8; ++l) {
                sums[t][l] += d * static_cast<float>(aux32[t][l]);
            }
        }
    }
    for (int t = 0; t < kT; ++t) {
        for (int l = 0; l < 8; ++l) {
            sumf[t] += sums[t][l];
        }
        out[t] = sumf[t];
    }
}

/**
 * row_k_q8k_sse4 for a Q6_K row and kT tokens: the block's 6-bit codes
 * decoded and shuffled once, each token's integer lane sums and f32
 * sums in the one-token kernel's order.
 */
template <int kT>
void row_q6_k_q8k_sse4_xt(const std::byte* row, const Q8kAct* a,
                          std::size_t nb, float* out) {
    const __m128i il = lane_pairs_mask();
    const __m128i u1 = _mm_set1_epi8(1);
    float sums[kT][8] = {};
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* blk = row + b * 210;
        const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
        alignas(16) std::uint8_t aux[256];
        q6_k_raw6_sse4(blk, aux);
        __m128i acc_lo[kT], acc_hi[kT];
        for (int t = 0; t < kT; ++t) {
            acc_lo[t] = _mm_setzero_si128();
            acc_hi[t] = _mm_setzero_si128();
        }
        for (int j = 0; j < 16; ++j) {
            const __m128i rv = _mm_shuffle_epi8(
                _mm_load_si128(reinterpret_cast<const __m128i*>(aux + j * 16)),
                il);
            for (int t = 0; t < kT; ++t) {
                const __m128i qv = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(a[t].qs + b * 256 +
                                                     j * 16));
                const __m128i s = _mm_sub_epi16(
                    _mm_maddubs_epi16(rv, qv),
                    _mm_slli_epi16(_mm_maddubs_epi16(u1, qv), 5));
                add_scaled_lanes(s, sc[j], acc_lo[t], acc_hi[t]);
            }
        }
        const float bd = f16_to_f32(load_u16(blk + 208));
        for (int t = 0; t < kT; ++t) {
            alignas(16) std::int32_t aux32[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(aux32), acc_lo[t]);
            _mm_store_si128(reinterpret_cast<__m128i*>(aux32 + 4), acc_hi[t]);
            const float d = bd * a[t].d[b];
            for (int l = 0; l < 8; ++l) {
                sums[t][l] += d * static_cast<float>(aux32[l]);
            }
        }
    }
    for (int t = 0; t < kT; ++t) {
        float sumf = 0.0f;
        for (int l = 0; l < 8; ++l) {
            sumf += sums[t][l];
        }
        out[t] = sumf;
    }
}

/**
 * Byte mask of two 8-weight sign groups: byte j (0..7) is 0xff when bit
 * j of s0 is set, byte 8 + j when bit j of s1 is.
 */
__m128i iq2_neg_mask(std::uint32_t s0, std::uint32_t s1) {
    const __m128i bits =
        _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64,
                      -128);
    const __m128i v = _mm_set_epi64x(
        static_cast<long long>(std::uint64_t{s1} * 0x0101010101010101ull),
        static_cast<long long>(std::uint64_t{s0} * 0x0101010101010101ull));
    return _mm_cmpeq_epi8(_mm_and_si128(v, bits), bits);
}

/**
 * Adds ls * (sum of the 16 signed products grid * q8) to the int32
 * lanes of acc, exactly: g holds 16 unsigned grid bytes (<= 43), neg
 * the bytes whose sign bit is set, q 16 activations. The signed sum is
 * maddubs(g, q) - 2 maddubs(g & neg, q): the activation is never
 * negated (Q8_K can hold -128), and each int16 lane's true value is at
 * most 2 * 43 * 128 in size, so the wrapping subtract lands on it.
 */
void iq2_group16(__m128i g, __m128i neg, __m128i q, int ls, __m128i& acc) {
    const __m128i all = _mm_maddubs_epi16(g, q);
    const __m128i ng = _mm_maddubs_epi16(_mm_and_si128(g, neg), q);
    const __m128i s = _mm_sub_epi16(all, _mm_add_epi16(ng, ng));
    acc = _mm_add_epi32(
        acc, _mm_madd_epi16(s, _mm_set1_epi16(static_cast<short>(ls))));
}

/**
 * One IQ2_XXS (66 B) or IQ2_S (82 B) row against a Q8_K activation (plain
 * order), in scalar vec_dot_iq2_xxs_q8_k / vec_dot_iq2_s_q8_k's float
 * order: each block's integer sum (exact, so its add order is free),
 * times f16(d) * d_x, added to sumf, then 0.125 * sumf.
 */
float row_iq2_q8k_sse4(bool xxs, const std::byte* row, const Q8kAct& a,
                       std::size_t nb) {
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * (xxs ? 66 : 82);
        const float d = f16_to_f32(load_u16(blk)) * a.d[i];
        const std::int8_t* q8 = a.qs + i * 256;
        const auto* u = reinterpret_cast<const std::uint8_t*>(blk);
        __m128i acc = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const __m128i q0 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(q8 + ib32 * 32));
            const __m128i q1 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(q8 + ib32 * 32 + 16));
            std::uint64_t g[4];
            std::uint32_t s[4];
            int ls1, ls2;
            if (xxs) {
                const std::uint8_t* p = u + 2 + 8 * ib32;
                std::uint32_t aux1;
                std::memcpy(&aux1, p + 4, 4);
                for (int l = 0; l < 4; ++l) {
                    g[l] = iq2xxs_grid[p[l]];
                    s[l] = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                }
                ls1 = ls2 = static_cast<int>(2 * (aux1 >> 28) + 1);
            } else {
                const std::uint8_t* qs = u + 2 + 4 * ib32;
                const std::uint8_t* signs = u + 34 + 4 * ib32;
                const std::uint32_t qh = u[66 + ib32];
                for (int l = 0; l < 4; ++l) {
                    g[l] = iq2s_grid[qs[l] | ((qh << (8 - 2 * l)) & 0x300)];
                    s[l] = signs[l];
                }
                ls1 = 1 + 2 * (u[74 + ib32] & 0xf);
                ls2 = 1 + 2 * (u[74 + ib32] >> 4);
            }
            iq2_group16(_mm_set_epi64x(static_cast<long long>(g[1]),
                                       static_cast<long long>(g[0])),
                        iq2_neg_mask(s[0], s[1]), q0, ls1, acc);
            iq2_group16(_mm_set_epi64x(static_cast<long long>(g[3]),
                                       static_cast<long long>(g[2])),
                        iq2_neg_mask(s[2], s[3]), q1, ls2, acc);
        }
        sumf += d * static_cast<float>(hsum_epi32_sse4(acc));
    }
    return 0.125f * sumf;
}

/**
 * One IQ3_XXS (98 B) or IQ3_S (110 B) row against a Q8_K activation
 * (plain order), in scalar vec_dot_iq3_xxs_q8_k / vec_dot_iq3_s_q8_k's
 * float order: each block's integer sum (exact), times f16(d) * d_x,
 * added to sumf; then 0.25 * sumf (XXS) or sumf (S). Each 8 weights are
 * two 4-byte grid entries and one sign byte; grid bytes are at most 62,
 * so iq2_group16's int16 lanes stay within 2 * 62 * 127 (doubled: 31496).
 */
float row_iq3_q8k_sse4(bool xxs, const std::byte* row, const Q8kAct& a,
                       std::size_t nb) {
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        const std::byte* blk = row + i * (xxs ? 98 : 110);
        const float d = f16_to_f32(load_u16(blk)) * a.d[i];
        const std::int8_t* q8 = a.qs + i * 256;
        const auto* u = reinterpret_cast<const std::uint8_t*>(blk);
        __m128i acc = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            std::uint32_t g[8];
            std::uint32_t s[4];
            int ls;
            if (xxs) {
                const std::uint8_t* q3 = u + 2 + 8 * ib32;
                std::uint32_t aux32;
                std::memcpy(&aux32, u + 66 + 4 * ib32, 4);
                for (int k = 0; k < 8; ++k) {
                    g[k] = iq3xxs_grid[q3[k]];
                }
                for (int l = 0; l < 4; ++l) {
                    s[l] = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                }
                ls = static_cast<int>(2 * (aux32 >> 28) + 1);
            } else {
                const std::uint8_t* qs = u + 2 + 8 * ib32;
                const std::uint32_t qh = u[66 + ib32];
                for (int l = 0; l < 4; ++l) {
                    g[2 * l] =
                        iq3s_grid[qs[2 * l] | ((qh << (8 - 2 * l)) & 256)];
                    g[2 * l + 1] = iq3s_grid[qs[2 * l + 1] |
                                             ((qh << (7 - 2 * l)) & 256)];
                    s[l] = u[74 + 4 * ib32 + l];
                }
                const std::uint32_t sc = u[106 + ib32 / 2];
                ls = static_cast<int>(
                    2 * ((ib32 & 1) == 0 ? (sc & 0xf) : (sc >> 4)) + 1);
            }
            for (int h = 0; h < 2; ++h) {
                const __m128i q = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(q8 + ib32 * 32 + h * 16));
                const __m128i gv = _mm_setr_epi32(
                    static_cast<int>(g[4 * h]), static_cast<int>(g[4 * h + 1]),
                    static_cast<int>(g[4 * h + 2]),
                    static_cast<int>(g[4 * h + 3]));
                iq2_group16(gv, iq2_neg_mask(s[2 * h], s[2 * h + 1]), q, ls,
                            acc);
            }
        }
        sumf += d * static_cast<float>(hsum_epi32_sse4(acc));
    }
    return xxs ? 0.25f * sumf : sumf;
}

/**
 * One IQ4_XS (136 B) row against a Q8_K activation (plain order), in
 * scalar vec_dot_iq4_xs_q8_k's float order: each 32-weight group's
 * integer sum (exact), times d4d8 * (ls - 32), added to sumf in group
 * order. The 4-bit codes map through kvalues_iq4nl with one pshufb;
 * the products go through maddubs(|v|, sign(q, v)), which is exact
 * because Q8_K activations stay in [-127, 127] (never -128).
 */
float row_iq4_xs_q8k_sse4(const std::byte* row, const Q8kAct& a,
                          std::size_t nb) {
    const __m128i table = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i lo4 = _mm_set1_epi8(0x0f);
    const __m128i ones = _mm_set1_epi16(1);
    // The signed sum of 16 codes (as values) times 16 activations.
    const auto dot16 = [&](__m128i codes, __m128i q) {
        const __m128i v = _mm_shuffle_epi8(table, codes);
        return _mm_madd_epi16(
            _mm_maddubs_epi16(_mm_abs_epi8(v), _mm_sign_epi8(q, v)), ones);
    };
    float sumf = 0.0f;
    for (std::size_t ibl = 0; ibl < nb; ++ibl) {
        const std::byte* blk = row + ibl * 136;
        const float d4d8 = f16_to_f32(load_u16(blk)) * a.d[ibl];
        std::uint16_t h;
        std::memcpy(&h, blk + 2, 2);
        const auto* scales_l = reinterpret_cast<const std::uint8_t*>(blk + 4);
        const auto* qs = reinterpret_cast<const std::uint8_t*>(blk + 8);
        const std::int8_t* q8 = a.qs + ibl * 256;
        for (int ib = 0; ib < 8; ++ib) {
            // ls: the low nibble (even ib) or high nibble (odd ib) of
            // scales_l[ib / 2], with 2 bits of h on top.
            const int sl = (ib & 1) == 0 ? (scales_l[ib / 2] & 0xf)
                                         : (scales_l[ib / 2] >> 4);
            const int ls = sl | (((h >> (2 * ib)) & 3) << 4);
            const __m128i b = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(qs + ib * 16));
            const __m128i q0 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(q8 + ib * 32));
            const __m128i q1 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(q8 + ib * 32 + 16));
            const __m128i s = _mm_add_epi32(
                dot16(_mm_and_si128(b, lo4), q0),
                dot16(_mm_and_si128(_mm_srli_epi16(b, 4), lo4), q1));
            sumf += d4d8 * static_cast<float>(ls - 32) *
                    static_cast<float>(hsum_epi32_sse4(s));
        }
    }
    return sumf;
}

/** Which row kernel a weight takes under Q8_K / Q8_0 activations. */
enum class Q8Path { kBlock32, kQ2K, kK, kIQ2, kIQ3, kIQ4XS, kOther };

Q8Path q8_path(gguf::TensorType type, std::size_t cols) {
    using T = gguf::TensorType;
    if (type == T::kQ8_0 && cols % 32 == 0) {
        return Q8Path::kBlock32;
    }
    if (cols % 256 != 0) {
        return Q8Path::kOther;
    }
    if (type == T::kQ2_K) {
        return Q8Path::kQ2K;
    }
    if (type == T::kQ3_K || type == T::kQ4_K || type == T::kQ5_K ||
        type == T::kQ6_K) {
        return Q8Path::kK;
    }
    if (type == T::kIQ2_XXS || type == T::kIQ2_S) {
        return Q8Path::kIQ2;
    }
    if (type == T::kIQ4_XS) {
        return Q8Path::kIQ4XS;
    }
    if (type == T::kIQ3_XXS || type == T::kIQ3_S) {
        return Q8Path::kIQ3;
    }
    return Q8Path::kOther;
}

/** The per-token row kernel for a kQ2K, kK or IQ (kIQ2, kIQ3, kIQ4XS)
 *  weight. */
float row_q8k_sse4(Q8Path path, gguf::TensorType type, const std::byte* row,
                   const Q8kAct& a, std::size_t nb) {
    if (path == Q8Path::kQ2K) {
        return row_q2_k_q8k_sse4(row, a, nb);
    }
    if (path == Q8Path::kIQ2) {
        return row_iq2_q8k_sse4(type == gguf::TensorType::kIQ2_XXS, row, a,
                                nb);
    }
    if (path == Q8Path::kIQ4XS) {
        return row_iq4_xs_q8k_sse4(row, a, nb);
    }
    if (path == Q8Path::kIQ3) {
        return row_iq3_q8k_sse4(type == gguf::TensorType::kIQ3_XXS, row, a,
                                nb);
    }
    return row_k_q8k_sse4(type, row, a, nb);
}

/**
 * Quantizes x to Q8_K into d, qs and bsums, with qs in lane_pairs_mask
 * order unless the weight is Q3_K (its kernel reads the plain order).
 */
void quantize_q8k_sse4(std::span<const float> x, bool shuffle, float* d,
                       std::int8_t* qs, std::int16_t* bsums) {
    quantize_activation_q8k(x, d, qs, bsums);
    if (shuffle) {
        const __m128i il = lane_pairs_mask();
        for (std::size_t i = 0; i < x.size(); i += 16) {
            auto* p = reinterpret_cast<__m128i*>(qs + i);
            _mm_storeu_si128(p, _mm_shuffle_epi8(_mm_loadu_si128(p), il));
        }
    }
}

}  // namespace

void matvec_sse4_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out) {
    const std::size_t cols = x.size();
    const Q8Path path = q8_path(w.type, cols);
    if (path == Q8Path::kOther) {
        matvec_q8k(w, x, out);  // other types: scalar Q8_K / Q8_0
        return;
    }
    if (path == Q8Path::kBlock32) {
        const std::size_t nb = cols / 32;
        std::vector<float> xd(nb);
        std::vector<std::int8_t> xq(cols);
        quantize_activation_q8_0(x, xd.data(), xq.data());
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out[r] = row_q8_0_q8_0_sse4(w.data + r * nb * 34, xd.data(),
                                        xq.data(), nb);
        }
        return;
    }
    const std::size_t nb = cols / 256;
    std::vector<float> hd(nb);
    std::vector<std::int8_t> hqs(cols);
    std::vector<std::int16_t> hbs(nb * 16);
    const bool q3k = w.type == gguf::TensorType::kQ3_K;
    // Q2_K, Q3_K and the IQ2 types read the plain order; Q4_K..Q6_K the
    // lane_pairs_mask one.
    quantize_q8k_sse4(x, path == Q8Path::kK && !q3k, hd.data(),
                      hqs.data(), hbs.data());
    const Q8kAct a{hd.data(), hqs.data(), hbs.data()};
    const std::size_t row_bytes = mat_row_bytes(w);
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        out[r] = row_q8k_sse4(path, w.type, w.data + r * row_bytes, a, nb);
    }
}

void matvec_batch_sse4_q8k(const Mat& w, std::span<const float> x_batch,
                           std::span<float> out_batch, std::uint32_t n) {
    const std::size_t cols = w.cols;
    const Q8Path path = q8_path(w.type, cols);
    if (path == Q8Path::kOther) {
        // Other types: n per-token calls, scattered row-major.
        thread_local std::vector<float> tmp;
        tmp.resize(w.rows);
        for (std::uint32_t t = 0; t < n; ++t) {
            self::matvec_sse4_q8k(w, x_batch.subspan(t * cols, cols), tmp);
            for (std::uint32_t r = 0; r < w.rows; ++r) {
                out_batch[std::size_t{r} * n + t] = tmp[r];
            }
        }
        return;
    }
    // Each token quantized once; then each weight row is dotted with a
    // tile of tokens while it stays in cache, the tile sized so its
    // quantized activations stay in L2 (about 96 KB). Every (row,
    // token) runs the per-token kernel, so the bits match it.
    const std::uint32_t tile = static_cast<std::uint32_t>(
        std::max<std::size_t>(1, (std::size_t{96} << 10) / cols));
    const std::size_t row_bytes = mat_row_bytes(w);
    if (path == Q8Path::kBlock32) {
        const std::size_t nb = cols / 32;
        thread_local std::vector<float> xd;
        thread_local std::vector<std::int8_t> xq;
        xd.resize(nb * n);
        xq.resize(cols * n);
        for (std::uint32_t t = 0; t < n; ++t) {
            quantize_activation_q8_0(x_batch.subspan(t * cols, cols),
                                     xd.data() + t * nb, xq.data() + t * cols);
        }
        for (std::uint32_t t0 = 0; t0 < n; t0 += tile) {
            const std::uint32_t t1 = std::min(n, t0 + tile);
            for (std::uint32_t r = 0; r < w.rows; ++r) {
                const std::byte* row = w.data + r * row_bytes;
                float* o = out_batch.data() + std::size_t{r} * n;
                std::uint32_t t = t0;
                constexpr int kT = 4;  // four tokens per widened block
                for (; t + kT <= t1; t += kT) {
                    const float* d4[kT];
                    const std::int8_t* q4[kT];
                    for (int k = 0; k < kT; ++k) {
                        d4[k] = xd.data() + (t + k) * nb;
                        q4[k] = xq.data() + (t + k) * cols;
                    }
                    row_q8_0_q8_0_sse4_xt<kT>(row, d4, q4, nb, o + t);
                }
                for (; t < t1; ++t) {
                    o[t] = row_q8_0_q8_0_sse4(row, xd.data() + t * nb,
                                              xq.data() + t * cols, nb);
                }
            }
        }
        return;
    }
    const std::size_t nb = cols / 256;
    thread_local std::vector<float> hd;
    thread_local std::vector<std::int8_t> hqs;
    thread_local std::vector<std::int16_t> hbs;
    hd.resize(nb * n);
    hqs.resize(cols * n);
    hbs.resize(nb * 16 * n);
    const bool shuffle =
        path == Q8Path::kK && w.type != gguf::TensorType::kQ3_K;
    for (std::uint32_t t = 0; t < n; ++t) {
        quantize_q8k_sse4(x_batch.subspan(t * cols, cols), shuffle,
                          hd.data() + t * nb, hqs.data() + t * cols,
                          hbs.data() + t * nb * 16);
    }
    const auto act = [&](std::uint32_t t) {
        return Q8kAct{hd.data() + t * nb, hqs.data() + t * cols,
                      hbs.data() + t * nb * 16};
    };
    const bool q4k = w.type == gguf::TensorType::kQ4_K;
    const bool q6k = w.type == gguf::TensorType::kQ6_K;
    for (std::uint32_t t0 = 0; t0 < n; t0 += tile) {
        const std::uint32_t t1 = std::min(n, t0 + tile);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            const std::byte* row = w.data + r * row_bytes;
            float* o = out_batch.data() + std::size_t{r} * n;
            std::uint32_t t = t0;
            if (q4k || q6k) {  // four tokens per decoded block, then the tail
                constexpr int kT = 4;
                for (; t + kT <= t1; t += kT) {
                    const Q8kAct a4[kT] = {act(t), act(t + 1), act(t + 2),
                                           act(t + 3)};
                    if (q4k) {
                        row_q4_k_q8k_sse4_xt<kT>(row, a4, nb, o + t);
                    } else {
                        row_q6_k_q8k_sse4_xt<kT>(row, a4, nb, o + t);
                    }
                }
            }
            for (; t < t1; ++t) {
                o[t] = row_q8k_sse4(path, w.type, row, act(t), nb);
            }
        }
    }
}

void matvec_batch_sse4(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch, std::uint32_t n) {
    if (w.type == gguf::TensorType::kQ4_K) {
        batch_q4_k_sse(w, x_batch.data(), out_batch.data(), n);
        return;
    }
    if (w.type == gguf::TensorType::kQ6_K) {
        batch_q6_k_sse(w, x_batch.data(), out_batch.data(), n);
        return;
    }
    // Other types: n per-token matvec_sse4() calls scattered into the
    // row-major layout. No weight-traffic amortization (not the hot
    // path), but bit-exact to the per-token sse4 kernel -- which for
    // F32/Q8_0 is SIMD, so delegating to matvec_batch_scalar would
    // diverge and break batched==per-token token-exactness.
    std::vector<float> col(w.rows);
    const std::size_t xc = w.cols;
    for (std::uint32_t t = 0; t < n; ++t) {
        self::matvec_sse4(
            w,
            x_batch.subspan(static_cast<std::size_t>(t) * xc, xc),
            col);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out_batch[static_cast<std::size_t>(r) * n + t] = col[r];
        }
    }
}

#if defined(LOCUS_SSE4_AVX_VARIANT)
}  // namespace sse4_avx
#endif
}  // namespace locus::backend

#endif  // __x86_64__ && __SSE4_1__
