#include "locus/backend/variants.hpp"

#if defined(__x86_64__) && defined(__SSE4_1__)

#include <smmintrin.h>  // SSE4.1

#include <cstring>
#include <vector>

namespace locus::backend {

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

/** Accumulates 16 unsigned nibble values (each byte 0..15) as
 * (d*v - m)*x into acc. */
__m128 accum16_q4k_sse(__m128 acc, __m128i vals, float d, float m,
                       const float* xp) {
    const __m128 dv = _mm_set1_ps(d), mv = _mm_set1_ps(m);
    for (int k = 0; k < 16; k += 4) {
        const __m128i v4 = _mm_srli_si128(vals, k);
        __m128 f = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(v4));
        f = _mm_sub_ps(_mm_mul_ps(f, dv), mv);
        acc = _mm_add_ps(acc,
                         _mm_mul_ps(f, _mm_loadu_ps(xp + k)));
    }
    return acc;
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

/** Accumulates 16 Q6_K combined values (each byte 0..63) as
 * (v - 32) * dsc * x into acc. */
__m128 accum16_q6k_sse(__m128 acc, __m128i vals, float dsc,
                       const float* xp) {
    const __m128i sv = _mm_sub_epi8(vals, _mm_set1_epi8(32));
    const __m128 s = _mm_set1_ps(dsc);
    for (int k = 0; k < 16; k += 4) {
        const __m128i v4 = _mm_srli_si128(sv, k);
        const __m128 f =
            _mm_cvtepi32_ps(_mm_cvtepi8_epi32(v4));
        acc = _mm_add_ps(
            acc, _mm_mul_ps(_mm_mul_ps(f, s),
                            _mm_loadu_ps(xp + k)));
    }
    return acc;
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

/** SSE4 Q4_K x Q8_K integer dot for one super-block: computes the
 *  scalar vec_dot_q4_k_q8_k's aux32[8] lanes with a widen-then-per-lane
 *  multiply (int16 q8*a products sign-extended to int32, x scale) --
 *  NO maddubs pair-collapse, so each lane equals the scalar's exactly.
 *  Fills aux32[8] and sumi (the bsums min term). */
void q4_k_q8k_aux32_sse4(const std::byte* blk, const std::int8_t* q8,
                         const std::int16_t* bsums,
                         std::int32_t aux32[8], int& sumi) {
    constexpr std::uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f,
                            kmask3 = 0x03030303;
    const auto* q4 = reinterpret_cast<const std::uint8_t*>(blk + 16);
    std::int8_t aux8[256];
    std::int8_t* a = aux8;
    for (int j = 0; j < 4; ++j) {
        for (int l = 0; l < 32; ++l) {
            a[l] = static_cast<std::int8_t>(q4[l] & 0xF);
        }
        a += 32;
        for (int l = 0; l < 32; ++l) {
            a[l] = static_cast<std::int8_t>(q4[l] >> 4);
        }
        a += 32;
        q4 += 32;
    }
    std::uint32_t utmp[4];
    std::memcpy(utmp, blk + 4, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const std::uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const auto* scales = reinterpret_cast<const std::uint8_t*>(utmp);
    const auto* mins = reinterpret_cast<const std::uint8_t*>(utmp + 2);
    sumi = 0;
    for (int j = 0; j < 16; ++j) {
        sumi += bsums[j] * mins[j / 2];
    }
    __m128i acc_lo = _mm_setzero_si128(), acc_hi = _mm_setzero_si128();
    const std::int8_t* ap = aux8;
    const std::int8_t* q8p = q8;
    int is = 0;
    for (int j = 0; j < 8; ++j) {
        const __m128i sc = _mm_set1_epi32(scales[is++]);
        for (int k = 0; k < 4; ++k) {
            const __m128i qv = _mm_cvtepi8_epi16(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q8p)));
            const __m128i av = _mm_cvtepi8_epi16(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ap)));
            const __m128i p = _mm_mullo_epi16(qv, av);  // 8x int16 q8*a
            const __m128i plo = _mm_cvtepi16_epi32(p);
            const __m128i phi =
                _mm_cvtepi16_epi32(_mm_srli_si128(p, 8));
            acc_lo = _mm_add_epi32(acc_lo, _mm_mullo_epi32(plo, sc));
            acc_hi = _mm_add_epi32(acc_hi, _mm_mullo_epi32(phi, sc));
            q8p += 8;
            ap += 8;
        }
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32), acc_lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aux32 + 4), acc_hi);
}

/** SSE4 Q6_K x Q8_K aux32[8]: same widen-per-lane core as Q4_K, but
 *  6-bit ql+qh unpack with the -32 offset and 16 int8 scales (2 k-iters
 *  per scale); no min term. Bit-identical to scalar vec_dot_q6_k_q8_k. */
void q6_k_q8k_aux32_sse4(const std::byte* blk, const std::int8_t* q8,
                         std::int32_t aux32[8]) {
    const auto* ql = reinterpret_cast<const std::uint8_t*>(blk);
    const auto* qh = reinterpret_cast<const std::uint8_t*>(blk + 128);
    const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
    std::int8_t aux8[256];
    std::int8_t* a = aux8;
    const std::uint8_t* q4 = ql;
    const std::uint8_t* qhp = qh;
    for (int j = 0; j < 256; j += 128) {
        for (int l = 0; l < 32; ++l) {
            a[l + 0] = static_cast<std::int8_t>(
                ((q4[l] & 0xF) | (((qhp[l] >> 0) & 3) << 4)) - 32);
            a[l + 32] = static_cast<std::int8_t>(
                ((q4[l + 32] & 0xF) | (((qhp[l] >> 2) & 3) << 4)) - 32);
            a[l + 64] = static_cast<std::int8_t>(
                ((q4[l] >> 4) | (((qhp[l] >> 4) & 3) << 4)) - 32);
            a[l + 96] = static_cast<std::int8_t>(
                ((q4[l + 32] >> 4) | (((qhp[l] >> 6) & 3) << 4)) - 32);
        }
        a += 128;
        q4 += 64;
        qhp += 32;
    }
    __m128i acc_lo = _mm_setzero_si128(), acc_hi = _mm_setzero_si128();
    const std::int8_t* ap = aux8;
    const std::int8_t* q8p = q8;
    int is = 0;
    for (int j = 0; j < 16; ++j) {
        const __m128i scv = _mm_set1_epi32(sc[is++]);
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

/** SSE4 Q5_K x Q8_K aux32[8]: Q4_K's widen-per-lane core with the hm
 *  5th-bit added to the unpack; utmp 6-bit scales/mins + bsums min term.
 *  Bit-identical to scalar vec_dot_q5_k_q8_k. */
void q5_k_q8k_aux32_sse4(const std::byte* blk, const std::int8_t* q8,
                         const std::int16_t* bsums,
                         std::int32_t aux32[8], int& sumi) {
    constexpr std::uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f,
                            kmask3 = 0x03030303;
    const auto* hm = reinterpret_cast<const std::uint8_t*>(blk + 16);
    const auto* q4 = reinterpret_cast<const std::uint8_t*>(blk + 48);
    std::int8_t aux8[256];
    std::int8_t* a = aux8;
    std::uint8_t m = 1;
    for (int j = 0; j < 4; ++j) {
        for (int l = 0; l < 32; ++l) {
            a[l] = static_cast<std::int8_t>(q4[l] & 0xF);
        }
        for (int l = 0; l < 32; ++l) {
            a[l] = static_cast<std::int8_t>(a[l] + ((hm[l] & m) ? 16 : 0));
        }
        a += 32;
        m <<= 1;
        for (int l = 0; l < 32; ++l) {
            a[l] = static_cast<std::int8_t>(q4[l] >> 4);
        }
        for (int l = 0; l < 32; ++l) {
            a[l] = static_cast<std::int8_t>(a[l] + ((hm[l] & m) ? 16 : 0));
        }
        a += 32;
        m <<= 1;
        q4 += 32;
    }
    std::uint32_t utmp[4];
    std::memcpy(utmp, blk + 4, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const std::uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const auto* scales = reinterpret_cast<const std::uint8_t*>(utmp);
    const auto* mins = reinterpret_cast<const std::uint8_t*>(utmp + 2);
    sumi = 0;
    for (int j = 0; j < 16; ++j) {
        sumi += bsums[j] * mins[j / 2];
    }
    __m128i acc_lo = _mm_setzero_si128(), acc_hi = _mm_setzero_si128();
    const std::int8_t* ap = aux8;
    const std::int8_t* q8p = q8;
    int is = 0;
    for (int j = 0; j < 8; ++j) {
        const __m128i scv = _mm_set1_epi32(scales[is++]);
        for (int k = 0; k < 4; ++k) {
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

void matvec_sse4_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out) {
    const bool q4k = w.type == gguf::TensorType::kQ4_K;
    const bool q5k = w.type == gguf::TensorType::kQ5_K;
    const bool q6k = w.type == gguf::TensorType::kQ6_K;
    if ((!q4k && !q5k && !q6k) || x.size() % 256 != 0) {
        matvec_q8k(w, x, out);  // other types: scalar Q8_K
        return;
    }
    const std::size_t nb = x.size() / 256;
    std::vector<float> hd(nb);
    std::vector<std::int8_t> hqs(x.size());
    std::vector<std::int16_t> hbs(nb * 16);
    quantize_activation_q8k(x, hd.data(), hqs.data(), hbs.data());
    const std::size_t blk_bytes = q4k ? 144 : (q5k ? 176 : 210);
    const std::size_t row_bytes = nb * blk_bytes;
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const std::byte* row = w.data + r * row_bytes;
        float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float sumf = 0.0f;
        for (std::size_t b = 0; b < nb; ++b) {
            const std::byte* blk = row + b * blk_bytes;
            std::int32_t aux32[8];
            if (q4k || q5k) {  // d,dmin f16 at blk+0/+2; bsums min term
                const float bd = f16_to_f32(load_u16(blk));
                const float bdmin = f16_to_f32(load_u16(blk + 2));
                int sumi = 0;
                if (q4k) {
                    q4_k_q8k_aux32_sse4(blk, hqs.data() + b * 256,
                                        hbs.data() + b * 16, aux32,
                                        sumi);
                } else {
                    q5_k_q8k_aux32_sse4(blk, hqs.data() + b * 256,
                                        hbs.data() + b * 16, aux32,
                                        sumi);
                }
                const float d = bd * hd[b];
                for (int l = 0; l < 8; ++l) {
                    sums[l] += d * static_cast<float>(aux32[l]);
                }
                sumf -= bdmin * hd[b] * static_cast<float>(sumi);
            } else {  // Q6_K: no min term, f16 scale at blk+208
                const float bd = f16_to_f32(load_u16(blk + 208));
                q6_k_q8k_aux32_sse4(blk, hqs.data() + b * 256, aux32);
                const float d = bd * hd[b];
                for (int l = 0; l < 8; ++l) {
                    sums[l] += d * static_cast<float>(aux32[l]);
                }
            }
        }
        for (int l = 0; l < 8; ++l) {
            sumf += sums[l];
        }
        out[r] = sumf;
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
        matvec_sse4(
            w,
            x_batch.subspan(static_cast<std::size_t>(t) * xc, xc),
            col);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out_batch[static_cast<std::size_t>(r) * n + t] = col[r];
        }
    }
}

}  // namespace locus::backend

#endif  // __x86_64__ && __SSE4_1__
