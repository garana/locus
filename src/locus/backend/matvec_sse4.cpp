#include "locus/backend/variants.hpp"

#if defined(__x86_64__) && defined(__SSE4_1__)

#include <smmintrin.h>  // SSE4.1

#include <cstring>

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

}  // namespace locus::backend

#endif  // __x86_64__ && __SSE4_1__
