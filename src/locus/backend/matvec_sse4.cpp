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
    matvec(w, x, out);  // other types: scalar reference
}

}  // namespace locus::backend

#endif  // __x86_64__ && __SSE4_1__
