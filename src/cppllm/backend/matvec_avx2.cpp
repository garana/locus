#include "cppllm/backend/variants.hpp"

#if defined(__x86_64__) && defined(__AVX2__)

#include <immintrin.h>

#include <cstring>

namespace cppllm::backend {

namespace {

std::uint16_t load_u16(const std::byte* p) {
    std::uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

float dot_f32_avx2(const float* w, const float* x,
                   std::size_t n) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + i),
                              _mm256_loadu_ps(x + i), acc);
    }
    float s = hsum(acc);
    for (; i < n; ++i) {
        s += w[i] * x[i];
    }
    return s;
}

/** One Q8_0 block: f16 scale + 32 int8 quants (34 bytes). */
float dot_q8_0_block_avx2(const std::byte* blk, const float* x) {
    const float d = f16_to_f32(load_u16(blk));
    const auto* q = reinterpret_cast<const std::int8_t*>(blk + 2);
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < 32; g += 8) {
        const __m128i q8 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(q + g));
        const __m256 wf = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(q8));
        acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + g), acc);
    }
    return d * hsum(acc);
}

}  // namespace

void matvec_avx2(const Mat& w, std::span<const float> x,
                 std::span<float> out) {
    if (w.type == gguf::TensorType::kF32) {
        const auto* rows = reinterpret_cast<const float*>(w.data);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out[r] = dot_f32_avx2(
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
                acc += dot_q8_0_block_avx2(row + b * 34,
                                           x.data() + b * 32);
            }
            out[r] = acc;
        }
        return;
    }
    matvec(w, x, out);  // other types: scalar reference
}

}  // namespace cppllm::backend

#endif  // __x86_64__ && __AVX2__
