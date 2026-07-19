#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "cppllm/gguf/gguf.hpp"

namespace cppllm::backend {

/**
 * Scalar reference CPU ops for the model runtime. These are the
 * correctness baseline every SIMD/Vulkan variant is tested against
 * (DESIGN.md 4.3); optimized variants land as per-source files
 * selected via cppllm::sys::detect().
 */

/** @returns The float value of an IEEE binary16 bit pattern. */
float f16_to_f32(std::uint16_t h);

/** @returns The binary16 bit pattern nearest to f. */
std::uint16_t f32_to_f16(float f);

/**
 * RMS-normalizes x and scales by weight, into out.
 *
 * @param x Input activations.
 * @param w Per-channel scale; same length as x.
 * @param eps Stabilizer added to the mean square.
 * @param out Output; same length as x, may alias x.
 */
void rmsnorm(std::span<const float> x, std::span<const float> w,
             float eps, std::span<float> out);

/** In-place numerically-stable softmax over x. */
void softmax_inplace(std::span<float> x);

/**
 * out = silu(gate) * up, elementwise (SwiGLU FFN activation).
 * All three spans have the same length; out may alias gate.
 */
void silu_mul(std::span<const float> gate, std::span<const float> up,
              std::span<float> out);

/**
 * Applies interleaved-pair ("norm") RoPE in place, as used by
 * Llama-family models: pairs (x[2i], x[2i+1]) within each head are
 * rotated by pos * freq_base^(-2i/head_dim) / factor[i].
 *
 * @param x Q or K activations, n_heads * head_dim floats.
 * @param factors Per-pair frequency divisors (head_dim/2 entries,
 *     the GGUF rope_freqs.weight tensor for llama3-scaled
 *     models); empty means all 1.
 */
void rope_norm(std::span<float> x, std::uint32_t n_heads,
               std::uint32_t head_dim, std::uint32_t pos,
               float freq_base,
               std::span<const float> factors = {});

/**
 * A 2-D weight view into mapped model memory. Rows are output
 * channels; `cols` elements per row, stored row-major in `type`'s
 * on-disk encoding (ggml layout: ne[0] = cols, ne[1] = rows).
 */
struct Mat {
    gguf::TensorType type = gguf::TensorType::kF32;
    const std::byte* data = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
};

/**
 * Dense matrix-vector product: out[r] = dot(row r of w, x).
 * Quantized weight rows are dequantized on the fly.
 *
 * @throws gguf::Error on a weight type without a matvec kernel.
 */
void matvec(const Mat& w, std::span<const float> x,
            std::span<float> out);

/**
 * Dequantizes one row of w into out (used for embedding lookup).
 */
void dequant_row(const Mat& w, std::uint32_t row,
                 std::span<float> out);

}  // namespace cppllm::backend
