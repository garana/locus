#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <span>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"

/** Token-major dense ops the image models share (PixArt DiT, VAE). */
namespace locus::image::dense {

/**
 * out[t][r] = w[r] . x[t] + b[r] for n tokens, token-major in and out
 * (b may be empty for no bias). Rows are split across threads when
 * backend::splits_rows(op); each slice runs the backend's batched
 * matvec over blocks of 64 tokens.
 */
void linear(const backend::Ops& op, const backend::Mat& w,
            std::span<const float> b, std::span<const float> x,
            std::uint32_t n, std::span<float> out);

/**
 * Multi-head attention without a mask: nq queries (q, token-major,
 * heads * hd wide) over nk keys and values (k, v, same layout),
 * softmax(q k^T / sqrt(hd)) v, into out (nq rows, token-major).
 *
 * Each head's keys are copied to a [nk][hd] matrix and its values to
 * [hd][nk], so the scores and the weighted sum run through the
 * backend's batched matvec, 64 queries at a time.
 */
void attention(const backend::Ops& op, std::span<const float> q,
               std::uint32_t nq, std::span<const float> k,
               std::span<const float> v, std::uint32_t nk,
               std::uint32_t heads, std::uint32_t hd, std::span<float> out);

/** Runs fn(i) for i in [0, n) on the thread pool. */
void parallel(std::size_t n, const std::function<void(std::size_t)>& fn);

/** One vector: out = w x + b. */
void linear1(const backend::Ops& op, const backend::Mat& w,
             std::span<const float> b, std::span<const float> x,
             std::span<float> out);

/**
 * LayerNorm without affine over each of n rows of width d, then
 * out = ln(x) * (1 + scale) + shift (scale and shift: d floats).
 */
void ln_modulate(std::span<const float> x, std::uint32_t n, std::uint32_t d,
                 float eps, const float* shift, const float* scale,
                 std::span<float> out);

/** torch's F.gelu(approximate="tanh"). */
inline float gelu_tanh(float v) {
    constexpr float kC = 0.79788456080286535588f;  // sqrt(2 / pi)
    return 0.5f * v * (1.0f + std::tanh(kC * (v + 0.044715f * v * v * v)));
}

/** v * sigmoid(v). */
inline float silu(float v) { return v / (1.0f + std::exp(-v)); }

}  // namespace locus::image::dense
