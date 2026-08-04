#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "locus/gguf/gguf.hpp"

namespace locus::backend {

/**
 * Scalar reference CPU ops for the model runtime. These are the
 * correctness baseline every SIMD/Vulkan variant is tested against
 * (DESIGN.md 4.3); optimized variants land as per-source files
 * selected via locus::sys::detect().
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

/**
 * LayerNorm (weight-only, no bias): centers x by its mean, scales by
 * 1/sqrt(variance + eps), then multiplies by weight, into out. Used
 * by arches whose norm is LayerNorm rather than RMSNorm (e.g. DBRX).
 *
 * @param x Input activations.
 * @param w Per-channel scale; same length as x.
 * @param eps Stabilizer added to the variance.
 * @param out Output; same length as x, may alias x.
 */
void layernorm(std::span<const float> x, std::span<const float> w,
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
 * Q8_K-activation matvec (ggml parity, DESIGN.md R16): the activation
 * x is quantized to Q8_K once and integer-dotted with the quantized
 * weight rows, matching llama.cpp bit-for-bit on quantized matvec (vs
 * the f32-activation dot in matvec). Currently the Q8_K path covers
 * Q4_K; other types delegate to matvec. `x.size()` must be a multiple
 * of 256 for the Q8_K types.
 */
void matvec_q8k(const Mat& w, std::span<const float> x,
                std::span<float> out);

/**
 * Scalar reference for the R11 batched matvec (backend::Ops::
 * matvec_batch). Applies w to n token vectors (x_batch is n*w.cols,
 * token t at t*w.cols) and writes results row-major: out_batch is
 * w.rows*n with [row r, token t] at r*n + t. Byte-identical to n
 * matvec() calls; the correctness baseline the SIMD register-
 * blocked kernels are tested against.
 */
void matvec_batch_scalar(const Mat& w, std::span<const float> x_batch,
                         std::span<float> out_batch,
                         std::uint32_t n);

/**
 * Transposed matrix-vector product: out[c] = sum_r w[r,c]*x[r]
 * (i.e. W^T x). Used by MLA's weight absorption; scalar only.
 */
void matvec_t(const Mat& w, std::span<const float> x,
              std::span<float> out);

/** YARN long-context rope correction parameters. */
struct Yarn {
    /** 1/scaling_factor; 1.0 disables all corrections. */
    float freq_scale = 1.0f;
    /** cos/sin magnitude scale (1.0 for DeepSeek: it cancels). */
    float mscale = 1.0f;
    /** Original (pre-scaling) training context. */
    std::uint32_t n_ctx_orig = 0;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
};

/**
 * Computes the YARN correction-dim range for a rope of head_dim
 * (identity yarn yields lo = hi = 0). Exposed so GPU dispatch can
 * precompute what the shaders need.
 */
void yarn_corr_range(std::uint32_t head_dim, float freq_base,
                     const Yarn& yarn, float& lo, float& hi);

/**
 * NEOX-style RoPE in place: within each head, pairs
 * (x[i], x[i + head_dim/2]) rotate by the YARN-corrected angle.
 */
void rope_neox_yarn(std::span<float> x, std::uint32_t n_heads,
                    std::uint32_t head_dim, std::uint32_t pos,
                    float freq_base, const Yarn& yarn);

/**
 * Interleaved-pair RoPE with YARN corrections: pairs
 * (x[2i], x[2i+1]) rotate by the corrected angle. This is what
 * deepseek2 applies to its q_pe/k_pe (the HF implementation's
 * de-interleaving view makes its rotate_half equivalent to
 * interleaved pairs; verified against llama.cpp tensors).
 */
void rope_norm_yarn(std::span<float> x, std::uint32_t n_heads,
                    std::uint32_t head_dim, std::uint32_t pos,
                    float freq_base, const Yarn& yarn);

/**
 * Dequantizes one row of w into out (used for embedding lookup).
 */
void dequant_row(const Mat& w, std::uint32_t row,
                 std::span<float> out);

/** @returns On-disk bytes of one row of w (for row slicing). */
std::size_t mat_row_bytes(const Mat& w);

/** @returns The nrows-row sub-matrix of w starting at row0. */
inline Mat mat_rows(const Mat& w, std::uint32_t row0,
                    std::uint32_t nrows) {
    Mat s = w;
    s.data = w.data + row0 * mat_row_bytes(w);
    s.rows = nrows;
    return s;
}

}  // namespace locus::backend
