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

/**
 * @returns The binary16 bit pattern nearest to f, with an exact tie
 * going to the even pattern (IEEE round-to-nearest-even), over the
 * whole range including subnormals. Matches the GPU's packHalf2x16,
 * so an f16 KV cache written on either side reads back the same.
 */
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
 * Weightless RMS norm in place: x /= sqrt(mean(x^2) + eps).
 *
 * @param x Activations, normalized in place.
 * @param eps Stabilizer added to the mean square.
 */
void rmsnorm_unit(std::span<float> x, float eps);

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
 * @returns True for the 15 QK_K super-block weight types that have a
 * Q8_K activation-dot (all k-quants, both ternary, the IQ2/IQ3/IQ1
 * families, IQ4_XS). False for everything else, including IQ4_NL
 * (32-element q8_0-style blocks, not QK_K) and the legacy/float types.
 * The model routes these types through Ops::matvec_q8k for ggml
 * parity; the rest stay on the f32 matvec.
 */
bool is_q8k_type(gguf::TensorType type);

/**
 * @returns Whether matvec_q8k dots w against a Q8_0-quantized
 *     activation (block-32 weights: Q8_0, Q4_0, IQ4_NL), llama.cpp's
 *     vec_dot_type for them.
 */
bool is_q8_0_act_type(gguf::TensorType type);

/**
 * @returns Whether matvec_q8k takes a quantized-activation dot for w
 *     (is_q8k_type or is_q8_0_act_type). The one predicate every
 *     dispatch to matvec_q8k uses, so decode and batched prefill agree.
 */
inline bool uses_quantized_activation(gguf::TensorType type) {
    return is_q8k_type(type) || is_q8_0_act_type(type);
}

/**
 * Quantizes activation x to Q8_K, writing the block fields as flat
 * arrays: d[n/256], qs[n] (int8), bsums[n/16] (int16), where n =
 * x.size() (must be a multiple of 256). Bit-identical to the internal
 * quant matvec_q8k applies, exposed so a GPU backend can upload the
 * same quants and match matvec_q8k's integer dot on device.
 */
void quantize_activation_q8k(std::span<const float> x, float* d,
                             std::int8_t* qs, std::int16_t* bsums);

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

/**
 * @returns The output length conv1d produces for these shape
 *     parameters: (in_len + 2*pad - dilation*(k - 1) - 1) / stride + 1.
 *     The one definition callers and the GPU variants share, so a
 *     caller's `out` buffer cannot disagree with the kernel. Domain:
 *     in_len >= 1, k >= 1, stride >= 1, and in_len + 2*pad >= the tap
 *     span dilation*(k - 1) + 1.
 */
std::uint32_t conv1d_out_len(std::uint32_t in_len, std::uint32_t k,
                             std::uint32_t stride,
                             std::uint32_t dilation, std::uint32_t pad);

/**
 * @returns The output length conv1d_transpose produces:
 *     (in_len - 1) * stride - 2*pad + dilation*(k - 1) + 1 (no
 *     output_padding). Domain: in_len >= 1, k >= 1, stride >= 1.
 */
std::uint32_t conv1d_transpose_out_len(std::uint32_t in_len,
                                       std::uint32_t k,
                                       std::uint32_t stride,
                                       std::uint32_t dilation,
                                       std::uint32_t pad);

/**
 * 1D convolution (F32), the building block of neural-codec decoders.
 * Channels-major, row-major: `in` is [in_ch][in_len], `weight` is
 * [out_ch][in_ch/groups][k], `bias` is [out_ch] or empty, `out` is
 * [out_ch][out_len] with
 *   out_len = (in_len + 2*pad - dilation*(k - 1) - 1) / stride + 1.
 * `groups` partitions channels: groups == in_ch is depthwise,
 * groups == 1 is dense; in_ch and out_ch must both divide by groups.
 *
 * @param in Input, in_ch * in_len floats.
 * @param in_ch Input channels.
 * @param in_len Input length.
 * @param weight Filters, out_ch * (in_ch / groups) * k floats.
 * @param out_ch Output channels.
 * @param k Kernel width.
 * @param bias Per-output-channel bias (out_ch), or empty for none.
 * @param stride Output stride.
 * @param dilation Spacing between kernel taps.
 * @param pad Zero padding added to both ends of the input.
 * @param groups Channel groups.
 * @param out Output, out_ch * out_len floats.
 */
void conv1d(std::span<const float> in, std::uint32_t in_ch,
            std::uint32_t in_len, std::span<const float> weight,
            std::uint32_t out_ch, std::uint32_t k,
            std::span<const float> bias, std::uint32_t stride,
            std::uint32_t dilation, std::uint32_t pad,
            std::uint32_t groups, std::span<float> out);

/**
 * Transposed (upsampling) 1D convolution (F32). Channels-major: `in`
 * is [in_ch][in_len], `weight` is [in_ch][out_ch/groups][k], `bias` is
 * [out_ch] or empty, `out` is [out_ch][out_len] with
 *   out_len = (in_len - 1) * stride - 2*pad + dilation*(k - 1) + 1.
 * No output_padding (the SNAC decoder upsamples with k = 2*stride and
 * even stride, so length scales by exactly stride at pad = (k-stride)/2).
 */
void conv1d_transpose(std::span<const float> in, std::uint32_t in_ch,
                      std::uint32_t in_len,
                      std::span<const float> weight,
                      std::uint32_t out_ch, std::uint32_t k,
                      std::span<const float> bias, std::uint32_t stride,
                      std::uint32_t dilation, std::uint32_t pad,
                      std::uint32_t groups, std::span<float> out);

/**
 * Snake activation in place (per-channel, F32), the SNAC codec's
 * nonlinearity: x += (1 / (alpha + eps)) * sin(alpha * x)^2, with one
 * learnable alpha per channel. `x` is [ch][len], `alpha` is [ch].
 */
void snake1d(std::span<float> x, std::uint32_t ch, std::uint32_t len,
             std::span<const float> alpha, float eps = 1e-9f);

/**
 * conv1d across the thread pool, with a loop order the compiler can
 * vectorize (one contiguous output run per tap). Each output sums the
 * same terms in the same order as conv1d, so the result is
 * bit-identical. Same arguments as conv1d; stride != 1 runs conv1d.
 */
void conv1d_mt(std::span<const float> in, std::uint32_t in_ch,
               std::uint32_t in_len, std::span<const float> weight,
               std::uint32_t out_ch, std::uint32_t k,
               std::span<const float> bias, std::uint32_t stride,
               std::uint32_t dilation, std::uint32_t pad,
               std::uint32_t groups, std::span<float> out);

/**
 * conv1d_transpose across the thread pool. Outputs are split by phase
 * (ox + pad) mod stride, so each phase is a contiguous run fed by every
 * stride-th tap; each output sums the same terms in the same order as
 * conv1d_transpose, so the result is bit-identical. Same arguments as
 * conv1d_transpose; dilation != 1 runs conv1d_transpose.
 */
void conv1d_transpose_mt(std::span<const float> in, std::uint32_t in_ch,
                         std::uint32_t in_len,
                         std::span<const float> weight,
                         std::uint32_t out_ch, std::uint32_t k,
                         std::span<const float> bias,
                         std::uint32_t stride, std::uint32_t dilation,
                         std::uint32_t pad, std::uint32_t groups,
                         std::span<float> out);

/** snake1d across the thread pool; bit-identical to snake1d. */
void snake1d_mt(std::span<float> x, std::uint32_t ch, std::uint32_t len,
                std::span<const float> alpha, float eps = 1e-9f);

/**
 * 2D convolution (F32), the image VQ decoder's workhorse (i#222):
 * cross-correlation (no kernel flip), grouped and strided, with zero
 * padding, the same stride and padding on both axes. Channels-major:
 * `in` is [in_ch][in_h][in_w], `weight` is [out_ch][in_ch/groups][kh]
 * [kw], `bias` is [out_ch] or empty, `out` is [out_ch][out_h][out_w]
 * with out_h = conv1d_out_len(in_h, kh, stride, 1, pad) (out_w alike).
 * The scalar reference for every other variant.
 *
 * @param in Input, in_ch * in_h * in_w floats.
 * @param in_ch Input channels.
 * @param in_h Input height.
 * @param in_w Input width.
 * @param weight Filters, out_ch * (in_ch / groups) * kh * kw floats.
 * @param out_ch Output channels.
 * @param kh Kernel height.
 * @param kw Kernel width.
 * @param bias Per-output-channel bias (out_ch), or empty for none.
 * @param stride Output stride (both axes).
 * @param pad Zero padding on every edge.
 * @param groups Channel groups.
 * @param out Output, out_ch * out_h * out_w floats.
 */
void conv2d(std::span<const float> in, std::uint32_t in_ch,
            std::uint32_t in_h, std::uint32_t in_w,
            std::span<const float> weight, std::uint32_t out_ch,
            std::uint32_t kh, std::uint32_t kw, std::span<const float> bias,
            std::uint32_t stride, std::uint32_t pad, std::uint32_t groups,
            std::span<float> out);

/**
 * Checks conv2d's argument contract, which conv2d itself only asserts:
 * groups non-zero and dividing both channel counts, stride non-zero,
 * and every span the size the layout needs. A device variant calls it
 * before its device attempt, so a bad shape fails loudly instead of
 * reaching the scalar fallback (Release has no asserts).
 *
 * @throws std::invalid_argument naming the first violated rule.
 */
void conv2d_check_args(std::span<const float> in, std::uint32_t in_ch,
                       std::uint32_t in_h, std::uint32_t in_w,
                       std::span<const float> weight, std::uint32_t out_ch,
                       std::uint32_t kh, std::uint32_t kw,
                       std::span<const float> bias, std::uint32_t stride,
                       std::uint32_t pad, std::uint32_t groups,
                       std::span<const float> out);

/**
 * conv2d across the thread pool, one item per (output channel, output
 * row), with the inner loop over contiguous output columns (one per
 * tap) so the compiler vectorizes it. Each output adds the same terms
 * in the same order as conv2d (bias, then ci, ky, kx), so the result is
 * bit-identical. Same arguments as conv2d; stride != 1 runs conv2d.
 */
void conv2d_mt(std::span<const float> in, std::uint32_t in_ch,
               std::uint32_t in_h, std::uint32_t in_w,
               std::span<const float> weight, std::uint32_t out_ch,
               std::uint32_t kh, std::uint32_t kw,
               std::span<const float> bias, std::uint32_t stride,
               std::uint32_t pad, std::uint32_t groups,
               std::span<float> out);

/**
 * GroupNorm in place (F32), as torch.nn.GroupNorm: `x` is [ch][plane];
 * channels are split into `groups` equal groups, each normalised over
 * all its channels' elements with the biased variance, then scaled per
 * channel: y = (x - mean) / sqrt(var + eps) * gamma[c] + beta[c]. The
 * mean and variance are accumulated in double.
 *
 * @param x Activations, ch * plane floats, normalised in place.
 * @param ch Channels (a multiple of groups).
 * @param plane Elements per channel (H * W for an image).
 * @param groups Channel groups.
 * @param gamma Per-channel scale (ch).
 * @param beta Per-channel shift (ch).
 * @param eps Added to the variance.
 */
void group_norm(std::span<float> x, std::uint32_t ch, std::uint32_t plane,
                std::uint32_t groups, std::span<const float> gamma,
                std::span<const float> beta, float eps);

/** SiLU / Swish in place: x = x * sigmoid(x) = x / (1 + exp(-x)). */
void silu(std::span<float> x);

/**
 * Nearest-neighbour 2x upsample (torch F.interpolate, scale 2,
 * "nearest"): out[c][y][x] = in[c][y / 2][x / 2].
 *
 * @param in [ch][h][w].
 * @param out [ch][2h][2w].
 */
void upsample_nearest2x(std::span<const float> in, std::uint32_t ch,
                        std::uint32_t h, std::uint32_t w,
                        std::span<float> out);

}  // namespace locus::backend
