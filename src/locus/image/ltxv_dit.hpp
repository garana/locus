#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"

namespace locus::image {

/**
 * LTX-Video's denoiser (i#357): the Transformer3DModel of
 * Lightricks/LTX-Video (ltx_video/models/transformers). One call
 * predicts the flow velocity of one video latent at one noise level,
 * given the T5 states of one prompt.
 *
 * - The latent [C][F][H][W] (C = 128) becomes F * H * W tokens, one per
 *   latent cell in (frame, row, column) order, through a Linear
 *   (patchify_proj; patch size 1).
 * - Positions enter as RoPE over the whole width: each token has three
 *   coordinates (time, y, x) divided by the max positions; see rope().
 * - The noise level sigma (0 to 1) times the timestep scale (1000)
 *   goes through sine/cosine features and an MLP into one modulation
 *   vector shared by all blocks (adaLN-single, as in PixArt).
 * - Each block: RMSNorm with shift and scale, self-attention (q and k
 *   RMS-normed over the whole width with a weight, then RoPE) scaled
 *   by a gate; cross-attention onto the projected T5 rows (q and k
 *   normed, no RoPE, no gate); RMSNorm with shift and scale, a
 *   tanh-GELU FFN scaled by a gate.
 * - A final modulated LayerNorm and a linear give C values per token.
 *
 * Weights come from an "ltxv" GGUF (scripts/convert-ltxv.py). Matrices
 * stay in the GGUF mapping, so the GgufFile must outlive the model.
 */
class LtxvDit {
  public:
    /**
     * @param g A GGUF whose general.architecture is "ltxv".
     * @throws gguf::Error on a missing or wrong key or tensor.
     */
    static LtxvDit load(const gguf::GgufFile& g);

    /** @returns The latent channels (128). */
    std::uint32_t channels() const { return in_ch_; }
    /** @returns The T5 width the caption projection takes (4096). */
    std::uint32_t caption_channels() const { return cap_ch_; }
    /** @returns The model width (2048). */
    std::uint32_t width() const { return width_; }

    /**
     * One denoiser call for one batch row.
     *
     * @param latent [channels][n] channel-major: n tokens in (frame,
     *     row, column) order.
     * @param n Tokens, at least 1.
     * @param coords [3][n]: each token's (time, y, x) coordinate as the
     *     reference's indices_grid (pixel units, time in seconds), before
     *     the division by the max positions.
     * @param sigma The noise level in [0, 1], one for all tokens (the
     *     reference's per-token timestep, for image or video
     *     conditioning, is not supported).
     * @param text n_text rows of caption_channels floats: the T5 rows
     *     whose mask is 1. A masked row gets weight exactly 0 in the
     *     reference (a -10000 score bias), so leaving it out gives the
     *     same output.
     * @param n_text Rows in text, at least 1.
     * @param op The backend ops the matrices run on.
     * @returns [channels][n], channel-major like latent.
     * @throws std::invalid_argument on wrong sizes.
     */
    std::vector<float> forward(std::span<const float> latent,
                               std::uint32_t n,
                               std::span<const float> coords, float sigma,
                               std::span<const float> text,
                               std::uint32_t n_text,
                               const backend::Ops& op) const;

    /**
     * The reference's precompute_freqs_cis (spacing "exp") for n
     * tokens: per token, dim cosines and dim sines. The first dim % 6
     * entries are cos 1, sin 0; then, for j = 0 .. dim / 6 * 3 - 1, the
     * pair (2 j, 2 j + 1) after them uses axis a = j % 3 and frequency
     * k = j / 3:
     *
     *     angle = theta^(k / (dim / 6 - 1)) * pi / 2 * (2 frac_a - 1),
     *     frac_a = coord_a / max_pos_a,
     *
     * in float32.
     *
     * @param coords [3][n] as forward() takes them.
     * @param cosv, sinv n * dim floats each, token-major.
     */
    static void rope(std::span<const float> coords, std::uint32_t n,
                     std::uint32_t dim, float theta,
                     const float (&max_pos)[3], std::span<float> cosv,
                     std::span<float> sinv);

  private:
    struct Linear {
        backend::Mat w;
        std::vector<float> b;
    };
    struct Block {
        std::vector<float> table;  ///< [6][width] scale_shift_table
        std::vector<float> qn1, kn1, qn2, kn2;  ///< q / k RMSNorm weights
        Linear q1, k1, v1, o1, q2, k2, v2, o2, up, down;
    };

    std::uint32_t width_ = 0, heads_ = 0, head_dim_ = 0, ffn_ = 0,
                  in_ch_ = 0, out_ch_ = 0, cap_ch_ = 0, t_ch_ = 0;
    float eps_ = 0.0f, theta_ = 0.0f, t_scale_ = 0.0f;
    float max_pos_[3] = {};
    Linear patchify_, t1_, t2_, ada_, cap1_, cap2_, proj_out_;
    std::vector<float> table_;  ///< [2][width] final scale_shift_table
    std::vector<Block> blocks_;
};

}  // namespace locus::image
