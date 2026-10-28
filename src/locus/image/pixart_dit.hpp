#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"

namespace locus::image {

/**
 * PixArt-alpha's denoiser (i#146): diffusers' PixArtTransformer2DModel
 * for the 256 and 512 models, as docs/models/pixart-alpha.md section 4
 * states it. One call predicts the noise (and the learned variance) of
 * one latent at one timestep, given the T5 states of one prompt.
 *
 * - The latent [4][64][64] becomes 32 x 32 = 1024 tokens through a
 *   stride-2 patch conv, plus a fixed 2-D sine/cosine position table.
 * - The timestep goes through sine/cosine features and an MLP into one
 *   modulation vector shared by all blocks (adaLN-single).
 * - Each block: LayerNorm with shift and scale, self-attention scaled
 *   by a gate; cross-attention onto the projected T5 rows (no norm, no
 *   gate); LayerNorm with shift and scale, a tanh-GELU FFN scaled by a
 *   gate.
 * - A final modulated LayerNorm and a linear give 2 x 2 x 8 values per
 *   token, unpatchified to [8][64][64]: 4 noise channels, then 4
 *   learned-variance channels.
 *
 * Weights come from a "pixart" GGUF (scripts/convert-pixart.py,
 * checkpoint tensor names). Matrices stay in the GGUF mapping, so the
 * GgufFile must outlive the model.
 */
class PixArtDit {
  public:
    /**
     * @param g A GGUF whose general.architecture is "pixart".
     * @throws gguf::Error on a missing or wrong key or tensor, or on
     *     the 1024 model (sample_size 128, size embedders).
     */
    static PixArtDit load(const gguf::GgufFile& g);

    /** @returns The latent side (64 for 512 px). */
    std::uint32_t latent_size() const { return sample_; }
    /** @returns The latent channels (4). */
    std::uint32_t in_channels() const { return in_ch_; }
    /** @returns The output channels (8: noise, then variance). */
    std::uint32_t out_channels() const { return out_ch_; }
    /** @returns The T5 width the caption projection takes (4096). */
    std::uint32_t caption_channels() const { return cap_ch_; }

    /**
     * One denoiser call for one batch row.
     *
     * @param latent [in_channels][latent_size][latent_size].
     * @param t The scheduler timestep (for example 999).
     * @param text n_text rows of caption_channels floats: the T5 rows
     *     whose mask is 1. A masked row gets weight exactly 0 in the
     *     reference, so leaving it out gives the same output.
     * @param n_text Rows in text, at least 1.
     * @param op The backend ops the matrices run on.
     * @returns [out_channels][latent_size][latent_size].
     * @throws std::invalid_argument on wrong sizes.
     */
    std::vector<float> forward(std::span<const float> latent, float t,
                               std::span<const float> text,
                               std::uint32_t n_text,
                               const backend::Ops& op) const;

    /**
     * diffusers' get_2d_sincos_pos_embed for a grid x grid patch grid:
     * the first half of each row encodes the column, the second half
     * the row, each as [sin(pos w_k), cos(pos w_k)] with
     * w_k = 10000^(-k / (dim / 4)), in double, cast to float.
     *
     * @returns grid^2 rows of dim floats.
     */
    static std::vector<float> pos_embed(std::uint32_t dim, std::uint32_t grid,
                                        std::uint32_t base_size,
                                        float interpolation_scale);

    /**
     * diffusers' Timesteps(channels, flip_sin_to_cos=True,
     * downscale_freq_shift=0) in float32: [cos(t f_k), sin(t f_k)],
     * f_k = exp(-ln(10000) k / (channels / 2)).
     */
    static std::vector<float> timestep_features(float t,
                                                std::uint32_t channels);

  private:
    struct Linear {
        backend::Mat w;
        std::vector<float> b;
    };
    struct Block {
        std::vector<float> table;  ///< [6][width] scale_shift_table
        Linear q1, k1, v1, o1, q2, k2, v2, o2, up, down;
    };

    std::uint32_t width_ = 0, heads_ = 0, head_dim_ = 0, ffn_ = 0,
                  in_ch_ = 0, out_ch_ = 0, patch_ = 0, sample_ = 0,
                  grid_ = 0, cap_ch_ = 0, t_ch_ = 0;
    float eps_ = 0.0f;
    Linear patch_embd_, t1_, t2_, ada_, cap1_, cap2_, proj_out_;
    std::vector<float> pos_;    ///< [grid^2][width]
    std::vector<float> table_;  ///< [2][width] final scale_shift_table
    std::vector<Block> blocks_;
};

}  // namespace locus::image
