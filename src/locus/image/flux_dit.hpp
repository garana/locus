#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"

namespace locus::image {

/**
 * FLUX.1's denoiser (i#372): diffusers' FluxTransformer2DModel, an
 * MMDiT. One call predicts the flow velocity of one packed latent at
 * one noise level, given one prompt's T5 rows and CLIP pooled vector.
 *
 * - Image tokens are the latent's 2x2 patches (64 values each: 16
 *   channels x 2 x 2), through a Linear (x_embedder). The T5 rows go
 *   through context_embedder.
 * - One conditioning vector: the sinusoidal features of the noise level
 *   times 1000 through an MLP, plus the same for the guidance scale
 *   (when the model has a guidance embedding), plus an MLP of the CLIP
 *   pooled vector.
 * - Positions are 3-axis RoPE: each token has three ids (text tokens
 *   (0, 0, 0); image tokens (0, row, column), Kontext reference tokens
 *   (1, row, column)), each axis rotating its own slice of every head.
 * - Double blocks keep separate image and text weights: per stream, a
 *   LayerNorm modulated by shift and scale from the conditioning
 *   vector, q / k / v with a per-head RMS norm on q and k, one joint
 *   attention over text then image tokens (RoPE on q and k), a gated
 *   residual, and a modulated, gated tanh-GELU FFN.
 * - Single blocks run on the joined sequence: one modulated LayerNorm,
 *   attention and an MLP side by side, one projection of both, gated.
 * - A final modulated LayerNorm (scale first, then shift) and a Linear
 *   on the image tokens give 64 values per token.
 *
 * Weights come from a "flux" GGUF (scripts/convert-flux.py). Matrices
 * stay in the GGUF mapping, so the GgufFile must outlive the model.
 */
class FluxDit {
  public:
    /** Intermediate states for tests: the image and text rows after the
     *  first double block, and the joined rows after the first single
     *  block, token-major. */
    struct Trace {
        std::vector<float> double0_img, double0_txt, single0;
    };

    /**
     * @param g A GGUF whose general.architecture is "flux".
     * @throws gguf::Error on a missing or wrong key or tensor.
     */
    static FluxDit load(const gguf::GgufFile& g);

    /** @returns The packed channels per image token (64). */
    std::uint32_t channels() const { return in_ch_; }
    /** @returns The T5 width context_embedder takes (4096). */
    std::uint32_t text_width() const { return joint_; }
    /** @returns The CLIP pooled width (768). */
    std::uint32_t pooled_width() const { return pooled_; }
    /** @returns The model width (3072). */
    std::uint32_t width() const { return width_; }
    /** @returns Whether the model takes a guidance scale (dev, Kontext). */
    bool guidance_embeds() const { return guidance_; }
    /** @returns The double block count (19). */
    std::uint32_t double_blocks() const {
        return static_cast<std::uint32_t>(doubles_.size());
    }
    /** @returns The single block count (38). */
    std::uint32_t single_blocks() const {
        return static_cast<std::uint32_t>(singles_.size());
    }

    /**
     * One denoiser call for one batch row.
     *
     * @param x n_img x channels() floats, token-major: the packed latent.
     * @param img_ids n_img x 3 position ids, token-major.
     * @param n_img Image tokens, at least 1.
     * @param text n_txt x text_width() floats: the T5 rows.
     * @param txt_ids n_txt x 3 position ids (zeros in the pipelines).
     * @param n_txt Text tokens, at least 1.
     * @param pooled pooled_width() floats: the CLIP pooled output.
     * @param sigma The noise level in [0, 1] (the pipeline's t / 1000).
     * @param guidance The guidance scale; ignored without a guidance
     *     embedding.
     * @param op The backend ops the matrices run on.
     * @param trace When not null, receives the intermediate states.
     * @returns n_img x channels() floats, token-major: the velocity.
     * @throws std::invalid_argument on wrong sizes.
     */
    std::vector<float> forward(std::span<const float> x,
                               std::span<const float> img_ids,
                               std::uint32_t n_img,
                               std::span<const float> text,
                               std::span<const float> txt_ids,
                               std::uint32_t n_txt,
                               std::span<const float> pooled, float sigma,
                               float guidance, const backend::Ops& op,
                               Trace* trace = nullptr) const;

    /**
     * The RoPE angles' cosines and sines for each token, as diffusers'
     * FluxPosEmbed: per axis i of width d_i, frequency j of the axis is
     * theta^(-2 j / d_i), computed in double; each angle is repeated
     * for the pair it rotates.
     *
     * @param ids n x 3 position ids, token-major.
     * @param n Tokens.
     * @param cos_out, sin_out n x head_dim floats each.
     */
    void rope(std::span<const float> ids, std::uint32_t n,
              std::vector<float>& cos_out,
              std::vector<float>& sin_out) const;

  private:
    struct Linear {
        backend::Mat w;
        std::vector<float> b;
    };
    struct Stream {
        Linear mod;  ///< norm1: 6 x width (shift, scale, gate, then MLP)
        Linear q, k, v, out;
        std::vector<float> q_norm, k_norm;
        Linear ff1, ff2;
    };
    struct DoubleBlock {
        Stream img, txt;
    };
    struct SingleBlock {
        Linear mod;  ///< 3 x width (shift, scale, gate)
        Linear q, k, v, mlp, out;
        std::vector<float> q_norm, k_norm;
    };
    struct Mlp {
        Linear l1, l2;
    };

    std::uint32_t in_ch_ = 0, joint_ = 0, pooled_ = 0, width_ = 0,
                  heads_ = 0, head_dim_ = 0, ffn_ = 0, t_ch_ = 0;
    bool guidance_ = false;
    float eps_ = 0.0f, theta_ = 0.0f;
    std::vector<std::uint32_t> axes_;
    Linear x_embed_, ctx_embed_, norm_out_, proj_out_;
    Mlp time_, guide_, text_;
    std::vector<DoubleBlock> doubles_;
    std::vector<SingleBlock> singles_;
};

}  // namespace locus::image
