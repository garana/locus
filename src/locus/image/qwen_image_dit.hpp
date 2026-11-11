#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"

namespace locus::image {

/**
 * Qwen-Image's denoiser (i#532): diffusers' QwenImageTransformer2DModel,
 * an MMDiT with dual-stream blocks only. One call predicts the flow
 * velocity of the image tokens at one noise level, given one prompt's
 * text-encoder rows.
 *
 * - Image tokens are 2x2 patches of the latents (64 values each: 16
 *   channels x 2 x 2) through img_in. The sequence may hold several
 *   segments (the noise latent, then reference images for an edit),
 *   each its own grid. The text rows go through an RMSNorm (txt_norm),
 *   then txt_in.
 * - The conditioning vector is the sinusoidal features of the noise
 *   level times 1000 through an MLP; silu of it feeds every modulation.
 * - Positions are 3-axis RoPE (frame, row, column), each axis rotating
 *   its own slice of every head's pairs (2 i, 2 i + 1). Segment s is
 *   frame s; rows and columns are centred (scale_rope): row r of h at
 *   r - (h - h / 2). Text token j sits at max(h / 2, w / 2) over the
 *   segments plus j on all three axes.
 * - Each block keeps separate image and text weights: per stream, a
 *   LayerNorm modulated by shift and scale, q / k / v with a per-head
 *   RMS norm on q and k, one joint attention over text then image
 *   tokens (RoPE on q and k), a gated residual, and a modulated, gated
 *   tanh-GELU FFN; the six modulation vectors per stream come from one
 *   Linear of silu(conditioning).
 * - A final modulated LayerNorm (scale first, then shift) and proj_out
 *   on the image tokens give 64 values per token.
 *
 * Weights come from a "qwen_image" GGUF (scripts/convert-qwen-image.py).
 * Matrices stay in the GGUF mapping, so the GgufFile must outlive the
 * model.
 */
class QwenImageDit {
  public:
    /** One image segment's grid, in patches (frames is 1 for images). */
    struct Segment {
        std::uint32_t frames = 1, rows = 0, cols = 0;
    };

    /** Intermediate states for tests: the image and text rows after the
     *  first block, token-major. */
    struct Trace {
        std::vector<float> block0_img, block0_txt;
    };

    /**
     * @param g A GGUF whose general.architecture is "qwen_image".
     * @throws gguf::Error on a missing or wrong key or tensor.
     */
    static QwenImageDit load(const gguf::GgufFile& g);

    /** @returns The packed channels per image token (64). */
    std::uint32_t channels() const { return in_ch_; }
    /** @returns The values per image token proj_out gives (64). */
    std::uint32_t out_channels() const { return out_; }
    /** @returns The text-encoder width txt_in takes (3584). */
    std::uint32_t text_width() const { return joint_; }
    /** @returns The model width (3072). */
    std::uint32_t width() const { return width_; }
    /** @returns The block count (60). */
    std::uint32_t blocks() const {
        return static_cast<std::uint32_t>(blocks_.size());
    }

    /**
     * One denoiser call for one batch row.
     *
     * @param x The image tokens, token-major, channels() floats each,
     *     segment after segment, each row-major over its grid.
     * @param segs The segments; their tokens must add up to x's.
     * @param text n_txt x text_width() floats: the text-encoder rows.
     * @param n_txt Text tokens, at least 1.
     * @param sigma The noise level in [0, 1] (the pipeline's t / 1000).
     * @param op The backend ops the matrices run on.
     * @param trace When not null, receives the intermediate states.
     * @returns One out_channels() row per image token, token-major (the
     *     pipeline keeps the first segment's).
     * @throws std::invalid_argument on wrong sizes.
     */
    std::vector<float> forward(std::span<const float> x,
                               std::span<const Segment> segs,
                               std::span<const float> text,
                               std::uint32_t n_txt, float sigma,
                               const backend::Ops& op,
                               Trace* trace = nullptr) const;

    /**
     * The RoPE cosines and sines of the joined sequence (text, then the
     * segments' image tokens), as diffusers' QwenEmbedRope in float32:
     * per axis of width d, pair j turns at pos * theta^(-2 j / d); each
     * angle is repeated for the two values of its pair.
     *
     * @param segs The image segments.
     * @param n_txt Text tokens.
     * @param cos_out, sin_out (n_txt + image tokens) x head_dim floats.
     */
    void rope(std::span<const Segment> segs, std::uint32_t n_txt,
              std::vector<float>& cos_out,
              std::vector<float>& sin_out) const;

  private:
    struct Linear {
        backend::Mat w;
        std::vector<float> b;
    };
    struct Stream {
        Linear mod;  ///< 6 x width: shift, scale, gate (attn), then MLP
        Linear q, k, v, out;
        std::vector<float> q_norm, k_norm;
        Linear ff1, ff2;
    };
    struct Block {
        Stream img, txt;
    };

    std::uint32_t in_ch_ = 0, out_ = 0, joint_ = 0, width_ = 0, heads_ = 0,
                  head_dim_ = 0, ffn_ = 0, t_ch_ = 0;
    float eps_ = 0.0f, theta_ = 0.0f;
    std::vector<std::uint32_t> axes_;
    std::vector<float> txt_norm_;
    Linear img_in_, txt_in_, t1_, t2_, norm_out_, proj_out_;
    std::vector<Block> blocks_;
};

}  // namespace locus::image
