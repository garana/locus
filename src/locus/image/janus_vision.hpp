#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::image {

/**
 * Janus-Pro's image-understanding encoder (i#241): the SigLIP-L/16
 * vision tower at 384 x 384 and the aligner that maps its 1024-wide
 * patch features into the language model's residual stream. The result
 * replaces the prompt's 576 <image_placeholder> embeddings.
 *
 * Tower: a 16 x 16 stride-16 patch conv (24 x 24 = 576 tokens), plus a
 * learned position table, then 24 pre-norm blocks (LayerNorm, 16-head
 * full attention, LayerNorm, GELU MLP) and a final LayerNorm. Aligner:
 * Linear, GELU, Linear. Weights come from the vit.* tensors that
 * scripts/convert-janus.py writes; matrices stay in the GGUF mapping,
 * so the GgufFile must outlive the encoder.
 */
class JanusVision {
  public:
    /**
     * @param g The Janus-Pro GGUF (with vit.* tensors and the
     *     janus.vision.* keys).
     * @param n_embd The language model's width (the aligner's output).
     * @throws gguf::Error on a missing or wrong-sized tensor or key.
     */
    static JanusVision load(const gguf::GgufFile& g, std::uint32_t n_embd);

    /** @returns The input side in pixels (384). */
    std::uint32_t image_size() const { return image_; }
    /** @returns Patch tokens per image (576). */
    std::uint32_t tokens() const { return grid_ * grid_; }
    /** @returns The <image_placeholder> id. */
    tok::TokenId placeholder() const { return placeholder_; }
    /** @returns The <end_of_image> id. */
    tok::TokenId end_of_image() const { return end_; }
    /** @returns The <begin_of_image> id. */
    tok::TokenId begin_of_image() const { return begin_; }

    /**
     * The vision tower's output for one image.
     *
     * @param chw [3][image_size][image_size] pixels in [-1, 1]
     *     (to_pixels).
     * @param op The backend ops the matrices run on.
     * @returns tokens() x 1024 floats, token-major.
     */
    std::vector<float> features(std::span<const float> chw,
                                const backend::Ops& op) const;

    /**
     * The aligner on the tower's output.
     *
     * @param feat tokens() x 1024 floats (features()).
     * @returns tokens() x n_embd floats, token-major: the input
     *     embeddings for the placeholder positions.
     */
    std::vector<float> align(std::span<const float> feat,
                             const backend::Ops& op) const;

    /**
     * The reference's normalization for an image that is already
     * image_size() square: x / 255, then (x - 0.5) / 0.5.
     *
     * @param rgb image_size()^2 * 3 bytes, row-major RGB.
     * @returns [3][image_size][image_size] floats.
     */
    std::vector<float> to_pixels(std::span<const std::uint8_t> rgb) const;

  private:
    struct Block {
        std::span<const float> ln1_w, ln1_b, qkv_b, out_b, ln2_w, ln2_b,
            up_b, down_b;
        backend::Mat qkv, out, up, down;
    };

    std::uint32_t image_ = 0, patch_ = 0, grid_ = 0, width_ = 0, heads_ = 0,
                  ffn_ = 0, n_embd_ = 0;
    float eps_ = 0.0f;
    tok::TokenId placeholder_ = 0, end_ = 0, begin_ = 0;
    std::span<const float> patch_w_, patch_b_, pos_, post_w_, post_b_,
        align0_b_, align2_b_;
    backend::Mat align0_, align2_;
    std::vector<Block> blocks_;
};

}  // namespace locus::image
