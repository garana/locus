#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "locus/gguf/gguf.hpp"

namespace locus::image {

/**
 * The VQ-16 image decoder (LlamaGen's, used by Janus-Pro; i#221 section
 * 4), from a "vq16" GGUF. Turns a grid of codebook ids into an RGB
 * image in about [-1, 1]: an l2-normalised codebook lookup, a 1x1
 * post-quant conv, a 3x3 conv_in, a middle (res, attention, res), one
 * level per ch_mult entry from the widest down (num_res_blocks + 1 res
 * blocks each, attention after each at the lowest resolution, a
 * nearest x2 upsample and a 3x3 conv between levels), then GroupNorm,
 * SiLU and a 3x3 conv to 3 channels. All F32, on the conv2d / GroupNorm
 * / SiLU CPU kernels; the structure is fixed, the sizes come from the
 * GGUF metadata.
 */
class Vq16Decoder {
  public:
    /**
     * Loads a "vq16" GGUF (copies the tensors it needs, so the GgufFile
     * need not outlive the decoder).
     *
     * @param g A parsed GGUF whose general.architecture is "vq16".
     * @throws gguf::Error on a wrong architecture, missing metadata, or
     *     a layout locus does not run (no l2-normalised codebook,
     *     attention at a level other than the lowest resolution, a
     *     channel count not divisible by the norm groups).
     */
    static Vq16Decoder load(const gguf::GgufFile& g);

    /**
     * Decodes a grid of codebook ids, row-major.
     *
     * @param codes grid_h * grid_w ids in [0, codebook_size).
     * @param grid_h Grid rows.
     * @param grid_w Grid columns.
     * @returns [3][grid_h * s][grid_w * s] floats, with s =
     *     upscale().
     * @throws std::invalid_argument on a wrong code count or a code
     *     outside the codebook.
     * @throws gguf::Error on a missing or wrong-sized tensor.
     */
    std::vector<float> decode(std::span<const std::int32_t> codes,
                              std::uint32_t grid_h,
                              std::uint32_t grid_w) const;

    /** @returns Output pixels per code along each axis (16 for VQ-16). */
    std::uint32_t upscale() const {
        return 1u << (ch_mult_.size() - 1);
    }

  private:
    /**
     * @returns The tensor `name`, which must hold exactly `n` floats.
     * @throws gguf::Error if it is missing or another size.
     */
    std::span<const float> t(const std::string& name, std::size_t n) const;

    std::unordered_map<std::string, std::vector<float>> w_;
    std::vector<std::uint32_t> ch_mult_;
    std::uint32_t codebook_size_ = 0;
    std::uint32_t embed_dim_ = 0;
    std::uint32_t z_channels_ = 0;
    std::uint32_t ch_ = 0;
    std::uint32_t num_res_blocks_ = 0;
    std::uint32_t norm_groups_ = 0;
    float norm_eps_ = 0.0f;
};

}  // namespace locus::image
