#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"

namespace locus::image {

/**
 * The decoder half of diffusers' AutoencoderKL, the Stable Diffusion
 * VAE that PixArt-alpha uses (i#146; docs/models/pixart-alpha.md
 * section 6). It turns a [4][h][w] latent into a [3][8h][8w] image in
 * [-1, 1] (for the four-level SD VAE).
 *
 * The latent is first divided by the scaling factor (0.18215), as the
 * pipeline does; then a 1x1 post-quant conv, a 3x3 conv_in, a mid
 * block (resnet, single-head attention, resnet), up blocks of
 * layers_per_block + 1 resnets each followed (but the last) by a
 * nearest 2x upsample and a 3x3 conv, and GroupNorm, SiLU, a 3x3
 * conv_out. Resnets are GroupNorm, SiLU, conv, twice, plus the input
 * (through a 1x1 conv when the width changes).
 *
 * Weights come from an "autoencoder_kl" GGUF
 * (scripts/convert-sd-vae.py, checkpoint names). Tensors stay in the
 * GGUF mapping, so the GgufFile must outlive the decoder.
 */
class SdVaeDecoder {
  public:
    /**
     * @param g A GGUF whose general.architecture is "autoencoder_kl".
     * @throws gguf::Error on a missing or wrong key or tensor.
     */
    static SdVaeDecoder load(const gguf::GgufFile& g);

    /** @returns The latent channels (4). */
    std::uint32_t latent_channels() const { return latent_; }
    /** @returns The image side over the latent side (8). */
    std::uint32_t upscale() const { return 1u << (chans_.size() - 1); }

    /**
     * @param latent [latent_channels][h][w], as the sampler leaves it
     *     (not yet divided by the scaling factor).
     * @param op The backend ops for the attention's matrices.
     * @returns [3][h * upscale][w * upscale], about [-1, 1], unclamped.
     * @throws std::invalid_argument on a wrong size.
     */
    std::vector<float> decode(std::span<const float> latent, std::uint32_t h,
                              std::uint32_t w, const backend::Ops& op) const;

    /**
     * The pipeline's postprocess: round(clamp(y / 2 + 0.5, 0, 1) * 255),
     * half to even, from [3][h][w] to row-major RGB bytes.
     */
    static std::vector<std::uint8_t> to_rgb8(std::span<const float> image,
                                             std::uint32_t h,
                                             std::uint32_t w);

  private:
    /** @returns tensor `name`, which must be F32 with n values. */
    std::span<const float> t(const std::string& name, std::size_t n) const;

    const gguf::GgufFile* g_ = nullptr;
    std::uint32_t latent_ = 0, out_ch_ = 0, layers_ = 0, groups_ = 0;
    float eps_ = 0.0f, scaling_ = 0.0f;
    std::vector<std::uint32_t> chans_;  ///< block_out_channels
};

}  // namespace locus::image
