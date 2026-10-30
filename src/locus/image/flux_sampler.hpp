#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/gguf/gguf.hpp"

namespace locus::image {

/**
 * FLUX.1's sampler (i#372): diffusers' FlowMatchEulerDiscreteScheduler
 * as FluxPipeline drives it, and the pipeline's latent packing.
 *
 * The schedule: sigmas linspace(1, 1 / steps, steps), then either the
 * dynamic shift exp(mu) / (exp(mu) + 1 / s - 1), with mu linear in the
 * image token count from base_shift at base_image_seq_len to max_shift
 * at max_image_seq_len (dev, Kontext), or the static shift
 * shift s / (1 + (shift - 1) s) (schnell, shift 1); computed in double,
 * stored as float, with a final 0. Each step is Euler:
 * x += (sigma_next - sigma) v.
 */
class FluxSampler {
  public:
    /**
     * @param g A "flux" GGUF; reads its flux.sampler.* keys.
     * @throws gguf::Error on a missing key.
     */
    static FluxSampler load(const gguf::GgufFile& g);

    /**
     * @param steps Denoising steps, at least 1.
     * @param image_tokens The packed latent's tokens, (h / 2) (w / 2).
     * @returns steps + 1 sigmas, from about 1 down to 0.
     * @throws std::invalid_argument on 0 steps.
     */
    std::vector<float> sigmas(std::uint32_t steps,
                              std::uint32_t image_tokens) const;

    /**
     * The noise level the pipeline gives the transformer for sigma:
     * (sigma * 1000) / 1000 in float, as timesteps are stored and then
     * divided.
     */
    static float model_sigma(float sigma);

    /** x += (sigma_next - sigma) * v, element-wise. */
    static void step(std::span<float> x, std::span<const float> v,
                     float sigma, float sigma_next);

    /**
     * FluxPipeline._pack_latents: a [c][h][w] latent (h, w even) to
     * (h / 2) (w / 2) tokens of c x 2 x 2 values, token-major (row-major
     * patches; each token is channel, then row and column in the patch).
     */
    static std::vector<float> pack(std::span<const float> latent,
                                   std::uint32_t c, std::uint32_t h,
                                   std::uint32_t w);

    /** The inverse of pack: tokens back to [c][h][w]. */
    static std::vector<float> unpack(std::span<const float> tokens,
                                     std::uint32_t c, std::uint32_t h,
                                     std::uint32_t w);

    /**
     * _prepare_latent_image_ids for an (h / 2) x (w / 2) patch grid:
     * (first, row, column) per token, token-major. first is 0 for the
     * image being made and 1 for a Kontext reference image.
     */
    static std::vector<float> image_ids(std::uint32_t h2, std::uint32_t w2,
                                        float first);

  private:
    bool dynamic_ = false;
    /** Doubles, as the pipeline's Python floats (the GGUF stores F64). */
    double shift_ = 1.0, base_shift_ = 0.5, max_shift_ = 1.15;
    std::uint32_t base_seq_ = 256, max_seq_ = 4096;
};

}  // namespace locus::image
