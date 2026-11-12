#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/qwen_image_dit.hpp"
#include "locus/image/qwen_vae.hpp"

namespace locus::image {

/**
 * Qwen-Image's sampler (i#615): diffusers' FlowMatchEulerDiscreteScheduler
 * with Qwen-Image's config, as QwenImageEditPipeline drives it.
 *
 * The schedule: sigmas linspace(1, 1 / steps, steps), then the dynamic
 * shift exp(mu) / (exp(mu) + 1 / s - 1), with mu linear in the image
 * token count from base_shift at base_image_seq_len to max_shift at
 * max_image_seq_len, then the stretch that ends the schedule at
 * shift_terminal: 1 - (1 - s) / ((1 - s_last) / (1 - shift_terminal)).
 * mu is computed in double (Python floats); the rest in float, as the
 * scheduler computes it on a float32 array. A final 0 follows.
 */
class QwenImageSampler {
  public:
    /**
     * @param g A "qwen_image" GGUF; reads its qwen_image.sampler.* keys.
     * @throws gguf::Error on a missing or bad key.
     */
    static QwenImageSampler load(const gguf::GgufFile& g);

    /**
     * @param steps Denoising steps: 1 or more, 2 or more with a
     *     shift_terminal (diffusers' stretch divides 0 by 0 on one step).
     * @param image_tokens The noise latent's packed tokens (not the
     *     reference's), (h / 2) (w / 2).
     * @returns steps + 1 sigmas, from 1 down to 0.
     * @throws std::invalid_argument on too few steps.
     */
    std::vector<float> sigmas(std::uint32_t steps,
                              std::uint32_t image_tokens) const;

  private:
    bool dynamic_ = true;
    /** Doubles, as the pipeline's Python floats (the GGUF stores F64). */
    double shift_ = 1.0, base_shift_ = 0.5, max_shift_ = 0.9;
    double terminal_ = 0.0;
    std::uint32_t base_seq_ = 256, max_seq_ = 8192;
};

/**
 * True classifier-free guidance as QwenImageEditPipeline mixes it:
 * comb = neg + scale (cond - neg), then each token rescaled to the
 * norm of its conditional prediction: cond = comb |cond| / |comb|.
 *
 * @param cond The conditional prediction, token-major; replaced by the
 *     guided one.
 * @param neg The prediction for the negative prompt, same size.
 * @param scale true_cfg_scale.
 * @param channels Values per token.
 * @throws std::invalid_argument on mismatched sizes.
 */
void qwen_true_cfg(std::span<float> cond, std::span<const float> neg,
                   float scale, std::uint32_t channels);

/** One prompt's text-encoder rows: n x the MMDiT's text width. */
struct QwenEditText {
    std::span<const float> rows;
    std::uint32_t n = 0;
};

/**
 * QwenImageEditPipeline's denoising loop: at each step the MMDiT sees
 * the noise latent's tokens followed by the reference image's (fixed),
 * at the noise level FluxSampler::model_sigma(sigma); the velocity of
 * the noise tokens alone steps them by Euler. With a negative prompt
 * (n > 0) and cfg_scale > 1, each step runs the MMDiT twice and mixes
 * the two by qwen_true_cfg.
 *
 * @param noise The initial noise, packed: noise_seg's tokens of
 *     dit.channels() values.
 * @param ref The reference image's packed latents
 *     (qwen_edit_reference_latents), ref_seg's tokens.
 * @param op The backend ops for the MMDiT.
 * @param alive When set, polled before each step; false cancels.
 * @returns The final packed latents, noise's size, or empty when
 *     cancelled.
 * @throws std::invalid_argument on wrong sizes or steps.
 */
std::vector<float> qwen_edit_denoise(
    const QwenImageDit& dit, const QwenImageSampler& sampler,
    std::span<const float> noise, QwenImageDit::Segment noise_seg,
    std::span<const float> ref, QwenImageDit::Segment ref_seg,
    QwenEditText prompt, QwenEditText negative, float cfg_scale,
    std::uint32_t steps,
    const backend::Ops& op = backend::best_backend().ops,
    const std::function<bool()>& alive = {});

/**
 * The pipeline's decode of the final latents (output_type "np"):
 * unpack, undo latents_mean / latents_std, the VAE decoder, then
 * x / 2 + 1 / 2 clamped to [0, 1].
 *
 * @param tokens The packed latents of an h x w pixel image.
 * @param h, w Pixels; multiples of 2 x vae.scale().
 * @returns [h][w][3] floats in [0, 1].
 * @throws std::invalid_argument on wrong sizes.
 */
std::vector<float> qwen_edit_decode(
    const QwenVae& vae, std::span<const float> tokens, std::uint32_t h,
    std::uint32_t w, const backend::Ops& op = backend::best_backend().ops);

}  // namespace locus::image
