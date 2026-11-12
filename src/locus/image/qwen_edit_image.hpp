#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/image/image_input.hpp"
#include "locus/image/qwen_vae.hpp"

namespace locus::image {

/**
 * The reference image's preprocessing in Qwen-Image-Edit (i#601), as
 * diffusers 0.35.1's QwenImageEditPipeline.__call__ does it: one
 * LANCZOS resize to about 1024 x 1024 pixels for the VAE, and that
 * image resized again for the Qwen2.5-VL processor (smart_resize).
 */

/**
 * diffusers' calculate_dimensions(1024 * 1024, w / h): the side of the
 * same aspect with about 1024 x 1024 pixels, each side rounded (half to
 * even, as Python's round) to a multiple of 32.
 *
 * @returns {width, height}.
 * @param area The pixel count to aim at (the pipeline's 1024 x 1024;
 *     smaller in tests).
 * @throws std::invalid_argument on a zero side or area.
 */
std::pair<std::uint32_t, std::uint32_t> qwen_edit_dimensions(
    std::uint32_t w, std::uint32_t h, std::uint64_t area = 1024 * 1024);

/**
 * transformers' Qwen2-VL smart_resize: both sides multiples of factor,
 * the area within [min_pixels, max_pixels], the aspect kept as closely
 * as possible.
 *
 * @returns {height, width}.
 * @throws std::invalid_argument on a zero side or an aspect over 200.
 */
std::pair<std::uint32_t, std::uint32_t> qwen_smart_resize(
    std::uint32_t h, std::uint32_t w, std::uint32_t factor = 28,
    std::uint64_t min_pixels = 3136, std::uint64_t max_pixels = 12845056);

/** One reference image, ready for the VAE and for the vision tower. */
struct EditReference {
    /** The LANCZOS resize to qwen_edit_dimensions(). */
    Rgb8 vae_image;
    /** vae_image as the VAE takes it: [3][h][w], x / 127.5 - 1. */
    std::vector<float> vae_input;
    /** vae_image resized to smart_resize's size with Pillow BICUBIC,
     *  for QwenVision::patches(). The pipeline's fast processor resizes
     *  with torchvision instead; see i#601 for how far apart they are. */
    Rgb8 text_image;
};

/**
 * @param img Any RGB image.
 * @param area The VAE image's pixel count to aim at
 *     (qwen_edit_dimensions).
 * @throws std::invalid_argument on an empty image.
 */
EditReference qwen_edit_reference(const Rgb8& img,
                                  std::uint64_t area = 1024 * 1024);

/** The reference image as the MMDiT's second segment (i#614). */
struct EditLatents {
    /** rows x cols tokens of z_dim * 4 values (FluxSampler::pack, which
     *  is QwenImageEditPipeline._pack_latents too). */
    std::vector<float> tokens;
    /** The token grid: the latents' sides / 2. */
    std::uint32_t rows = 0, cols = 0;
};

/**
 * _encode_vae_image then _pack_latents: the VAE encoder's mean (the
 * posterior's mode), normalized with the VAE's latents mean and std,
 * packed into tokens.
 *
 * @param chw [3][h][w] in [-1, 1] (EditReference::vae_input).
 * @param h, w Sides; multiples of 2 * vae.scale().
 * @param op The backend ops for the VAE encoder (the Vulkan image ops
 *     run its convs on the GPU).
 * @throws std::invalid_argument on a bad size or a VAE without its
 *     encoder.
 */
EditLatents qwen_edit_reference_latents(
    const QwenVae& vae, std::span<const float> chw, std::uint32_t h,
    std::uint32_t w, const backend::Ops& op = backend::best_backend().ops);

}  // namespace locus::image
