#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "locus/image/image_input.hpp"

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
 * @throws std::invalid_argument on a zero side.
 */
std::pair<std::uint32_t, std::uint32_t> qwen_edit_dimensions(std::uint32_t w,
                                                             std::uint32_t h);

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
 * @throws std::invalid_argument on an empty image.
 */
EditReference qwen_edit_reference(const Rgb8& img);

}  // namespace locus::image
