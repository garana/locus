#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace locus::image {

/**
 * The reference's pixel conversion for a decoder output in about
 * [-1, 1]: clip((x + 1) / 2 * 255, 0, 255) in f32, then a TRUNCATING
 * cast to uint8 (numpy float -> uint8 assignment), interleaved to RGB.
 *
 * @param chw [3][h][w] floats.
 * @returns h * w * 3 bytes, row-major RGB.
 */
std::vector<std::uint8_t> to_rgb8(std::span<const float> chw,
                                  std::uint32_t h, std::uint32_t w);

/**
 * An 8-bit RGB PNG with no compression: IHDR, one IDAT holding a zlib
 * stream of stored deflate blocks (filter 0 per row), IEND, with the
 * CRC-32 and Adler-32 the format needs. No dependency; larger than a
 * compressed PNG (about 3 bytes per pixel), which is fine for an API
 * response.
 *
 * @param rgb h * w * 3 bytes, row-major.
 * @returns The PNG file bytes.
 */
std::vector<std::byte> encode_png(std::span<const std::uint8_t> rgb,
                                  std::uint32_t w, std::uint32_t h);

}  // namespace locus::image
