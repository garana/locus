#pragma once

#include <cstddef>
#include <span>

#include "locus/image/image_input.hpp"

namespace locus::image {

/** @returns true when `data` starts with the PNG signature. */
bool is_png(std::span<const std::byte> data);

/**
 * Decodes a PNG (untrusted bytes) to 8-bit RGB, the way stb_image does
 * when asked for 3 channels:
 * - every color type and bit depth of the spec: gray (1, 2, 4, 8, 16),
 *   RGB (8, 16), palette (1, 2, 4, 8), gray + alpha and RGBA (8, 16);
 * - 16-bit samples keep their high byte; gray below 8 bits scales to
 *   0..255; gray is copied to R, G and B; alpha (and tRNS) is dropped;
 * - Adam7 interlace, and the five row filters.
 *
 * The chunk CRCs and the zlib Adler-32 must match. The caps are
 * checked on IHDR, before any buffer the file can size, and the
 * inflated data may not exceed what the header implies.
 *
 * @throws std::invalid_argument on a malformed, truncated or
 *     unsupported file, or one over the caps.
 */
Rgb8 decode_png(std::span<const std::byte> data, const DecodeLimits& lim);

}  // namespace locus::image
