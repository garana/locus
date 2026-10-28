#pragma once

#include <cstddef>
#include <span>

#include "locus/image/image_input.hpp"

namespace locus::image {

/** @returns true when `data` starts with a JPEG SOI marker. */
bool is_jpeg(std::span<const std::byte> data);

/**
 * Decodes a JPEG (untrusted bytes) to 8-bit RGB, the way stb_image does
 * when asked for 3 channels:
 * - baseline and extended sequential (SOF0, SOF1) and progressive
 *   (SOF2) Huffman JPEGs, 8-bit precision; arithmetic coding, 12-bit
 *   and lossless JPEGs are refused;
 * - 1, 3 or 4 components with sampling factors 1 to 4 in integer
 *   ratios, restart intervals, DNL;
 * - stb_image's integer IDCT (jidctint), its upsampling (the
 *   triangle filter for 2x horizontal and 2x2, nearest otherwise) and
 *   its fixed-point YCbCr to RGB; RGB-coded and Adobe CMYK / YCCK
 *   files as stb_image converts them; gray copied to R, G and B.
 *
 * Like stb_image, a scan that runs out of data decodes the rest as
 * zero bits, so a truncated photo still gives an image. The caps are
 * checked on the frame header, before any buffer it sizes.
 *
 * @throws std::invalid_argument on a malformed or unsupported file, or
 *     one over the caps.
 */
Rgb8 decode_jpeg(std::span<const std::byte> data, const DecodeLimits& lim);

}  // namespace locus::image
