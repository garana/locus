#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace locus::image {

/** An 8-bit RGB image, row-major, 3 bytes per pixel. */
struct Rgb8 {
    std::uint32_t w = 0, h = 0;
    std::vector<std::uint8_t> rgb;
};

/** Caps for decode_image, checked on the header before any pixel
 * buffer is allocated. */
struct DecodeLimits {
    std::size_t max_bytes = 16u << 20;   ///< encoded input size
    std::uint64_t max_pixels = 4096ull * 4096;  ///< width x height
};

/**
 * Decodes a PNG or JPEG (request input, so untrusted bytes) to RGB with
 * locus's own decoders (png_decode.hpp, jpeg_decode.hpp), which give
 * stb_image's 3-channel result. Gray and palette images become RGB; an
 * alpha channel is dropped, as Pillow's convert("RGB") does.
 *
 * @param data The encoded file.
 * @param lim Size caps.
 * @throws std::invalid_argument on an unsupported format, a corrupt or
 *     truncated file, or an image over the caps.
 */
Rgb8 decode_image(std::span<const std::byte> data,
                  const DecodeLimits& lim = {});

namespace detail {
/** decode_image through stb_image for any PNG or JPEG: the reference
 * the tests compare locus's PNG decoder with, until stb_image goes
 * (i#250). */
Rgb8 decode_image_stb(std::span<const std::byte> data,
                      const DecodeLimits& lim = {});
}  // namespace detail

/**
 * Pillow's Image.resize((w, h), BICUBIC) for an RGB image, bit-exact:
 * the same separable passes (horizontal, then vertical, each only when
 * that side changes), antialiased support, 22-bit fixed-point weights
 * and rounding to uint8 after each pass.
 *
 * @throws std::invalid_argument on a zero size.
 */
Rgb8 resize_bicubic(const Rgb8& in, std::uint32_t w, std::uint32_t h);

/**
 * The Janus reference preprocessing up to the normalization
 * (VLMImageProcessor.resize): resize so the longer side is `side`,
 * each side max(int(side_px / max_side * side), min_size), with
 * resize_bicubic, then pad to a side x side square with `background`,
 * centered at (side - w) / 2, (side - h) / 2 (integer division).
 */
Rgb8 fit_square(const Rgb8& in, std::uint32_t side, std::uint32_t min_size,
                std::array<std::uint8_t, 3> background);

}  // namespace locus::image
