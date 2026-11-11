#include "locus/image/qwen_edit_image.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace locus::image {

namespace {

/** Python's round(x) for a float: to the nearest, ties to even (the
 *  default rounding mode). */
double py_round(double x) { return std::nearbyint(x); }

}  // namespace

std::pair<std::uint32_t, std::uint32_t> qwen_edit_dimensions(std::uint32_t w,
                                                             std::uint32_t h) {
    if (w == 0 || h == 0) {
        throw std::invalid_argument("qwen edit: empty image");
    }
    const double ratio = static_cast<double>(w) / static_cast<double>(h);
    const double width = std::sqrt(1024.0 * 1024.0 * ratio);
    const double height = width / ratio;
    return {static_cast<std::uint32_t>(py_round(width / 32) * 32),
            static_cast<std::uint32_t>(py_round(height / 32) * 32)};
}

std::pair<std::uint32_t, std::uint32_t> qwen_smart_resize(
    std::uint32_t h, std::uint32_t w, std::uint32_t factor,
    std::uint64_t min_pixels, std::uint64_t max_pixels) {
    if (h == 0 || w == 0 || factor == 0) {
        throw std::invalid_argument("smart_resize: empty image");
    }
    const double hd = h, wd = w, f = factor;
    if (std::max(hd, wd) / std::min(hd, wd) > 200) {
        throw std::invalid_argument(
            "smart_resize: the aspect ratio must be under 200");
    }
    double hb = py_round(hd / f) * f, wb = py_round(wd / f) * f;
    if (hb * wb > static_cast<double>(max_pixels)) {
        const double beta =
            std::sqrt(hd * wd / static_cast<double>(max_pixels));
        hb = std::max(f, std::floor(hd / beta / f) * f);
        wb = std::max(f, std::floor(wd / beta / f) * f);
    } else if (hb * wb < static_cast<double>(min_pixels)) {
        const double beta =
            std::sqrt(static_cast<double>(min_pixels) / (hd * wd));
        hb = std::ceil(hd * beta / f) * f;
        wb = std::ceil(wd * beta / f) * f;
    }
    return {static_cast<std::uint32_t>(hb), static_cast<std::uint32_t>(wb)};
}

EditReference qwen_edit_reference(const Rgb8& img) {
    if (img.w == 0 || img.h == 0 ||
        img.rgb.size() != std::size_t{img.w} * img.h * 3) {
        throw std::invalid_argument("qwen edit: empty or bad image");
    }
    EditReference r;
    const auto [vw, vh] = qwen_edit_dimensions(img.w, img.h);
    // Pillow returns a copy when the size does not change.
    r.vae_image = (vw == img.w && vh == img.h) ? img
                                               : resize_lanczos(img, vw, vh);
    const std::size_t plane = std::size_t{vw} * vh;
    r.vae_input.resize(3 * plane);
    for (std::size_t p = 0; p < plane; ++p) {
        for (std::size_t c = 0; c < 3; ++c) {
            // VaeImageProcessor: to [0, 1] in float32, then 2x - 1.
            const float v = static_cast<float>(r.vae_image.rgb[p * 3 + c]) /
                            255.0f;
            r.vae_input[c * plane + p] = 2.0f * v - 1.0f;
        }
    }
    const auto [th, tw] = qwen_smart_resize(vh, vw);
    r.text_image = (tw == vw && th == vh)
                       ? r.vae_image
                       : resize_bicubic(r.vae_image, tw, th);
    return r;
}

namespace {

void check_latents(std::size_t n, std::uint32_t c, std::uint32_t h,
                   std::uint32_t w) {
    if (c == 0 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0 ||
        n != std::size_t{c} * h * w) {
        throw std::invalid_argument(
            "qwen edit: latents must be [c][h][w] with even sides");
    }
}

}  // namespace

std::vector<float> qwen_pack_latents(std::span<const float> z,
                                     std::uint32_t c, std::uint32_t h,
                                     std::uint32_t w) {
    check_latents(z.size(), c, h, w);
    const std::uint32_t rows = h / 2, cols = w / 2;
    std::vector<float> out(z.size());
    std::size_t o = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t q = 0; q < cols; ++q) {
            for (std::uint32_t ch = 0; ch < c; ++ch) {
                for (std::uint32_t dy = 0; dy < 2; ++dy) {
                    for (std::uint32_t dx = 0; dx < 2; ++dx) {
                        out[o++] = z[(std::size_t{ch} * h + 2 * r + dy) * w +
                                     2 * q + dx];
                    }
                }
            }
        }
    }
    return out;
}

std::vector<float> qwen_unpack_latents(std::span<const float> tokens,
                                       std::uint32_t c, std::uint32_t h,
                                       std::uint32_t w) {
    check_latents(tokens.size(), c, h, w);
    const std::uint32_t rows = h / 2, cols = w / 2;
    std::vector<float> z(tokens.size());
    std::size_t o = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t q = 0; q < cols; ++q) {
            for (std::uint32_t ch = 0; ch < c; ++ch) {
                for (std::uint32_t dy = 0; dy < 2; ++dy) {
                    for (std::uint32_t dx = 0; dx < 2; ++dx) {
                        z[(std::size_t{ch} * h + 2 * r + dy) * w + 2 * q +
                          dx] = tokens[o++];
                    }
                }
            }
        }
    }
    return z;
}

EditLatents qwen_edit_reference_latents(const QwenVae& vae,
                                        std::span<const float> chw,
                                        std::uint32_t h, std::uint32_t w) {
    const std::uint32_t s = vae.scale();
    if (h == 0 || w == 0 || h % (2 * s) != 0 || w % (2 * s) != 0) {
        throw std::invalid_argument(
            "qwen edit: the reference's sides must be multiples of 2 x "
            "the VAE scale");
    }
    auto z = vae.encode(chw, h, w);
    vae.normalize(z);
    EditLatents out;
    out.rows = h / s / 2;
    out.cols = w / s / 2;
    out.tokens = qwen_pack_latents(z, vae.z_dim(), h / s, w / s);
    return out;
}

}  // namespace locus::image
