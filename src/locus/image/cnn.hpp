#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/image/image_input.hpp"

/** What the CNN image models (ResNet, Inception; i#274) share: the eval
 *  preprocessing, the labels and the top-k. */
namespace locus::image::cnn {

/** timm's eval transform for a checkpoint, from the GGUF metadata. */
struct Preprocess {
    std::uint32_t size = 0;    ///< the network's input side
    std::uint32_t resize = 0;  ///< the shorter side before the crop
    bool bicubic = false;      ///< else bilinear
    float mean[3] = {}, std[3] = {};
};

/**
 * Reads `prefix` input_size, resize, interpolation ("bilinear" or
 * "bicubic"), mean and std (scripts/convert-cnn.py writes them).
 *
 * @throws gguf::Error on a missing or bad key.
 */
Preprocess read_preprocess(const gguf::GgufFile& g,
                           const std::string& prefix);

/**
 * Reads `prefix` labels (a string array) and checks it against
 * `prefix` num_classes.
 *
 * @throws gguf::Error on a missing key or a count mismatch.
 */
std::vector<std::string> read_labels(const gguf::GgufFile& g,
                                     const std::string& prefix);

/**
 * timm's eval transform: the shorter side resized to p.resize (Pillow's
 * filter), a center crop of p.size (offsets rounded half to even, as
 * Python's round), then x / 255 and (x - mean) / std.
 *
 * @returns [3][p.size][p.size].
 * @throws std::invalid_argument on an empty image.
 */
std::vector<float> preprocess(const Rgb8& img, const Preprocess& p);

/**
 * @returns The k classes with the highest softmax probability, as
 *     (class, probability), highest first; a tie keeps the lower class
 *     first.
 */
std::vector<std::pair<std::uint32_t, float>> top_k(
    std::span<const float> logits, std::uint32_t k);

/** A conv with its batch norm folded in: [out][in][kh][kw], bias. */
struct Conv {
    std::span<const float> w, b;
    std::uint32_t in = 0, out = 0, kh = 0, kw = 0;
};

/**
 * Reads NAME.weight ([out][in][kh][kw], F32) and NAME.bias. A nonzero
 * in or out must match the tensor's.
 *
 * @throws gguf::Error on a missing tensor or a wrong shape.
 */
Conv read_conv(const gguf::GgufFile& g, const std::string& name,
               std::uint32_t in = 0, std::uint32_t out = 0);

/** @returns NAME (F32, n values) from g. @throws gguf::Error. */
std::span<const float> read_f32(const gguf::GgufFile& g,
                                const std::string& name, std::size_t n);

/**
 * The conv, [in][h][w] to [out][h'][w'] (h and w updated), padding ph
 * rows and pw columns on each side, then ReLU when relu.
 */
std::vector<float> conv(const Conv& c, std::span<const float> x,
                        std::uint32_t& h, std::uint32_t& w,
                        std::uint32_t stride, std::uint32_t ph,
                        std::uint32_t pw, bool relu);

/** torch's max_pool2d(3, stride, padding pad) over c planes; the
 *  padding is never chosen. h and w are updated. */
std::vector<float> max_pool3(std::span<const float> x, std::uint32_t c,
                             std::uint32_t& h, std::uint32_t& w,
                             std::uint32_t stride, std::uint32_t pad);

/** torch's avg_pool2d(3, stride 1, padding 1) with count_include_pad:
 *  every output is its 3 x 3 window's sum over 9. */
std::vector<float> avg_pool3(std::span<const float> x, std::uint32_t c,
                             std::uint32_t h, std::uint32_t w);

/** @returns Each of c planes' mean (global average pooling). */
std::vector<float> global_avg(std::span<const float> x, std::uint32_t c,
                              std::size_t plane);

/** @returns w x + b, w [out][x.size()] (the classifier). */
std::vector<float> linear(std::span<const float> w,
                          std::span<const float> b,
                          std::span<const float> x);

}  // namespace locus::image::cnn
