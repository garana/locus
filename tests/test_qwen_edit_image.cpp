#include "locus/image/qwen_edit_image.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/flux_sampler.hpp"
#include "locus/image/qwen_vae.hpp"
#include "png_parse.hpp"
#include "qwen_tiny.hpp"

namespace {

using locus::image::qwen_edit_dimensions;
using locus::image::qwen_smart_resize;

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";

// Keep in sync with scripts/make-qwen-edit-image-golden.py.
const std::vector<std::pair<std::uint32_t, std::uint32_t>> kSizes = {
    {384, 288},   {384, 384},  {288, 384}, {1920, 1080}, {1080, 1920},
    {640, 480},   {1, 1},      {37, 1000}, {5000, 3333}, {1023, 1025}};
const std::vector<std::pair<std::uint32_t, std::uint32_t>> kSmart = {
    {896, 1184}, {1024, 1024}, {20, 30},   {4000, 4000}, {14, 2700},
    {55, 57},    {1000, 100},  {896, 1194}, {897, 1203}};
constexpr std::uint32_t kCropW = 384, kCropH = 288;

std::vector<char> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f),
            std::istreambuf_iterator<char>()};
}

template <class T>
std::vector<T> read_as(const std::string& path) {
    const auto c = read_bytes(path);
    std::vector<T> v(c.size() / sizeof(T));
    std::memcpy(v.data(), c.data(), v.size() * sizeof(T));
    return v;
}

}  // namespace

// i#601: diffusers' calculate_dimensions and transformers' smart_resize
// (with Qwen-Image-Edit's processor limits), size for size, including
// Python's round-half-to-even and both area limits.
TEST_CASE("qwen edit sizes match diffusers and transformers",
          "[qwen-edit][image]") {
    const auto ref = read_as<std::int32_t>(kDir + "golden/qwen_edit_sizes.i32");
    REQUIRE(ref.size() == 2 * (kSizes.size() + kSmart.size()));
    std::size_t i = 0;
    for (const auto& [w, h] : kSizes) {
        INFO("calculate_dimensions " << w << " x " << h);
        const auto [cw, ch] = qwen_edit_dimensions(w, h);
        CHECK(cw == static_cast<std::uint32_t>(ref[i++]));
        CHECK(ch == static_cast<std::uint32_t>(ref[i++]));
    }
    for (const auto& [h, w] : kSmart) {
        INFO("smart_resize " << h << " x " << w);
        const auto [sh, sw] = qwen_smart_resize(h, w);
        CHECK(sh == static_cast<std::uint32_t>(ref[i++]));
        CHECK(sw == static_cast<std::uint32_t>(ref[i++]));
    }
    REQUIRE_THROWS_AS(qwen_edit_dimensions(0, 5), std::invalid_argument);
    REQUIRE_THROWS_AS(qwen_smart_resize(1, 300), std::invalid_argument);
}

// i#601: a 384 x 288 crop of janus_apple.png through the pipeline's
// preprocessing: the VAE input (LANCZOS to 1184 x 896, x / 127.5 - 1)
// and the text side's image (Pillow BICUBIC to 1176 x 896), sampled
// rows, bit for bit.
TEST_CASE("qwen edit reference matches the pipeline's preprocessing",
          "[qwen-edit][image]") {
    const auto png = png_test::parse_stored_png(
        [] {
            const auto c = read_bytes(kDir + "golden/janus_apple.png");
            std::vector<std::byte> b(c.size());
            std::memcpy(b.data(), c.data(), c.size());
            return b;
        }());
    locus::image::Rgb8 crop{kCropW, kCropH, {}};
    crop.rgb.resize(std::size_t{kCropW} * kCropH * 3);
    for (std::uint32_t y = 0; y < kCropH; ++y) {
        std::memcpy(&crop.rgb[std::size_t{y} * kCropW * 3],
                    &png.rgb[std::size_t{y} * png.w * 3], kCropW * 3);
    }
    const auto r = locus::image::qwen_edit_reference(crop);
    REQUIRE(r.vae_image.w == 1184);
    REQUIRE(r.vae_image.h == 896);
    REQUIRE(r.text_image.w == 1176);
    REQUIRE(r.text_image.h == 896);

    const auto ref = read_as<float>(kDir + "golden/qwen_edit_image.f32");
    const std::uint32_t vw = r.vae_image.w, vh = r.vae_image.h;
    const std::uint32_t tw = r.text_image.w, th = r.text_image.h;
    REQUIRE(ref.size() == 9 * std::size_t{vw} + 9 * std::size_t{tw});
    std::size_t i = 0;
    for (std::uint32_t c = 0; c < 3; ++c) {
        for (std::uint32_t y : {0u, vh / 2, vh - 1}) {
            for (std::uint32_t x = 0; x < vw; ++x, ++i) {
                INFO("vae channel " << c << " row " << y << " col " << x);
                REQUIRE(r.vae_input[(std::size_t{c} * vh + y) * vw + x] ==
                        ref[i]);
            }
        }
    }
    for (std::uint32_t c = 0; c < 3; ++c) {
        for (std::uint32_t y : {0u, th / 2, th - 1}) {
            for (std::uint32_t x = 0; x < tw; ++x, ++i) {
                INFO("text channel " << c << " row " << y << " col " << x);
                REQUIRE(static_cast<float>(
                            r.text_image.rgb[(std::size_t{y} * tw + x) * 3 +
                                             c]) == ref[i]);
            }
        }
    }
    REQUIRE_THROWS_AS(locus::image::qwen_edit_reference({}),
                      std::invalid_argument);
}

// i#614: the reference image's latents, as the pipeline's
// _encode_vae_image (the posterior's mode, normalized) and
// _pack_latents (scripts/make-qwen-edit-latents-golden.py): the
// synthetic VAE of tests/golden/qwen_vae_synth.gguf on a [3][32][24]
// LCG image, so 8 x 6 latents and 4 x 3 tokens of 16 values.
TEST_CASE("qwen edit reference latents match the pipeline",
          "[qwen-edit][image]") {
    constexpr std::uint32_t kH = 32, kW = 24;
    const auto ref = read_as<float>(kDir + "golden/qwen_edit_latents.f32");
    const auto g =
        locus::gguf::GgufFile::open(kDir + "golden/qwen_vae_synth.gguf");
    const auto vae = locus::image::QwenVae::load(g);
    REQUIRE(vae.scale() == 4);
    std::uint32_t state = 777;
    auto img = qwen_tiny::lcg(state, std::size_t{3} * kH * kW);
    for (float& x : img) x *= 4.0f;
    const auto lat = locus::image::qwen_edit_reference_latents(vae, img, kH,
                                                               kW);
    CHECK(lat.rows == 4);
    CHECK(lat.cols == 3);
    REQUIRE(lat.tokens.size() == ref.size());
    double err = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = double{lat.tokens[i]} - ref[i];
        err += d * d;
        norm += double{ref[i]} * ref[i];
    }
    WARN("error rms " << std::sqrt(err / ref.size()) << ", reference rms "
                      << std::sqrt(norm / ref.size()));
    REQUIRE(std::sqrt(err) <= 1e-5 * std::sqrt(norm));
    // A side that is not a multiple of 2 x the VAE scale is refused.
    REQUIRE_THROWS_AS(
        locus::image::qwen_edit_reference_latents(vae, img, kH, kW - 4),
        std::invalid_argument);
}

// Qwen-Image-Edit packs its latents with FluxSampler::pack (i#630):
// QwenImageEditPipeline._pack_latents is FLUX's permutation, and the
// unpack undoes it.
TEST_CASE("qwen latents pack and unpack", "[qwen-edit][image]") {
    using locus::image::FluxSampler;
    // [2][4][6]: value = channel * 100 + row * 10 + column.
    constexpr std::uint32_t c = 2, h = 4, w = 6;
    std::vector<float> z(std::size_t{c} * h * w);
    for (std::uint32_t ch = 0; ch < c; ++ch) {
        for (std::uint32_t y = 0; y < h; ++y) {
            for (std::uint32_t x = 0; x < w; ++x) {
                z[(std::size_t{ch} * h + y) * w + x] =
                    static_cast<float>(ch * 100 + y * 10 + x);
            }
        }
    }
    const auto t = FluxSampler::pack(z, c, h, w);
    // Token (row 1, column 2) is token 1 * 3 + 2 = 5; its values are
    // channel 0's 2 x 2 patch at (2, 4), then channel 1's.
    const std::vector<float> want = {24, 25, 34, 35, 124, 125, 134, 135};
    for (std::size_t i = 0; i < want.size(); ++i) {
        CHECK(t[5 * 8 + i] == want[i]);
    }
    CHECK(FluxSampler::unpack(t, c, h, w) == z);
    REQUIRE_THROWS_AS(FluxSampler::pack(z, c, h + 1, w),
                      std::invalid_argument);
}
