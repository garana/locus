#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/sd_vae.hpp"

using locus::image::SdVaeDecoder;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/sd-vae-f32.gguf";

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

/** @returns max |got - ref| over the RMS of ref. */
double rel_error(std::span<const float> got, std::span<const float> ref) {
    REQUIRE(got.size() == ref.size());
    double ss = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        ss += double{ref[i]} * ref[i];
        worst = std::max(worst, std::fabs(double{got[i]} - ref[i]));
    }
    return worst / std::sqrt(ss / static_cast<double>(ref.size()));
}

}  // namespace

TEST_CASE("vae to_rgb8 follows the pipeline postprocess", "[pixart][image]") {
    // Planes R {-1, 1}, G {0, 5}, B {-5, 0.5}: -1 -> 0, 1 -> 255,
    // 0 -> 127.5 -> 128 (half to even), 0.5 -> 191.25 -> 191, and out of
    // range clamps.
    const std::vector<float> img = {-1.0f, 1.0f, 0.0f, 5.0f, -5.0f, 0.5f};
    const auto rgb = SdVaeDecoder::to_rgb8(img, 1, 2);
    REQUIRE(rgb == std::vector<std::uint8_t>{0, 128, 0, 255, 255, 191});
}

// VALUES, on every box: a tiny AutoencoderKL (block_out_channels [8, 16],
// one layer per block) converted by scripts/convert-sd-vae.py, against
// torch's decode of the same latent (scripts/make-pixart-vae-goldens.py
// synth): the scaling, post-quant conv, mid resnets and attention, the
// up blocks with a conv shortcut and an upsample, and conv_out.
TEST_CASE("vae decoder matches torch on a synthetic model",
          "[pixart][image]") {
    auto g = locus::gguf::GgufFile::open(kDir + "golden/vae_synth.gguf");
    const auto vae = SdVaeDecoder::load(g);
    REQUIRE(vae.latent_channels() == 4);
    REQUIRE(vae.upscale() == 2);
    const auto z = read_f32(kDir + "golden/vae_synth_in.f32");
    const auto ref = read_f32(kDir + "golden/vae_synth_out.f32");
    REQUIRE(z.size() == 4 * 8 * 8);
    const auto& op = locus::backend::best_backend().ops;
    const auto got = vae.decode(z, 8, 8, op);
    REQUIRE(rel_error(got, ref) <= 1e-5);
    // The GPU convolutions, linears and attention, when Vulkan is usable.
    if (locus::backend::vulkan_backend_usable()) {
        const auto gpu = vae.decode(z, 8, 8, locus::backend::vulkan_image_ops());
        REQUIRE(rel_error(gpu, ref) <= 1e-5);
    }
    REQUIRE_THROWS_AS(vae.decode(z, 4, 8, op), std::invalid_argument);
    // latent_fits: at most 256 a side and 128^2 positions.
    const std::vector<float> wide(std::size_t{4} * 257 * 1);
    REQUIRE_THROWS_AS(vae.decode(wide, 257, 1, op), std::invalid_argument);
    const std::vector<float> big(std::size_t{4} * 129 * 129);
    REQUIRE_THROWS_AS(vae.decode(big, 129, 129, op), std::invalid_argument);
}

// The PixArt VAE on the golden final latent, against the pipeline's
// image (0..255 floats before rounding) at every 4th row and column.
TEST_CASE("pixart vae decode matches the reference image",
          "[pixart][image][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("sd-vae-f32.gguf absent; see scripts/convert-sd-vae.py");
    }
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto vae = SdVaeDecoder::load(g);
    REQUIRE(vae.upscale() == 8);
    const auto z = read_f32(kDir + "golden/pixart/latent_final.f32");
    const auto ref = read_f32(kDir + "golden/pixart/vae_image_s4.f32");
    REQUIRE(ref.size() == std::size_t{128} * 128 * 3);
    // The CPU backend, and the GPU when a Vulkan device is usable.
    std::vector<std::pair<const char*, const locus::backend::Ops*>> legs = {
        {"cpu", &locus::backend::best_backend().ops}};
    if (locus::backend::vulkan_backend_usable()) {
        legs.push_back({"vulkan", &locus::backend::vulkan_image_ops()});
    }
    for (const auto& [name, op] : legs) {
        INFO(name);
        const auto t0 = std::chrono::steady_clock::now();
        const auto img = vae.decode(z, 64, 64, *op);
        const double secs = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
        REQUIRE(img.size() == std::size_t{3} * 512 * 512);
        // clamp(y / 2 + 0.5, 0, 1) * 255, HWC, at the sampled pixels.
        std::vector<float> got(ref.size());
        const std::size_t plane = std::size_t{512} * 512;
        double worst = 0.0;
        for (std::uint32_t r = 0; r < 128; ++r) {
            for (std::uint32_t c = 0; c < 128; ++c) {
                for (std::uint32_t ch = 0; ch < 3; ++ch) {
                    const float y = img[ch * plane +
                                        std::size_t{4 * r} * 512 + 4 * c];
                    const float px =
                        std::clamp(y / 2.0f + 0.5f, 0.0f, 1.0f) * 255.0f;
                    const std::size_t i =
                        (std::size_t{r} * 128 + c) * 3 + ch;
                    got[i] = px;
                    worst = std::max(worst, double{std::fabs(px - ref[i])});
                }
            }
        }
        // Measured on macOS arm64 (CPU): 0.0015 of 255 (1.2e-5 of the
        // RMS).
        WARN("pixart vae " << name << ": max |diff| " << worst
                           << " of 255, max diff / rms "
                           << rel_error(got, ref) << ", " << secs << " s");
        CHECK(worst <= 0.1);
    }
}
