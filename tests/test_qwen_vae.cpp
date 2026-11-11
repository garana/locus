#include "locus/image/qwen_vae.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/gguf/gguf.hpp"

namespace {

using locus::image::QwenVae;

const std::string kGolden = std::string(LOCUS_SOURCE_DIR) + "/tests/golden/";

// Keep in sync with scripts/make-qwen-vae-goldens.py.
constexpr std::uint32_t kZ = 4, kSize = 32, kLat = 8;

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

/** The script's LCG: n values, advancing state. */
std::vector<float> lcg(std::uint32_t state, std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 0.5f;
    }
    return v;
}

/** REQUIREs the RMS of got - ref within rel of ref's RMS. */
void check_close(std::span<const float> got, std::span<const float> ref,
                 double rel) {
    REQUIRE(got.size() == ref.size());
    double err = 0.0, norm = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = double{got[i]} - ref[i];
        err += d * d;
        norm += double{ref[i]} * ref[i];
        worst = std::max(worst, std::abs(d));
    }
    err = std::sqrt(err / static_cast<double>(ref.size()));
    norm = std::sqrt(norm / static_cast<double>(ref.size()));
    WARN("error rms " << err << ", worst " << worst << ", reference rms "
                      << norm);
    REQUIRE(err <= rel * norm);
}

struct Leg {
    const char* name;
    const locus::backend::Ops* op;
    double bound;
};

/** The CPU, plus the Vulkan image ops when there is a device (i#587:
 *  the convs on the GPU, the attention block's matmuls too, so a looser
 *  bound for their other summation order). */
std::vector<Leg> legs() {
    std::vector<Leg> v = {{"cpu", &locus::backend::best_backend().ops, 1e-5}};
    if (locus::backend::vk::VulkanContext::available()) {
        v.push_back({"vulkan", &locus::backend::vulkan_image_ops(), 1e-4});
    }
    return v;
}

}  // namespace

// i#533 VALUES, on every box: a tiny AutoencoderKLQwenImage from
// diffusers (scripts/make-qwen-vae-goldens.py, its GGUF in
// tests/golden) on one frame: the encoder's mean and the decoder's
// output against diffusers' own encode and decode, with the first-frame
// path of the 3D resamplers.
TEST_CASE("qwen vae matches diffusers on a synthetic model",
          "[qwen-vae][image]") {
    const auto g = locus::gguf::GgufFile::open(kGolden + "qwen_vae_synth.gguf");
    const auto v = QwenVae::load(g);
    REQUIRE(v.z_dim() == kZ);
    REQUIRE(v.scale() == kSize / kLat);
    REQUIRE(v.has_encoder());
    const auto ref = read_f32(kGolden + "qwen_vae_synth.f32");
    const std::size_t nz = std::size_t{kZ} * kLat * kLat;
    REQUIRE(ref.size() == nz + std::size_t{3} * kSize * kSize);

    auto img = lcg(777, std::size_t{3} * kSize * kSize);
    for (float& x : img) x *= 4.0f;
    auto z = lcg(999, nz);
    for (float& x : z) x *= 4.0f;
    for (const Leg& leg : legs()) {
        DYNAMIC_SECTION(leg.name) {
            const auto mean = v.encode(img, kSize, kSize, *leg.op);
            check_close(mean, std::span<const float>(ref).first(nz),
                        leg.bound);
            const auto out = v.decode(z, kLat, kLat, *leg.op);
            check_close(out, std::span<const float>(ref).subspan(nz),
                        leg.bound);
        }
    }
}

TEST_CASE("qwen vae latents normalize and back", "[qwen-vae][image]") {
    const auto g = locus::gguf::GgufFile::open(kGolden + "qwen_vae_synth.gguf");
    const auto v = QwenVae::load(g);
    // The script's latents_mean and latents_std.
    std::vector<float> z(kZ * 2, 1.0f);
    v.normalize(z);
    REQUIRE(z[0] == Catch::Approx((1.0f - 0.1f) / 1.5f));
    REQUIRE(z[2] == Catch::Approx((1.0f + 0.2f) / 0.5f));
    v.denormalize(z);
    for (float x : z) REQUIRE(x == Catch::Approx(1.0f));
}

TEST_CASE("qwen vae rejects bad sizes", "[qwen-vae][image]") {
    const auto g = locus::gguf::GgufFile::open(kGolden + "qwen_vae_synth.gguf");
    const auto v = QwenVae::load(g);
    std::vector<float> img(std::size_t{3} * 30 * 32);
    REQUIRE_THROWS_AS(v.encode(img, 30, 32), std::invalid_argument);
    std::vector<float> z(3);
    REQUIRE_THROWS_AS(v.decode(z, 1, 1), std::invalid_argument);
}

// i#531 VALUES, with the real VAE: Qwen/Qwen-Image-Edit at ac7f931's
// vae/, converted by scripts/convert-qwen-image.py --vae-out to
// tests/models/qwen-image-vae.gguf, against diffusers' own encode and
// decode of the same weights in float32
// (scripts/make-qwen-image-vae-golden.py): the converter's layout and
// values, and QwenVae at the real widths.
TEST_CASE("qwen vae matches diffusers on the real Qwen-Image VAE",
          "[qwen-vae][image][e2e]") {
    const std::string path =
        std::string(LOCUS_SOURCE_DIR) + "/tests/models/qwen-image-vae.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("qwen-image-vae.gguf absent; make it with "
             "scripts/convert-qwen-image.py --vae-out");
    }
    constexpr std::uint32_t z = 16, lat = 4;
    const auto g = locus::gguf::GgufFile::open(path);
    const auto v = QwenVae::load(g);
    REQUIRE(v.z_dim() == z);
    REQUIRE(v.scale() == kSize / lat);
    REQUIRE(v.has_encoder());
    const auto ref = read_f32(kGolden + "qwen_image_vae.f32");
    const std::size_t nz = std::size_t{z} * lat * lat;
    REQUIRE(ref.size() == nz + std::size_t{3} * kSize * kSize);

    auto img = lcg(777, std::size_t{3} * kSize * kSize);
    for (float& x : img) x *= 4.0f;
    auto zv = lcg(999, nz);
    for (float& x : zv) x *= 4.0f;
    for (const Leg& leg : legs()) {
        DYNAMIC_SECTION(leg.name) {
            const auto mean = v.encode(img, kSize, kSize, *leg.op);
            check_close(mean, std::span<const float>(ref).first(nz),
                        leg.bound);
            const auto out = v.decode(zv, lat, lat, *leg.op);
            check_close(out, std::span<const float>(ref).subspan(nz),
                        leg.bound);
        }
    }
}
