#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/pixart_dit.hpp"

using locus::image::PixArtDit;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/pixart-xl-2-512-f32.gguf";

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

nlohmann::json read_json(const std::string& path) {
    std::ifstream f(path);
    REQUIRE(f.good());
    return nlohmann::json::parse(f);
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

// docs/models/pixart-alpha.md 4.3: token n = 32 i + j; the first half
// encodes the column j, the second the row i; w_k = 10000^(-k/288).
TEST_CASE("pixart position table follows the reference layout",
          "[pixart][image]") {
    const auto pe = PixArtDit::pos_embed(1152, 32, 32, 1.0f);
    REQUIRE(pe.size() == std::size_t{1024} * 1152);
    const auto at = [&](std::uint32_t i, std::uint32_t j, std::uint32_t k) {
        return pe[(std::size_t{i} * 32 + j) * 1152 + k];
    };
    REQUIRE(at(0, 0, 0) == 0.0f);
    REQUIRE(at(0, 0, 288) == 1.0f);
    const double w5 = std::pow(10000.0, -5.0 / 288.0);
    const double w287 = std::pow(10000.0, -287.0 / 288.0);
    // Row 3, column 7.
    REQUIRE(at(3, 7, 5) == static_cast<float>(std::sin(7 * w5)));
    REQUIRE(at(3, 7, 288 + 5) == static_cast<float>(std::cos(7 * w5)));
    REQUIRE(at(3, 7, 576 + 287) == static_cast<float>(std::sin(3 * w287)));
    REQUIRE(at(3, 7, 864 + 287) == static_cast<float>(std::cos(3 * w287)));
    // A 16-wide base halves the coordinates of a 32 grid.
    const auto half = PixArtDit::pos_embed(8, 32, 16, 1.0f);
    REQUIRE(half[(std::size_t{0} * 32 + 4) * 8 + 0] ==
            static_cast<float>(std::sin(2.0)));
}

// 4.4: [cos(t f_k), sin(t f_k)], f_k = exp(-ln(10000) k / 128).
TEST_CASE("pixart timestep features put cos first", "[pixart][image]") {
    const auto tf = PixArtDit::timestep_features(999.0f, 256);
    REQUIRE(tf.size() == 256);
    REQUIRE(tf[0] == std::cos(999.0f));
    REQUIRE(tf[128] == std::sin(999.0f));
    const float f1 = std::exp(static_cast<float>(-std::log(10000.0)) / 128.0f);
    REQUIRE(tf[1] == Catch::Approx(std::cos(999.0f * f1)).margin(1e-6));
    REQUIRE(tf[129] == Catch::Approx(std::sin(999.0f * f1)).margin(1e-6));
}

// VALUES, on every box: a tiny PixArtTransformer2DModel converted by
// scripts/convert-pixart.py, against its torch output
// (scripts/make-pixart-dit-goldens.py synth). The two batch rows use
// different timesteps (999, 500) and caption masks (1 and 4 valid
// rows), so the modulation, both attentions, the FFN, the final layer
// and the unpatchify are all pinned.
TEST_CASE("pixart dit matches torch on a synthetic model",
          "[pixart][image]") {
    auto g = locus::gguf::GgufFile::open(kDir + "golden/pixart_synth.gguf");
    const auto dit = PixArtDit::load(g);
    REQUIRE(dit.latent_size() == 8);
    REQUIRE(dit.in_channels() == 4);
    REQUIRE(dit.out_channels() == 8);
    REQUIRE(dit.caption_channels() == 32);
    const auto j = read_json(kDir + "golden/pixart_synth.json");
    const auto in = read_f32(kDir + "golden/pixart_synth_in.f32");
    const auto ref = read_f32(kDir + "golden/pixart_synth_out.f32");
    const std::size_t lat = 4 * 8 * 8, cap = 6 * 32, out = 8 * 8 * 8;
    REQUIRE(in.size() == 2 * (lat + cap));
    REQUIRE(ref.size() == 2 * out);
    const auto& op = locus::backend::best_backend().ops;
    for (std::size_t b = 0; b < 2; ++b) {
        INFO("batch row " << b);
        const auto mask = j["masks"][b].get<std::vector<int>>();
        std::uint32_t valid = 0;
        while (valid < mask.size() && mask[valid] == 1) ++valid;
        const std::span<const float> x(&in[b * lat], lat);
        const std::span<const float> y(&in[2 * lat + b * cap], valid * 32);
        const auto got = dit.forward(x, j["timesteps"][b].get<float>(), y,
                                     valid, op);
        REQUIRE(rel_error(got, std::span<const float>(&ref[b * out], out)) <=
                1e-5);
    }
    REQUIRE_THROWS_AS(dit.forward(std::span<const float>(in.data(), lat), 1.0f,
                                  {}, 0, op),
                      std::invalid_argument);
}

// PixArt-XL-2-512x512 at step 0 (t = 999) against the diffusers
// pipeline (scripts/make-pixart-goldens.py, trimmed by
// scripts/make-pixart-dit-goldens.py real): row 0 is the "" negative,
// row 1 the prompt, both on the same start latent; all 8 channels.
TEST_CASE("pixart dit matches the reference on the first step",
          "[pixart][image][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("pixart-xl-2-512-f32.gguf absent; see scripts/convert-pixart.py");
    }
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto dit = PixArtDit::load(g);
    REQUIRE(dit.latent_size() == 64);
    const auto j = read_json(kDir + "golden/pixart/dit.json");
    const auto text = read_f32(kDir + "golden/pixart/dit_text.f32");
    const auto x = read_f32(kDir + "golden/pixart/dit_in.f32");
    const auto ref = read_f32(kDir + "golden/pixart/dit_out.f32");
    const auto rows = j["rows"].get<std::vector<std::uint32_t>>();
    const float t = j["timestep"].get<float>();
    const std::size_t out = std::size_t{8} * 64 * 64;
    REQUIRE(ref.size() == 2 * out);
    const auto& op = locus::backend::best_backend().ops;
    std::size_t at = 0;
    for (std::size_t b = 0; b < 2; ++b) {
        INFO("batch row " << b);
        const std::span<const float> y(&text[at], rows[b] * 4096);
        const auto t0 = std::chrono::steady_clock::now();
        const auto got = dit.forward(x, t, y, rows[b], op);
        const double secs = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
        const double err =
            rel_error(got, std::span<const float>(&ref[b * out], out));
        // Measured on macOS arm64: 3.9e-6 (row 0), 4.3e-5 (row 1).
        WARN("pixart dit row " << b << ": max diff / rms " << err << ", "
                               << secs << " s");
        CHECK(err <= 5e-4);
        at += std::size_t{rows[b]} * 4096;
    }
}
