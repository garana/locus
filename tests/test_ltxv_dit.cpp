#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/ltxv_dit.hpp"

using locus::image::LtxvDit;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/ltxv-2b-dit.gguf";

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

// precompute_freqs_cis: dim % 6 leading identity entries, then pairs of
// equal angles theta^(k / (dim/6 - 1)) * pi/2 * (2 frac - 1), with the
// axis cycling time, y, x before the frequency steps.
TEST_CASE("ltxv rope follows the reference layout", "[ltxv][image]") {
    const float max_pos[3] = {20.0f, 2048.0f, 2048.0f};
    // One token at (time 0.32 s, y 64, x 1024); dim 16: 4 padding
    // entries, 2 frequencies x 3 axes x 2.
    const std::vector<float> coords = {0.32f, 64.0f, 1024.0f};
    std::vector<float> c(16), s(16);
    LtxvDit::rope(coords, 1, 16, 10000.0f, max_pos, c, s);
    for (int i = 0; i < 4; ++i) {
        REQUIRE(c[i] == 1.0f);
        REQUIRE(s[i] == 0.0f);
    }
    const float pi2 = 3.14159265358979323846f / 2.0f;
    const float frac[3] = {0.32f / 20.0f * 2 - 1, 64.0f / 2048.0f * 2 - 1,
                           1024.0f / 2048.0f * 2 - 1};
    for (int j = 0; j < 6; ++j) {
        const int k = j / 3;  // the frequency index: 0 or 1
        const float f = std::pow(10000.0f, static_cast<float>(k)) * pi2;
        INFO("pair " << j);
        REQUIRE(c[4 + 2 * j] == Catch::Approx(std::cos(f * frac[j % 3])));
        REQUIRE(c[4 + 2 * j + 1] == c[4 + 2 * j]);
        REQUIRE(s[4 + 2 * j] == Catch::Approx(std::sin(f * frac[j % 3]))
                                    .margin(1e-6));
        REQUIRE(s[4 + 2 * j + 1] == s[4 + 2 * j]);
    }
    // x at the middle of its range has angle 0 for both frequencies.
    REQUIRE(c[4 + 2 * 2] == 1.0f);
    REQUIRE(s[4 + 2 * 5] == 0.0f);
    REQUIRE_THROWS_AS(LtxvDit::rope(coords, 2, 16, 10000.0f, max_pos, c, s),
                      std::invalid_argument);
}

// VALUES, on every box: a tiny Transformer3DModel converted by
// scripts/convert-ltxv.py, against its torch output
// (scripts/make-ltxv-dit-goldens.py synth). The two batch rows use
// different noise levels (1.0, 0.725) and caption masks (2 and 5 valid
// rows) over 24 tokens of a 2 x 3 x 4 latent, so the modulation, RoPE,
// both attentions with their q / k norms, the FFN and the final layer
// are all pinned.
TEST_CASE("ltxv dit matches torch on a synthetic model", "[ltxv][image]") {
    auto g = locus::gguf::GgufFile::open(kDir + "golden/ltxv_synth.gguf");
    const auto dit = LtxvDit::load(g);
    REQUIRE(dit.channels() == 8);
    REQUIRE(dit.caption_channels() == 32);
    REQUIRE(dit.width() == 16);
    const auto j = read_json(kDir + "golden/ltxv_synth.json");
    const auto in = read_f32(kDir + "golden/ltxv_synth_in.f32");
    const auto ref = read_f32(kDir + "golden/ltxv_synth_out.f32");
    constexpr std::uint32_t n = 24;
    const std::size_t lat = 8 * n, cap = 6 * 32;
    REQUIRE(in.size() == 2 * (lat + cap));
    REQUIRE(ref.size() == 2 * lat);
    std::vector<float> coords;
    for (const auto& axis : j["coords"]) {
        for (const auto& v : axis) coords.push_back(v.get<float>());
    }
    REQUIRE(coords.size() == 3 * n);
    std::vector<std::pair<const char*, const locus::backend::Ops*>> legs = {
        {"cpu", &locus::backend::best_backend().ops}};
    if (locus::backend::vulkan_backend_usable()) {
        legs.push_back({"vulkan", &locus::backend::vulkan_image_ops()});
    }
    for (const auto& [name, op] : legs) {
        for (std::size_t b = 0; b < 2; ++b) {
            INFO(name << " batch row " << b);
            const auto mask = j["masks"][b].get<std::vector<int>>();
            std::uint32_t valid = 0;
            while (valid < mask.size() && mask[valid] == 1) ++valid;
            const std::span<const float> x(&in[b * lat], lat);
            const std::span<const float> y(&in[2 * lat + b * cap],
                                           valid * 32);
            const auto got = dit.forward(x, n, coords,
                                         j["sigmas"][b].get<float>(), y,
                                         valid, *op);
            REQUIRE(rel_error(got, std::span<const float>(&ref[b * lat],
                                                          lat)) <= 1e-5);
        }
    }
    const auto& cpu = locus::backend::best_backend().ops;
    const std::span<const float> x(in.data(), lat);
    const std::span<const float> y(&in[2 * lat], 32);
    REQUIRE_THROWS_AS(dit.forward(x, n, coords, 1.0f, {}, 0, cpu),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(dit.forward(x, n - 1, coords, 1.0f, y, 1, cpu),
                      std::invalid_argument);
}

// The real 2B DiT (BF16 weights) against the reference upcast to
// float32 (scripts/make-ltxv-dit-goldens.py real): one call at sigma
// 0.9094 on a 2 x 4 x 6 latent with 7 seeded caption rows. Both sides
// read the same BF16 weight values, so they differ only in summation
// order.
TEST_CASE("ltxv dit matches the reference on the 2B model",
          "[ltxv][image][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("ltxv-2b-dit.gguf absent; see scripts/convert-ltxv.py");
    }
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto dit = LtxvDit::load(g);
    REQUIRE(dit.channels() == 128);
    REQUIRE(dit.width() == 2048);
    const auto j = read_json(kDir + "golden/ltxv/dit.json");
    const auto in = read_f32(kDir + "golden/ltxv/dit_in.f32");
    const auto ref = read_f32(kDir + "golden/ltxv/dit_out.f32");
    constexpr std::uint32_t n = 48, n_text = 7;
    const std::size_t lat = std::size_t{128} * n;
    REQUIRE(in.size() == lat + std::size_t{n_text} * 4096);
    REQUIRE(ref.size() == lat);
    std::vector<float> coords;
    for (const auto& axis : j["coords"]) {
        for (const auto& v : axis) coords.push_back(v.get<float>());
    }
    REQUIRE(coords.size() == 3 * n);
    std::vector<std::pair<const char*, const locus::backend::Ops*>> legs = {
        {"cpu", &locus::backend::best_backend().ops}};
    if (locus::backend::vulkan_backend_usable()) {
        legs.push_back({"vulkan", &locus::backend::vulkan_image_ops()});
    }
    for (const auto& [name, op] : legs) {
        INFO(name);
        const auto got = dit.forward(
            std::span<const float>(in.data(), lat), n, coords,
            j["sigma"].get<float>(),
            std::span<const float>(&in[lat], std::size_t{n_text} * 4096),
            n_text, *op);
        const double err = rel_error(got, ref);
        CAPTURE(err);
        REQUIRE(err <= 1e-4);  // measured 1.5e-5 (cpu), 1.6e-5 (vulkan)
    }
}
