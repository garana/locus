#include "locus/image/qwen_image_sampler.hpp"

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
#include "locus/image/qwen_image_dit.hpp"
#include "locus/image/qwen_vae.hpp"

namespace {

using locus::image::QwenEditText;
using locus::image::QwenImageDit;
using locus::image::QwenImageSampler;

const std::string kGolden = std::string(LOCUS_SOURCE_DIR) + "/tests/golden/";

// Keep in sync with scripts/make-qwen-edit-denoise-golden.py.
const std::vector<std::pair<std::uint32_t, std::uint32_t>> kSigmaCases = {
    {2, 12}, {3, 256}, {4, 1024}, {20, 4096}, {50, 8192}, {28, 20000}};
constexpr std::uint32_t kRefH = 32, kRefW = 24, kOutH = 16, kOutW = 32;
constexpr std::uint32_t kTxt = 5, kNeg = 3, kWidth = 64, kCh = 16;
constexpr std::uint32_t kSteps = 2;
constexpr float kCfg = 4.0f;
// The synthetic VAE's scale is 4, so tokens are 8 x 8 pixels.
constexpr std::uint32_t kRefTok = (kRefH / 8) * (kRefW / 8);
constexpr std::uint32_t kOutTok = (kOutH / 8) * (kOutW / 8);

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

/** The script's LCG, scaled by 4 as the inputs are. */
std::vector<float> lcg4(std::uint32_t state, std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 0.5f *
            4.0f;
    }
    return v;
}

/** REQUIREs the RMS of got - ref within rel of ref's RMS. */
void check_close(std::span<const float> got, std::span<const float> ref,
                 double rel) {
    REQUIRE(got.size() == ref.size());
    double err = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = double{got[i]} - ref[i];
        err += d * d;
        norm += double{ref[i]} * ref[i];
    }
    err = std::sqrt(err / static_cast<double>(ref.size()));
    norm = std::sqrt(norm / static_cast<double>(ref.size()));
    WARN("error rms " << err << ", reference rms " << norm);
    REQUIRE(err <= rel * norm);
}

}  // namespace

// i#615: the sigma schedule of diffusers' scheduler with Qwen-Image's
// config (dynamic shift 0.5 to 0.9 over 256 to 8192 tokens, the
// stretch to 0.02), bit for bit, at several step and token counts.
TEST_CASE("qwen image sigmas match diffusers", "[qwen-edit][image]") {
    const auto g =
        locus::gguf::GgufFile::open(kGolden + "qwen_image_dit_synth.gguf");
    const auto s = QwenImageSampler::load(g);
    const auto ref = read_f32(kGolden + "qwen_edit_sigmas.f32");
    std::size_t i = 0;
    for (const auto& [steps, tokens] : kSigmaCases) {
        const auto got = s.sigmas(steps, tokens);
        REQUIRE(got.size() == steps + 1);
        REQUIRE(i + got.size() <= ref.size());
        for (std::size_t k = 0; k < got.size(); ++k, ++i) {
            INFO(steps << " steps, " << tokens << " tokens, sigma " << k);
            CHECK(got[k] == ref[i]);
        }
    }
    CHECK(i == ref.size());
    // One step would divide 0 by 0 in the stretch to shift_terminal.
    REQUIRE_THROWS_AS(s.sigmas(1, 256), std::invalid_argument);
}

// The guided prediction keeps each token's conditional norm.
TEST_CASE("qwen true cfg mixes and rescales per token",
          "[qwen-edit][image]") {
    std::vector<float> cond = {3, 4, 1, 0};
    const std::vector<float> neg = {1, 1, 1, 2};
    locus::image::qwen_true_cfg(cond, neg, 2.0f, 2);
    // Token 0: comb (5, 7), rescaled to norm 5. Token 1: comb (1, -2),
    // rescaled to norm 1.
    const float k0 = 5.0f / std::sqrt(74.0f), k1 = 1.0f / std::sqrt(5.0f);
    CHECK(cond[0] == Catch::Approx(5 * k0));
    CHECK(cond[1] == Catch::Approx(7 * k0));
    CHECK(cond[2] == Catch::Approx(1 * k1));
    CHECK(cond[3] == Catch::Approx(-2 * k1));
    REQUIRE_THROWS_AS(locus::image::qwen_true_cfg(cond, neg, 2.0f, 3),
                      std::invalid_argument);
}

// i#615: diffusers' QwenImageEditPipeline.__call__ on the synthetic
// MMDiT and VAE (scripts/make-qwen-edit-denoise-golden.py): 2 steps of
// true CFG (scale 4) on an 16 x 32 edit of a 32 x 24 reference, its
// prompt 5 and its negative prompt 3 text rows. The final latents and
// the decoded image against the pipeline's.
TEST_CASE("qwen edit denoising matches the pipeline",
          "[qwen-edit][image]") {
    const auto gd =
        locus::gguf::GgufFile::open(kGolden + "qwen_image_dit_synth.gguf");
    const auto dit = QwenImageDit::load(gd);
    const auto sampler = QwenImageSampler::load(gd);
    const auto gv =
        locus::gguf::GgufFile::open(kGolden + "qwen_vae_synth.gguf");
    const auto vae = locus::image::QwenVae::load(gv);
    REQUIRE(vae.scale() == 4);
    REQUIRE(dit.channels() == kCh);

    const auto blob = read_f32(kGolden + "qwen_edit_denoise.f32");
    const std::size_t n_ref = std::size_t{kRefTok} * kCh;
    const std::size_t n_lat = std::size_t{kOutTok} * kCh;
    const std::size_t n_img = std::size_t{kOutH} * kOutW * 3;
    REQUIRE(blob.size() == n_ref + n_lat + n_img);
    const std::span<const float> ref(blob.data(), n_ref);
    const std::span<const float> want_lat(blob.data() + n_ref, n_lat);
    const std::span<const float> want_img(blob.data() + n_ref + n_lat,
                                          n_img);

    const auto noise = lcg4(31337, n_lat);
    const auto pos = lcg4(999, std::size_t{kTxt} * kWidth);
    const auto neg = lcg4(555, std::size_t{kNeg} * kWidth);
    const auto lat = locus::image::qwen_edit_denoise(
        dit, sampler, noise, {1, kOutH / 8, kOutW / 8}, ref,
        {1, kRefH / 8, kRefW / 8}, QwenEditText{pos, kTxt},
        QwenEditText{neg, kNeg}, kCfg, kSteps);
    check_close(lat, want_lat, 1e-4);
    const auto img = locus::image::qwen_edit_decode(vae, lat, kOutH, kOutW);
    check_close(img, want_img, 1e-4);

    // A noise size that does not match its segment is refused.
    REQUIRE_THROWS_AS(
        locus::image::qwen_edit_denoise(
            dit, sampler, noise, {1, kOutH / 8, kOutW / 8 + 1}, ref,
            {1, kRefH / 8, kRefW / 8}, QwenEditText{pos, kTxt},
            QwenEditText{neg, kNeg}, kCfg, kSteps),
        std::invalid_argument);
}
