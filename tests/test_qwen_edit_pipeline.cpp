#include "locus/image/qwen_edit_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "qwen_tiny.hpp"

namespace {

using locus::image::QwenEditPipeline;

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kTokModel =
    kDir + "models/qwen2.5-0.5b-instruct-q8_0.gguf";

// Keep in sync with scripts/make-qwen-edit-pipeline-golden.py.
constexpr std::uint64_t kVocab = 151936, kArea = 96 * 96;
constexpr std::uint32_t kRefW = 80, kRefH = 56, kSteps = 2;
constexpr float kCfg = 4.0f;

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

/** The scripts' LCG image: (x + 0.25) * 510 clipped to bytes. */
locus::image::Rgb8 reference() {
    std::uint32_t state = 777;
    const auto v = qwen_tiny::lcg(state, std::size_t{kRefW} * kRefH * 3);
    locus::image::Rgb8 img{kRefW, kRefH, {}};
    img.rgb.resize(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        // numpy's astype(uint8) truncates the clipped float.
        const float x = std::clamp((v[i] + 0.25f) * 510.0f, 0.0f, 255.0f);
        img.rgb[i] = static_cast<std::uint8_t>(x);
    }
    return img;
}

}  // namespace

// i#616: diffusers' QwenImageEditPipeline.__call__ end to end on tiny
// parts (scripts/make-qwen-edit-pipeline-golden.py): the tiny
// Qwen2.5-VL with the real vocabulary, the synthetic MMDiT and VAE, an
// 80 x 56 reference at a 96 x 96 pixel area, "Make the sky purple."
// against a " " negative prompt, 2 steps of true CFG (scale 4). The
// vision tower's image is resized with Pillow's BICUBIC here and with
// torchvision in the pipeline's fast processor (i#601), so the bound
// is the measured error with a margin.
TEST_CASE("qwen edit pipeline matches diffusers on tiny parts",
          "[qwen-edit][image]") {
    if (!std::filesystem::exists(kTokModel)) {
        SKIP("qwen2.5-0.5b-instruct-q8_0.gguf absent (its tokenizer)");
    }
    const auto tok_g = locus::gguf::GgufFile::open(kTokModel);
    const auto lm_img = qwen_tiny::lm("qwen2vl", kVocab);
    const auto lm_g = locus::gguf::GgufFile::parse(lm_img);
    const auto v_img = qwen_tiny::mmproj(qwen_tiny::kE);
    const auto v_g = locus::gguf::GgufFile::parse(v_img);
    const auto dit_g =
        locus::gguf::GgufFile::open(kDir + "golden/qwen_image_dit_synth.gguf");
    const auto vae_g =
        locus::gguf::GgufFile::open(kDir + "golden/qwen_vae_synth.gguf");
    const auto p =
        QwenEditPipeline::from_parts(lm_g, v_g, dit_g, vae_g, tok_g);

    QwenEditPipeline::Request req;
    req.prompt = "Make the sky purple.";
    req.negative_prompt = " ";
    req.reference = reference();
    req.steps = kSteps;
    req.true_cfg_scale = kCfg;
    req.area = kArea;
    const auto [w, h] = p->output_size(req);
    const auto ref = read_f32(kDir + "golden/qwen_edit_pipeline.f32");
    REQUIRE(ref.size() >= 2);
    REQUIRE(h == static_cast<std::uint32_t>(ref[0]));
    REQUIRE(w == static_cast<std::uint32_t>(ref[1]));
    REQUIRE(ref.size() == 2 + std::size_t{w} * h * 3);

    const std::uint32_t s = p->vae_scale();
    std::uint32_t state = 31337;
    auto noise = qwen_tiny::lcg(
        state, std::size_t{h / s / 2} * (w / s / 2) * p->channels());
    for (float& x : noise) x *= 4.0f;
    const auto& op = locus::backend::best_backend().ops;
    const auto img = p->edit_from(noise, req, op, {});
    REQUIRE(img.size() + 2 == ref.size());
    double err = 0.0, norm = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < img.size(); ++i) {
        const double d = double{img[i]} - ref[i + 2];
        err += d * d;
        norm += double{ref[i + 2]} * ref[i + 2];
        worst = std::max(worst, std::abs(d));
    }
    err = std::sqrt(err / static_cast<double>(img.size()));
    norm = std::sqrt(norm / static_cast<double>(img.size()));
    WARN("error rms " << err << ", worst " << worst << ", reference rms "
                      << norm);
    // Measured on an M2 Pro: 1.8e-4 of the RMS, the worst value 8e-4.
    REQUIRE(err <= 1e-3 * norm);

    // A cancelled edit returns nothing.
    CHECK(p->edit_from(noise, req, op, [] { return false; }).empty());
    // The byte output has the same size, and a request without a
    // reference is refused.
    std::uint32_t ow = 0, oh = 0;
    const auto rgb = p->edit(req, op, {}, ow, oh);
    CHECK(ow == w);
    CHECK(oh == h);
    CHECK(rgb.size() == std::size_t{w} * h * 3);
    req.reference = {};
    REQUIRE_THROWS_AS(p->edit(req, op, {}, ow, oh), std::invalid_argument);
}

// i#670: auto_resize false encodes the reference at the output's area
// when that is smaller than `area`. At an output whose area is `area`
// or more it is the default, bit for bit; at a smaller output the
// reference shrinks with it, so the edit changes while the output size
// does not.
TEST_CASE("qwen edit auto_resize false encodes the reference at the "
          "output's area",
          "[qwen-edit][image]") {
    if (!std::filesystem::exists(kTokModel)) {
        SKIP("qwen2.5-0.5b-instruct-q8_0.gguf absent (its tokenizer)");
    }
    const auto tok_g = locus::gguf::GgufFile::open(kTokModel);
    const auto lm_img = qwen_tiny::lm("qwen2vl", kVocab);
    const auto lm_g = locus::gguf::GgufFile::parse(lm_img);
    const auto v_img = qwen_tiny::mmproj(qwen_tiny::kE);
    const auto v_g = locus::gguf::GgufFile::parse(v_img);
    const auto dit_g =
        locus::gguf::GgufFile::open(kDir + "golden/qwen_image_dit_synth.gguf");
    const auto vae_g =
        locus::gguf::GgufFile::open(kDir + "golden/qwen_vae_synth.gguf");
    const auto p =
        QwenEditPipeline::from_parts(lm_g, v_g, dit_g, vae_g, tok_g);
    const auto& op = locus::backend::best_backend().ops;

    auto run = [&](std::uint32_t side, bool auto_resize) {
        QwenEditPipeline::Request req;
        req.prompt = "Make the sky purple.";
        req.reference = reference();
        req.steps = kSteps;
        req.true_cfg_scale = 1.0f;  // no CFG: one pass per step
        req.area = kArea;
        req.width = req.height = side;
        req.auto_resize = auto_resize;
        const auto [w, h] = p->output_size(req);
        REQUIRE(w == side);
        REQUIRE(h == side);
        const std::uint32_t s = p->vae_scale();
        std::uint32_t state = 31337;
        auto noise = qwen_tiny::lcg(
            state, std::size_t{h / s / 2} * (w / s / 2) * p->channels());
        return p->edit_from(noise, req, op, {});
    };
    // 96 x 96 is kArea: the same reference either way; a larger
    // output keeps the default's reference.
    CHECK(run(96, false) == run(96, true));
    CHECK(run(128, false) == run(128, true));
    // 48 x 48: a quarter of the reference pixels, a different edit.
    const auto small = run(48, false);
    CHECK(small.size() == std::size_t{48} * 48 * 3);
    CHECK(small != run(48, true));
}
