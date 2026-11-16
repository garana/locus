#include "locus/image/qwen_image_pipeline.hpp"

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

using locus::image::QwenImagePipeline;

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kTokModel =
    kDir + "models/qwen2.5-0.5b-instruct-q8_0.gguf";
const std::string kGolden = kDir + "golden/qwen_image_pipeline.f32";

// Keep in sync with scripts/image/make-qwen-image-pipeline-golden.py.
constexpr std::uint64_t kVocab = 151936;
constexpr std::uint32_t kWidth = 96, kHeight = 80, kSteps = 2;
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

}  // namespace

// The template's system part, which the pipeline drops, is
// prompt_template_encode_start_idx tokens; with another count the
// conditioning would keep template tokens or lose prompt ones.
TEST_CASE("qwen image template drops exactly its system part",
          "[qwen-image][image]") {
    if (!std::filesystem::exists(kTokModel)) {
        SKIP("qwen2.5-0.5b-instruct-q8_0.gguf absent (its tokenizer)");
    }
    const auto g = locus::gguf::GgufFile::open(kTokModel);
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    const auto head =
        tok->encode(std::string(locus::image::kImageTemplateHead), false);
    CHECK(head.size() == locus::image::kImageDropTokens);
    const auto ids = locus::image::qwen_image_prompt(*tok, "A cat.");
    REQUIRE(ids.size() > head.size());
    CHECK(std::equal(head.begin(), head.end(), ids.begin()));
    // A prompt past the tokenizer's cap is cut at the tail.
    std::string longp;
    for (int i = 0; i < 1200; ++i) longp += " cat";
    CHECK(locus::image::qwen_image_prompt(*tok, longp).size() ==
          locus::image::kImageTokenizerMax +
              locus::image::kImageDropTokens);
}

// Text only: n - drop rows, cut to max_sequence_length.
TEST_CASE("qwen image text embeds keep the prompt rows",
          "[qwen-image][image]") {
    const auto lm_img = qwen_tiny::lm("qwen2vl", 64);
    const auto lm_g = locus::gguf::GgufFile::parse(lm_img);
    const auto lm = locus::model::TransformerModel::load(lm_g);
    const std::uint32_t E = lm.hparams().n_embd;
    std::vector<locus::tok::TokenId> ids(12);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<locus::tok::TokenId>(i * 5 % 64);
    }
    const auto all = locus::image::qwen_image_text_embeds(lm, ids, 4, 512);
    CHECK(all.size() == std::size_t{8} * E);
    const auto cut = locus::image::qwen_image_text_embeds(lm, ids, 4, 3);
    REQUIRE(cut.size() == std::size_t{3} * E);
    // The cut keeps the first rows.
    CHECK(std::equal(cut.begin(), cut.end(), all.begin()));
    REQUIRE_THROWS_AS(locus::image::qwen_image_text_embeds(lm, ids, 12, 512),
                      std::invalid_argument);
}

// i#1091: diffusers' QwenImagePipeline.__call__ end to end on tiny parts
// (scripts/image/make-qwen-image-pipeline-golden.py): the tiny
// Qwen2.5-VL with the real vocabulary, the synthetic MMDiT and VAE, a
// 96 x 80 image of "A red fox in fresh snow." against a " " negative
// prompt, 2 steps of true CFG (scale 4).
TEST_CASE("qwen image pipeline matches diffusers on tiny parts",
          "[qwen-image][image]") {
    if (!std::filesystem::exists(kTokModel)) {
        SKIP("qwen2.5-0.5b-instruct-q8_0.gguf absent (its tokenizer)");
    }
    const auto tok_g = locus::gguf::GgufFile::open(kTokModel);
    const auto lm_img = qwen_tiny::lm("qwen2vl", kVocab);
    const auto lm_g = locus::gguf::GgufFile::parse(lm_img);
    const auto dit_g =
        locus::gguf::GgufFile::open(kDir + "golden/qwen_image_dit_synth.gguf");
    const auto vae_g =
        locus::gguf::GgufFile::open(kDir + "golden/qwen_vae_synth.gguf");
    const auto p = QwenImagePipeline::from_parts(lm_g, dit_g, vae_g, tok_g);

    QwenImagePipeline::Request req;
    req.prompt = "A red fox in fresh snow.";
    req.negative_prompt = " ";
    req.width = kWidth;
    req.height = kHeight;
    req.steps = kSteps;
    req.true_cfg_scale = kCfg;
    const auto [w, h] = p->output_size(req);
    REQUIRE(w == kWidth);
    REQUIRE(h == kHeight);
    const auto ref = read_f32(kGolden);
    REQUIRE(ref.size() == 2 + std::size_t{w} * h * 3);
    REQUIRE(h == static_cast<std::uint32_t>(ref[0]));
    REQUIRE(w == static_cast<std::uint32_t>(ref[1]));

    const std::uint32_t s = p->vae_scale();
    std::uint32_t state = 31337;
    auto noise = qwen_tiny::lcg(
        state, std::size_t{h / s / 2} * (w / s / 2) * p->channels());
    for (float& x : noise) x *= 4.0f;
    const auto& op = locus::backend::best_backend().ops;
    const auto img = p->generate_from(noise, req, op, {});
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
    // Measured on an M2 Pro: 2.1e-7 of the RMS, the worst value 8.9e-7.
    REQUIRE(err <= 1e-3 * norm);

    // A cancelled run returns nothing; the byte output has the size;
    // a size that floors to 0 is refused.
    CHECK(p->generate_from(noise, req, op, [] { return false; }).empty());
    std::uint32_t ow = 0, oh = 0;
    const auto rgb = p->generate(req, op, {}, ow, oh);
    CHECK(ow == w);
    CHECK(oh == h);
    CHECK(rgb.size() == std::size_t{w} * h * 3);
    req.width = 1;  // floors to 0 at any VAE scale
    REQUIRE_THROWS_AS(p->generate(req, op, {}, ow, oh),
                      std::invalid_argument);
}
