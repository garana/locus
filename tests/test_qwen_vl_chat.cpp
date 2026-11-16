#include "locus/image/qwen_vl_chat.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "qwen_tiny.hpp"

namespace {

const std::string kTokModel = std::string(LOCUS_SOURCE_DIR) +
                              "/tests/models/qwen2.5-0.5b-instruct-q8_0.gguf";
const std::string kGolden =
    std::string(LOCUS_SOURCE_DIR) + "/tests/golden/qwen_vl_chat.f32";

using Grid = std::pair<std::uint32_t, std::uint32_t>;

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

// chat_template.jinja with add_generation_prompt: the default system
// turn, each image as a vision span, no separator added between parts.
TEST_CASE("qwen vl chat text follows the chat template", "[qwen-vl]") {
    using locus::image::ChatTurn;
    using locus::image::kImagePlaceholder;
    const std::vector<ChatTurn> turns = {
        {"user", std::string(kImagePlaceholder) + "What is this?"},
        {"assistant", "A cat."},
        {"user", "And its color?"}};
    CHECK(locus::image::qwen_vl_chat_text(turns, std::nullopt) ==
          "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
          "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
          "What is this?<|im_end|>\n"
          "<|im_start|>assistant\nA cat.<|im_end|>\n"
          "<|im_start|>user\nAnd its color?<|im_end|>\n"
          "<|im_start|>assistant\n");
    CHECK(locus::image::qwen_vl_chat_text({turns.data() + 2, 1},
                                          std::string("Be brief.")) ==
          "<|im_start|>system\nBe brief.<|im_end|>\n"
          "<|im_start|>user\nAnd its color?<|im_end|>\n"
          "<|im_start|>assistant\n");
    REQUIRE_THROWS_AS(
        locus::image::qwen_vl_chat_text({turns.data(), 2}, std::nullopt),
        std::invalid_argument);
}

// One image: the same positions as mrope_positions (pinned to
// transformers' get_rope_index by the qwen mrope golden); two images:
// the second starts after the first's largest position + 1.
TEST_CASE("qwen vl chat ids expand pads and place M-RoPE positions",
          "[qwen-vl]") {
    if (!std::filesystem::exists(kTokModel)) {
        SKIP("qwen2.5-0.5b-instruct-q8_0.gguf absent (its tokenizer)");
    }
    const auto g = locus::gguf::GgufFile::open(kTokModel);
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    const auto pad = tok->encode("<|image_pad|>", false);
    REQUIRE(pad.size() == 1);
    using locus::image::ChatTurn;
    using locus::image::kImagePlaceholder;
    const std::string ph(kImagePlaceholder);

    const std::vector<ChatTurn> one = {{"user", ph + "Describe it."}};
    const auto text1 = locus::image::qwen_vl_chat_text(one, std::nullopt);
    const Grid g1[] = {{3, 5}};
    const auto p1 = locus::image::qwen_vl_chat_ids(*tok, text1, g1);
    REQUIRE(p1.image_at.size() == 1);
    const std::uint32_t at = p1.image_at[0];
    const auto n = static_cast<std::uint32_t>(p1.ids.size());
    for (std::uint32_t i = 0; i < 15; ++i) CHECK(p1.ids[at + i] == pad[0]);
    CHECK(p1.ids[at + 15] != pad[0]);
    const auto want = locus::model::mrope_positions(at, 3, 5, n - at - 15);
    CHECK(p1.pos == want);
    // The first generated token follows the last prompt position.
    CHECK(p1.next == want.back()[0] + 1);

    const std::vector<ChatTurn> two = {
        {"user", ph + "A" + ph + "B"}};
    const auto text2 = locus::image::qwen_vl_chat_text(two, std::nullopt);
    const Grid g2[] = {{2, 4}, {3, 1}};
    const auto p2 = locus::image::qwen_vl_chat_ids(*tok, text2, g2);
    REQUIRE(p2.image_at.size() == 2);
    const std::uint32_t a = p2.image_at[0], b = p2.image_at[1];
    // First image at s = a, then "<|vision_end|>A<|vision_start|>"
    // (b - a - 8 text tokens) from a + max(2, 4).
    CHECK(p2.pos[a] == locus::model::MropePos{a, a, a});
    CHECK(p2.pos[a + 7] == locus::model::MropePos{a, a + 1, a + 3});
    const std::uint32_t t0 = a + 4;
    CHECK(p2.pos[a + 8] == locus::model::MropePos{t0, t0, t0});
    const std::uint32_t s2 = t0 + (b - a - 8);
    CHECK(p2.pos[b] == locus::model::MropePos{s2, s2, s2});
    CHECK(p2.pos[b + 2] == locus::model::MropePos{s2, s2 + 2, s2});
    CHECK(p2.pos[b + 3] ==
          locus::model::MropePos{s2 + 3, s2 + 3, s2 + 3});

    // Pads and images must agree in count.
    REQUIRE_THROWS_AS(locus::image::qwen_vl_chat_ids(*tok, text2, g1),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(locus::image::qwen_vl_chat_ids(*tok, text1, g2),
                      std::invalid_argument);
}

// i#1092: transformers' Qwen2_5_VLForConditionalGeneration with its
// processor and chat template (scripts/image/make-qwen-vl-chat-golden.py)
// on the tiny Qwen2.5-VL with the real vocabulary: an 84 x 56 LCG image
// then "What is in the picture?", greedy. The prompt's last logits and
// three steps after it match, and so do the four greedy tokens.
TEST_CASE("qwen vl chat matches transformers on a tiny model",
          "[qwen-vl]") {
    if (!std::filesystem::exists(kTokModel)) {
        SKIP("qwen2.5-0.5b-instruct-q8_0.gguf absent (its tokenizer)");
    }
    constexpr std::uint64_t kVocab = 151936;
    constexpr std::uint32_t kW = 84, kH = 56;
    const auto tok_g = locus::gguf::GgufFile::open(kTokModel);
    const auto tok = locus::tok::tokenizer_from_gguf(tok_g);
    const auto lm_img = qwen_tiny::lm("qwen2vl", kVocab);
    const auto lm_g = locus::gguf::GgufFile::parse(lm_img);
    const auto lm = locus::model::TransformerModel::load(lm_g);
    // The processor's image_mean and image_std, as the real mmproj has.
    const auto v_img =
        qwen_tiny::mmproj(qwen_tiny::kE, "qwen2.5vl_merger", true);
    const auto v_g = locus::gguf::GgufFile::parse(v_img);
    const auto vision = locus::image::QwenVision::load(v_g);

    // The scripts' LCG image: (x + 0.25) * 510 clipped to bytes.
    std::uint32_t state = 777;
    const auto v = qwen_tiny::lcg(state, std::size_t{kW} * kH * 3);
    locus::image::Rgb8 img{kW, kH, {}};
    img.rgb.resize(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        // numpy's astype(uint8) truncates the clipped float.
        const float x = std::clamp((v[i] + 0.25f) * 510.0f, 0.0f, 255.0f);
        img.rgb[i] = static_cast<std::uint8_t>(x);
    }
    const auto& op = locus::backend::best_backend().ops;
    const locus::image::QwenVlImage im[] = {
        locus::image::qwen_vl_image(vision, img)};
    REQUIRE(im[0].gh == kH / 14);
    REQUIRE(im[0].gw == kW / 14);
    const std::uint32_t m = vision.merge_size();
    const Grid grid[] = {{im[0].gh / m, im[0].gw / m}};
    const std::vector<locus::image::ChatTurn> turns = {
        {"user",
         std::string(locus::image::kImagePlaceholder) +
             "What is in the picture?"}};
    const auto p = locus::image::qwen_vl_chat_ids(
        *tok, locus::image::qwen_vl_chat_text(turns, std::nullopt), grid);

    const auto ref = read_f32(kGolden);
    REQUIRE(ref.size() >= 2);
    const auto n_prompt = static_cast<std::size_t>(ref[0]);
    const auto steps = static_cast<std::size_t>(ref[1]);
    REQUIRE(ref.size() == 2 + steps + steps * kVocab);
    REQUIRE(p.ids.size() == n_prompt);

    auto cache = lm.make_cache();
    auto ws = lm.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    REQUIRE(cache.ensure_capacity(
        seq, static_cast<std::uint32_t>(n_prompt + steps)));
    std::vector<float> logits(kVocab);
    REQUIRE(locus::image::qwen_vl_prefill(lm, vision, p, im, cache, seq, ws,
                                          logits, op));
    for (std::size_t s = 0; s < steps; ++s) {
        CAPTURE(s);
        const float* want = ref.data() + 2 + steps + s * kVocab;
        double err = 0.0, norm = 0.0;
        for (std::size_t i = 0; i < kVocab; ++i) {
            const double d = double{logits[i]} - want[i];
            err += d * d;
            norm += double{want[i]} * want[i];
        }
        WARN("step " << s << ": error rms " << std::sqrt(err / kVocab)
                     << ", reference rms " << std::sqrt(norm / kVocab));
        // Measured on an M2 Pro: 2.3e-7 to 3.4e-7 of the RMS.
        CHECK(std::sqrt(err) <= 1e-4 * std::sqrt(norm));
        const auto greedy = static_cast<locus::tok::TokenId>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        REQUIRE(greedy == static_cast<locus::tok::TokenId>(ref[2 + s]));
        if (s + 1 < steps) {
            locus::image::qwen_vl_step(
                lm, greedy, p.next + static_cast<std::uint32_t>(s), cache,
                seq, ws, logits);
        }
    }
    cache.release(seq);
}
