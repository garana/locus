#include "locus/image/qwen_edit_text.hpp"

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
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"
#include "qwen_tiny.hpp"

namespace {

using locus::image::QwenVision;
using locus::model::TransformerModel;

// Keep in sync with scripts/make-qwen-edit-text-golden.py.
constexpr std::uint32_t kGh = 4, kGw = 6, kDrop = 64;
const std::vector<std::int32_t> kAfter = {5, 33, 21, 58, 12};

/** The script's prompt: 70 text ids and <|vision_start|>, the image
 *  span, 5 text ids. */
std::vector<locus::tok::TokenId> prompt_ids() {
    std::vector<locus::tok::TokenId> ids;
    for (int i = 0; i < 70; ++i) ids.push_back((7 * i + 3) % 60);
    ids.push_back(qwen_tiny::kStartId);
    ids.insert(ids.end(), kGh * kGw / 4, qwen_tiny::kImageId);
    ids.insert(ids.end(), kAfter.begin(), kAfter.end());
    return ids;
}

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> c((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(c.size() / 4);
    std::memcpy(v.data(), c.data(), v.size() * 4);
    return v;
}

}  // namespace

// i#574: the edit template with an instruction and a 4 x 6 patch image
// through the Qwen2 tokenizer (Qwen2.5's, from a Qwen2.5 GGUF) gives
// the reference processor's ids (scripts/make-qwen-edit-text-golden.py
// ids: Qwen/Qwen-Image-Edit's processor at ac7f931, the template read
// from diffusers), the image span where the processor put it.
TEST_CASE("qwen edit prompt matches the reference processor's ids",
          "[qwen-edit][image]") {
    const std::string model = std::string(LOCUS_SOURCE_DIR) +
                              "/tests/models/qwen2.5-0.5b-instruct-q8_0.gguf";
    if (!std::filesystem::exists(model)) {
        SKIP("qwen2.5-0.5b-instruct-q8_0.gguf absent (its tokenizer)");
    }
    std::ifstream f(std::string(LOCUS_SOURCE_DIR) +
                        "/tests/golden/qwen_edit_prompt_ids.i32",
                    std::ios::binary);
    std::vector<char> c((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<std::int32_t> want(c.size() / 4);
    std::memcpy(want.data(), c.data(), want.size() * 4);
    REQUIRE(want.size() == 89);

    const auto g = locus::gguf::GgufFile::open(model);
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    const auto p = locus::image::qwen_edit_prompt(
        *tok, "Make the sky purple, and add a small red kite.", 6);
    CHECK(p.image_at == 65);
    CHECK(p.n_image == 6);
    REQUIRE(p.ids.size() == want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
        INFO("position " << i);
        CHECK(p.ids[i] == want[i]);
    }
    REQUIRE_THROWS_AS(locus::image::qwen_edit_prompt(*tok, "x", 0),
                      std::invalid_argument);
}

// i#574: the whole conditioning step on a tiny Qwen2.5-VL (the tower
// merging to the model's width, the qwen2vl language model) against
// transformers' Qwen2_5_VLForConditionalGeneration: the image span's
// rows replaced by the tower's output, M-RoPE positions, hidden_states
// [-1] after the final norm, the first 64 positions dropped.
TEST_CASE("qwen edit text embeds match transformers on a tiny model",
          "[qwen-edit][image]") {
    const auto ref = read_f32(std::string(LOCUS_SOURCE_DIR) +
                              "/tests/golden/qwen_edit_text_synth.f32");
    const auto ids = prompt_ids();
    REQUIRE(ref.size() == (ids.size() - kDrop) * qwen_tiny::kE);

    const auto lm_img = qwen_tiny::lm();
    const auto lm_g = locus::gguf::GgufFile::parse(lm_img);
    const auto lm = TransformerModel::load(lm_g);
    const auto v_img = qwen_tiny::mmproj(qwen_tiny::kE);
    const auto v_g = locus::gguf::GgufFile::parse(v_img);
    const auto v = QwenVision::load(v_g);

    std::uint32_t state = 777;
    auto px = qwen_tiny::lcg(state, std::size_t{kGh} * kGw * v.patch_dim());
    for (float& x : px) x *= 4.0f;
    const auto& op = locus::backend::best_backend().ops;
    const auto got = locus::image::qwen_edit_text_embeds(
        lm, v, ids, 71, px, kGh, kGw, kDrop, op);
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
    REQUIRE(err <= 1e-5 * norm);

    // Sizes that do not match are refused.
    REQUIRE_THROWS_AS(locus::image::qwen_edit_text_embeds(
                          lm, v, ids, 0, px, kGh, kGw, kDrop, op),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(locus::image::qwen_edit_text_embeds(
                          lm, v, ids, 71, px, 3, kGw, kDrop, op),
                      std::invalid_argument);
    const auto narrow_img = qwen_tiny::mmproj(24);
    const auto narrow_g = locus::gguf::GgufFile::parse(narrow_img);
    const auto narrow = QwenVision::load(narrow_g);
    REQUIRE_THROWS_AS(locus::image::qwen_edit_text_embeds(
                          lm, narrow, ids, 71, px, kGh, kGw, kDrop, op),
                      std::invalid_argument);

    // i#671: the overload on the tower's rows, encoded once, gives the
    // same bits; rows that do not match the grid are refused.
    const std::uint32_t mg = v.merge_size();
    const auto rows = v.encode(px, kGh, kGw, op);
    CHECK(locus::image::qwen_edit_text_embeds(lm, rows, ids, 71, kGh / mg,
                                              kGw / mg, kDrop) == got);
    REQUIRE_THROWS_AS(
        locus::image::qwen_edit_text_embeds(
            lm, std::span<const float>(rows).first(rows.size() - 1), ids,
            71, kGh / mg, kGw / mg, kDrop),
        std::invalid_argument);
}
