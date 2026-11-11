#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "qwen_tiny.hpp"

namespace {

using locus::model::MropePos;
using locus::model::TransformerModel;
using qwen_tiny::kE;
using qwen_tiny::kImageId;
using qwen_tiny::kStartId;
using qwen_tiny::kVocab;
using qwen_tiny::lcg;

// Keep in sync with scripts/make-qwen-mrope-golden.py.
const std::vector<std::int32_t> kBefore = {3, 17, 42, 9, kStartId};
constexpr std::uint32_t kGh = 2, kGw = 3;
const std::vector<std::int32_t> kAfter = {5, 33, 21, 58};

/** The script's tiny language model (qwen_tiny.hpp) under `arch`. */
std::vector<std::byte> tiny_model(const std::string& arch = "qwen2vl") {
    return qwen_tiny::lm(arch);
}

/** The prompt's token ids: text, the image span, text. */
std::vector<locus::tok::TokenId> prompt_ids() {
    std::vector<locus::tok::TokenId> ids(kBefore.begin(), kBefore.end());
    ids.insert(ids.end(), kGh * kGw, kImageId);
    ids.insert(ids.end(), kAfter.begin(), kAfter.end());
    return ids;
}

/** Every position's logits of one forward_batch over ids. */
std::vector<float> run(const TransformerModel& m,
                       const std::vector<locus::tok::TokenId>& ids,
                       std::span<const float* const> embd,
                       std::span<const MropePos> pos) {
    auto cache = m.make_cache();
    auto ws = m.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    REQUIRE(cache.ensure_capacity(seq, ids.size()));
    std::vector<float> logits(ids.size() * kVocab);
    m.forward_batch(ids, cache, seq, ws, logits, true, embd, pos);
    cache.release(seq);
    return logits;
}

}  // namespace

// i#545: mrope_positions is transformers' get_rope_index for a prompt
// with one image (the golden's positions).
TEST_CASE("qwen2vl mrope positions match get_rope_index",
          "[qwen2vl][mrope]") {
    std::ifstream f(std::string(LOCUS_SOURCE_DIR) +
                        "/tests/golden/qwen_mrope.bin",
                    std::ios::binary);
    REQUIRE(f.good());
    const auto pos = locus::model::mrope_positions(
        static_cast<std::uint32_t>(kBefore.size()), kGh, kGw,
        static_cast<std::uint32_t>(kAfter.size()));
    const std::size_t n = prompt_ids().size();
    REQUIRE(pos.size() == n);
    std::vector<std::int32_t> ref(3 * n);
    f.read(reinterpret_cast<char*>(ref.data()),
           static_cast<std::streamsize>(ref.size() * 4));
    REQUIRE(f.good());
    for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t i = 0; i < n; ++i) {
            INFO("axis " << a << " token " << i);
            REQUIRE(static_cast<std::int32_t>(pos[i][a]) == ref[a * n + i]);
        }
    }
}

// i#545 VALUES: the tiny qwen2vl model on text, an image span of input
// embeddings with 3D positions, and text again, against transformers'
// Qwen2_5_VLForConditionalGeneration with the same weights
// (scripts/make-qwen-mrope-golden.py), every position's logits.
TEST_CASE("qwen2vl matches transformers with an image span",
          "[qwen2vl][mrope]") {
    std::ifstream f(std::string(LOCUS_SOURCE_DIR) +
                        "/tests/golden/qwen_mrope.bin",
                    std::ios::binary);
    REQUIRE(f.good());
    const auto ids = prompt_ids();
    const std::size_t n = ids.size();
    std::vector<float> ref(n * kVocab);
    f.seekg(static_cast<std::streamoff>(3 * n * 4));
    f.read(reinterpret_cast<char*>(ref.data()),
           static_cast<std::streamsize>(ref.size() * 4));
    REQUIRE(f.good());

    const auto img = tiny_model();
    const auto g = locus::gguf::GgufFile::parse(img);
    const auto m = TransformerModel::load(g);
    REQUIRE(m.hparams().rope_sections[1] == 3);
    std::uint32_t state = 777;
    auto e = lcg(state, std::size_t{kGh} * kGw * kE);
    for (float& x : e) x *= 4.0f;
    std::vector<const float*> embd(n, nullptr);
    for (std::uint32_t i = 0; i < kGh * kGw; ++i) {
        embd[kBefore.size() + i] = e.data() + std::size_t{i} * kE;
    }
    const auto pos = locus::model::mrope_positions(
        static_cast<std::uint32_t>(kBefore.size()), kGh, kGw,
        static_cast<std::uint32_t>(kAfter.size()));
    const auto got = run(m, ids, embd, pos);

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

    // Plain positions instead of 3D ones move the image span's logits:
    // the test above is not passing by ignoring them.
    const auto plain = run(m, ids, embd, {});
    double moved = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        moved = std::max(moved, std::abs(double{plain[i]} - got[i]));
    }
    REQUIRE(moved > 1e-3);
}

// A text token's three positions are equal, so qwen2vl on text equals
// qwen2 on the same weights, bit for bit, with 3D positions or without.
TEST_CASE("qwen2vl on text is qwen2", "[qwen2vl][mrope]") {
    const std::vector<locus::tok::TokenId> ids = {3, 17, 42, 9, 5, 33};
    const auto a = tiny_model("qwen2"), b = tiny_model();
    const auto ga = locus::gguf::GgufFile::parse(a);
    const auto gb = locus::gguf::GgufFile::parse(b);
    const auto qwen2 = TransformerModel::load(ga);
    const auto vl = TransformerModel::load(gb);
    const auto ref = run(qwen2, ids, {}, {});
    REQUIRE(run(vl, ids, {}, {}) == ref);
    const auto pos = locus::model::mrope_positions(
        static_cast<std::uint32_t>(ids.size()), 0, 0, 0);
    REQUIRE(run(vl, ids, {}, pos) == ref);
}

TEST_CASE("qwen2vl mrope positions are checked", "[qwen2vl][mrope]") {
    const auto a = tiny_model("qwen2"), b = tiny_model();
    const auto ga = locus::gguf::GgufFile::parse(a);
    const auto gb = locus::gguf::GgufFile::parse(b);
    const auto qwen2 = TransformerModel::load(ga);
    const auto vl = TransformerModel::load(gb);
    const std::vector<locus::tok::TokenId> ids = {3, 17, 42};
    const auto pos = locus::model::mrope_positions(3, 0, 0, 0);
    // An arch without M-RoPE sections, and a count that is off.
    REQUIRE_THROWS_AS(run(qwen2, ids, {}, pos), std::invalid_argument);
    const auto two = locus::model::mrope_positions(2, 0, 0, 0);
    REQUIRE_THROWS_AS(run(vl, ids, {}, two), std::invalid_argument);
}
