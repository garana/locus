#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"

namespace {

using locus::model::MropePos;
using locus::model::TransformerModel;
using Shape = std::vector<std::uint64_t>;

// Keep in sync with scripts/make-qwen-mrope-golden.py.
constexpr std::uint32_t kE = 64, kHeads = 4, kKv = 2, kFf = 96,
                        kLayers = 2, kVocab = 64;
constexpr std::int32_t kSections[] = {2, 3, 3, 0};
constexpr std::int32_t kImageId = 60, kStartId = 61;
const std::vector<std::int32_t> kBefore = {3, 17, 42, 9, kStartId};
constexpr std::uint32_t kGh = 2, kGw = 3;
const std::vector<std::int32_t> kAfter = {5, 33, 21, 58};

/** The script's LCG: n values, advancing state. */
std::vector<float> lcg(std::uint32_t& state, std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 0.5f;
    }
    return v;
}

/**
 * The script's tiny Qwen2.5-VL language model as a GGUF image under
 * `arch` (qwen2vl, or qwen2 without the rope sections): the script's
 * gguf_shapes, F32, filled from the LCG in sorted name order.
 */
std::vector<std::byte> tiny_model(const std::string& arch = "qwen2vl") {
    const std::uint64_t hd = kE / kHeads;
    std::map<std::string, Shape> shapes;
    shapes["token_embd.weight"] = {kVocab, kE};
    shapes["output_norm.weight"] = {kE};
    shapes["output.weight"] = {kVocab, kE};
    for (std::uint32_t i = 0; i < kLayers; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        shapes[b + "attn_norm.weight"] = {kE};
        shapes[b + "ffn_norm.weight"] = {kE};
        shapes[b + "attn_q.weight"] = {kE, kE};
        shapes[b + "attn_q.bias"] = {kE};
        for (const char* t : {"attn_k", "attn_v"}) {
            shapes[b + t + ".weight"] = {kKv * hd, kE};
            shapes[b + t + ".bias"] = {kKv * hd};
        }
        shapes[b + "attn_output.weight"] = {kE, kE};
        shapes[b + "ffn_gate.weight"] = {kFf, kE};
        shapes[b + "ffn_up.weight"] = {kFf, kE};
        shapes[b + "ffn_down.weight"] = {kE, kFf};
    }
    std::uint32_t state = 12345;
    std::map<std::string, std::vector<float>> data;
    for (const auto& [name, shape] : shapes) {
        std::uint64_t n = 1;
        for (auto d : shape) n *= d;
        data[name] = lcg(state, n);
    }

    const bool vl = arch == "qwen2vl";
    GgufBuilder g;
    g.header(data.size(), vl ? 10 : 9);
    g.kv_string("general.architecture", arch);
    g.kv_u32(arch + ".embedding_length", kE);
    g.kv_u32(arch + ".block_count", kLayers);
    g.kv_u32(arch + ".attention.head_count", kHeads);
    g.kv_u32(arch + ".attention.head_count_kv", kKv);
    g.kv_u32(arch + ".feed_forward_length", kFf);
    g.kv_u32(arch + ".context_length", 128);
    g.kv_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
    g.kv_f32(arch + ".rope.freq_base", 10000.0f);
    if (vl) g.kv_i32_array(arch + ".rope.dimension_sections", kSections);
    std::uint64_t offset = 0;
    for (const auto& [name, v] : data) {
        const Shape& s = shapes[name];
        const std::vector<std::uint64_t> dims(s.rbegin(), s.rend());
        g.tensor(name, dims, 0 /* F32 */, offset);
        offset = (offset + v.size() * 4 + 31) & ~31ull;
    }
    g.pad();
    for (const auto& [name, v] : data) {
        for (float x : v) g.f32(x);
        while (g.bytes().size() % 32 != 0) g.zeros(1);
    }
    auto span = g.bytes();
    return {span.begin(), span.end()};
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
