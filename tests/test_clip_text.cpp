#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/clip_text.hpp"
#include "locus/tok/clip_tokenizer.hpp"

using locus::image::ClipTextEncoder;
using locus::tok::ClipTokenizer;
using locus::tok::TokenId;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/flux/clip-l.gguf";

// The synthetic model of scripts/make-clip-goldens.py synth.
constexpr std::uint32_t kVocab = 40, kWidth = 16, kHeads = 4, kFfn = 24,
                        kLayers = 2, kContext = 10;

struct Tensor {
    std::string name;
    std::vector<std::uint64_t> shape;  ///< torch order: [out, in]
};

/** The script's tensor list, in its LCG order. */
std::vector<Tensor> synth_tensors() {
    const std::uint64_t w = kWidth, f = kFfn;
    const std::string m = "text_model.";
    std::vector<Tensor> t = {
        {m + "embeddings.token_embedding.weight", {kVocab, w}},
        {m + "embeddings.position_embedding.weight", {kContext, w}}};
    for (std::uint32_t n = 0; n < kLayers; ++n) {
        const std::string p = m + "encoder.layers." + std::to_string(n) + ".";
        t.push_back({p + "layer_norm1.weight", {w}});
        t.push_back({p + "layer_norm1.bias", {w}});
        for (const char* x : {"q", "k", "v", "out"}) {
            t.push_back({p + "self_attn." + x + "_proj.weight", {w, w}});
            t.push_back({p + "self_attn." + x + "_proj.bias", {w}});
        }
        t.push_back({p + "layer_norm2.weight", {w}});
        t.push_back({p + "layer_norm2.bias", {w}});
        t.push_back({p + "mlp.fc1.weight", {f, w}});
        t.push_back({p + "mlp.fc1.bias", {f}});
        t.push_back({p + "mlp.fc2.weight", {w, f}});
        t.push_back({p + "mlp.fc2.bias", {w}});
    }
    t.push_back({m + "final_layer_norm.weight", {w}});
    t.push_back({m + "final_layer_norm.bias", {w}});
    return t;
}

/** A clip_text GGUF image with the script's F32 weights. */
std::vector<std::byte> synth_clip() {
    const auto tensors = synth_tensors();
    std::vector<std::vector<float>> data;
    std::uint32_t lcg = 372;
    for (const auto& t : tensors) {
        std::size_t n = 1;
        for (auto d : t.shape) n *= d;
        std::vector<float> v(n);
        for (auto& x : v) {
            lcg = lcg * 1664525u + 1013904223u;
            x = (static_cast<float>(lcg >> 8) / 16777216.0f - 0.5f) * 0.5f;
        }
        data.push_back(std::move(v));
    }
    GgufBuilder g;
    g.header(tensors.size(), 9);
    g.kv_string("general.architecture", "clip_text");
    g.kv_u32("clip_text.embedding_length", kWidth);
    g.kv_u32("clip_text.attention.head_count", kHeads);
    g.kv_u32("clip_text.feed_forward_length", kFfn);
    g.kv_u32("clip_text.context_length", kContext);
    g.kv_u32("clip_text.block_count", kLayers);
    g.kv_u32("clip_text.config_eos_token_id", 2);
    g.kv_f32("clip_text.norm_epsilon", 1e-5f);
    g.kv_u32("tokenizer.ggml.eos_token_id", kVocab - 1);
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& s = tensors[i].shape;
        const std::vector<std::uint64_t> dims(s.rbegin(), s.rend());
        g.tensor(tensors[i].name, dims, 0 /* F32 */, offset);
        offset = (offset + data[i].size() * 4 + 31) & ~31ull;
    }
    g.pad();
    for (const auto& v : data) {
        for (float x : v) g.f32(x);
        while (g.bytes().size() % 32 != 0) g.zeros(1);
    }
    auto span = g.bytes();
    return {span.begin(), span.end()};
}

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

/**
 * @returns The largest difference between two rows of `width` floats,
 *     over the reference row's RMS.
 */
double row_error(const float* got, const float* ref, std::uint32_t width) {
    double ss = 0.0, worst = 0.0;
    for (std::uint32_t i = 0; i < width; ++i) {
        ss += double{ref[i]} * ref[i];
        worst = std::max(worst, std::fabs(double{got[i]} - ref[i]));
    }
    return worst / std::sqrt(ss / width);
}

}  // namespace

TEST_CASE("clip text encoder matches transformers on a tiny model",
          "[clip][image]") {
    const auto img = synth_clip();
    const auto g = locus::gguf::GgufFile::parse(img);
    const auto enc = ClipTextEncoder::load(g);
    const std::vector<TokenId> ids = {38, 5, 17, 3, 22, 9, 39, 39, 39, 39};
    const auto out = enc.encode(ids, locus::backend::best_backend().ops);
    const auto ref = read_f32(kDir + "golden/clip_synth.f32");
    REQUIRE(ref.size() == (kContext + 1) * kWidth);
    for (std::uint32_t t = 0; t < kContext; ++t) {
        INFO("row " << t);
        REQUIRE(row_error(out.hidden.data() + t * kWidth,
                          ref.data() + t * kWidth, kWidth) < 1e-4);
    }
    REQUIRE(row_error(out.pooled.data(), ref.data() + kContext * kWidth,
                      kWidth) < 1e-4);
}

TEST_CASE("clip text encoder rejects bad inputs", "[clip][image]") {
    const auto img = synth_clip();
    const auto g = locus::gguf::GgufFile::parse(img);
    const auto enc = ClipTextEncoder::load(g);
    const auto& op = locus::backend::best_backend().ops;
    REQUIRE_THROWS_AS(enc.encode({}, op), std::invalid_argument);
    const std::vector<TokenId> too_long(kContext + 1, 1);
    REQUIRE_THROWS_AS(enc.encode(too_long, op), std::invalid_argument);
    const std::vector<TokenId> bad = {1, kVocab};
    REQUIRE_THROWS_AS(enc.encode(bad, op), std::invalid_argument);
}

// FLUX.1's CLIP-L tokenizer against CLIPTokenizer: plain prompts,
// contractions, spacing, digits, punctuation runs, Latin-1, Greek,
// Cyrillic, emoji, the special tokens in the text, and truncation.
TEST_CASE("clip tokenizer matches CLIPTokenizer", "[clip][tokenizer]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("no " << kModel << " (scripts/convert-flux.py --clip-out)");
    }
    const auto g = locus::gguf::GgufFile::open(kModel);
    const auto tok = ClipTokenizer::from_gguf(g);
    std::ifstream f(kDir + "golden/clip_tokenizer.jsonl");
    REQUIRE(f.good());
    std::string line;
    int cases = 0;
    while (std::getline(f, line)) {
        const auto j = nlohmann::json::parse(line);
        const std::string text = j["text"];
        const std::vector<TokenId> want = j["ids"];
        INFO("text: " << text);
        REQUIRE(tok.encode(text, 77) == want);
        ++cases;
    }
    REQUIRE(cases >= 10);
}

TEST_CASE("clip-l pooled output matches transformers", "[clip][image]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("no " << kModel << " (scripts/convert-flux.py --clip-out)");
    }
    const auto g = locus::gguf::GgufFile::open(kModel);
    const auto tok = ClipTokenizer::from_gguf(g);
    const auto enc = ClipTextEncoder::load(g);
    std::ifstream f(kDir + "golden/flux/clip_pooled.json");
    REQUIRE(f.good());
    const auto j = nlohmann::json::parse(f);
    const auto ref = read_f32(kDir + "golden/flux/clip_pooled.f32");
    const std::uint32_t w = j["width"];
    REQUIRE(w == enc.width());
    std::size_t c = 0;
    for (const auto& cs : j["cases"]) {
        const std::string text = cs["text"];
        const std::vector<TokenId> want = cs["ids"];
        INFO("text: " << text);
        const auto ids = tok.encode(text, enc.context());
        REQUIRE(ids == want);
        const auto out = enc.encode(ids, locus::backend::best_backend().ops);
        REQUIRE(row_error(out.pooled.data(), ref.data() + c * w, w) < 1e-3);
        ++c;
    }
    REQUIRE(c == 4);
}
