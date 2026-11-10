#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/embed/bert.hpp"
#include "locus/embed/embedding_model.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::embed::BertEncoder;
using locus::embed::Pooling;
using locus::tok::TokenId;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/bge-small-en-v1.5-q8_0.gguf";

constexpr std::uint32_t kVocab = 30, kWidth = 16, kHeads = 4, kFfn = 24,
                        kLayers = 2, kContext = 12;

struct Tensor {
    std::string name;
    std::vector<std::uint64_t> shape;  ///< [out, in]
    std::vector<float> data;
};

std::vector<Tensor> synth_tensors() {
    const std::uint64_t w = kWidth, f = kFfn;
    std::vector<Tensor> t = {{"token_embd.weight", {kVocab, w}, {}},
                             {"position_embd.weight", {kContext, w}, {}},
                             {"token_types.weight", {2, w}, {}},
                             {"token_embd_norm.weight", {w}, {}},
                             {"token_embd_norm.bias", {w}, {}}};
    for (std::uint32_t n = 0; n < kLayers; ++n) {
        const std::string p = "blk." + std::to_string(n) + ".";
        for (const char* x : {"attn_q", "attn_k", "attn_v", "attn_output"}) {
            t.push_back({p + x + ".weight", {w, w}, {}});
            t.push_back({p + x + ".bias", {w}, {}});
        }
        t.push_back({p + "attn_output_norm.weight", {w}, {}});
        t.push_back({p + "attn_output_norm.bias", {w}, {}});
        t.push_back({p + "ffn_up.weight", {f, w}, {}});
        t.push_back({p + "ffn_up.bias", {f}, {}});
        t.push_back({p + "ffn_down.weight", {w, f}, {}});
        t.push_back({p + "ffn_down.bias", {w}, {}});
        t.push_back({p + "layer_output_norm.weight", {w}, {}});
        t.push_back({p + "layer_output_norm.bias", {w}, {}});
    }
    std::uint32_t lcg = 502;
    for (auto& x : t) {
        std::size_t n = 1;
        for (auto d : x.shape) n *= d;
        x.data.resize(n);
        const bool gain = x.name.find("norm.weight") != std::string::npos;
        for (auto& v : x.data) {
            lcg = lcg * 1664525u + 1013904223u;
            const float r = static_cast<float>(lcg >> 8) / 16777216.0f - 0.5f;
            v = gain ? 1.0f + 0.2f * r : r;
        }
    }
    return t;
}

/** A bert GGUF image with the synthetic F32 weights. */
std::vector<std::byte> synth_bert(const std::vector<Tensor>& tensors,
                                  std::uint32_t pooling) {
    GgufBuilder g;
    g.header(tensors.size(), 8);
    g.kv_string("general.architecture", "bert");
    g.kv_u32("bert.embedding_length", kWidth);
    g.kv_u32("bert.attention.head_count", kHeads);
    g.kv_u32("bert.feed_forward_length", kFfn);
    g.kv_u32("bert.context_length", kContext);
    g.kv_u32("bert.block_count", kLayers);
    g.kv_f32("bert.attention.layer_norm_epsilon", 1e-12f);
    g.kv_u32("bert.pooling_type", pooling);
    std::uint64_t offset = 0;
    for (const auto& t : tensors) {
        const std::vector<std::uint64_t> dims(t.shape.rbegin(),
                                              t.shape.rend());
        g.tensor(t.name, dims, 0 /* F32 */, offset);
        offset = (offset + t.data.size() * 4 + 31) & ~31ull;
    }
    g.pad();
    for (const auto& t : tensors) {
        for (float x : t.data) g.f32(x);
        while (g.bytes().size() % 32 != 0) g.zeros(1);
    }
    auto span = g.bytes();
    return {span.begin(), span.end()};
}

/** A plain reference of the encoder: one sequence, doubles. */
std::vector<double> reference(const std::vector<Tensor>& tensors,
                              const std::vector<TokenId>& ids) {
    const auto get = [&](const std::string& name) -> const auto& {
        for (const auto& t : tensors) {
            if (t.name == name) return t.data;
        }
        FAIL("no tensor " << name);
        return tensors[0].data;
    };
    const std::size_t n = ids.size(), w = kWidth, hd = kWidth / kHeads;
    const auto norm = [&](std::vector<double>& x, const std::string& p) {
        const auto& g = get(p + ".weight");
        const auto& b = get(p + ".bias");
        for (std::size_t t = 0; t < n; ++t) {
            double m = 0, v = 0;
            for (std::size_t i = 0; i < w; ++i) m += x[t * w + i];
            m /= w;
            for (std::size_t i = 0; i < w; ++i) {
                v += (x[t * w + i] - m) * (x[t * w + i] - m);
            }
            v /= w;
            for (std::size_t i = 0; i < w; ++i) {
                x[t * w + i] = (x[t * w + i] - m) / std::sqrt(v + 1e-12) * g[i] +
                               b[i];
            }
        }
    };
    const auto lin = [&](const std::vector<double>& x, std::size_t in,
                         const std::string& p, std::size_t out) {
        const auto& W = get(p + ".weight");
        const auto& b = get(p + ".bias");
        std::vector<double> y(n * out);
        for (std::size_t t = 0; t < n; ++t) {
            for (std::size_t r = 0; r < out; ++r) {
                double s = b[r];
                for (std::size_t c = 0; c < in; ++c) {
                    s += W[r * in + c] * x[t * in + c];
                }
                y[t * out + r] = s;
            }
        }
        return y;
    };
    std::vector<double> x(n * w);
    const auto& te = get("token_embd.weight");
    const auto& pe = get("position_embd.weight");
    const auto& ty = get("token_types.weight");
    for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t i = 0; i < w; ++i) {
            x[t * w + i] = te[static_cast<std::size_t>(ids[t]) * w + i] +
                           pe[t * w + i] + ty[i];
        }
    }
    norm(x, "token_embd_norm");
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const auto q = lin(x, w, p + "attn_q", w);
        const auto k = lin(x, w, p + "attn_k", w);
        const auto v = lin(x, w, p + "attn_v", w);
        std::vector<double> a(n * w, 0.0);
        for (std::size_t h = 0; h < kHeads; ++h) {
            for (std::size_t i = 0; i < n; ++i) {
                std::vector<double> s(n);
                double mx = -1e300, sum = 0;
                for (std::size_t j = 0; j < n; ++j) {
                    double d = 0;
                    for (std::size_t e = 0; e < hd; ++e) {
                        d += q[i * w + h * hd + e] * k[j * w + h * hd + e];
                    }
                    s[j] = d / std::sqrt(static_cast<double>(hd));
                    mx = std::max(mx, s[j]);
                }
                for (auto& z : s) sum += (z = std::exp(z - mx));
                for (std::size_t j = 0; j < n; ++j) {
                    for (std::size_t e = 0; e < hd; ++e) {
                        a[i * w + h * hd + e] +=
                            s[j] / sum * v[j * w + h * hd + e];
                    }
                }
            }
        }
        const auto o = lin(a, w, p + "attn_output", w);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];
        norm(x, p + "attn_output_norm");
        auto f = lin(x, w, p + "ffn_up", kFfn);
        for (auto& z : f) z = 0.5 * z * (1.0 + std::erf(z / std::sqrt(2.0)));
        const auto d = lin(f, kFfn, p + "ffn_down", w);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += d[i];
        norm(x, p + "layer_output_norm");
    }
    return x;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        ab += double{a[i]} * b[i];
        aa += double{a[i]} * a[i];
        bb += double{b[i]} * b[i];
    }
    return ab / std::sqrt(aa * bb);
}

}  // namespace

TEST_CASE("bert encoder matches a plain reference on a tiny model",
          "[bert]") {
    const auto tensors = synth_tensors();
    const auto img = synth_bert(tensors, 2);
    const auto g = locus::gguf::GgufFile::parse(img);
    const auto enc = BertEncoder::load(g);
    REQUIRE(enc.width() == kWidth);
    REQUIRE(enc.context() == kContext);
    REQUIRE(enc.pooling() == Pooling::kCls);
    // Three sequences of different lengths in one batch: attention must
    // stay within each one.
    const std::vector<std::vector<TokenId>> batch = {
        {1, 7, 12, 29, 2}, {1, 3, 2}, {1, 5, 5, 8, 9, 10, 11, 4, 0, 2}};
    const auto& op = locus::backend::best_backend().ops;
    const auto out = enc.encode(batch, op);
    REQUIRE(out.size() == batch.size());
    for (std::size_t s = 0; s < batch.size(); ++s) {
        const auto ref = reference(tensors, batch[s]);
        REQUIRE(out[s].size() == ref.size());
        double worst = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            worst = std::max(worst, std::fabs(out[s][i] - ref[i]));
        }
        INFO("sequence " << s);
        REQUIRE(worst < 1e-4);
    }
    // embed() is CLS pooling, unit length.
    const auto e = enc.embed(batch, op);
    const auto ref0 = reference(tensors, batch[0]);
    double nn = 0;
    for (std::uint32_t i = 0; i < kWidth; ++i) nn += ref0[i] * ref0[i];
    for (std::uint32_t i = 0; i < kWidth; ++i) {
        REQUIRE(std::fabs(e[0][i] - ref0[i] / std::sqrt(nn)) < 1e-4);
    }
}

TEST_CASE("bert encoder rejects bad input and bad models", "[bert]") {
    const auto tensors = synth_tensors();
    const auto& op = locus::backend::best_backend().ops;
    {
        const auto img = synth_bert(tensors, 2);
        const auto g = locus::gguf::GgufFile::parse(img);
        const auto enc = BertEncoder::load(g);
        const std::vector<std::vector<TokenId>> empty = {{}};
        REQUIRE_THROWS_AS(enc.encode(empty, op), std::invalid_argument);
        const std::vector<std::vector<TokenId>> big = {
            std::vector<TokenId>(kContext + 1, 1)};
        REQUIRE_THROWS_AS(enc.encode(big, op), std::invalid_argument);
        const std::vector<std::vector<TokenId>> oov = {{1, kVocab}};
        REQUIRE_THROWS_AS(enc.encode(oov, op), std::invalid_argument);
        REQUIRE(enc.encode({}, op).empty());
    }
    for (const std::uint32_t pooling : {0u, 4u, 9u}) {
        const auto img = synth_bert(tensors, pooling);
        const auto g = locus::gguf::GgufFile::parse(img);
        INFO("pooling " << pooling);
        REQUIRE_THROWS_AS(BertEncoder::load(g), locus::gguf::Error);
    }
    auto missing = tensors;
    missing.pop_back();
    const auto img = synth_bert(missing, 2);
    const auto g = locus::gguf::GgufFile::parse(img);
    REQUIRE_THROWS_AS(BertEncoder::load(g), locus::gguf::Error);
}

TEST_CASE("bert pooling averages or picks a token then normalizes",
          "[bert]") {
    // Two tokens of width 2.
    const std::vector<float> s = {3, 4, 1, 0};
    REQUIRE(BertEncoder::pool(s, 2, 2, Pooling::kCls) ==
            std::vector<float>{0.6f, 0.8f});
    REQUIRE(BertEncoder::pool(s, 2, 2, Pooling::kLast) ==
            std::vector<float>{1.0f, 0.0f});
    const auto m = BertEncoder::pool(s, 2, 2, Pooling::kMean);  // (2, 2)
    REQUIRE(std::fabs(m[0] - std::sqrt(0.5f)) < 1e-6f);
    REQUIRE(std::fabs(m[1] - std::sqrt(0.5f)) < 1e-6f);
    const std::vector<float> z = {0, 0};
    REQUIRE(BertEncoder::pool(z, 1, 2, Pooling::kCls) == z);
    REQUIRE_THROWS_AS(BertEncoder::pool(s, 2, 2, Pooling::kRank),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(BertEncoder::pool(s, 3, 2, Pooling::kCls),
                      std::invalid_argument);
}

TEST_CASE("bge-small embeddings match llama.cpp and transformers",
          "[bert][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("bge-small-en-v1.5-q8_0.gguf absent; see "
             "scripts/make-bert-embd-golden.py");
    }
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    const auto enc = BertEncoder::load(g);
    REQUIRE(enc.width() == 384);
    REQUIRE(enc.pooling() == Pooling::kCls);
    std::ifstream f(kDir + "golden/bge_small_embd.json");
    REQUIRE(f.good());
    const auto j = nlohmann::json::parse(f);
    const auto texts = j["texts"].get<std::vector<std::string>>();
    const auto want = j["embeddings"].get<std::vector<std::vector<float>>>();
    REQUIRE(texts.size() == 7);
    std::ifstream fh(kDir + "golden/bge_small_embd_hf.json");
    REQUIRE(fh.good());
    const auto hf = nlohmann::json::parse(fh)["embeddings"]
                        .get<std::vector<std::vector<float>>>();
    std::vector<std::vector<TokenId>> batch;
    for (const auto& t : texts) batch.push_back(tok->encode(t, true));
    const auto& op = locus::backend::best_backend().ops;
    // All in one batch, and each alone: both match.
    const auto got = enc.embed(batch, op);
    for (std::size_t i = 0; i < texts.size(); ++i) {
        INFO("text " << texts[i]);
        const auto one = enc.embed(
            std::span<const std::vector<TokenId>>(&batch[i], 1), op);
        // llama.cpp rounds the activations to 8 bits before each Q8_0
        // product, locus keeps them in f32: against the original
        // weights in f32 (transformers), locus is the nearer of the
        // two (0.99989 to 0.99995, llama.cpp 0.99971 to 0.99989).
        const double c = cosine(got[i], want[i]);
        const double ch = cosine(got[i], hf[i]);
        const double lh = cosine(want[i], hf[i]);
        UNSCOPED_INFO("cosine llama " << c << " hf " << ch << " llama-hf "
                                      << lh);
        REQUIRE(c >= 0.9998);
        REQUIRE(ch >= 0.9998);
        REQUIRE(ch >= lh);
        REQUIRE(cosine(one[0], got[i]) >= 0.99999);
    }
    // The question is nearer its answer than an unrelated sentence.
    REQUIRE(cosine(got[2], got[3]) > cosine(got[2], got[4]));
}

TEST_CASE("embedding model groups inputs into passes", "[bert][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("bge-small-en-v1.5-q8_0.gguf absent");
    }
    const auto m = locus::embed::EmbeddingModel::load(kModel);
    REQUIRE(m->name() == "bge-small-en-v1.5-q8_0.gguf");
    REQUIRE(m->context() == 512);
    // Ten inputs of 502 tokens: 5020 tokens, more than one pass holds.
    std::string text;
    for (int i = 0; i < 500; ++i) text += i % 2 ? "cat " : "dog ";
    std::vector<std::string> inputs(10, text);
    inputs[3] = "a short one";
    const auto& op = locus::backend::best_backend().ops;
    std::size_t tokens = 0;
    const auto all = m->embed(inputs, op, &tokens);
    REQUIRE(tokens == 9 * 502 + 5);
    REQUIRE(all.size() == 10);
    const auto one = m->embed({inputs[9]}, op);
    REQUIRE(cosine(all[9], one[0]) >= 0.99999);
    REQUIRE(cosine(all[0], all[9]) >= 0.99999);
    const auto short1 = m->embed({inputs[3]}, op);
    REQUIRE(cosine(all[3], short1[0]) >= 0.99999);
    inputs.push_back(text + text);
    REQUIRE_THROWS_WITH(m->embed(inputs, op),
                        Catch::Matchers::ContainsSubstring(
                            "input 10 has 1002 tokens"));
    REQUIRE_THROWS_AS(locus::embed::EmbeddingModel::load(
                          kDir + "golden/bert-vocab.gguf"),
                      locus::gguf::Error);
}
