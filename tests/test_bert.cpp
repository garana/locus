#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
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

/** A bert (or nomic-bert, with rope.freq_base 1000) GGUF image with
 *  the synthetic F32 weights. */
std::vector<std::byte> synth_bert(const std::vector<Tensor>& tensors,
                                  std::uint32_t pooling,
                                  const std::string& arch = "bert") {
    const bool nomic = arch == "nomic-bert";
    const std::string p = arch + ".";
    GgufBuilder g;
    g.header(tensors.size(), nomic ? 9 : 8);
    g.kv_string("general.architecture", arch);
    g.kv_u32(p + "embedding_length", kWidth);
    g.kv_u32(p + "attention.head_count", kHeads);
    g.kv_u32(p + "feed_forward_length", kFfn);
    g.kv_u32(p + "context_length", kContext);
    g.kv_u32(p + "block_count", kLayers);
    g.kv_f32(p + "attention.layer_norm_epsilon", 1e-12f);
    g.kv_u32(p + "pooling_type", pooling);
    if (nomic) g.kv_f32(p + "rope.freq_base", 1000.0f);
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

TEST_CASE("nomic-bert encoder matches a plain reference on a tiny model",
          "[bert]") {
    // No position table and no biases; fused qkv, NEOX rope (base
    // 1000), down * (silu(gate x) * (up x)).
    const std::uint64_t w = kWidth, f = kFfn;
    std::vector<Tensor> t = {{"token_embd.weight", {kVocab, w}, {}},
                             {"token_types.weight", {2, w}, {}},
                             {"token_embd_norm.weight", {w}, {}},
                             {"token_embd_norm.bias", {w}, {}}};
    for (std::uint32_t n = 0; n < kLayers; ++n) {
        const std::string p = "blk." + std::to_string(n) + ".";
        t.push_back({p + "attn_qkv.weight", {3 * w, w}, {}});
        t.push_back({p + "attn_output.weight", {w, w}, {}});
        t.push_back({p + "attn_output_norm.weight", {w}, {}});
        t.push_back({p + "attn_output_norm.bias", {w}, {}});
        t.push_back({p + "ffn_up.weight", {f, w}, {}});
        t.push_back({p + "ffn_gate.weight", {f, w}, {}});
        t.push_back({p + "ffn_down.weight", {w, f}, {}});
        t.push_back({p + "layer_output_norm.weight", {w}, {}});
        t.push_back({p + "layer_output_norm.bias", {w}, {}});
    }
    std::uint32_t lcg = 1502;
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
    const auto get = [&](const std::string& name) -> const auto& {
        for (const auto& x : t) {
            if (x.name == name) return x.data;
        }
        FAIL("no tensor " << name);
        return t[0].data;
    };
    const auto reference = [&](const std::vector<TokenId>& ids) {
        const std::size_t n = ids.size(), hd = kWidth / kHeads;
        const auto norm = [&](std::vector<double>& x, const std::string& p) {
            const auto& g = get(p + ".weight");
            const auto& b = get(p + ".bias");
            for (std::size_t r = 0; r < n; ++r) {
                double m = 0, v = 0;
                for (std::size_t i = 0; i < w; ++i) m += x[r * w + i];
                m /= w;
                for (std::size_t i = 0; i < w; ++i) {
                    v += (x[r * w + i] - m) * (x[r * w + i] - m);
                }
                v /= w;
                for (std::size_t i = 0; i < w; ++i) {
                    x[r * w + i] =
                        (x[r * w + i] - m) / std::sqrt(v + 1e-12) * g[i] + b[i];
                }
            }
        };
        const auto lin = [&](const std::vector<double>& x, std::size_t in,
                             const std::string& name, std::size_t out) {
            const auto& W = get(name);
            std::vector<double> y(n * out, 0.0);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t o = 0; o < out; ++o) {
                    double s = 0;
                    for (std::size_t c = 0; c < in; ++c) {
                        s += W[o * in + c] * x[r * in + c];
                    }
                    y[r * out + o] = s;
                }
            }
            return y;
        };
        // NEOX rope on one head's row at position pos.
        const auto rope = [&](double* h, std::size_t pos) {
            const std::size_t half = hd / 2;
            for (std::size_t i = 0; i < half; ++i) {
                const double a =
                    pos * std::pow(1000.0, -2.0 * static_cast<double>(i) / hd);
                const double x0 = h[i], x1 = h[i + half];
                h[i] = x0 * std::cos(a) - x1 * std::sin(a);
                h[i + half] = x0 * std::sin(a) + x1 * std::cos(a);
            }
        };
        std::vector<double> x(n * w);
        const auto& te = get("token_embd.weight");
        const auto& ty = get("token_types.weight");
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t i = 0; i < w; ++i) {
                x[r * w + i] = te[static_cast<std::size_t>(ids[r]) * w + i] +
                               ty[i];
            }
        }
        norm(x, "token_embd_norm");
        for (std::uint32_t l = 0; l < kLayers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const auto qkv = lin(x, w, p + "attn_qkv.weight", 3 * w);
            std::vector<double> q(n * w), k(n * w), v(n * w);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t i = 0; i < w; ++i) {
                    q[r * w + i] = qkv[r * 3 * w + i];
                    k[r * w + i] = qkv[r * 3 * w + w + i];
                    v[r * w + i] = qkv[r * 3 * w + 2 * w + i];
                }
                for (std::size_t h = 0; h < kHeads; ++h) {
                    rope(&q[r * w + h * hd], r);
                    rope(&k[r * w + h * hd], r);
                }
            }
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
            const auto o = lin(a, w, p + "attn_output.weight", w);
            for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];
            norm(x, p + "attn_output_norm");
            const auto up = lin(x, w, p + "ffn_up.weight", f);
            auto gt = lin(x, w, p + "ffn_gate.weight", f);
            for (std::size_t i = 0; i < gt.size(); ++i) {
                gt[i] = gt[i] / (1.0 + std::exp(-gt[i])) * up[i];
            }
            const auto d = lin(gt, f, p + "ffn_down.weight", w);
            for (std::size_t i = 0; i < x.size(); ++i) x[i] += d[i];
            norm(x, p + "layer_output_norm");
        }
        return x;
    };
    const auto img = synth_bert(t, 1, "nomic-bert");
    const auto g = locus::gguf::GgufFile::parse(img);
    const auto enc = BertEncoder::load(g);
    REQUIRE(enc.pooling() == Pooling::kMean);
    const std::vector<std::vector<TokenId>> batch = {
        {1, 7, 12, 29, 2}, {1, 3, 2}, {1, 5, 5, 8, 9, 10, 11, 4, 0, 2}};
    const auto out = enc.encode(batch, locus::backend::best_backend().ops);
    for (std::size_t s = 0; s < batch.size(); ++s) {
        const auto ref = reference(batch[s]);
        REQUIRE(out[s].size() == ref.size());
        double worst = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            worst = std::max(worst, std::fabs(out[s][i] - ref[i]));
        }
        INFO("sequence " << s);
        REQUIRE(worst < 1e-4);
    }
    // A position-sensitive check: the same tokens in another order give
    // other states (rope is on).
    const std::vector<std::vector<TokenId>> swapped = {{1, 29, 12, 7, 2}};
    const auto other = enc.encode(swapped, locus::backend::best_backend().ops);
    REQUIRE(std::fabs(other[0][0] - out[0][0]) > 1e-6);
}

TEST_CASE("bert reranker head scores the CLS state", "[bert][rerank]") {
    // The synthetic model plus a RoBERTa head with two labels:
    // logits = cls.output * tanh(cls * x0 + b) + b.
    auto tensors = synth_tensors();
    std::vector<Tensor> head = {{"cls.weight", {kWidth, kWidth}, {}},
                                {"cls.bias", {kWidth}, {}},
                                {"cls.output.weight", {2, kWidth}, {}},
                                {"cls.output.bias", {2}, {}}};
    std::uint32_t lcg = 503;
    for (auto& t : head) {
        std::size_t n = 1;
        for (auto d : t.shape) n *= d;
        t.data.resize(n);
        for (auto& v : t.data) {
            lcg = lcg * 1664525u + 1013904223u;
            v = static_cast<float>(lcg >> 8) / 16777216.0f - 0.5f;
        }
        tensors.push_back(t);
    }
    const auto img = synth_bert(tensors, 4);
    const auto g = locus::gguf::GgufFile::parse(img);
    const auto enc = BertEncoder::load(g);
    REQUIRE(enc.pooling() == Pooling::kRank);
    REQUIRE(enc.labels() == 2);
    const std::vector<std::vector<TokenId>> batch = {{1, 7, 12, 2},
                                                     {1, 3, 9, 9, 4, 2}};
    const auto& op = locus::backend::best_backend().ops;
    const auto got = enc.rank(batch, op);
    REQUIRE(got.size() == 2);
    for (std::size_t s = 0; s < batch.size(); ++s) {
        const auto x = reference(tensors, batch[s]);
        std::vector<double> h(kWidth);
        for (std::uint32_t r = 0; r < kWidth; ++r) {
            double z = head[1].data[r];
            for (std::uint32_t c = 0; c < kWidth; ++c) {
                z += head[0].data[r * kWidth + c] * x[c];
            }
            h[r] = std::tanh(z);
        }
        REQUIRE(got[s].size() == 2);
        for (std::uint32_t l = 0; l < 2; ++l) {
            double z = head[3].data[l];
            for (std::uint32_t c = 0; c < kWidth; ++c) {
                z += head[2].data[l * kWidth + c] * h[c];
            }
            INFO("sequence " << s << " label " << l);
            REQUIRE(std::fabs(got[s][l] - z) < 1e-4);
        }
    }
    // A reranker does not embed, and an embedding model does not rank.
    REQUIRE_THROWS_AS(enc.embed(batch, op), std::invalid_argument);
    const auto img2 = synth_bert(synth_tensors(), 2);
    const auto g2 = locus::gguf::GgufFile::parse(img2);
    REQUIRE_THROWS_AS(BertEncoder::load(g2).rank(batch, op),
                      std::invalid_argument);
    // Rank pooling without cls.output is refused.
    const auto img3 = synth_bert(synth_tensors(), 4);
    const auto g3 = locus::gguf::GgufFile::parse(img3);
    REQUIRE_THROWS_AS(BertEncoder::load(g3), locus::gguf::Error);
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

namespace {

/**
 * Embeds the golden texts with a model, all in one batch and each
 * alone, and checks them against three references:
 *
 * - q8ref (scripts/make-bert-q8-ref.py): the GGUF's own weights in
 *   float64 arithmetic; locus must match it to a cosine of 0.99999.
 * - llama: llama.cpp's vectors, at least 0.9995. llama.cpp rounds the
 *   activations to 8 bits before each Q8_0 product, locus keeps them
 *   in f32, so llama.cpp is the one off the q8ref: 0.99984 to 0.99993
 *   on the 12-layer models, 0.99965 to 0.99973 on bge-m3's 24.
 * - hf, when given: the unquantized original (f32 in transformers, or
 *   for nomic its own F16 GGUF in float64), at least hf_bar and at least
 *   as near as llama.cpp.
 *
 * The bars are per model: nomic-embed v1.5 loses more to Q8_0 (its
 * Q8_0 weights alone are 0.99887 to 0.99935 from its F16 ones), so its
 * llama_bar and hf_bar are 0.9985.
 */
void check_golden(const std::string& file, const char* llama,
                  const char* q8ref, const char* hf, std::uint32_t width,
                  Pooling pooling, double llama_bar = 0.9995,
                  double hf_bar = 0.9998) {
    const std::string path = kDir + "models/" + file;
    if (!std::filesystem::exists(path)) {
        SKIP(file << " absent; see scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(path);
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    const auto enc = BertEncoder::load(g);
    REQUIRE(enc.width() == width);
    REQUIRE(enc.pooling() == pooling);
    std::ifstream f(kDir + "golden/" + llama);
    REQUIRE(f.good());
    const auto j = nlohmann::json::parse(f);
    const auto texts = j["texts"].get<std::vector<std::string>>();
    const auto want = j["embeddings"].get<std::vector<std::vector<float>>>();
    REQUIRE(texts.size() == 7);
    const auto read = [](const char* name) {
        std::ifstream fr(kDir + "golden/" + name);
        REQUIRE(fr.good());
        return nlohmann::json::parse(fr)["embeddings"]
            .get<std::vector<std::vector<float>>>();
    };
    const auto exact = read(q8ref);
    REQUIRE(exact.size() == texts.size());
    std::vector<std::vector<float>> ref;
    if (hf != nullptr) ref = read(hf);
    std::vector<std::vector<TokenId>> batch;
    for (const auto& t : texts) batch.push_back(tok->encode(t, true));
    const auto& op = locus::backend::best_backend().ops;
    const auto got = enc.embed(batch, op);
    for (std::size_t i = 0; i < texts.size(); ++i) {
        INFO("text " << texts[i]);
        const auto one = enc.embed(
            std::span<const std::vector<TokenId>>(&batch[i], 1), op);
        const double e = cosine(got[i], exact[i]);
        const double c = cosine(got[i], want[i]);
        UNSCOPED_INFO("cosine q8ref " << e << " llama " << c);
        REQUIRE(e >= 0.99999);
        REQUIRE(c >= llama_bar);
        if (hf != nullptr) {
            const double ch = cosine(got[i], ref[i]);
            const double lh = cosine(want[i], ref[i]);
            UNSCOPED_INFO("cosine hf " << ch << " llama-hf " << lh);
            REQUIRE(ch >= hf_bar);
            REQUIRE(ch >= lh);
        }
        REQUIRE(cosine(one[0], got[i]) >= 0.99999);
    }
    // The question is nearer its answer than an unrelated sentence.
    REQUIRE(cosine(got[2], got[3]) > cosine(got[2], got[4]));
}

}  // namespace

TEST_CASE("bge-small embeddings match llama.cpp and transformers",
          "[bert][e2e]") {
    // CLS pooling, WordPiece.
    check_golden("bge-small-en-v1.5-q8_0.gguf", "bge_small_embd.json",
                 "bge_small_embd_q8ref.json", "bge_small_embd_hf.json", 384,
                 Pooling::kCls);
}

TEST_CASE("e5-small-v2 embeddings match llama.cpp and transformers",
          "[bert][e2e]") {
    // Mean pooling, WordPiece.
    check_golden("e5-small-v2-q8_0.gguf", "e5_small_embd.json",
                 "e5_small_embd_q8ref.json", "e5_small_embd_hf.json", 384,
                 Pooling::kMean);
}

TEST_CASE("bge-m3 embeddings match exact arithmetic and llama.cpp",
          "[bert][e2e]") {
    // XLM-R: SentencePiece unigram, one token type, 24 layers of 1024.
    // Its original weights are a pickle on BAAI's main branch, so it
    // has no transformers reference.
    check_golden("bge-m3-q8_0.gguf", "bge_m3_embd.json",
                 "bge_m3_embd_q8ref.json", nullptr, 1024, Pooling::kCls);
}

TEST_CASE("nomic-embed-text v1.5 embeddings match exact arithmetic",
          "[bert][e2e]") {
    // nomic-bert: NEOX rope, fused qkv, SwiGLU, no biases, mean pooling.
    // The unquantized reference is nomic-ai's F16 GGUF in float64
    // (scripts/make-bert-q8-ref.py), which llama.cpp's F16 run matches
    // to 0.999999; its transformers model needs trust_remote_code.
    check_golden("nomic-embed-text-v1.5.Q8_0.gguf", "nomic_embd.json",
                 "nomic_embd_q8ref.json", "nomic_embd_f16ref.json", 768,
                 Pooling::kMean, 0.9985, 0.9985);
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

TEST_CASE("encoder models on the Vulkan image ops match the CPU",
          "[bert][vulkan][e2e]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no Vulkan device");
    }
    const auto& cpu = locus::backend::best_backend().ops;
    const auto& gpu = locus::backend::vulkan_image_ops();
    const std::vector<std::string> texts = {
        "what is the capital of china?",
        "Beijing is the capital of the People's Republic of China.",
        "A small red apple on a wooden table, soft morning light."};
    // Eight inputs of about 120 tokens: past kGpuMinTokens and kGpuMacs
    // for every model, so the gpu argument runs the passes on Vulkan,
    // with each input's attention on the CPU (kCpuAttnTokens); and one
    // of about 360 tokens, whose attention runs on the GPU.
    std::string para;
    for (int i = 0; i < 12; ++i) para += texts[i % 3] + " ";
    const std::vector<std::string> big = [&] {
        std::vector<std::string> v(8, para);
        v.push_back(para + para + para);
        return v;
    }();
    for (const char* file :
         {"bge-small-en-v1.5-q8_0.gguf", "bge-m3-q8_0.gguf",
          "nomic-embed-text-v1.5.Q8_0.gguf"}) {
        const std::string path = kDir + "models/" + file;
        if (!std::filesystem::exists(path)) continue;
        const auto m = locus::embed::EmbeddingModel::load(path);
        for (const auto* in : {&texts, &big}) {
            const auto c = m->embed(*in, cpu);
            const auto g = m->embed(*in, gpu);  // every pass on Vulkan
            const auto p = m->embed(*in, cpu, nullptr, &gpu);  // by size
            for (std::size_t i = 0; i < in->size(); ++i) {
                INFO(file << " input " << i);
                REQUIRE(cosine(c[i], g[i]) >= 0.99999);
                REQUIRE(cosine(c[i], p[i]) >= 0.99999);
            }
        }
    }
    const std::string rr = kDir + "models/bge-reranker-v2-m3-q8_0.gguf";
    if (std::filesystem::exists(rr)) {
        const auto m = locus::embed::EmbeddingModel::load(rr);
        const auto c = m->rank(texts[0], big, cpu);
        const auto g = m->rank(texts[0], big, cpu, nullptr, &gpu);
        for (std::size_t i = 0; i < big.size(); ++i) {
            REQUIRE(std::fabs(c[i] - g[i]) < 2e-3);
        }
    }
}

// i#519: on the CUDA ops the matrices run through matvec_batch_cuda
// (each weight read once per pass) and the attention stays on the CPU
// (dense::attention never hands a self-parallel backend its host
// buffers). The same inputs and bars as the Vulkan case above.
TEST_CASE("encoder models on the CUDA ops match the CPU",
          "[bert][cuda][e2e]") {
    const auto* cuda = locus::backend::find_backend("cuda");
    if (cuda == nullptr || !cuda->available) {
        SKIP("no usable CUDA device");
    }
    locus::backend::cuda_pool_reset();
    const auto& cpu = locus::backend::best_backend().ops;
    const auto& gpu = cuda->ops;
    const std::vector<std::string> texts = {
        "what is the capital of china?",
        "Beijing is the capital of the People's Republic of China.",
        "A small red apple on a wooden table, soft morning light."};
    std::string para;
    for (int i = 0; i < 12; ++i) para += texts[i % 3] + " ";
    const std::vector<std::string> big(8, para);
    std::size_t ran = 0;
    for (const char* file :
         {"bge-small-en-v1.5-q8_0.gguf", "bge-m3-q8_0.gguf",
          "nomic-embed-text-v1.5.Q8_0.gguf"}) {
        const std::string path = kDir + "models/" + file;
        if (!std::filesystem::exists(path)) continue;
        ++ran;
        const auto m = locus::embed::EmbeddingModel::load(path);
        for (const auto* in : {&texts, &big}) {
            const auto c = m->embed(*in, cpu);
            const auto g = m->embed(*in, gpu);  // every pass on CUDA
            const auto p = m->embed(*in, cpu, nullptr, &gpu);  // by size
            for (std::size_t i = 0; i < in->size(); ++i) {
                INFO(file << " input " << i);
                REQUIRE(cosine(c[i], g[i]) >= 0.99999);
                REQUIRE(cosine(c[i], p[i]) >= 0.99999);
            }
        }
    }
    const std::string rr = kDir + "models/bge-reranker-v2-m3-q8_0.gguf";
    if (std::filesystem::exists(rr)) {
        ++ran;
        const auto m = locus::embed::EmbeddingModel::load(rr);
        const auto c = m->rank(texts[0], big, cpu);
        const auto g = m->rank(texts[0], big, cpu, nullptr, &gpu);
        for (std::size_t i = 0; i < big.size(); ++i) {
            REQUIRE(std::fabs(c[i] - g[i]) < 2e-3);
        }
    }
    if (ran == 0) {
        SKIP("no encoder model present");
    }
    locus::backend::cuda_pool_reset();
}
