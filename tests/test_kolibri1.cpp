// i#613: the kolibri1 arch (Kolibri-1) on a tiny synthetic GGUF with
// the real tensor names and metadata keys (llama.cpp's converter),
// against a double-precision reference written from llama.cpp's
// llama_model_kolibri1 graph.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "gguf_builder.hpp"

using locus::model::TransformerModel;

namespace {

/** Deterministic small pseudo-random values in [-0.5, 0.5]. */
std::vector<float> weights(std::size_t n, std::uint32_t seed) {
    std::vector<float> v(n);
    std::uint32_t s = seed * 2654435761u + 12345u;
    for (auto& f : v) {
        s = s * 1664525u + 1013904223u;
        f = static_cast<float>(s >> 8) / static_cast<float>(1u << 24) -
            0.5f;
    }
    return v;
}

// Two layers: 0 sliding-window (window 3, NEOX RoPE), 1 global (NoPE).
// embd 16, 4 q / 2 kv heads of 8, 6 experts (top 2) of width 12, a
// shared expert of width 10, vocab 32.
constexpr std::uint32_t kE = 16, kH = 4, kKv = 2, kHd = 8, kNe = 6;
constexpr std::uint32_t kUsed = 2, kFfe = 12, kFsh = 10, kV = 32;
constexpr std::uint32_t kWin = 3, kL = 2;
constexpr float kEps = 1e-5f, kBase = 10000.0f;

struct Tensor {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::vector<float> data;
};

/** The model's tensors; norms near 1, the selection bias about +-1 so
 *  it changes which experts win. */
std::vector<Tensor> make_tensors() {
    std::vector<Tensor> ts;
    std::uint32_t seed = 1;
    auto add = [&](std::string name, std::vector<std::uint64_t> dims,
                   float scale = 1.0f, float shift = 0.0f) {
        std::size_t n = 1;
        for (auto d : dims) n *= d;
        auto v = weights(n, seed++);
        for (auto& f : v) f = f * scale + shift;
        ts.push_back({std::move(name), std::move(dims), std::move(v)});
    };
    add("token_embd.weight", {kE, kV}, 2.0f);
    add("output.weight", {kE, kV});
    add("output_norm.weight", {kE}, 0.4f, 1.0f);
    for (std::uint32_t l = 0; l < kL; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        add(b + "attn_norm.weight", {kE}, 0.4f, 1.0f);
        add(b + "attn_q.weight", {kE, kH * kHd});
        add(b + "attn_k.weight", {kE, kKv * kHd});
        add(b + "attn_v.weight", {kE, kKv * kHd});
        add(b + "attn_output.weight", {kH * kHd, kE});
        add(b + "attn_q_norm.weight", {kHd}, 0.4f, 1.0f);
        add(b + "attn_k_norm.weight", {kHd}, 0.4f, 1.0f);
        add(b + "post_attention_norm.weight", {kE}, 0.4f, 1.0f);
        add(b + "ffn_norm.weight", {kE}, 0.4f, 1.0f);
        add(b + "post_ffw_norm.weight", {kE}, 0.4f, 1.0f);
        add(b + "ffn_gate_inp.weight", {kE, kNe}, 2.0f);
        add(b + "exp_probs_b.bias", {kNe}, 4.0f);
        add(b + "ffn_gate_exps.weight", {kE, kFfe, kNe});
        add(b + "ffn_up_exps.weight", {kE, kFfe, kNe});
        add(b + "ffn_down_exps.weight", {kFfe, kE, kNe});
        add(b + "ffn_gate_shexp.weight", {kE, kFsh});
        add(b + "ffn_up_shexp.weight", {kE, kFsh});
        add(b + "ffn_down_shexp.weight", {kFsh, kE});
    }
    return ts;
}

std::vector<std::byte> build_image(const std::vector<Tensor>& ts, bool norm) {
    const std::string p = "kolibri1.";
    GgufBuilder g;
    g.header(ts.size(), 19)
        .kv_string("general.architecture", "kolibri1")
        .kv_u32(p + "embedding_length", kE)
        .kv_u32(p + "block_count", kL)
        .kv_u32(p + "attention.head_count", kH)
        .kv_u32(p + "attention.head_count_kv", kKv)
        .kv_u32(p + "context_length", 64)
        .kv_f32(p + "attention.layer_norm_rms_epsilon", kEps)
        .kv_f32(p + "rope.freq_base", kBase)
        .kv_u32(p + "expert_count", kNe)
        .kv_u32(p + "expert_used_count", kUsed)
        .kv_u32(p + "attention.key_length", kHd)
        .kv_u32(p + "attention.value_length", kHd)
        .kv_u32(p + "expert_feed_forward_length", kFfe)
        .kv_u32(p + "expert_shared_count", 1)
        .kv_u32(p + "expert_shared_feed_forward_length", kFsh)
        .kv_u32(p + "expert_gating_func", 2)
        .kv_bool(p + "expert_weights_norm", norm)
        .kv_u32(p + "attention.sliding_window", kWin);
    // sliding_window_pattern: a bool array (GGUF type 9 of 7).
    g.str(p + "attention.sliding_window_pattern").u32(9).u32(7).u64(kL);
    for (std::uint32_t l = 0; l < kL; ++l) g.u8(l == 0 ? 1 : 0);
    std::uint64_t offset = 0;
    for (const auto& t : ts) {
        g.tensor(t.name, t.dims, 0 /* F32 */, offset);
        offset = (offset + t.data.size() * 4 + 31) & ~31ull;
    }
    g.pad();
    for (const auto& t : ts) {
        for (float v : t.data) g.f32(v);
        while (g.bytes().size() % 32 != 0) g.zeros(1);
    }
    auto span = g.bytes();
    return {span.begin(), span.end()};
}

/** The reference, in double: llama.cpp's graph written out. */
class Reference {
  public:
    Reference(const std::vector<Tensor>& ts, bool norm) : norm_(norm) {
        for (const auto& t : ts) w_[t.name] = &t.data;
    }

    /** @returns The logits after each of `toks`. */
    std::vector<std::vector<double>> run(const std::vector<int>& toks) {
        const std::size_t kvd = kKv * kHd;
        std::vector<std::vector<std::vector<double>>> kc(kL), vc(kL);
        std::vector<std::vector<double>> out;
        for (std::size_t pos = 0; pos < toks.size(); ++pos) {
            std::vector<double> x(kE);
            for (std::uint32_t i = 0; i < kE; ++i) {
                x[i] = at("token_embd.weight")[toks[pos] * kE + i];
            }
            for (std::uint32_t l = 0; l < kL; ++l) {
                const std::string b = "blk." + std::to_string(l) + ".";
                const bool swa = l == 0;
                auto h = rms(x, at(b + "attn_norm.weight"));
                auto q = mv(at(b + "attn_q.weight"), h, kE, kH * kHd);
                auto k = mv(at(b + "attn_k.weight"), h, kE, kvd);
                auto v = mv(at(b + "attn_v.weight"), h, kE, kvd);
                head_rms(q, kH, at(b + "attn_q_norm.weight"));
                head_rms(k, kKv, at(b + "attn_k_norm.weight"));
                if (swa) {
                    neox(q, kH, pos);
                    neox(k, kKv, pos);
                }
                kc[l].push_back(k);
                vc[l].push_back(v);
                std::vector<double> o(kH * kHd, 0.0);
                for (std::uint32_t hh = 0; hh < kH; ++hh) {
                    const std::uint32_t kvh = hh / (kH / kKv);
                    std::vector<double> s;
                    std::vector<std::size_t> ts;
                    for (std::size_t t = 0; t <= pos; ++t) {
                        if (swa && pos - t >= kWin) continue;
                        double d = 0.0;
                        for (std::uint32_t i = 0; i < kHd; ++i) {
                            d += q[hh * kHd + i] * kc[l][t][kvh * kHd + i];
                        }
                        s.push_back(d / std::sqrt(double(kHd)));
                        ts.push_back(t);
                    }
                    const double mx = *std::max_element(s.begin(), s.end());
                    double sum = 0.0;
                    for (auto& e : s) sum += (e = std::exp(e - mx));
                    for (std::size_t j = 0; j < ts.size(); ++j) {
                        for (std::uint32_t i = 0; i < kHd; ++i) {
                            o[hh * kHd + i] +=
                                s[j] / sum * vc[l][ts[j]][kvh * kHd + i];
                        }
                    }
                }
                auto a = mv(at(b + "attn_output.weight"), o, kH * kHd, kE);
                a = rms(a, at(b + "post_attention_norm.weight"));
                for (std::uint32_t i = 0; i < kE; ++i) x[i] += a[i];

                h = rms(x, at(b + "ffn_norm.weight"));
                auto logit = mv(at(b + "ffn_gate_inp.weight"), h, kE, kNe);
                const auto& bias = at(b + "exp_probs_b.bias");
                std::vector<std::uint32_t> order(kNe);
                std::iota(order.begin(), order.end(), 0u);
                std::stable_sort(order.begin(), order.end(),
                                 [&](std::uint32_t p1, std::uint32_t p2) {
                                     return logit[p1] + bias[p1] >
                                            logit[p2] + bias[p2];
                                 });
                std::vector<double> wt(kUsed);
                double wsum = 0.0;
                for (std::uint32_t j = 0; j < kUsed; ++j) {
                    wt[j] = 1.0 / (1.0 + std::exp(-logit[order[j]]));
                    wsum += wt[j];
                }
                std::vector<double> f(kE, 0.0);
                for (std::uint32_t j = 0; j < kUsed; ++j) {
                    const double wj = norm_ ? wt[j] / wsum : wt[j];
                    const auto y = expert(b, order[j], h);
                    for (std::uint32_t i = 0; i < kE; ++i) f[i] += wj * y[i];
                }
                const auto sh = glu(at(b + "ffn_gate_shexp.weight"),
                                    at(b + "ffn_up_shexp.weight"),
                                    at(b + "ffn_down_shexp.weight"), 0, kFsh,
                                    h);
                for (std::uint32_t i = 0; i < kE; ++i) f[i] += sh[i];
                f = rms(f, at(b + "post_ffw_norm.weight"));
                for (std::uint32_t i = 0; i < kE; ++i) x[i] += f[i];
            }
            const auto h = rms(x, at("output_norm.weight"));
            out.push_back(mv(at("output.weight"), h, kE, kV));
        }
        return out;
    }

  private:
    const std::vector<float>& at(const std::string& n) { return *w_.at(n); }

    static std::vector<double> rms(const std::vector<double>& x,
                                   const std::vector<float>& w) {
        double ss = 0.0;
        for (double v : x) ss += v * v;
        const double r = 1.0 / std::sqrt(ss / double(x.size()) + kEps);
        std::vector<double> y(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] * r * w[i];
        return y;
    }
    static void head_rms(std::vector<double>& x, std::uint32_t heads,
                         const std::vector<float>& w) {
        for (std::uint32_t h = 0; h < heads; ++h) {
            std::vector<double> s(x.begin() + h * kHd,
                                  x.begin() + (h + 1) * kHd);
            s = rms(s, w);
            std::copy(s.begin(), s.end(), x.begin() + h * kHd);
        }
    }
    /** NEOX: pair (i, i + hd / 2) turns by pos * base^(-2 i / hd). */
    static void neox(std::vector<double>& x, std::uint32_t heads,
                     std::size_t pos) {
        for (std::uint32_t h = 0; h < heads; ++h) {
            for (std::uint32_t i = 0; i < kHd / 2; ++i) {
                const double th = double(pos) *
                                  std::pow(double(kBase), -2.0 * i / kHd);
                double& a = x[h * kHd + i];
                double& b = x[h * kHd + i + kHd / 2];
                const double a0 = a, b0 = b;
                a = a0 * std::cos(th) - b0 * std::sin(th);
                b = a0 * std::sin(th) + b0 * std::cos(th);
            }
        }
    }
    /** y = W x for a ggml [in, out] weight at `off` floats. */
    static std::vector<double> mv(const std::vector<float>& w,
                                  const std::vector<double>& x,
                                  std::uint32_t in, std::uint32_t out,
                                  std::size_t off = 0) {
        std::vector<double> y(out, 0.0);
        for (std::uint32_t o = 0; o < out; ++o) {
            for (std::uint32_t i = 0; i < in; ++i) {
                y[o] += double(w[off + std::size_t{o} * in + i]) * x[i];
            }
        }
        return y;
    }
    static std::vector<double> glu(const std::vector<float>& g,
                                   const std::vector<float>& u,
                                   const std::vector<float>& d,
                                   std::size_t e, std::uint32_t ff,
                                   const std::vector<double>& x) {
        auto a = mv(g, x, kE, ff, e * kE * ff);
        const auto c = mv(u, x, kE, ff, e * kE * ff);
        for (std::uint32_t i = 0; i < ff; ++i) {
            a[i] = a[i] / (1.0 + std::exp(-a[i])) * c[i];
        }
        return mv(d, a, ff, kE, e * ff * kE);
    }
    std::vector<double> expert(const std::string& b, std::uint32_t e,
                               const std::vector<double>& x) {
        return glu(at(b + "ffn_gate_exps.weight"),
                   at(b + "ffn_up_exps.weight"),
                   at(b + "ffn_down_exps.weight"), e, kFfe, x);
    }

    std::unordered_map<std::string, const std::vector<float>*> w_;
    bool norm_;
};

/** @returns Each position's logits from `model` over `toks`. */
std::vector<std::vector<float>> run(const TransformerModel& model,
                                    const std::vector<int>& toks) {
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<std::vector<float>> out;
    std::vector<float> logits(model.hparams().n_vocab);
    for (int t : toks) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(t, cache, seq, ws, logits);
        out.push_back(logits);
    }
    cache.release(seq);
    return out;
}

}  // namespace

TEST_CASE("kolibri1 loads its hparams and tensors", "[model][kolibri1]") {
    const auto ts = make_tensors();
    auto img = build_image(ts, false);
    auto g = locus::gguf::GgufFile::parse(img);
    auto model = TransformerModel::load(g);
    const auto& hp = model.hparams();
    REQUIRE(hp.custom_layer);
    REQUIRE(hp.rope_neox);
    REQUIRE(hp.head_dim == kHd);
    REQUIRE(hp.n_ff_exp == kFfe);
    REQUIRE(hp.n_ff_shexp == kFsh);
    REQUIRE(hp.n_expert_shared == 1);
    REQUIRE(hp.swa_window == kWin);
    REQUIRE(hp.swa_layers == std::vector<bool>{true, false});
    REQUIRE(hp.gating == locus::model::GatingFunc::kSigmoid);
    REQUIRE_FALSE(hp.expert_weights_norm);
    REQUIRE(model.layers()[1].post_attn_norm.size() == kE);
    REQUIRE(model.layers()[1].exp_probs_b.size() == kNe);
    // Its own layer body: no batch paths, no GPU full forward.
    REQUIRE_FALSE(model.supports_batch());
    REQUIRE_FALSE(locus::backend::vulkan_forward_supported(model));
}

// Ten positions: past the window of 3 on layer 0, every position's
// logits against the reference, with and without renormalized expert
// weights; on Vulkan too when there is a device (its matvecs sum in
// another order).
TEST_CASE("kolibri1 matches the reference graph", "[model][kolibri1]") {
    const std::vector<int> toks = {3, 17, 9, 30, 1, 22, 5, 14, 27, 8};
    const auto ts = make_tensors();
    for (const bool norm : {false, true}) {
        INFO("expert_weights_norm " << norm);
        const auto want = Reference(ts, norm).run(toks);
        auto img = build_image(ts, norm);
        auto g = locus::gguf::GgufFile::parse(img);
        auto model = TransformerModel::load(g);
        using locus::backend::Backend;
        std::vector<std::pair<const Backend*, double>> legs = {
            {&locus::backend::best_backend(), 1e-4}};
        if (locus::backend::vulkan_backend_usable()) {
            legs.push_back({locus::backend::find_backend("vulkan"), 1e-3});
        }
        for (const auto& [be, bound] : legs) {
            INFO(be->name);
            model.use_backend(*be);
            const auto got = run(model, toks);
            for (std::size_t p = 0; p < toks.size(); ++p) {
                INFO("position " << p);
                double err = 0.0, ref = 0.0;
                for (std::uint32_t i = 0; i < kV; ++i) {
                    err = std::max(err, std::abs(got[p][i] - want[p][i]));
                    ref = std::max(ref, std::abs(want[p][i]));
                }
                CHECK(err <= bound * ref);
                CHECK(std::max_element(got[p].begin(), got[p].end()) -
                          got[p].begin() ==
                      std::max_element(want[p].begin(), want[p].end()) -
                          want[p].begin());
            }
        }
    }
}
