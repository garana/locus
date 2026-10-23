#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "gguf_builder.hpp"

using locus::model::TransformerModel;

namespace {

/** Deterministic small pseudo-random weights in [-0.5, 0.5]. */
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

// Tiny muse-spark: embd 16, 4 q / 2 kv heads of 4, 4 experts (top 2)
// over a latent of 8 with expert width 12, vocab 32, sliding window 2.
constexpr std::uint32_t kE = 16, kH = 4, kKv = 2, kHd = 4, kNe = 4;
constexpr std::uint32_t kLat = 8, kFfe = 12, kV = 32, kWin = 2;
constexpr float kCap = 3.0f;

/**
 * @returns A one-layer muse-spark image whose layer is sliding-window
 *     (swa) or global (NoPE, full attention).
 */
std::vector<std::byte> build_image(bool swa) {
    struct T {
        std::string name;
        std::vector<std::uint64_t> dims;
        std::vector<float> data;
    };
    std::vector<T> ts;
    std::uint32_t seed = 1;
    auto add = [&](std::string name, std::vector<std::uint64_t> dims) {
        std::size_t n = 1;
        for (auto d : dims) {
            n *= d;
        }
        ts.push_back({std::move(name), std::move(dims), weights(n, seed++)});
    };
    auto ones = [&](std::string name, std::uint64_t n) {
        ts.push_back({std::move(name), {n}, std::vector<float>(n, 1.0f)});
    };
    add("token_embd.weight", {kE, kV});
    add("output.weight", {kE, kV});
    ones("output_norm.weight", kE);
    const std::string b = "blk.0.";
    ones(b + "attn_norm.weight", kE);
    add(b + "attn_q.weight", {kE, kH * kHd});
    add(b + "attn_k.weight", {kE, kKv * kHd});
    add(b + "attn_v.weight", {kE, kKv * kHd});
    add(b + "attn_output.weight", {kH * kHd, kE});
    add(b + "attn_gate.weight", {kE, kH * kHd});
    ones(b + "attn_q_norm.weight", kHd);
    ones(b + "attn_k_norm.weight", kHd);
    ones(b + "attn_resid_alpha.weight", kE);
    ones(b + "attn_resid_beta.weight", kE);
    ones(b + "ffn_resid_alpha.weight", kE);
    ones(b + "ffn_resid_beta.weight", kE);
    ones(b + "ffn_norm.weight", kE);
    ones(b + "post_ffw_norm.weight", kE);
    add(b + "ffn_gate_inp.weight", {kE, kNe});
    ts.push_back({b + "exp_probs_b.bias", {kNe}, std::vector<float>(kNe)});
    add(b + "ffn_latent_down.weight", {kE, kLat});
    add(b + "ffn_latent_up.weight", {kLat, kE});
    ones(b + "ffn_pre_exp_norm.weight", kLat);
    ones(b + "ffn_post_exp_norm.weight", kLat);
    add(b + "ffn_gate_exps.weight", {kLat, kFfe, kNe});
    add(b + "ffn_up_exps.weight", {kLat, kFfe, kNe});
    add(b + "ffn_down_exps.weight", {kFfe, kLat, kNe});

    const std::string p = "muse-spark.";
    GgufBuilder g;
    g.header(ts.size(), 20)
        .kv_string("general.architecture", "muse-spark")
        .kv_u32(p + "embedding_length", kE)
        .kv_u32(p + "block_count", 1)
        .kv_u32(p + "attention.head_count", kH)
        .kv_u32(p + "attention.head_count_kv", kKv)
        .kv_u32(p + "feed_forward_length", 32)
        .kv_u32(p + "context_length", 32)
        .kv_f32(p + "attention.layer_norm_rms_epsilon", 1e-5f)
        .kv_u32(p + "expert_count", kNe)
        .kv_u32(p + "expert_used_count", 2)
        .kv_u32(p + "attention.key_length", kHd)
        .kv_u32(p + "attention.value_length", kHd)
        .kv_u32(p + "expert_feed_forward_length", kFfe)
        .kv_u32(p + "moe_latent_size", kLat)
        .kv_u32(p + "expert_gating_func", 2)
        .kv_f32(p + "post_norm_epsilon", 1e-8f)
        .kv_f32(p + "logit_scale", 2.0f)
        .kv_f32(p + "final_logit_softcapping", kCap)
        .kv_u32(p + "attention.sliding_window", kWin);
    // sliding_window_pattern: a bool array (GGUF type 9 of 7).
    g.str(p + "attention.sliding_window_pattern")
        .u32(9)
        .u32(7)
        .u64(1)
        .u8(swa ? 1 : 0);
    std::uint64_t offset = 0;
    for (const auto& t : ts) {
        g.tensor(t.name, t.dims, 0 /* F32 */, offset);
        offset = (offset + t.data.size() * 4 + 31) & ~31ull;
    }
    g.pad();
    for (const auto& t : ts) {
        for (float v : t.data) {
            g.f32(v);
        }
        while (g.bytes().size() % 32 != 0) {
            g.zeros(1);
        }
    }
    auto span = g.bytes();
    return {span.begin(), span.end()};
}

/** Last-position logits after feeding `tokens`. */
std::vector<float> last_logits(const TransformerModel& model,
                               const std::vector<locus::tok::TokenId>& toks) {
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    for (auto t : toks) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(t, cache, seq, ws, logits);
    }
    cache.release(seq);
    return logits;
}

}  // namespace

TEST_CASE("muse-spark loads its hparams and runs single-token only",
          "[model][muse]") {
    auto img = build_image(true);
    auto g = locus::gguf::GgufFile::parse(img);
    auto model = TransformerModel::load(g);
    const auto& hp = model.hparams();
    REQUIRE(hp.custom_layer);
    REQUIRE(hp.embd_norm);
    REQUIRE(hp.moe_latent == kLat);
    REQUIRE(hp.n_ff_exp == kFfe);
    REQUIRE(hp.swa_window == kWin);
    REQUIRE(hp.swa_layers == std::vector<bool>{true});
    REQUIRE(hp.gating == locus::model::GatingFunc::kSigmoid);
    REQUIRE(hp.final_logit_softcap == kCap);
    REQUIRE(model.layers()[0].gate_exps.base.cols == kLat);
    // Its own layer body: no batch paths, no GPU full forward.
    REQUIRE_FALSE(model.supports_batch());
    REQUIRE_FALSE(locus::backend::vulkan_forward_supported(model));
}

TEST_CASE("muse-spark logits are finite and within the softcap",
          "[model][muse]") {
    auto img = build_image(false);
    auto g = locus::gguf::GgufFile::parse(img);
    auto model = TransformerModel::load(g);
    const auto logits = last_logits(model, {1, 5, 9, 2});
    for (float v : logits) {
        REQUIRE(std::isfinite(v));
        REQUIRE(std::fabs(v) <= kCap);
    }
}

TEST_CASE("muse-spark sliding-window layers see only the window",
          "[model][muse]") {
    // Window 2: at position 3 a sliding-window layer attends to
    // positions 2 and 3 only, so the token at position 0 cannot
    // change the logits; a global layer sees it.
    const std::vector<locus::tok::TokenId> a = {1, 5, 9, 2};
    const std::vector<locus::tok::TokenId> b = {7, 5, 9, 2};
    for (const bool swa : {true, false}) {
        INFO("swa " << swa);
        auto img = build_image(swa);
        auto g = locus::gguf::GgufFile::parse(img);
        auto model = TransformerModel::load(g);
        const auto la = last_logits(model, a);
        const auto lb = last_logits(model, b);
        if (swa) {
            REQUIRE(la == lb);
        } else {
            REQUIRE(la != lb);
        }
    }
}
