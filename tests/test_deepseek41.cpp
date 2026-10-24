#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/deepseek41.hpp"
#include "locus/model/transformer.hpp"
#include "gguf_builder.hpp"

using locus::model::TransformerModel;
namespace ds41 = locus::model::deepseek41;

namespace {

std::vector<float> weights(std::size_t n, std::uint32_t seed, float amp) {
    std::vector<float> v(n);
    std::uint32_t s = seed * 2654435761u + 12345u;
    for (auto& f : v) {
        s = s * 1664525u + 1013904223u;
        f = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24) -
             0.5f) *
            2.0f * amp;
    }
    return v;
}

// Tiny deepseek41: embd 16, 2 heads of 8 (rope 4), q rank 8, indexer
// 2 x 4, output LoRA 2 groups x 4, 4 hyper-connection streams, 4
// experts (top 2) of width 8 plus a shared one, vocab 32, window 3.
// Layers: 0 window only; 1 ratio-2 kv / index source; 2 its consumer;
// 3 ratio-1 kv / index / candidate source; 4 an index source that
// scores inside layer 3's candidate blocks. Engram on layer 1.
constexpr std::uint32_t kE = 16, kH = 2, kDh = 8, kQr = 8, kHi = 2, kDi = 4;
constexpr std::uint32_t kG = 2, kRank = 4, kHc = 4, kNe = 4, kFf = 8;
constexpr std::uint32_t kV = 32, kL = 5;
const std::uint64_t kSizes[] = {5, 7, 11, 13, 17, 19};
const std::uint64_t kOffs[] = {0, 5, 12, 23, 36, 53};
constexpr std::uint32_t kTableRows = 72;

std::vector<std::byte> build_image() {
    struct T {
        std::string name;
        std::vector<std::uint64_t> dims;
        std::vector<float> data;
    };
    std::vector<T> ts;
    std::uint32_t seed = 1;
    auto add = [&](std::string name, std::vector<std::uint64_t> dims,
                   float amp = 0.5f) {
        std::size_t n = 1;
        for (auto d : dims) {
            n *= d;
        }
        ts.push_back({std::move(name), std::move(dims), weights(n, seed++, amp)});
    };
    auto ones = [&](std::string name, std::uint64_t n) {
        ts.push_back({std::move(name), {n}, std::vector<float>(n, 1.0f)});
    };
    add("token_embd.weight", {kE, kV});
    add("output.weight", {kE, kV});
    ones("output_norm.weight", kE);
    add("per_layer_token_embd.weight", {4, kTableRows});
    for (std::uint32_t l = 0; l < kL; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        ones(b + "attn_norm.weight", kE);
        add(b + "attn_sinks.weight", {kH});
        add(b + "attn_q_a.weight", {kE, kQr});
        ones(b + "attn_q_a_norm.weight", kQr);
        add(b + "attn_q_b.weight", {kQr, kH * kDh});
        add(b + "attn_kv.weight", {kE, kDh});
        ones(b + "attn_kv_a_norm.weight", kDh);
        add(b + "attn_output_a.weight", {kH * kDh / kG, kG * kRank});
        add(b + "attn_output_b.weight", {kG * kRank, kE});
        for (const char* sub : {"attn", "ffn"}) {
            const std::string h = b + "hc_" + sub;
            add(h + "_fn.weight", {kHc * kE, 24}, 0.1f);
            add(h + "_base.weight", {24}, 0.1f);
            add(h + "_scale.weight", {3}, 0.5f);
        }
        ones(b + "ffn_norm.weight", kE);
        add(b + "ffn_gate_inp.weight", {kE, kNe});
        add(b + "exp_probs_b.bias", {kNe}, 0.1f);
        add(b + "ffn_gate_exps.weight", {kE, kFf, kNe});
        add(b + "ffn_up_exps.weight", {kE, kFf, kNe});
        add(b + "ffn_down_exps.weight", {kFf, kE, kNe});
        add(b + "ffn_gate_shexp.weight", {kE, kFf});
        add(b + "ffn_up_shexp.weight", {kE, kFf});
        add(b + "ffn_down_shexp.weight", {kFf, kE});
        if (l == 1 || l == 3) {  // kv sources
            add(b + "attn_compressor_kv.weight", {kE, kDh});
            ones(b + "attn_compressor_norm.weight", kDh);
            if (l == 1) {
                add(b + "attn_compressor_gate.weight", {kE, kDh});
            }
            add(b + "indexer.attn_k.weight", {kDh, kDi});
            ones(b + "indexer.k_norm.weight", kDi);
        }
        if (l == 1 || l == 3 || l == 4) {  // index sources
            add(b + "indexer.attn_q_b.weight", {kQr, kHi * kDi});
            add(b + "indexer.proj.weight", {kE, kHi});
        }
        if (l == 1) {
            add(b + "engram_kv.weight", {6 * 4, (kHc + 1) * kE});
            add(b + "engram_gate.weight", {kE, kHc});
        }
    }

    const std::string p = "deepseek41.";
    std::vector<std::function<void(GgufBuilder&)>> kv;
    auto u32 = [&](std::string k, std::uint32_t v) {
        kv.push_back([k, v](GgufBuilder& g) { g.kv_u32(k, v); });
    };
    auto f32 = [&](std::string k, float v) {
        kv.push_back([k, v](GgufBuilder& g) { g.kv_f32(k, v); });
    };
    // GGUF arrays: type 9, then the element type and count.
    auto arr = [&](std::string k, std::uint32_t type,
                   std::vector<std::uint64_t> vals) {
        kv.push_back([k, type, vals](GgufBuilder& g) {
            g.str(k).u32(9).u32(type).u64(vals.size());
            for (auto v : vals) {
                if (type == 10) {
                    g.u64(v);
                } else {
                    g.u32(static_cast<std::uint32_t>(v));
                }
            }
        });
    };
    kv.push_back([](GgufBuilder& g) {
        g.kv_string("general.architecture", "deepseek41");
    });
    u32(p + "embedding_length", kE);
    u32(p + "block_count", kL);
    u32(p + "attention.head_count", kH);
    u32(p + "attention.head_count_kv", 1);
    u32(p + "context_length", 64);
    f32(p + "attention.layer_norm_rms_epsilon", 1e-6f);
    u32(p + "expert_count", kNe);
    u32(p + "expert_used_count", 2);
    u32(p + "expert_shared_count", 1);
    u32(p + "expert_feed_forward_length", kFf);
    u32(p + "expert_gating_func", 4);
    f32(p + "expert_weights_scale", 1.5f);
    kv.push_back([p](GgufBuilder& g) {
        g.kv_bool(p + "expert_weights_norm", true);
    });
    u32(p + "attention.key_length", kDh);
    u32(p + "attention.value_length", kDh);
    u32(p + "rope.dimension_count", 4);
    f32(p + "rope.scaling.factor", 16.0f);
    u32(p + "rope.scaling.original_context_length", 32);
    u32(p + "attention.sliding_window", 3);
    arr(p + "attention.compress_ratios", 5, {0, 2, 2, 1, 1});  // i32
    f32(p + "attention.compress_rope_freq_base", 160000.0f);
    arr(p + "attention.kv_source_layers", 4, {1, 3});
    arr(p + "attention.indexer.source_layers", 4, {1, 3, 4});
    u32(p + "attention.indexer.head_count", kHi);
    u32(p + "attention.indexer.key_length", kDi);
    u32(p + "attention.indexer.top_k", 3);
    kv.push_back([p](GgufBuilder& g) {
        g.str(p + "attention.indexer.candidate_source_layer").u32(5).i32(3);
    });
    u32(p + "attention.indexer.candidate_top_k_blocks", 2);
    u32(p + "attention.indexer.candidate_block_size", 2);
    u32(p + "attention.output_group_count", kG);
    u32(p + "attention.output_lora_rank", kRank);
    u32(p + "hyper_connection.count", kHc);
    u32(p + "hyper_connection.sinkhorn_iterations", 20);
    f32(p + "hyper_connection.epsilon", 1e-6f);
    kv.push_back([p](GgufBuilder& g) {
        const std::vector<float> c(kL, 10.0f);
        g.kv_f32_array(p + "swiglu_clamp_exp", c);
    });
    arr(p + "ple.layers", 4, {1});
    u32(p + "ple.ngram_size", 4);
    u32(p + "ple.heads_per_ngram", 2);
    u32(p + "embedding_length_per_layer_input", 4);
    arr(p + "ple.layer_multipliers", 10, {1, 3, 5, 7});
    arr(p + "ple.head_offsets", 10, {kOffs, kOffs + 6});
    arr(p + "ple.head_vocab_sizes", 10, {kSizes, kSizes + 6});
    u32(p + "engram.pad_token_id", 2);
    {
        std::vector<std::uint64_t> map(kV);
        for (std::uint32_t t = 0; t < kV; ++t) {
            map[t] = t % 20;
        }
        arr(p + "engram.token_map", 5, map);
    }

    GgufBuilder g;
    g.header(ts.size(), kv.size());
    for (auto& w : kv) {
        w(g);
    }
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

std::vector<std::vector<float>> run(const TransformerModel& model,
                                    const std::vector<locus::tok::TokenId>& toks) {
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<std::vector<float>> out;
    std::vector<float> logits(model.hparams().n_vocab);
    for (auto t : toks) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(t, cache, seq, ws, logits);
        out.push_back(logits);
    }
    cache.release(seq);
    return out;
}

}  // namespace

TEST_CASE("deepseek41 sinkhorn normalizes rows and columns", "[ds41]") {
    std::vector<float> m = {0.3f, -1.0f, 2.0f, 0.5f, 1.5f, 0.0f, -0.7f, 0.9f,
                            -2.0f, 0.4f, 1.1f, 0.2f, 0.8f, -0.3f, 0.6f, 1.9f};
    ds41::sinkhorn(m, 4, 20, 1e-6f);
    for (std::uint32_t i = 0; i < 4; ++i) {
        float row = 0.0f, col = 0.0f;
        for (std::uint32_t j = 0; j < 4; ++j) {
            row += m[i + 4 * j];  // dst i over src
            col += m[j + 4 * i];  // src i over dst
            REQUIRE(m[i + 4 * j] > 0.0f);
        }
        REQUIRE(row == Catch::Approx(1.0f).margin(1e-4));
        REQUIRE(col == Catch::Approx(1.0f).margin(1e-3));
    }
}

TEST_CASE("deepseek41 expert weights renormalize only past one expert",
          "[ds41]") {
    // ec2's #166 note: the reference normalizes the top-k weights only
    // with expert_weights_norm and n_expert_used > 1.
    locus::model::Hparams hp;
    hp.expert_weights_norm = true;
    hp.expert_weights_scale = 1.5f;
    const float probs[] = {0.2f, 0.9f, 0.3f};  // picked pair sums 1.2
    const std::uint32_t two[] = {1, 2};
    auto w = ds41::moe_weights(hp, probs, two);
    REQUIRE(w[0] == Catch::Approx(0.9f / 1.2f * 1.5f));
    REQUIRE(w[1] == Catch::Approx(0.3f / 1.2f * 1.5f));
    const std::uint32_t one[] = {1};
    w = ds41::moe_weights(hp, probs, one);  // a lone expert: raw prob
    REQUIRE(w[0] == Catch::Approx(0.9f * 1.5f));
    hp.expert_weights_norm = false;  // flag off: never normalized
    w = ds41::moe_weights(hp, probs, two);
    REQUIRE(w[0] == Catch::Approx(0.9f * 1.5f));
    REQUIRE(w[1] == Catch::Approx(0.3f * 1.5f));
}

TEST_CASE("deepseek41 engram rows hash the n-gram per order and head",
          "[ds41]") {
    locus::model::Hparams::Ds41 d;
    d.ngram = 4;
    d.ngram_heads = 2;
    d.engram_mult = {1, 3, 5, 7};
    d.engram_size = {kSizes, kSizes + 6};
    d.engram_off = {kOffs, kOffs + 6};
    // rolling: 3*1 = 3; ^ 1*3 -> 0; ^ 2*5 -> 10; ^ 2*7 -> 4.
    const std::int64_t toks[] = {3, 1, 2, 2};
    std::uint32_t rows[6];
    ds41::engram_rows(d, 0, toks, rows);
    const std::uint32_t want[] = {0 % 5 + 0,   0 % 7 + 5,   10 % 11 + 12,
                                  10 % 13 + 23, 4 % 17 + 36, 4 % 19 + 53};
    for (int c = 0; c < 6; ++c) {
        REQUIRE(rows[c] == want[c]);
    }
}

TEST_CASE("deepseek41 tiny model decodes through every layer class",
          "[ds41][model]") {
    auto img = build_image();
    auto g = locus::gguf::GgufFile::parse(img);
    auto model = TransformerModel::load(g);
    const auto& d = model.hparams().ds41;
    REQUIRE(d.kv_src == std::vector<std::int32_t>{-1, 1, 1, 3, 3});
    REQUIRE(d.topk_src == std::vector<std::int32_t>{-1, 1, 1, 3, 4});
    REQUIRE(model.cache_layer_count() == kL + 2);
    REQUIRE_FALSE(model.supports_batch());
    REQUIRE_FALSE(locus::backend::vulkan_forward_supported(model));
    // 14 tokens: past the window (3), past top-k (3 rows) on every
    // compressed layer, and past layer 3's 2 candidate blocks of 2.
    std::vector<locus::tok::TokenId> toks;
    for (int i = 0; i < 14; ++i) {
        toks.push_back((i * 7 + 3) % static_cast<int>(kV));
    }
    const auto a = run(model, toks);
    for (const auto& row : a) {
        for (float v : row) {
            REQUIRE(std::isfinite(v));
        }
    }
    // Decoding is a function of the sequence alone (no state leaks
    // between runs through the per-thread scratch).
    REQUIRE(run(model, toks) == a);
    // A different first token changes later logits (Engram history
    // and the compressed rows carry it past the window).
    auto toks2 = toks;
    toks2[0] = (toks[0] + 1) % static_cast<int>(kV);
    REQUIRE(run(model, toks2).back() != a.back());
}
