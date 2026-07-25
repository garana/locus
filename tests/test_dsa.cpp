#include <cmath>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "gguf_builder.hpp"

using locus::model::LlamaModel;

namespace {

/** Deterministic small pseudo-random weights in [-0.1, 0.1]. */
std::vector<float> weights(std::size_t n, std::uint32_t seed) {
    std::vector<float> v(n);
    std::uint32_t s = seed * 2654435761u + 12345u;
    for (auto& f : v) {
        s = s * 1664525u + 1013904223u;
        f = (static_cast<float>(s >> 8) /
                 static_cast<float>(1u << 24) -
             0.5f) *
            0.2f;
    }
    return v;
}

// Mini MLA model: embd 8, 2 heads, qk 4 (nope 2 + rope 2),
// v_head 2, q_lora 4, kv_lora 4; DSA indexer 2 heads x 4 dims,
// top_k 4. The SAME tensor data is emitted under "glm-dsa"
// (indexer active) and "deepseek2" (indexer tensors unused), so
// below top_k the two must be bit-identical.
constexpr std::uint32_t kE = 8;
constexpr std::uint32_t kV = 16;
constexpr std::uint32_t kFf = 16;
constexpr std::uint32_t kTopK = 4;

std::vector<std::byte> build_image(const std::string& arch) {
    struct T {
        std::string name;
        std::vector<std::uint64_t> dims;
        std::vector<float> data;
    };
    std::vector<T> ts;
    auto add = [&](std::string name,
                   std::vector<std::uint64_t> dims,
                   std::uint32_t seed) {
        std::size_t n = 1;
        for (auto d : dims) {
            n *= d;
        }
        ts.push_back({std::move(name), std::move(dims),
                      weights(n, seed)});
    };
    add("token_embd.weight", {kE, kV}, 1);
    ts.push_back({"output_norm.weight",
                  {kE},
                  std::vector<float>(kE, 1.0f)});
    ts.push_back({"blk.0.attn_norm.weight",
                  {kE},
                  std::vector<float>(kE, 1.0f)});
    ts.push_back({"blk.0.ffn_norm.weight",
                  {kE},
                  std::vector<float>(kE, 1.0f)});
    add("blk.0.attn_q_a.weight", {kE, 4}, 2);
    ts.push_back({"blk.0.attn_q_a_norm.weight",
                  {4},
                  std::vector<float>(4, 1.0f)});
    add("blk.0.attn_q_b.weight", {4, 8}, 3);
    add("blk.0.attn_kv_a_mqa.weight", {kE, 6}, 4);
    ts.push_back({"blk.0.attn_kv_a_norm.weight",
                  {4},
                  std::vector<float>(4, 1.0f)});
    add("blk.0.attn_kv_b.weight", {4, 8}, 5);
    add("blk.0.attn_output.weight", {4, kE}, 6);
    add("blk.0.ffn_gate.weight", {kE, kFf}, 7);
    add("blk.0.ffn_up.weight", {kE, kFf}, 8);
    add("blk.0.ffn_down.weight", {kFf, kE}, 9);
    add("blk.0.indexer.proj.weight", {kE, 2}, 10);
    add("blk.0.indexer.attn_k.weight", {kE, 4}, 11);
    add("blk.0.indexer.attn_q_b.weight", {4, 8}, 12);
    ts.push_back({"blk.0.indexer.k_norm.weight",
                  {4},
                  std::vector<float>(4, 1.0f)});
    ts.push_back({"blk.0.indexer.k_norm.bias",
                  {4},
                  std::vector<float>(4, 0.0f)});

    const bool glm = arch == "glm-dsa";
    const std::string p = arch + ".";
    GgufBuilder b;
    b.header(ts.size(), glm ? 15 : 12)
        .kv_string("general.architecture", arch)
        .kv_u32(p + "embedding_length", kE)
        .kv_u32(p + "block_count", 1)
        .kv_u32(p + "attention.head_count", 2)
        .kv_u32(p + "feed_forward_length", kFf)
        .kv_u32(p + "context_length", 32)
        .kv_f32(p + "attention.layer_norm_rms_epsilon", 1e-5f)
        .kv_u32(p + "attention.q_lora_rank", 4)
        .kv_u32(p + "attention.kv_lora_rank", 4)
        .kv_u32(p + "rope.dimension_count", 2)
        .kv_u32(p + "attention.key_length", 4)
        .kv_u32(p + "attention.value_length", 2);
    if (glm) {
        b.kv_u32(p + "attention.indexer.head_count", 2)
            .kv_u32(p + "attention.indexer.key_length", 4)
            .kv_u32(p + "attention.indexer.top_k", kTopK);
    }
    std::uint64_t offset = 0;
    for (const auto& t : ts) {
        b.tensor(t.name, t.dims, 0 /* F32 */, offset);
        offset = (offset + t.data.size() * 4 + 31) & ~31ull;
    }
    b.pad();
    for (const auto& t : ts) {
        const auto* p8 = reinterpret_cast<const std::byte*>(
            t.data.data());
        for (std::size_t i = 0; i < t.data.size() * 4; ++i) {
            b.u8(static_cast<std::uint8_t>(p8[i]));
        }
        while (b.bytes().size() % 32 != 0) {
            b.zeros(1);
        }
    }
    auto span = b.bytes();
    return {span.begin(), span.end()};
}

/** Per-position logits for `tokens` fed through the model. */
std::vector<std::vector<float>> run(
    const std::vector<std::byte>& image,
    const std::vector<locus::tok::TokenId>& tokens) {
    auto g = locus::gguf::GgufFile::parse(image);
    auto model = LlamaModel::load(g);
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<std::vector<float>> out;
    std::vector<float> logits(model.hparams().n_vocab);
    for (auto t : tokens) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(t, cache, seq, ws, logits);
        out.push_back(logits);
    }
    cache.release(seq);
    return out;
}

}  // namespace

TEST_CASE("batched forward matches sequential (deepseek2 MLA)",
          "[batch]") {
    // R10: forward_batch must be byte-identical to N forward()
    // calls on the MLA path (attention reused per token, dense
    // FFN batched).
    auto img = build_image("deepseek2");
    auto g = locus::gguf::GgufFile::parse(img);
    auto model = LlamaModel::load(g);
    REQUIRE(model.supports_batch());
    const std::vector<locus::tok::TokenId> toks = {3, 7, 1, 9};
    const auto& hp = model.hparams();
    // MLA caches one latent row (kv_lora + rope) per token; the V
    // half is unused, so compare only the K (latent) rows.
    const std::size_t kvd = hp.kv_lora_rank + hp.qk_rope_dim;

    auto snapshot = [&](bool batched) {
        auto cache = model.make_cache();
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<float> logits(hp.n_vocab);
        if (batched) {
            REQUIRE(cache.ensure_capacity(seq, toks.size()));
            model.forward_batch(toks, cache, seq, ws, logits);
        } else {
            for (auto t : toks) {
                REQUIRE(cache.ensure_capacity(seq, 1));
                model.forward(t, cache, seq, ws, logits);
            }
        }
        std::vector<float> kv;
        for (std::uint32_t l = 0; l < hp.n_layers; ++l) {
            for (std::uint32_t p = 0; p < seq.n_tokens; ++p) {
                const float* k = cache.k(seq, l, p);
                kv.insert(kv.end(), k, k + kvd);
            }
        }
        cache.release(seq);
        return std::pair{logits, kv};
    };

    auto [log_seq, kv_seq] = snapshot(false);
    auto [log_bat, kv_bat] = snapshot(true);
    REQUIRE(log_bat == log_seq);
    REQUIRE(kv_bat == kv_seq);
}

TEST_CASE("dsa index score applies relu and head weights",
          "[dsa]") {
    // 2 heads x 2 dims. Head 0 dot = 1*1 + 0*2 = 1 (kept),
    // head 1 dot = -1*1 + 0*2 = -1 (relu drops it).
    const std::vector<float> q = {1.0f, 0.0f, -1.0f, 0.0f};
    const std::vector<float> w = {0.5f, 100.0f};
    const std::vector<float> k = {1.0f, 2.0f};
    REQUIRE(locus::model::dsa_index_score(q, w, k) ==
            Catch::Approx(0.5f));
    // Negative head weight passes through unchanged.
    const std::vector<float> w2 = {-2.0f, 3.0f};
    REQUIRE(locus::model::dsa_index_score(q, w2, k) ==
            Catch::Approx(-2.0f));
}

TEST_CASE("glm-dsa equals dense MLA below indexer top_k",
          "[dsa]") {
    auto glm = build_image("glm-dsa");
    auto ds = build_image("deepseek2");
    const std::vector<locus::tok::TokenId> toks = {3, 7, 1, 9};
    REQUIRE(toks.size() <= kTopK);
    auto a = run(glm, toks);
    auto b = run(ds, toks);
    for (std::size_t i = 0; i < toks.size(); ++i) {
        REQUIRE(a[i] == b[i]);  // bit-identical
    }
}

TEST_CASE("glm-dsa selects top_k positions beyond the cap",
          "[dsa]") {
    auto glm = build_image("glm-dsa");
    auto ds = build_image("deepseek2");
    const std::vector<locus::tok::TokenId> toks = {3, 7, 1, 9,
                                                   2, 11};
    auto a = run(glm, toks);
    auto b = run(ds, toks);
    // Below the cap: identical (selection inactive).
    for (std::size_t i = 0; i < kTopK; ++i) {
        REQUIRE(a[i] == b[i]);
    }
    // Beyond: the indexer drops positions, dense does not.
    bool diverged = false;
    for (std::size_t i = kTopK; i < toks.size(); ++i) {
        for (std::size_t j = 0; j < a[i].size(); ++j) {
            REQUIRE(std::isfinite(a[i][j]));
            diverged =
                diverged || std::abs(a[i][j] - b[i][j]) > 1e-7f;
        }
    }
    REQUIRE(diverged);
}
