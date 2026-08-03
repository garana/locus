#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
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

/** Assembles a tiny 1-layer llama GGUF image in memory. */
class ModelBuilder {
  public:
    static constexpr std::uint32_t kEmbd = 8;
    static constexpr std::uint32_t kFf = 16;
    static constexpr std::uint32_t kVocab = 16;

    void tensor(const std::string& name,
                std::vector<std::uint64_t> dims,
                std::vector<float> data) {
        std::size_t n = 1;
        for (auto d : dims) {
            n *= d;
        }
        REQUIRE(n == data.size());
        tensors_.push_back(
            {name, std::move(dims), std::move(data)});
    }

    /** Standard non-FFN tensors shared by every test model. */
    void common() {
        tensor("token_embd.weight", {kEmbd, kVocab},
               weights(kEmbd * kVocab, 1));
        tensor("blk.0.attn_norm.weight", {kEmbd},
               std::vector<float>(kEmbd, 1.0f));
        tensor("blk.0.ffn_norm.weight", {kEmbd},
               std::vector<float>(kEmbd, 1.0f));
        tensor("output_norm.weight", {kEmbd},
               std::vector<float>(kEmbd, 1.0f));
        for (const char* w : {"attn_q", "attn_k", "attn_v",
                              "attn_output"}) {
            tensor("blk.0." + std::string(w) + ".weight",
                   {kEmbd, kEmbd}, weights(kEmbd * kEmbd, 2));
        }
    }

    /** @returns The complete GGUF image. */
    std::vector<std::byte> build(std::uint32_t n_expert,
                                 std::uint32_t n_used) {
        GgufBuilder b;
        const std::uint32_t n_kv =
            n_expert > 0 ? 10u : 8u;
        b.header(tensors_.size(), n_kv)
            .kv_string("general.architecture", "llama")
            .kv_u32("llama.embedding_length", kEmbd)
            .kv_u32("llama.block_count", 1)
            .kv_u32("llama.attention.head_count", 2)
            .kv_u32("llama.attention.head_count_kv", 2)
            .kv_u32("llama.feed_forward_length", kFf)
            .kv_u32("llama.context_length", 32)
            .kv_f32("llama.attention.layer_norm_rms_epsilon",
                    1e-5f);
        if (n_expert > 0) {
            b.kv_u32("llama.expert_count", n_expert)
                .kv_u32("llama.expert_used_count", n_used);
        }
        std::uint64_t offset = 0;
        for (const auto& t : tensors_) {
            b.tensor(t.name, t.dims, 0 /* F32 */, offset);
            offset = (offset + t.data.size() * 4 + 31) & ~31ull;
        }
        b.pad();
        for (const auto& t : tensors_) {
            bytes_append(b,
                         reinterpret_cast<const std::byte*>(
                             t.data.data()),
                         t.data.size() * 4);
            while (b.bytes().size() % 32 != 0) {
                b.zeros(1);
            }
        }
        auto span = b.bytes();
        return {span.begin(), span.end()};
    }

    /**
     * Assembles a qwen2moe image (gated shared expert + q/k/v bias).
     * The caller adds token_embd/norms via common(), the attention
     * biases, the routed experts, the shared expert, and the shared-
     * expert gate; routed/shared FFN widths come from tensor shapes.
     */
    std::vector<std::byte> build_qwen2moe(std::uint32_t n_expert,
                                          std::uint32_t n_used) {
        GgufBuilder b;
        b.header(tensors_.size(), 10)
            .kv_string("general.architecture", "qwen2moe")
            .kv_u32("qwen2moe.embedding_length", kEmbd)
            .kv_u32("qwen2moe.block_count", 1)
            .kv_u32("qwen2moe.attention.head_count", 2)
            .kv_u32("qwen2moe.attention.head_count_kv", 2)
            .kv_u32("qwen2moe.feed_forward_length", kFf)
            .kv_u32("qwen2moe.context_length", 32)
            .kv_f32("qwen2moe.attention.layer_norm_rms_epsilon",
                    1e-5f)
            .kv_u32("qwen2moe.expert_count", n_expert)
            .kv_u32("qwen2moe.expert_used_count", n_used);
        std::uint64_t offset = 0;
        for (const auto& t : tensors_) {
            b.tensor(t.name, t.dims, 0 /* F32 */, offset);
            offset = (offset + t.data.size() * 4 + 31) & ~31ull;
        }
        b.pad();
        for (const auto& t : tensors_) {
            bytes_append(b,
                         reinterpret_cast<const std::byte*>(
                             t.data.data()),
                         t.data.size() * 4);
            while (b.bytes().size() % 32 != 0) {
                b.zeros(1);
            }
        }
        auto span = b.bytes();
        return {span.begin(), span.end()};
    }

  private:
    struct T {
        std::string name;
        std::vector<std::uint64_t> dims;
        std::vector<float> data;
    };

    static void bytes_append(GgufBuilder& b, const std::byte* p,
                             std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            b.u8(static_cast<std::uint8_t>(p[i]));
        }
    }

    std::vector<T> tensors_;
};

/** Greedy logits after feeding `tokens` through the model. */
std::vector<float> run(const std::vector<std::byte>& image,
                       const std::vector<locus::tok::TokenId>&
                           tokens,
                       const char* backend = nullptr) {
    auto g = locus::gguf::GgufFile::parse(image);
    auto model = LlamaModel::load(g);
    if (backend != nullptr) {
        model.use_backend(
            *locus::backend::find_backend(backend));
    }
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    for (auto t : tokens) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(t, cache, seq, ws, logits);
    }
    cache.release(seq);
    return logits;
}

constexpr std::uint32_t kE = ModelBuilder::kEmbd;
constexpr std::uint32_t kF = ModelBuilder::kFf;

/**
 * A tiny qwen2moe image with a gated shared expert. `gate_scale`
 * multiplies the shared-expert gate weights (so the sigmoid gate
 * differs between builds); `zero_shared` zeroes the shared down
 * projection so the shared expert output is exactly 0.
 */
std::vector<std::byte> qwen_image(float gate_scale, bool zero_shared) {
    constexpr std::uint32_t kNe = 4;    // experts
    constexpr std::uint32_t kFsh = 32;  // shared ff (distinct from kF)
    ModelBuilder mb;
    mb.common();  // token_embd, norms, attn_{q,k,v,output}.weight
    // Qwen carries q/k/v projection bias (kv_dim == kEmbd here).
    mb.tensor("blk.0.attn_q.bias", {kE}, weights(kE, 40));
    mb.tensor("blk.0.attn_k.bias", {kE}, weights(kE, 41));
    mb.tensor("blk.0.attn_v.bias", {kE}, weights(kE, 42));
    // Router + routed experts (width kF).
    mb.tensor("blk.0.ffn_gate_inp.weight", {kE, kNe},
              weights(kE * kNe, 43));
    mb.tensor("blk.0.ffn_gate_exps.weight", {kE, kF, kNe},
              weights(kE * kF * kNe, 44));
    mb.tensor("blk.0.ffn_up_exps.weight", {kE, kF, kNe},
              weights(kE * kF * kNe, 45));
    mb.tensor("blk.0.ffn_down_exps.weight", {kF, kE, kNe},
              weights(kF * kE * kNe, 46));
    // Shared expert at a DISTINCT width (kFsh != kF).
    mb.tensor("blk.0.ffn_gate_shexp.weight", {kE, kFsh},
              weights(kE * kFsh, 47));
    mb.tensor("blk.0.ffn_up_shexp.weight", {kE, kFsh},
              weights(kE * kFsh, 48));
    auto down = weights(kFsh * kE, 49);
    if (zero_shared) {
        std::fill(down.begin(), down.end(), 0.0f);
    }
    mb.tensor("blk.0.ffn_down_shexp.weight", {kFsh, kE}, down);
    // Shared-expert gate [n_embd -> 1].
    auto gate = weights(kE, 50);
    for (auto& g : gate) {
        g *= gate_scale;
    }
    mb.tensor("blk.0.ffn_gate_inp_shexp.weight", {kE, 1}, gate);
    return mb.build_qwen2moe(kNe, 2);
}

}  // namespace

TEST_CASE("identical experts reproduce the dense model exactly",
          "[moe]") {
    const auto wg = weights(kE * kF, 10);
    const auto wu = weights(kE * kF, 11);
    const auto wd = weights(kF * kE, 12);

    ModelBuilder dense;
    dense.common();
    dense.tensor("blk.0.ffn_gate.weight", {kE, kF}, wg);
    dense.tensor("blk.0.ffn_up.weight", {kE, kF}, wu);
    dense.tensor("blk.0.ffn_down.weight", {kF, kE}, wd);
    auto dense_img = dense.build(0, 0);

    // 4 experts, all with the dense weights: whatever the router
    // picks, the normalized mixture must equal the dense FFN.
    auto rep = [](const std::vector<float>& w, int n) {
        std::vector<float> out;
        for (int i = 0; i < n; ++i) {
            out.insert(out.end(), w.begin(), w.end());
        }
        return out;
    };
    ModelBuilder moe;
    moe.common();
    moe.tensor("blk.0.ffn_gate_inp.weight", {kE, 4},
               weights(kE * 4, 13));
    moe.tensor("blk.0.ffn_gate_exps.weight", {kE, kF, 4},
               rep(wg, 4));
    moe.tensor("blk.0.ffn_up_exps.weight", {kE, kF, 4},
               rep(wu, 4));
    moe.tensor("blk.0.ffn_down_exps.weight", {kF, kE, 4},
               rep(wd, 4));
    auto moe_img = moe.build(4, 2);

    const std::vector<locus::tok::TokenId> toks = {3, 7, 1};
    auto a = run(dense_img, toks);
    auto b = run(moe_img, toks);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(b[i] == Catch::Approx(a[i]).margin(1e-5));
    }
}

TEST_CASE("router forced to one expert matches that expert's "
          "dense model",
          "[moe]") {
    const auto wg = weights(kE * kF, 20);
    const auto wu = weights(kE * kF, 21);
    const auto wd = weights(kF * kE, 22);

    ModelBuilder dense;
    dense.common();
    dense.tensor("blk.0.ffn_gate.weight", {kE, kF}, wg);
    dense.tensor("blk.0.ffn_up.weight", {kE, kF}, wu);
    dense.tensor("blk.0.ffn_down.weight", {kF, kE}, wd);
    auto dense_img = dense.build(0, 0);

    // Expert 0 gets the dense weights, the others garbage. An
    // all-zero router makes the softmax uniform; the top-1
    // tie-break deterministically picks the lowest index (0) and
    // renormalization makes its weight exactly 1 -- independent
    // of the activations' sign, unlike a "huge logit row" trick.
    std::vector<float> gate_inp(kE * 4, 0.0f);
    auto splice = [&](const std::vector<float>& good,
                      std::size_t n, std::uint32_t seed) {
        std::vector<float> all;
        for (int e = 0; e < 4; ++e) {
            if (e == 0) {
                all.insert(all.end(), good.begin(), good.end());
            } else {
                auto ww = weights(
                    n, seed + static_cast<std::uint32_t>(e));
                all.insert(all.end(), ww.begin(), ww.end());
            }
        }
        return all;
    };
    ModelBuilder moe;
    moe.common();
    moe.tensor("blk.0.ffn_gate_inp.weight", {kE, 4}, gate_inp);
    moe.tensor("blk.0.ffn_gate_exps.weight", {kE, kF, 4},
               splice(wg, kE * kF, 30));
    moe.tensor("blk.0.ffn_up_exps.weight", {kE, kF, 4},
               splice(wu, kE * kF, 40));
    moe.tensor("blk.0.ffn_down_exps.weight", {kF, kE, 4},
               splice(wd, kF * kE, 50));
    auto moe_img = moe.build(4, 1);

    const std::vector<locus::tok::TokenId> toks = {5, 2};
    auto a = run(dense_img, toks);
    auto b = run(moe_img, toks);
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(b[i] == Catch::Approx(a[i]).margin(1e-4));
    }
}

TEST_CASE("moe on the vulkan backend matches dense",
          "[moe][vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    const auto wg = weights(kE * kF, 10);
    const auto wu = weights(kE * kF, 11);
    const auto wd = weights(kF * kE, 12);

    ModelBuilder dense;
    dense.common();
    dense.tensor("blk.0.ffn_gate.weight", {kE, kF}, wg);
    dense.tensor("blk.0.ffn_up.weight", {kE, kF}, wu);
    dense.tensor("blk.0.ffn_down.weight", {kF, kE}, wd);
    auto dense_img = dense.build(0, 0);

    auto rep = [](const std::vector<float>& w, int n) {
        std::vector<float> out;
        for (int i = 0; i < n; ++i) {
            out.insert(out.end(), w.begin(), w.end());
        }
        return out;
    };
    ModelBuilder moe;
    moe.common();
    moe.tensor("blk.0.ffn_gate_inp.weight", {kE, 4},
               weights(kE * 4, 13));
    moe.tensor("blk.0.ffn_gate_exps.weight", {kE, kF, 4},
               rep(wg, 4));
    moe.tensor("blk.0.ffn_up_exps.weight", {kE, kF, 4},
               rep(wu, 4));
    moe.tensor("blk.0.ffn_down_exps.weight", {kF, kE, 4},
               rep(wd, 4));
    auto moe_img = moe.build(4, 2);

    const std::vector<locus::tok::TokenId> toks = {3, 7, 1};
    auto a = run(dense_img, toks);  // CPU dense reference
    auto b = run(moe_img, toks, "vulkan");
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(b[i] == Catch::Approx(a[i]).margin(1e-3));
    }
}

TEST_CASE("readahead hints leave outputs bit-identical",
          "[moe]") {
    // R8: readahead (default on) is pure madvise hinting;
    // logits must not change at all vs LOCUS_NO_READAHEAD=1.
    // The in-memory image is not file-backed, so this also
    // covers the best-effort path where madvise fails.
    const auto wg = weights(kE * kF, 70);
    const auto wu = weights(kE * kF, 71);
    const auto wd = weights(kF * kE, 72);

    ModelBuilder dense;
    dense.common();
    dense.tensor("blk.0.ffn_gate.weight", {kE, kF}, wg);
    dense.tensor("blk.0.ffn_up.weight", {kE, kF}, wu);
    dense.tensor("blk.0.ffn_down.weight", {kF, kE}, wd);
    auto dense_img = dense.build(0, 0);

    ModelBuilder moe;
    moe.common();
    moe.tensor("blk.0.ffn_gate_inp.weight", {kE, 4},
               weights(kE * 4, 73));
    moe.tensor("blk.0.ffn_gate_exps.weight", {kE, kF, 4},
               weights(kE * kF * 4, 74));
    moe.tensor("blk.0.ffn_up_exps.weight", {kE, kF, 4},
               weights(kE * kF * 4, 75));
    moe.tensor("blk.0.ffn_down_exps.weight", {kF, kE, 4},
               weights(kF * kE * 4, 76));
    auto moe_img = moe.build(4, 2);

    const std::vector<locus::tok::TokenId> toks = {3, 7, 1};
    auto dense_on = run(dense_img, toks);  // default: hints on
    auto moe_on = run(moe_img, toks);

    setenv("LOCUS_NO_READAHEAD", "1", 1);
    auto dense_off = run(dense_img, toks);
    auto moe_off = run(moe_img, toks);
    unsetenv("LOCUS_NO_READAHEAD");

    REQUIRE(dense_on == dense_off);
    REQUIRE(moe_on == moe_off);
}

TEST_CASE("weight window drops file-backed experts losslessly",
          "[moe]") {
    ModelBuilder moe;
    moe.common();
    moe.tensor("blk.0.ffn_gate_inp.weight", {kE, 4},
               weights(kE * 4, 80));
    moe.tensor("blk.0.ffn_gate_exps.weight", {kE, kF, 4},
               weights(kE * kF * 4, 81));
    moe.tensor("blk.0.ffn_up_exps.weight", {kE, kF, 4},
               weights(kE * kF * 4, 82));
    moe.tensor("blk.0.ffn_down_exps.weight", {kF, kE, 4},
               weights(kF * kE * 4, 83));
    auto img = moe.build(4, 2);
    const std::vector<locus::tok::TokenId> toks = {3, 7, 1, 5};
    auto ref = run(img, toks);

    const auto path = std::filesystem::temp_directory_path() /
                      "locus_weight_window_test.gguf";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(img.data()),
                static_cast<std::streamsize>(img.size()));
    }
    setenv("LOCUS_WEIGHT_WINDOW", "1", 1);
    // File-backed: DONTNEED drops clean pages, later forwards
    // re-read them from the file -- logits must not change.
    {
        auto g = locus::gguf::GgufFile::open(path.string());
        REQUIRE(g.file_backed());
        auto model = LlamaModel::load(g);
        auto cache = model.make_cache();
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<float> logits(model.hparams().n_vocab);
        for (auto t : toks) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(t, cache, seq, ws, logits);
        }
        cache.release(seq);
        REQUIRE(logits == ref);
    }
    // In-memory image: the file_backed gate must keep DONTNEED
    // away from anonymous pages (it would DISCARD them).
    auto mem = run(img, toks);
    unsetenv("LOCUS_WEIGHT_WINDOW");
    REQUIRE(mem == ref);
    std::filesystem::remove(path);
}

TEST_CASE("batched forward matches N sequential forwards",
          "[batch]") {
    // R10: forward_batch(N) must be byte-identical to N
    // forward() calls -- logits AND the KV cache -- on the dense
    // llama path.
    ModelBuilder mb;
    mb.common();
    mb.tensor("blk.0.ffn_gate.weight", {kE, kF},
              weights(kE * kF, 40));
    mb.tensor("blk.0.ffn_up.weight", {kE, kF},
              weights(kE * kF, 41));
    mb.tensor("blk.0.ffn_down.weight", {kF, kE},
              weights(kF * kE, 42));
    auto img = mb.build(0, 0);  // dense llama
    auto g = locus::gguf::GgufFile::parse(img);
    auto model = LlamaModel::load(g);
    REQUIRE(model.supports_batch());

    const std::vector<locus::tok::TokenId> toks = {3, 7, 1, 5};
    const auto& hp = model.hparams();
    const std::size_t kvd =
        static_cast<std::size_t>(hp.n_kv_heads) * hp.head_dim;

    auto seq_run = [&](bool batched) {
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
        // Snapshot the KV cache for every layer and position.
        std::vector<float> kv;
        for (std::uint32_t l = 0; l < hp.n_layers; ++l) {
            for (std::uint32_t p = 0; p < seq.n_tokens; ++p) {
                const float* k = cache.k(seq, l, p);
                const float* v = cache.v(seq, l, p);
                kv.insert(kv.end(), k, k + kvd);
                kv.insert(kv.end(), v, v + kvd);
            }
        }
        cache.release(seq);
        return std::pair{logits, kv};
    };

    auto [log_seq, kv_seq] = seq_run(false);
    auto [log_bat, kv_bat] = seq_run(true);
    REQUIRE(log_bat == log_seq);  // bit-identical logits
    REQUIRE(kv_bat == kv_seq);    // bit-identical KV cache
}

TEST_CASE("batched decode matches per-sequence decode",
          "[batch]") {
    // R10 step 4b: forward_batch_decode over N independent
    // sequences must be byte-identical to N separate forward()
    // decode calls -- logits and each sequence's KV.
    ModelBuilder mb;
    mb.common();
    mb.tensor("blk.0.ffn_gate.weight", {kE, kF},
              weights(kE * kF, 90));
    mb.tensor("blk.0.ffn_up.weight", {kE, kF},
              weights(kE * kF, 91));
    mb.tensor("blk.0.ffn_down.weight", {kF, kE},
              weights(kF * kE, 92));
    auto img = mb.build(0, 0);
    auto g = locus::gguf::GgufFile::parse(img);
    auto model = LlamaModel::load(g);
    REQUIRE(model.supports_batch());
    const auto& hp = model.hparams();
    const std::uint32_t V = hp.n_vocab;
    const std::size_t kvd =
        static_cast<std::size_t>(hp.n_kv_heads) * hp.head_dim;

    // N sequences with distinct prefill contexts + a decode token.
    const std::vector<std::vector<locus::tok::TokenId>> ctx = {
        {3, 7}, {1, 5, 9}, {2}, {4, 8, 6, 0}};
    const std::vector<locus::tok::TokenId> next = {5, 2, 7, 1};
    const std::uint32_t N = 4;

    auto prefill = [&](locus::kv::PagedKvCache& cache,
                       std::vector<locus::kv::PagedKvCache::Seq>&
                           seqs,
                       LlamaModel::Workspace& ws) {
        std::vector<float> logits(V);
        for (std::uint32_t i = 0; i < N; ++i) {
            for (auto t : ctx[i]) {
                REQUIRE(cache.ensure_capacity(seqs[i], 1));
                model.forward(t, cache, seqs[i], ws, logits);
            }
        }
    };

    // Sequential decode.
    auto cacheA = model.make_cache(8);
    auto wsA = model.make_workspace();
    std::vector<locus::kv::PagedKvCache::Seq> seqsA(N);
    prefill(cacheA, seqsA, wsA);
    std::vector<std::vector<float>> logA(N,
                                         std::vector<float>(V));
    for (std::uint32_t i = 0; i < N; ++i) {
        REQUIRE(cacheA.ensure_capacity(seqsA[i], 1));
        model.forward(next[i], cacheA, seqsA[i], wsA, logA[i]);
    }

    // Batched decode.
    auto cacheB = model.make_cache(8);
    auto wsB = model.make_workspace();
    std::vector<locus::kv::PagedKvCache::Seq> seqsB(N);
    prefill(cacheB, seqsB, wsB);
    std::vector<locus::kv::PagedKvCache::Seq*> ptrs;
    for (std::uint32_t i = 0; i < N; ++i) {
        REQUIRE(cacheB.ensure_capacity(seqsB[i], 1));
        ptrs.push_back(&seqsB[i]);
    }
    std::vector<float> logB(static_cast<std::size_t>(N) * V);
    model.forward_batch_decode(next, cacheB, ptrs, wsB, logB);

    for (std::uint32_t i = 0; i < N; ++i) {
        REQUIRE(seqsB[i].n_tokens == seqsA[i].n_tokens);
        std::vector<float> got(
            logB.begin() + static_cast<std::ptrdiff_t>(i) * V,
            logB.begin() + static_cast<std::ptrdiff_t>(i + 1) * V);
        REQUIRE(got == logA[i]);  // byte-identical logits
        // Byte-identical KV for this sequence.
        for (std::uint32_t l = 0; l < hp.n_layers; ++l) {
            for (std::uint32_t p = 0; p < seqsA[i].n_tokens; ++p) {
                const float* ka = cacheA.k(seqsA[i], l, p);
                const float* kb = cacheB.k(seqsB[i], l, p);
                for (std::size_t j = 0; j < kvd; ++j) {
                    REQUIRE(ka[j] == kb[j]);
                }
            }
        }
    }
    for (std::uint32_t i = 0; i < N; ++i) {
        cacheA.release(seqsA[i]);
        cacheB.release(seqsB[i]);
    }
}

TEST_CASE("batched forward matches sequential (llama MoE)",
          "[batch]") {
    // R10: MoE FFN runs per token inside forward_batch, so it
    // stays byte-identical to the sequential path.
    ModelBuilder mb;
    mb.common();
    mb.tensor("blk.0.ffn_gate_inp.weight", {kE, 4},
              weights(kE * 4, 50));
    mb.tensor("blk.0.ffn_gate_exps.weight", {kE, kF, 4},
              weights(kE * kF * 4, 51));
    mb.tensor("blk.0.ffn_up_exps.weight", {kE, kF, 4},
              weights(kE * kF * 4, 52));
    mb.tensor("blk.0.ffn_down_exps.weight", {kF, kE, 4},
              weights(kF * kE * 4, 53));
    auto img = mb.build(4, 2);  // 4 experts, 2 routed
    auto g = locus::gguf::GgufFile::parse(img);
    auto model = LlamaModel::load(g);
    REQUIRE(model.supports_batch());

    const std::vector<locus::tok::TokenId> toks = {3, 7, 1, 5};
    const auto& hp = model.hparams();
    const std::size_t kvd =
        static_cast<std::size_t>(hp.n_kv_heads) * hp.head_dim;

    auto seq_run = [&](bool batched) {
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
                const float* v = cache.v(seq, l, p);
                kv.insert(kv.end(), k, k + kvd);
                kv.insert(kv.end(), v, v + kvd);
            }
        }
        cache.release(seq);
        return std::pair{logits, kv};
    };

    auto [log_seq, kv_seq] = seq_run(false);
    auto [log_bat, kv_bat] = seq_run(true);
    REQUIRE(log_bat == log_seq);
    REQUIRE(kv_bat == kv_seq);
}

TEST_CASE("distinct experts diverge from the dense model",
          "[moe]") {
    const auto wg = weights(kE * kF, 60);
    const auto wu = weights(kE * kF, 61);
    const auto wd = weights(kF * kE, 62);

    ModelBuilder dense;
    dense.common();
    dense.tensor("blk.0.ffn_gate.weight", {kE, kF}, wg);
    dense.tensor("blk.0.ffn_up.weight", {kE, kF}, wu);
    dense.tensor("blk.0.ffn_down.weight", {kF, kE}, wd);
    auto dense_img = dense.build(0, 0);

    ModelBuilder moe;
    moe.common();
    moe.tensor("blk.0.ffn_gate_inp.weight", {kE, 4},
               weights(kE * 4, 63));
    moe.tensor("blk.0.ffn_gate_exps.weight", {kE, kF, 4},
               weights(kE * kF * 4, 64));
    moe.tensor("blk.0.ffn_up_exps.weight", {kE, kF, 4},
               weights(kE * kF * 4, 65));
    moe.tensor("blk.0.ffn_down_exps.weight", {kF, kE, 4},
               weights(kF * kE * 4, 66));
    auto moe_img = moe.build(4, 2);

    const std::vector<locus::tok::TokenId> toks = {9};
    auto a = run(dense_img, toks);
    auto b = run(moe_img, toks);
    bool differs = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::isfinite(b[i]));
        differs = differs || std::abs(a[i] - b[i]) > 1e-6;
    }
    REQUIRE(differs);
}

namespace {

/**
 * Quantizes an f32 row-major matrix (rows x cols, cols a multiple
 * of 32) to Q8_0 blocks: [f16 scale][32 int8 quants] per 32 cols.
 */
std::vector<std::byte> quantize_q8_0(const std::vector<float>& w,
                                     std::uint32_t rows,
                                     std::uint32_t cols) {
    const std::uint32_t blocks = cols / 32;
    std::vector<std::byte> out(static_cast<std::size_t>(rows) *
                               blocks * 34);
    std::size_t o = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t b = 0; b < blocks; ++b) {
            const float* v = w.data() +
                             static_cast<std::size_t>(r) * cols +
                             b * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = std::max(amax, std::abs(v[i]));
            }
            const float d = amax / 127.0f;
            const float id = d > 0.0f ? 1.0f / d : 0.0f;
            std::uint16_t h = locus::backend::f32_to_f16(d);
            std::memcpy(out.data() + o, &h, 2);
            for (int i = 0; i < 32; ++i) {
                int q = static_cast<int>(std::lround(v[i] * id));
                q = std::max(-127, std::min(127, q));
                out[o + 2 + static_cast<std::size_t>(i)] =
                    static_cast<std::byte>(
                        static_cast<std::int8_t>(q));
            }
            o += 34;
        }
    }
    return out;
}

}  // namespace

TEST_CASE("dequant-amortized batch matvec matches per-token "
          "(f32 byte-identical, q8_0 token-exact)", "[batch]") {
    using locus::backend::Mat;
    // rows > kMinRowsPerSlice (64) so the threaded row-split path
    // is actually exercised below.
    constexpr std::uint32_t rows = 200, cols = 64, n = 5;
    const auto& op = locus::backend::find_backend("scalar")->ops;

    std::vector<float> w = weights(
        static_cast<std::size_t>(rows) * cols, 71);
    std::vector<float> xb = weights(
        static_cast<std::size_t>(n) * cols, 72);

    // F32: dequant_row is a plain copy and the row dot is the same
    // scalar accumulation matvec() uses, so the amortized kernel is
    // byte-identical to n sequential matvec() calls.
    {
        Mat m{locus::gguf::TensorType::kF32,
              reinterpret_cast<const std::byte*>(w.data()), rows,
              cols};
        std::vector<float> ref(static_cast<std::size_t>(n) * rows);
        for (std::uint32_t t = 0; t < n; ++t) {
            op.matvec(m, {xb.data() + t * cols, cols},
                      {ref.data() + t * rows, rows});
        }
        std::vector<float> got(static_cast<std::size_t>(n) * rows);
        locus::model::matvec_batch_deq(op, m, xb, got, n);
        REQUIRE(got == ref);
    }

    // Q8_0: fused matvec interleaves dequant with accumulation; the
    // amortized kernel dequants the whole row first. Same math,
    // reordered sums -> token-exact within a tight tolerance, not
    // bitwise equal.
    {
        auto q = quantize_q8_0(w, rows, cols);
        Mat m{locus::gguf::TensorType::kQ8_0, q.data(), rows, cols};
        std::vector<float> ref(static_cast<std::size_t>(n) * rows);
        for (std::uint32_t t = 0; t < n; ++t) {
            op.matvec(m, {xb.data() + t * cols, cols},
                      {ref.data() + t * rows, rows});
        }
        std::vector<float> got(static_cast<std::size_t>(n) * rows);
        locus::model::matvec_batch_deq(op, m, xb, got, n);
        for (std::size_t i = 0; i < ref.size(); ++i) {
            REQUIRE(got[i] == Catch::Approx(ref[i]).margin(1e-5));
        }
    }

    // Fused matvec_batch is the default (byte-identical) path: its
    // threaded row-split must stay bitwise equal to n sequential
    // matvec() calls at every thread count (like matvec_mt).
    {
        Mat m{locus::gguf::TensorType::kF32,
              reinterpret_cast<const std::byte*>(w.data()), rows,
              cols};
        std::vector<float> ref(static_cast<std::size_t>(n) * rows);
        for (std::uint32_t t = 0; t < n; ++t) {
            op.matvec(m, {xb.data() + t * cols, cols},
                      {ref.data() + t * rows, rows});
        }
        for (const char* nt : {"1", "2", "4"}) {
            setenv("LOCUS_THREADS", nt, 1);
            std::vector<float> got(
                static_cast<std::size_t>(n) * rows);
            locus::model::matvec_batch(op, m, xb, got, n);
            REQUIRE(got == ref);
        }
        unsetenv("LOCUS_THREADS");
    }
}

TEST_CASE("qwen2moe gated shared expert scales only the shared "
          "expert",
          "[moe]") {
    const std::vector<locus::tok::TokenId> toks = {2, 5, 1};

    // With an active shared expert, changing the gate changes the
    // output (the sigmoid gate is wired into the forward).
    auto a = run(qwen_image(+8.0f, /*zero_shared=*/false), toks);
    auto b = run(qwen_image(-8.0f, /*zero_shared=*/false), toks);
    REQUIRE(a.size() == b.size());
    bool differ = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        differ = differ || std::fabs(a[i] - b[i]) > 1e-4f;
    }
    REQUIRE(differ);

    // With the shared expert output forced to zero, the gate has no
    // effect at all -- so it touches ONLY the shared expert, nothing
    // in the routed/attention path.
    auto c = run(qwen_image(+8.0f, /*zero_shared=*/true), toks);
    auto d = run(qwen_image(-8.0f, /*zero_shared=*/true), toks);
    REQUIRE(c.size() == d.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
        REQUIRE(c[i] == Catch::Approx(d[i]).margin(1e-6));
    }
}
