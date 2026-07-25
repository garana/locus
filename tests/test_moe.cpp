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
