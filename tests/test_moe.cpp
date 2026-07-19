#include <cstring>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "cppllm/gguf/gguf.hpp"
#include "cppllm/model/llama.hpp"
#include "gguf_builder.hpp"

using cppllm::model::LlamaModel;

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
                       const std::vector<cppllm::tok::TokenId>&
                           tokens) {
    auto g = cppllm::gguf::GgufFile::parse(image);
    auto model = LlamaModel::load(g);
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    cppllm::kv::PagedKvCache::Seq seq;
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

    const std::vector<cppllm::tok::TokenId> toks = {3, 7, 1};
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

    const std::vector<cppllm::tok::TokenId> toks = {5, 2};
    auto a = run(dense_img, toks);
    auto b = run(moe_img, toks);
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(b[i] == Catch::Approx(a[i]).margin(1e-4));
    }
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

    const std::vector<cppllm::tok::TokenId> toks = {9};
    auto a = run(dense_img, toks);
    auto b = run(moe_img, toks);
    bool differs = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::isfinite(b[i]));
        differs = differs || std::abs(a[i] - b[i]) > 1e-6;
    }
    REQUIRE(differs);
}
