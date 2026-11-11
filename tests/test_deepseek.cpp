#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/deepseek-v2-lite-q4_k_m.gguf";
}

}  // namespace

TEST_CASE("deepseek-v2-lite matches llama.cpp token-exact",
          "[deepseek][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present (deepseek-v2-lite-q4_k_m.gguf, "
             "~10GB)");
    }
    // Heavy: loads a ~10GB model and generates on CPU AND Vulkan.
    // Minutes on x86, but HOURS on a Pi (streaming + V3D Vulkan)
    // with OOM risk -- so it is opt-in, never run in a default
    // suite pass on a host that happens to have the model.
    if (std::getenv("LOCUS_HEAVY_TESTS") == nullptr) {
        SKIP("deepseek e2e is heavy (~10GB, CPU+Vulkan generation); "
             "set LOCUS_HEAVY_TESTS=1 to run it");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    // Pinned to the f32-activation dot: this golden is llama.cpp's
    // greedy text, which the f32 path reproduces on every leg; with
    // the R16 Q8 activations (the CPU default) the neon leg parts from
    // it after 15 generated tokens (" a little bit shy" vs " a girl who
    // was born"), as qwen2moe's does.
    model.use_q8k_activations(false);
    const auto& hp = model.hparams();
    REQUIRE(hp.arch == locus::model::Arch::kDeepseek2);
    REQUIRE(hp.kv_lora_rank == 512);
    REQUIRE(hp.qk_rope_dim == 64);
    REQUIRE(hp.n_expert == 64);
    REQUIRE(hp.n_expert_used == 6);
    REQUIRE(hp.n_expert_shared == 2);
    REQUIRE(hp.n_dense_lead == 1);

    auto tok = locus::tok::tokenizer_from_gguf(g);
    // llama-tokenize golden.
    REQUIRE(tok->encode("The capital of France is", true) ==
            std::vector<locus::tok::TokenId>{100000, 549, 6077,
                                              280, 7239, 317});

    // llama-completion --temp 0 --no-conversation golden.
    const std::string want =
        "Once upon a time, there was a girl who was a little "
        "bit different. She was a little bit shy, a little bit "
        "awkward,";

    // Run on the default (CPU) backend and, when usable, the
    // full GPU path (MLA + MoE dispatch) -- both token-exact.
    std::vector<const locus::backend::Backend*> backends = {
        &model.active_backend()};
    if (locus::backend::vulkan_backend_usable()) {
        backends.push_back(
            locus::backend::find_backend("vulkan"));
    }
    for (const auto* b : backends) {
        INFO("backend " << b->name);
        model.use_backend(*b);
        auto cache = model.make_cache();
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<float> logits(hp.n_vocab);
        auto ids = tok->encode("Once upon a time", true);
        for (auto id : ids) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(id, cache, seq, ws, logits);
        }
        for (int i = 0; i < 24; ++i) {
            auto next = locus::model::argmax(logits);
            if (next == tok->eos_id()) {
                break;
            }
            ids.push_back(next);
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(next, cache, seq, ws, logits);
        }
        cache.release(seq);
        REQUIRE(tok->decode(ids) == want);
        REQUIRE(cache.free_blocks() == cache.total_blocks());
    }
}

// i#63: a quantized KV cache on the real MLA model (CPU). It used to
// crash (the latents read the byte pool as floats); now each position's
// logits track the F32 cache's. Measured on an M2 Pro, worst position
// of "The capital of France is Paris": cosine 0.9989 (F16), 0.9987
// (Q8), 0.9951 (Q4). The greedy next token after " Paris" is a near tie
// in F32 (24.468 for "," against 24.432 for "."), which the quantized
// caches flip, so this compares logits, not text.
TEST_CASE("deepseek-v2-lite runs with a quantized KV cache",
          "[deepseek][e2e][kv]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present (deepseek-v2-lite-q4_k_m.gguf, ~10GB)");
    }
    if (std::getenv("LOCUS_HEAVY_TESTS") == nullptr) {
        SKIP("deepseek e2e is heavy (~10GB); set LOCUS_HEAVY_TESTS=1");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    model.use_backend(*locus::backend::find_backend("scalar"));
    auto tok = locus::tok::tokenizer_from_gguf(g);
    const auto ids = tok->encode("The capital of France is Paris", true);
    const auto run = [&](locus::kv::KvType kt) {
        auto cache = model.make_cache(0, kt);
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<float> logits(model.hparams().n_vocab);
        std::vector<std::vector<float>> out;
        for (auto id : ids) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(id, cache, seq, ws, logits);
            out.push_back(logits);
        }
        cache.release(seq);
        return out;
    };
    const auto cosine = [](const std::vector<float>& a,
                           const std::vector<float>& b) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            dot += static_cast<double>(a[i]) * b[i];
            na += static_cast<double>(a[i]) * a[i];
            nb += static_cast<double>(b[i]) * b[i];
        }
        return dot / std::sqrt(na * nb);
    };
    using locus::kv::KvType;
    const auto f32 = run(KvType::kF32);
    for (const auto& [kt, bound] :
         {std::pair{KvType::kF16, 0.998}, std::pair{KvType::kQ8, 0.998},
          std::pair{KvType::kQ4, 0.99}}) {
        const auto q = run(kt);
        for (std::size_t i = 0; i < f32.size(); ++i) {
            INFO("kv type " << static_cast<int>(kt) << " position " << i);
            CHECK(cosine(f32[i], q[i]) > bound);
        }
    }
}
