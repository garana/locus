#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
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
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
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
