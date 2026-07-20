#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "cppllm/gguf/gguf.hpp"
#include "cppllm/model/llama.hpp"
#include "cppllm/tok/tokenizer.hpp"

namespace {

std::string model_path() {
    return std::string(CPPLLM_SOURCE_DIR) +
           "/tests/models/deepseek-v2-lite-q4_k_m.gguf";
}

}  // namespace

TEST_CASE("deepseek-v2-lite matches llama.cpp token-exact",
          "[deepseek][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present (deepseek-v2-lite-q4_k_m.gguf, "
             "~10GB)");
    }
    auto g = cppllm::gguf::GgufFile::open(model_path());
    auto model = cppllm::model::LlamaModel::load(g);
    const auto& hp = model.hparams();
    REQUIRE(hp.arch == cppllm::model::Arch::kDeepseek2);
    REQUIRE(hp.kv_lora_rank == 512);
    REQUIRE(hp.qk_rope_dim == 64);
    REQUIRE(hp.n_expert == 64);
    REQUIRE(hp.n_expert_used == 6);
    REQUIRE(hp.n_expert_shared == 2);
    REQUIRE(hp.n_dense_lead == 1);

    auto tok = cppllm::tok::tokenizer_from_gguf(g);
    // llama-tokenize golden.
    REQUIRE(tok->encode("The capital of France is", true) ==
            std::vector<cppllm::tok::TokenId>{100000, 549, 6077,
                                              280, 7239, 317});

    // llama-completion --temp 0 --no-conversation golden.
    const std::string want =
        "Once upon a time, there was a girl who was a little "
        "bit different. She was a little bit shy, a little bit "
        "awkward,";

    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    cppllm::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(hp.n_vocab);
    auto ids = tok->encode("Once upon a time", true);
    for (auto id : ids) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(id, cache, seq, ws, logits);
    }
    for (int i = 0; i < 24; ++i) {
        auto next = cppllm::model::argmax(logits);
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
