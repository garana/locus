#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "cppllm/backend/registry.hpp"
#include "cppllm/backend/variants.hpp"
#include "cppllm/engine/engine.hpp"
#include "cppllm/gguf/gguf.hpp"
#include "cppllm/model/llama.hpp"
#include "cppllm/tok/tokenizer.hpp"

using cppllm::engine::Engine;
using cppllm::engine::Status;

namespace {

std::string model_path() {
    return std::string(CPPLLM_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

/** Single-sequence reference generation via the raw model API. */
std::vector<cppllm::tok::TokenId> reference_generate(
    const cppllm::model::LlamaModel& model,
    const cppllm::tok::SpmTokenizer& tok,
    const std::vector<cppllm::tok::TokenId>& prompt,
    std::uint32_t max_new) {
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    cppllm::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    for (auto id : prompt) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(id, cache, seq, ws, logits);
    }
    std::vector<cppllm::tok::TokenId> out;
    for (std::uint32_t i = 0; i < max_new; ++i) {
        auto next = cppllm::model::argmax(logits);
        out.push_back(next);
        if (next == tok.eos_id()) {
            break;
        }
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(next, cache, seq, ws, logits);
    }
    return out;
}

}  // namespace

TEST_CASE("concurrent streams match single-sequence output",
          "[engine][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = cppllm::gguf::GgufFile::open(model_path());
    auto model = cppllm::model::LlamaModel::load(g);
    auto tok = cppllm::tok::SpmTokenizer::from_gguf(g);

    const std::vector<std::string> prompts = {
        "Once upon a time", "The little dog", "One day, Tom",
        "Once upon a time"};
    std::vector<std::vector<cppllm::tok::TokenId>> want;
    for (const auto& p : prompts) {
        want.push_back(reference_generate(
            model, tok, tok.encode(p, true), 24));
    }

    Engine engine(model, tok.eos_id());
    std::vector<std::uint64_t> ids;
    for (const auto& p : prompts) {
        ids.push_back(engine.submit(tok.encode(p, true), 24));
    }
    engine.run_to_completion();

    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto* r = engine.get(ids[i]);
        REQUIRE(r != nullptr);
        REQUIRE(r->status == Status::kDone);
        REQUIRE(r->generated == want[i]);
    }
    // Interleaved execution must leak nothing.
    REQUIRE(engine.free_blocks() == engine.total_blocks());
}

TEST_CASE("preemption recomputes and still matches",
          "[engine][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = cppllm::gguf::GgufFile::open(model_path());
    auto model = cppllm::model::LlamaModel::load(g);
    auto tok = cppllm::tok::SpmTokenizer::from_gguf(g);

    auto p0 = tok.encode("Once upon a time", true);
    auto p1 = tok.encode("The little dog", true);
    auto want0 = reference_generate(model, tok, p0, 24);
    auto want1 = reference_generate(model, tok, p1, 24);

    // Pool sized so two growing sequences collide: each needs up
    // to ~2-3 blocks (prompt + 24 tokens, block_tokens 16); 4
    // blocks force the newer sequence to be preempted.
    Engine::Config cfg;
    cfg.n_blocks = 4;
    cfg.decode_headroom = 1;
    Engine engine(model, tok.eos_id(), cfg);
    auto id0 = engine.submit(p0, 24);
    auto id1 = engine.submit(p1, 24);
    engine.run_to_completion();

    REQUIRE(engine.get(id0)->status == Status::kDone);
    REQUIRE(engine.get(id1)->status == Status::kDone);
    REQUIRE(engine.get(id0)->generated == want0);
    REQUIRE(engine.get(id1)->generated == want1);
    REQUIRE(engine.free_blocks() == engine.total_blocks());
}

TEST_CASE("engine on the vulkan backend matches CPU output",
          "[engine][e2e][vulkan]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    if (!cppllm::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    auto g = cppllm::gguf::GgufFile::open(model_path());
    auto model = cppllm::model::LlamaModel::load(g);
    auto tok = cppllm::tok::SpmTokenizer::from_gguf(g);

    auto p0 = tok.encode("Once upon a time", true);
    auto p1 = tok.encode("The little dog", true);
    // References on the default (CPU) backend.
    auto want0 = reference_generate(model, tok, p0, 20);
    auto want1 = reference_generate(model, tok, p1, 20);

    model.use_backend(
        *cppllm::backend::find_backend("vulkan"));
    Engine engine(model, tok.eos_id());
    auto id0 = engine.submit(p0, 20);
    auto id1 = engine.submit(p1, 20);
    engine.run_to_completion();

    REQUIRE(engine.get(id0)->status == Status::kDone);
    REQUIRE(engine.get(id1)->status == Status::kDone);
    REQUIRE(engine.get(id0)->generated == want0);
    REQUIRE(engine.get(id1)->generated == want1);
    REQUIRE(engine.free_blocks() == engine.total_blocks());
}

TEST_CASE("oversized request fails instead of wedging",
          "[engine][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = cppllm::gguf::GgufFile::open(model_path());
    auto model = cppllm::model::LlamaModel::load(g);
    auto tok = cppllm::tok::SpmTokenizer::from_gguf(g);

    Engine::Config cfg;
    cfg.n_blocks = 1;  // 16 positions total
    cfg.decode_headroom = 0;
    Engine engine(model, tok.eos_id(), cfg);
    auto id = engine.submit(tok.encode("Once upon a time", true),
                            64);
    engine.run_to_completion();

    REQUIRE(engine.get(id)->status == Status::kFailed);
    REQUIRE(engine.free_blocks() == engine.total_blocks());
}
