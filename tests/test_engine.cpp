#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/engine/engine.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::engine::Engine;
using locus::engine::Status;

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

/** Single-sequence reference generation via the raw model API. */
std::vector<locus::tok::TokenId> reference_generate(
    const locus::model::LlamaModel& model,
    const locus::tok::SpmTokenizer& tok,
    const std::vector<locus::tok::TokenId>& prompt,
    std::uint32_t max_new) {
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    for (auto id : prompt) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(id, cache, seq, ws, logits);
    }
    std::vector<locus::tok::TokenId> out;
    for (std::uint32_t i = 0; i < max_new; ++i) {
        auto next = locus::model::argmax(logits);
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
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    const std::vector<std::string> prompts = {
        "Once upon a time", "The little dog", "One day, Tom",
        "Once upon a time"};
    std::vector<std::vector<locus::tok::TokenId>> want;
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

TEST_CASE("batched prefill matches per-token prefill (engine)",
          "[engine][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    REQUIRE(model.supports_batch());  // stories260K is dense llama
    const auto prompt =
        tok.encode("Once upon a time, there was", true);

    auto run = [&](bool batched) {
        Engine::Config cfg;
        cfg.batched_prefill = batched;
        Engine engine(model, tok.eos_id(), cfg);
        auto id = engine.submit(prompt, 24);
        engine.run_to_completion();
        return engine.get(id)->generated;
    };
    // R10: batched prefill is byte-identical, so identical tokens.
    REQUIRE(run(true) == run(false));
}

TEST_CASE("batched decode matches the per-sequence scheduler",
          "[engine][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    REQUIRE(model.supports_batch());

    const std::vector<std::string> prompts = {
        "Once upon a time", "The little dog", "One day, Tom",
        "Once upon a time"};
    auto run = [&](bool batched, std::uint32_t n_blocks,
                   std::uint32_t headroom) {
        Engine::Config cfg;
        cfg.batched_decode = batched;
        cfg.n_blocks = n_blocks;
        cfg.decode_headroom = headroom;
        Engine engine(model, tok.eos_id(), cfg);
        std::vector<std::uint64_t> ids;
        for (const auto& p : prompts) {
            ids.push_back(engine.submit(tok.encode(p, true), 20));
        }
        engine.run_to_completion();
        std::vector<std::vector<locus::tok::TokenId>> outs;
        for (auto id : ids) {
            const auto* r = engine.get(id);
            REQUIRE(r->status == Status::kDone);
            outs.push_back(r->generated);
        }
        REQUIRE(engine.free_blocks() == engine.total_blocks());
        return outs;
    };
    // R10 4b: batched decode is byte-identical, so identical
    // tokens -- comfortable pool (pure batching) and a tight pool
    // that forces preemption + recompute in both schedulers.
    REQUIRE(run(true, 0, 16) == run(false, 0, 16));
    REQUIRE(run(true, 6, 1) == run(false, 6, 1));
}

TEST_CASE("preemption recomputes and still matches",
          "[engine][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

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

TEST_CASE("prefix cache reuses KV and stays byte-exact",
          "[engine][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    // A prompt longer than one block (16 tokens) so a full block is
    // cacheable.
    auto prompt = tok.encode(
        "Once upon a time, there was a little girl named Lily who "
        "loved to explore the forest near her home every day.",
        true);
    REQUIRE(prompt.size() > 16);

    Engine baseline(model, tok.eos_id(), Engine::Config{});
    const auto id = baseline.submit(prompt, 20);
    baseline.run_to_completion();
    const auto want = baseline.get(id)->generated;

    Engine::Config cfg;
    cfg.prefix_cache = true;
    Engine engine(model, tok.eos_id(), cfg);
    // First run: cache is empty, nothing reused, output matches.
    const auto a = engine.submit(prompt, 20);
    engine.run_to_completion();
    REQUIRE(engine.get(a)->generated == want);
    REQUIRE(engine.prefix_reused_tokens() == 0);
    // Second run of the same prompt: adopts the cached prefix (>= 1
    // block) and produces byte-identical output.
    const auto b = engine.submit(prompt, 20);
    engine.run_to_completion();
    REQUIRE(engine.get(b)->generated == want);
    REQUIRE(engine.prefix_reused_tokens() >= 16);
    REQUIRE(engine.free_blocks() < engine.total_blocks());  // pinned
}

TEST_CASE("speculative decoding matches greedy output",
          "[engine][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    // A repetitive prompt makes the greedy continuation echo earlier
    // n-grams, so prompt-lookup drafts get accepted (exercises the
    // accept path, not just the correction path). ngram=2 widens
    // matches on the toy vocab.
    auto prompt = tok.encode(
        "the cat sat on the mat the cat sat on the mat the cat sat "
        "on the mat the cat sat on the",
        true);

    Engine baseline(model, tok.eos_id(), Engine::Config{});
    const auto id = baseline.submit(prompt, 48);
    baseline.run_to_completion();
    const auto want = baseline.get(id)->generated;

    Engine::Config cfg;
    cfg.speculative = true;
    cfg.spec_ngram = 2;
    Engine engine(model, tok.eos_id(), cfg);
    const auto a = engine.submit(prompt, 48);
    engine.run_to_completion();
    // Greedy spec decode is byte-exact to plain greedy.
    REQUIRE(engine.get(a)->generated == want);
    REQUIRE(engine.spec_steps() > 0);            // verify ran
    REQUIRE(engine.spec_accepted_tokens() > 0);  // and accepted
}

TEST_CASE("engine on the vulkan backend matches CPU output",
          "[engine][e2e][vulkan]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    auto p0 = tok.encode("Once upon a time", true);
    auto p1 = tok.encode("The little dog", true);
    // References on the default (CPU) backend.
    auto want0 = reference_generate(model, tok, p0, 20);
    auto want1 = reference_generate(model, tok, p1, 20);

    model.use_backend(
        *locus::backend::find_backend("vulkan"));
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
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

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

TEST_CASE("admit_error rejects prompts the pool can never hold",
          "[engine]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    Engine engine(model, tok.eos_id());  // default full-context pool
    const std::uint32_t n_ctx = model.hparams().n_ctx;

    // A short prompt is admissible.
    REQUIRE(engine.admit_error(1).empty());
    // A prompt at or beyond the context window is not.
    REQUIRE_FALSE(engine.admit_error(n_ctx).empty());
    REQUIRE_FALSE(engine.admit_error(n_ctx + 100).empty());

    // A tiny KV pool rejects a prompt that would not fit it even
    // though the prompt is within the context window.
    Engine::Config tiny;
    tiny.n_blocks = 1;
    Engine small(model, tok.eos_id(), tiny);
    REQUIRE_FALSE(small.admit_error(n_ctx - 1).empty());
}

TEST_CASE("un-admissible prompt parks the loop instead of wedging it",
          "[engine]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    Engine engine(model, tok.eos_id());
    const std::uint32_t n_ctx = model.hparams().n_ctx;

    // Bypass admit_error and hand the engine a prompt larger than the
    // whole context: it can never be admitted. Pre-fix, step() would
    // report "more work" forever and run_to_completion would spin
    // (the server's worker held its mutex the whole time -> wedge).
    std::vector<locus::tok::TokenId> huge(
        static_cast<std::size_t>(n_ctx) + 50, 1);
    const auto id = engine.submit(huge, 8);

    REQUIRE_FALSE(engine.has_runnable_work());
    engine.run_to_completion();  // must return, not hang
    REQUIRE(engine.get(id)->status == Status::kWaiting);

    // A well-sized request submitted alongside is still runnable.
    Engine ok(model, tok.eos_id());
    ok.submit(tok.encode("Once upon a time", true), 4);
    REQUIRE(ok.has_runnable_work());
}

// R2 (#59): a finished request must release its heavy scratch/input
// buffers -- r.logits (n_vocab floats) and r.prompt -- or a long-lived
// server retains them for every completed request and grows without
// bound. The client result (generated / logprobs) is retained.
TEST_CASE("finished requests reclaim their scratch buffers",
          "[engine][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    Engine engine(model, tok.eos_id());
    const auto id =
        engine.submit(tok.encode("Once upon a time", true), 24);
    engine.run_to_completion();

    const auto* r = engine.get(id);
    REQUIRE(r->status == Status::kDone);
    REQUIRE_FALSE(r->generated.empty());       // result kept
    // Pre-fix: logits.capacity()==n_vocab, prompt still populated.
    REQUIRE(r->logits.capacity() == 0);        // n_vocab scratch freed
    REQUIRE(r->prompt.empty());                // prompt freed
}
