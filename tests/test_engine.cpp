#include <filesystem>
#include <span>
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

// R2 follow-up: even the small terminal Request RECORD must not
// accumulate forever. release() drops it on consume, and a backstop
// cap evicts the oldest terminal records so an abandoned request
// (client never consumes) cannot grow requests_ without bound.
TEST_CASE("finished request records are evicted (cap + release)",
          "[engine]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    Engine::Config cfg;
    cfg.max_retained_terminal = 3;  // tiny cap for the test
    Engine engine(model, tok.eos_id(), cfg);

    // Submit + drain one at a time so finish order == submit order.
    std::vector<std::uint64_t> ids;
    for (int i = 0; i < 8; ++i) {
        ids.push_back(
            engine.submit(tok.encode("Once upon a time", true), 2));
        engine.run_to_completion();  // no release: "abandoned"
    }

    // Only the last `cap` terminal records survive; older ones are
    // evicted oldest-first by the backstop.
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const bool retained = i >= ids.size() - cfg.max_retained_terminal;
        if (retained) {
            REQUIRE(engine.get(ids[i]) != nullptr);
        } else {
            REQUIRE(engine.get(ids[i]) == nullptr);  // evicted
        }
    }

    // release() drops a retained terminal record on consume.
    const std::uint64_t last = ids.back();
    REQUIRE(engine.get(last) != nullptr);
    engine.release(last);
    REQUIRE(engine.get(last) == nullptr);
    // release() on an unknown id is a harmless no-op.
    engine.release(999999);
}

// Prompt caching (KV prefix reuse): the per-request accounting the API
// usage block reports. First request writes the block-aligned prefix
// to the cache (cached > 0, reused == 0); an identical second request
// adopts it (reused > 0, cached == 0).
TEST_CASE("prefix cache reports reused/cached token accounting",
          "[engine]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    Engine::Config cfg;
    cfg.prefix_cache = true;
    Engine engine(model, tok.eos_id(), cfg);

    // 64 tokens spans several KV blocks whatever the block size, so a
    // cacheable block-aligned prefix exists. Token id 5 avoids BOS/EOS.
    const std::vector<locus::tok::TokenId> prompt(64, 5);

    const auto id1 = engine.submit(prompt, 2);
    engine.run_to_completion();
    const auto* r1 = engine.get(id1);
    REQUIRE(r1 != nullptr);
    REQUIRE(r1->status == Status::kDone);
    REQUIRE(r1->reused_prefix_tokens == 0);  // nothing cached yet
    REQUIRE(r1->cached_prefix_tokens > 0);   // wrote the prefix

    const auto id2 = engine.submit(prompt, 2);
    engine.run_to_completion();
    const auto* r2 = engine.get(id2);
    REQUIRE(r2 != nullptr);
    REQUIRE(r2->status == Status::kDone);
    REQUIRE(r2->reused_prefix_tokens > 0);   // adopted the cached prefix
    REQUIRE(r2->cached_prefix_tokens == 0);  // already fully cached
    // Reused can't exceed the prompt, and leaves >=1 token to reprefill.
    REQUIRE(r2->reused_prefix_tokens < prompt.size());
}

TEST_CASE("prefix cache adopts a long prompt without self-eviction",
          "[engine]") {
    // Regression: admission must adopt the cached prefix BEFORE it
    // sizes (and evicts for) the incoming prompt. A prompt long enough
    // that its pinned prefix leaves too few free blocks for its own
    // FULL length used to evict that prefix to admit itself, then find
    // nothing to adopt -- so the second identical request re-prefilled
    // from scratch instead of reusing.
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    std::string s;
    for (int i = 0; i < 40; ++i) {
        s += "the quick brown fox jumps over the lazy dog. ";
    }
    const auto prompt = tok.encode(s, true);

    Engine::Config cfg;
    cfg.prefix_cache = true;
    Engine engine(model, tok.eos_id(), cfg);
    CAPTURE(prompt.size(), engine.total_blocks());

    const auto id1 = engine.submit(prompt, 2);
    engine.run_to_completion();
    CAPTURE(static_cast<int>(engine.get(id1)->status),
            engine.free_blocks());
    REQUIRE(engine.get(id1)->status == Status::kDone);

    const auto id2 = engine.submit(prompt, 2);
    engine.run_to_completion();
    CAPTURE(static_cast<int>(engine.get(id2)->status),
            engine.get(id2)->reused_prefix_tokens,
            engine.free_blocks());
    REQUIRE(engine.get(id2)->status == Status::kDone);
    // The second identical request must ADOPT the cached prefix, not
    // evict it to admit itself. Pre-fix this was 0.
    REQUIRE(engine.get(id2)->reused_prefix_tokens > 0);
}

// Multi-server (i#3) increment 1: running the layer stack in slices
// via forward_layers -- [0,k) then [k,L) against one shared cache --
// must be byte-identical to a single forward(). This is the in-process
// proof of the pipeline-parallel hand-off before any networking: the
// hidden state handed between slices reconstructs the exact same logits
// and the exact same greedy continuation.
TEST_CASE("execute-slice: layer-range forward matches full forward",
          "[engine][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    const std::uint32_t L = model.hparams().n_layers;
    const std::uint32_t E = model.hparams().n_embd;
    const std::uint32_t V = model.hparams().n_vocab;
    REQUIRE(L >= 2);
    const auto prompt =
        tok.encode("Once upon a time, there was a little", true);
    constexpr int kGen = 12;

    // Drives prefill + greedy generation, feeding each token through
    // `step`. Captures the next-token logits right after prefill (a
    // tight bitwise check) and the generated token sequence.
    auto drive = [&](auto&& step, std::vector<float>& first_logits,
                     std::vector<locus::tok::TokenId>& gen) {
        auto cache = model.make_cache();
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<float> logits(V);
        for (auto t : prompt) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            step(t, cache, seq, ws, logits);
        }
        first_logits.assign(logits.begin(), logits.end());
        for (int i = 0; i < kGen; ++i) {
            const auto n = locus::model::argmax(logits);
            gen.push_back(n);
            if (n == tok.eos_id()) {
                break;
            }
            REQUIRE(cache.ensure_capacity(seq, 1));
            step(n, cache, seq, ws, logits);
        }
    };

    // Reference: the real single-call forward().
    auto mono = [&](locus::tok::TokenId t, locus::kv::PagedKvCache& c,
                    locus::kv::PagedKvCache::Seq& s,
                    locus::model::LlamaModel::Workspace& w,
                    std::span<float> out) {
        model.forward(t, c, s, w, out);
    };
    std::vector<float> ref_logits;
    std::vector<locus::tok::TokenId> ref_gen;
    drive(mono, ref_logits, ref_gen);
    REQUIRE(ref_gen.size() >= 1);

    // Runs one token through the ordered stage boundaries `bounds`
    // (0 .. L), handing the residual stream between slices.
    auto sliced = [&](const std::vector<std::uint32_t>& bounds) {
        return [&, bounds](locus::tok::TokenId t,
                           locus::kv::PagedKvCache& c,
                           locus::kv::PagedKvCache::Seq& s,
                           locus::model::LlamaModel::Workspace& w,
                           std::span<float> out) {
            std::vector<float> hidden(E);
            std::vector<float> prev;
            for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
                const std::uint32_t a = bounds[i], b = bounds[i + 1];
                const bool firstS = i == 0;
                const bool lastS = i + 2 == bounds.size();
                std::span<const float> hin =
                    firstS ? std::span<const float>{}
                           : std::span<const float>(prev);
                std::span<float> ob =
                    lastS ? out : std::span<float>(hidden);
                model.forward_layers(firstS ? t : 0, hin, a, b, c, s,
                                     w, ob);
                if (!lastS) {
                    prev = hidden;  // hand off to the next slice
                }
            }
        };
    };

    // Every 2-way split point must reproduce the reference exactly.
    for (std::uint32_t k = 1; k < L; ++k) {
        std::vector<float> k_logits;
        std::vector<locus::tok::TokenId> k_gen;
        drive(sliced({0, k, L}), k_logits, k_gen);
        CAPTURE(k);
        REQUIRE(k_gen == ref_gen);
        REQUIRE(k_logits == ref_logits);  // bitwise-identical logits
    }

    // A 3-way split exercises a middle slice (hidden in AND out).
    if (L >= 3) {
        std::vector<float> m_logits;
        std::vector<locus::tok::TokenId> m_gen;
        drive(sliced({0, 1, L - 1, L}), m_logits, m_gen);
        REQUIRE(m_gen == ref_gen);
        REQUIRE(m_logits == ref_logits);
    }
}
