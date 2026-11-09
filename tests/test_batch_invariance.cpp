// i#403: a request's text must not depend on the batch it runs in. One
// sampled request (fixed seed) runs alone, then among other requests
// (batches of 2, 3 and 8, prompts of different lengths, one longer
// than a step's prefill budget); its tokens and logprobs must match
// bit for bit on every selectable backend.

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/engine/engine.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"

using locus::model::TransformerModel;
using locus::tok::TokenId;

namespace {

/** Prompt i: deterministic token ids below n_vocab; prompt 0 has `len`
 *  tokens, the others 3 + 5 i. */
std::vector<TokenId> prompt(std::uint32_t i, std::uint32_t len,
                            std::uint32_t n_vocab) {
    const std::uint32_t n = i == 0 ? len : 3 + 5 * i;
    std::vector<TokenId> p(n);
    for (std::uint32_t k = 0; k < n; ++k) {
        p[k] = static_cast<TokenId>((1 + 37 * k + 101 * i) % (n_vocab - 1) +
                                    1);
    }
    return p;
}

/** Request 0's tokens and logprobs (chosen, then the top 5) when it runs
 *  with n_req - 1 others. */
std::pair<std::vector<TokenId>, std::vector<float>> run(
    TransformerModel& model, std::uint32_t n_req, std::uint32_t len,
    std::uint32_t n_gen) {
    locus::model::SamplingParams sp;
    sp.temperature = 0.8f;
    sp.top_p = 0.95f;
    // No EOS, so every request runs n_gen tokens.
    locus::engine::Engine engine(model, static_cast<TokenId>(-1), {});
    std::vector<std::uint64_t> ids;
    for (std::uint32_t i = 0; i < n_req; ++i) {
        ids.push_back(engine.submit(prompt(i, len, model.hparams().n_vocab),
                                    n_gen, sp, 1000 + i, nullptr,
                                    {true, 5}));
    }
    engine.run_to_completion();
    const auto* r = engine.get(ids[0]);
    std::vector<float> lp;
    for (const auto& e : r->logprobs) {
        lp.push_back(e.logprob);
        for (const auto& t : e.top) {
            lp.push_back(t.logprob);
        }
    }
    return {r->generated, lp};
}

void check_model(const std::string& path, std::uint32_t n_gen) {
    if (!std::filesystem::exists(path)) {
        WARN(path + " absent: skipped");
        return;
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = TransformerModel::load(g);
    for (const auto& b : locus::backend::backends()) {
        if (!b.available || !b.selectable) {
            continue;
        }
        DYNAMIC_SECTION("backend " << b.name) {
            model.use_backend(b);
            // A short target, and one longer than a step's prefill
            // budget, so its chunks depend on the other requests.
            for (const std::uint32_t len : {7u, 600u}) {
                const auto alone = run(model, 1, len, n_gen);
                REQUIRE(alone.first.size() == n_gen);
                for (const std::uint32_t n : {2u, 3u, 8u}) {
                    INFO("prompt " << len << " tokens, batch of " << n);
                    const auto got = run(model, n, len, n_gen);
                    CHECK(got.first == alone.first);
                    CHECK(got.second == alone.second);
                }
            }
        }
    }
}

const std::string kModels = std::string(LOCUS_SOURCE_DIR) + "/tests/models/";

}  // namespace

TEST_CASE("a request decodes the same alone or batched",
          "[batch][invariance]") {
    check_model(kModels + "stories260K.gguf", 24);
}

TEST_CASE("a request decodes the same alone or batched: Q8_0 and Q4_K_M",
          "[batch][invariance][e2e]") {
    for (const char* f :
         {"llama-3.2-1b-q8_0.gguf", "llama-3.2-1b-q4_k_m.gguf"}) {
        DYNAMIC_SECTION(f) { check_model(kModels + f, 16); }
    }
}
