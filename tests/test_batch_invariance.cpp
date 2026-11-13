// i#403: a request's text must not depend on the batch it runs in. One
// sampled request (fixed seed) runs alone, then among other requests
// (batches of 2, 3 and 8, prompts of different lengths, one longer
// than a step's prefill budget); its tokens and logprobs must match
// bit for bit on every selectable backend.

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"
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
    // No EOS, so every request runs n_gen tokens. A 32-token prefill
    // budget per step makes the 80-token prompt's chunks depend on the
    // other requests. 32 blocks of 16 positions hold the largest batch;
    // the default, one full context window per Engine, does not fit a
    // small GPU's memory.
    locus::engine::Engine::Config cfg;
    cfg.n_blocks = 32;
    cfg.prefill_budget = 32;
    locus::engine::Engine engine(model, static_cast<TokenId>(-1), cfg);
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

/** run(), then free the Vulkan state its Engine leaves behind (a KV
 *  pool per cache stays allocated until a reset). */
std::pair<std::vector<TokenId>, std::vector<float>> run_and_reset(
    TransformerModel& model, std::uint32_t n_req, std::uint32_t len,
    std::uint32_t n_gen, bool vulkan) {
    auto out = run(model, n_req, len, n_gen);
    if (vulkan) {
        locus::backend::vulkan_reset_state();
    }
    return out;
}

void check_model(const std::string& path, std::uint32_t n_gen,
                 bool scalar = true, bool vulkan_ok = true) {
    if (!std::filesystem::exists(path)) {
        WARN(path + " absent: skipped");
        return;
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = TransformerModel::load(g);
    for (const auto& b : locus::backend::backends()) {
        if (!b.available || !b.selectable ||
            (!scalar && std::string_view(b.name) == "scalar") ||
            (!vulkan_ok && std::string_view(b.name) == "vulkan")) {
            continue;
        }
        DYNAMIC_SECTION("backend " << b.name) {
            model.use_backend(b);
            const bool vulkan = std::string_view(b.name) == "vulkan";
            // A short target, and one longer than a step's prefill
            // budget, so its chunks depend on the other requests.
            for (const std::uint32_t len : {7u, 80u}) {
                const auto alone = run_and_reset(model, 1, len, n_gen, vulkan);
                REQUIRE(alone.first.size() == n_gen);
                for (const std::uint32_t n : {2u, 3u, 8u}) {
                    INFO("prompt " << len << " tokens, batch of " << n);
                    const auto got =
                        run_and_reset(model, n, len, n_gen, vulkan);
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
    // The scalar backend takes hours on these; stories260K covers it.
    // On Vulkan a model is batch-invariant in batch-invariant mode
    // (#432; the default on Apple GPUs only, and there not for a lone
    // K-quant sequence), which these legs set.
    struct Mode {
        Mode() { locus::backend::vulkan_set_batch_invariant(true); }
        ~Mode() { locus::backend::vulkan_reset_batch_invariant(); }
    };
    DYNAMIC_SECTION("llama-3.2-1b-q8_0.gguf") {
        const Mode mode;
        check_model(kModels + "llama-3.2-1b-q8_0.gguf", 16, false);
    }
    DYNAMIC_SECTION("llama-3.2-1b-q4_k_m.gguf") {
        const Mode mode;
        check_model(kModels + "llama-3.2-1b-q4_k_m.gguf", 16, false);
    }
}

// i#682: a driver may hand a new buffer back with a freed one's old
// bytes, so a kernel that reads memory nothing wrote decodes the same in
// a fresh process (zeroed pages) and differently after other work. New
// buffers filled with NaN bytes, then with small finite ones, must give
// the bits of unfilled ones, alone and batched, in batch-invariant mode.
TEST_CASE("vulkan decode reads only memory it wrote",
          "[vulkan][invariance][e2e]") {
    const std::string path = kModels + "llama-3.2-1b-q8_0.gguf";
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no Vulkan device");
    }
    if (!std::filesystem::exists(path)) {
        SKIP("llama-3.2-1b-q8_0.gguf absent");
    }
    using locus::backend::vk::VulkanContext;
    struct Restore {
        ~Restore() {
            VulkanContext::set_poison(-1);
            locus::backend::vulkan_reset_batch_invariant();
        }
    } const restore;
    locus::backend::vulkan_set_batch_invariant(true);
    auto g = locus::gguf::GgufFile::open(path);
    auto model = TransformerModel::load(g);
    model.use_backend(*locus::backend::find_backend("vulkan"));
    const auto decode = [&](int poison, std::uint32_t n_req) {
        locus::backend::vulkan_reset_state();
        VulkanContext::set_poison(poison);
        return run_and_reset(model, n_req, 7, 8, true);
    };
    for (const std::uint32_t n : {1u, 2u}) {
        INFO("batch of " << n);
        const auto clean = decode(-1, n);
        for (const int poison : {0xff, 0x3c}) {
            INFO("poison " << poison);
            const auto got = decode(poison, n);
            CHECK(got.first == clean.first);
            CHECK(got.second == clean.second);
        }
    }
}
