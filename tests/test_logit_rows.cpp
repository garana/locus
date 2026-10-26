#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"

namespace {

std::string model_dir() {
    return std::string(LOCUS_SOURCE_DIR) + "/tests/models/";
}

/**
 * Requires that forward() with a head row range gives, over three
 * tokens, the same logits bit for bit inside the range as a full
 * forward, and -inf outside it.
 */
void require_rows_match(locus::model::TransformerModel& model,
                        locus::model::LogitRows rows) {
    const std::uint32_t V = model.hparams().n_vocab;
    auto ws = model.make_workspace();
    auto full_cache = model.make_cache(64);
    auto rows_cache = model.make_cache(64);
    locus::kv::PagedKvCache::Seq fs, rs;
    std::vector<float> full(V), part(V);
    for (const locus::tok::TokenId t : {1, 9, 23}) {
        INFO("token " << t);
        REQUIRE(full_cache.ensure_capacity(fs, 1));
        REQUIRE(rows_cache.ensure_capacity(rs, 1));
        model.forward(t, full_cache, fs, ws, full);
        model.forward(t, rows_cache, rs, ws, part, rows);
        REQUIRE(std::memcmp(full.data() + rows.begin,
                            part.data() + rows.begin,
                            (rows.end - rows.begin) * sizeof(float)) == 0);
        for (std::uint32_t i = 0; i < V; ++i) {
            if (i < rows.begin || i >= rows.end) {
                REQUIRE(std::isinf(part[i]));
                REQUIRE(part[i] < 0.0f);
            }
        }
    }
}

}  // namespace

// i#173: speech computes only the audio band of the head. The computed
// rows must equal the full head's, and every other logit must be -inf,
// on every backend (the head is the same kernel over a row view).
TEST_CASE("forward with a logit row range matches the full head",
          "[model][e2e]") {
    const std::string path = model_dir() + "stories260K.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    const std::uint32_t V = model.hparams().n_vocab;
    REQUIRE(V > 300);
    require_rows_match(model, {100, 300});
    require_rows_match(model, {0, 1});
    require_rows_match(model, {V - 7, V});

    locus::model::TransformerModel::Workspace ws = model.make_workspace();
    auto cache = model.make_cache(64);
    locus::kv::PagedKvCache::Seq s;
    std::vector<float> out(V);
    REQUIRE(cache.ensure_capacity(s, 1));
    REQUIRE_THROWS_AS(model.forward(1, cache, s, ws, out, {5, 5}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(model.forward(1, cache, s, ws, out, {0, V + 1}),
                      std::invalid_argument);
}

// i#247: forward_embedding from a token gives forward()'s logits, and
// checks the token and the rows before any backend runs (the Vulkan
// forward reads both unchecked).
TEST_CASE("forward_embedding checks its token and rows like forward",
          "[model][e2e]") {
    const std::string path = model_dir() + "stories260K.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    const auto V = static_cast<std::int32_t>(model.hparams().n_vocab);
    auto ws = model.make_workspace();
    auto a = model.make_cache(64), b = model.make_cache(64);
    locus::kv::PagedKvCache::Seq sa, sb;
    std::vector<float> want(V), got(V), normed(model.hparams().n_embd);
    REQUIRE(a.ensure_capacity(sa, 1));
    REQUIRE(b.ensure_capacity(sb, 1));
    model.forward(9, a, sa, ws, want);
    model.forward_embedding(9, {}, b, sb, ws, got, {}, normed);
    REQUIRE(std::memcmp(want.data(), got.data(), V * sizeof(float)) == 0);

    std::vector<locus::model::TransformerModel> models;
    models.push_back(std::move(model));
    if (locus::backend::vulkan_backend_usable()) {
        models.push_back(locus::model::TransformerModel::load(g));
        models.back().use_backend(*locus::backend::find_backend("vulkan"));
    }
    for (const auto& m : models) {
        INFO("backend " << m.active_backend().name);
        auto c = m.make_cache(64);
        locus::kv::PagedKvCache::Seq s;
        REQUIRE(c.ensure_capacity(s, 1));
        for (const locus::tok::TokenId t : {V, -1}) {
            REQUIRE_THROWS_AS(
                m.forward_embedding(t, {}, c, s, ws, got, {}, normed),
                std::invalid_argument);
        }
        const std::uint32_t U = static_cast<std::uint32_t>(V);
        for (const locus::model::LogitRows r :
             {locus::model::LogitRows{5, 5},
              locus::model::LogitRows{0, U + 1}}) {
            REQUIRE_THROWS_AS(
                m.forward_embedding(1, {}, c, s, ws, got, r, normed),
                std::invalid_argument);
        }
        REQUIRE(s.n_tokens == 0);  // nothing ran
    }
}

// The same on the Vulkan full forward, on a model it runs (a Q6_K tied
// head, like Orpheus's), so the row view's byte offset into the
// uploaded head is exercised.
TEST_CASE("vulkan forward with a logit row range matches the full head",
          "[model][vulkan][e2e]") {
    const std::string path = model_dir() + "llama-3.2-1b-q4_k_m.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("llama-3.2-1b-q4_k_m.gguf not present");
    }
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    model.use_backend(*locus::backend::find_backend("vulkan"));
    REQUIRE(locus::backend::vulkan_forward_supported(model));
    const std::uint32_t V = model.hparams().n_vocab;
    require_rows_match(model, {128000, V});
    require_rows_match(model, {1000, 9000});
}
