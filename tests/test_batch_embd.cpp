#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"
#include "qwen_tiny.hpp"

using locus::model::TransformerModel;
using Seq = locus::kv::PagedKvCache::Seq;

namespace {
std::string stories_path() {
    return std::string(LOCUS_SOURCE_DIR) + "/tests/models/stories260K.gguf";
}
}  // namespace

// i#257: forward_batch takes input embeddings in place of some tokens
// (Janus image slots), and must equal one forward_layers per position
// with the same embedding, bit for bit, on every CPU backend.
TEST_CASE("forward_batch with input embeddings equals per-position forward",
          "[model][batch]") {
    if (!std::filesystem::exists(stories_path())) {
        SKIP("stories260K.gguf absent");
    }
    auto g = locus::gguf::GgufFile::open(stories_path());
    auto model = TransformerModel::load(g);
    const auto& hp = model.hparams();
    const std::uint32_t V = hp.n_vocab;
    const std::uint32_t E = hp.n_embd;

    // Positions 1 and 3 are embeddings; the others are tokens.
    const std::vector<locus::tok::TokenId> toks = {3, 0, 7, 0, 2};
    const auto m = static_cast<std::uint32_t>(toks.size());
    std::mt19937 rng(257);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<std::vector<float>> rows(2, std::vector<float>(E));
    for (auto& r : rows) {
        for (auto& v : r) v = dist(rng);
    }
    const std::vector<const float*> embd = {nullptr, rows[0].data(),
                                            nullptr, rows[1].data(),
                                            nullptr};

    for (const auto& be : locus::backend::backends()) {
        if (!be.available || !be.selectable || be.name == "vulkan") {
            continue;
        }
        DYNAMIC_SECTION("backend " << be.name) {
            model.use_backend(be);
            auto cR = model.make_cache(16);
            auto wR = model.make_workspace();
            Seq sR;
            std::vector<std::vector<float>> want(m, std::vector<float>(V));
            for (std::uint32_t i = 0; i < m; ++i) {
                REQUIRE(cR.ensure_capacity(sR, 1));
                const std::span<const float> e =
                    embd[i] != nullptr ? std::span<const float>(embd[i], E)
                                       : std::span<const float>();
                model.forward_layers(toks[i], e, 0, hp.n_layers, cR, sR, wR,
                                     want[i]);
            }

            auto cB = model.make_cache(16);
            auto wB = model.make_workspace();
            Seq sB;
            REQUIRE(cB.ensure_capacity(sB, m));
            std::vector<float> got(static_cast<std::size_t>(m) * V);
            model.forward_batch(toks, cB, sB, wB, got, /*all_logits=*/true,
                                embd);
            REQUIRE(sB.n_tokens == m);
            for (std::uint32_t i = 0; i < m; ++i) {
                INFO("position " << i);
                for (std::uint32_t r = 0; r < V; ++r) {
                    REQUIRE(got[static_cast<std::size_t>(i) * V + r] ==
                            want[i][r]);
                }
            }

            // Last-position prefill shape gives the same last logits.
            auto cP = model.make_cache(16);
            auto wP = model.make_workspace();
            Seq sP;
            REQUIRE(cP.ensure_capacity(sP, m));
            std::vector<float> last(V);
            model.forward_batch(toks, cP, sP, wP, last, false, embd);
            REQUIRE(last == want[m - 1]);

            // Misuse: a wrong embd length; an out-of-vocab id without an
            // embedding (an id under an embedding is ignored).
            auto cX = model.make_cache(16);
            auto wX = model.make_workspace();
            Seq sX;
            REQUIRE(cX.ensure_capacity(sX, m));
            const std::vector<const float*> short_embd = {nullptr};
            REQUIRE_THROWS_AS(model.forward_batch(toks, cX, sX, wX, last,
                                                  false, short_embd),
                              std::invalid_argument);
            std::vector<locus::tok::TokenId> bad = toks;
            bad[1] = static_cast<locus::tok::TokenId>(V + 5);  // embedded
            REQUIRE_NOTHROW(
                model.forward_batch(bad, cX, sX, wX, last, false, embd));
            cX.release(sX);
            Seq sY;
            REQUIRE(cX.ensure_capacity(sY, m));
            bad[0] = static_cast<locus::tok::TokenId>(V + 5);  // a token
            REQUIRE_THROWS_AS(
                model.forward_batch(bad, cX, sY, wX, last, false, embd),
                std::invalid_argument);
            cX.release(sY);
            cR.release(sR);
            cB.release(sB);
            cP.release(sP);
        }
    }
}

// i#257: on vulkan, a chunk with input embeddings runs as one GPU batch
// (vulkan_forward_batch with the embedding rows), not one GPU forward
// per position. It must track the CPU's per-position forward_layers
// (GPU rounding: cosine and argmax per position, not bit equality).
TEST_CASE("vulkan forward_batch with input embeddings tracks the CPU",
          "[model][batch][vulkan]") {
    const auto* vk = locus::backend::find_backend("vulkan");
    if (vk == nullptr || !vk->available) {
        SKIP("no usable Vulkan device");
    }
    if (!std::filesystem::exists(stories_path())) {
        SKIP("stories260K.gguf absent");
    }
    auto g = locus::gguf::GgufFile::open(stories_path());
    auto model = TransformerModel::load(g);
    const auto& hp = model.hparams();
    const std::uint32_t V = hp.n_vocab;
    const std::uint32_t E = hp.n_embd;
    const std::vector<locus::tok::TokenId> toks = {3, 0, 7, 0, 2, 9, 0, 4};
    const auto m = static_cast<std::uint32_t>(toks.size());
    std::mt19937 rng(257);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<std::vector<float>> rows(3, std::vector<float>(E));
    for (auto& r : rows) {
        for (auto& v : r) v = dist(rng);
    }
    const std::vector<const float*> embd = {
        nullptr, rows[0].data(), nullptr, rows[1].data(),
        nullptr, nullptr,        rows[2].data(), nullptr};

    // The reference: the CPU, one forward_layers per position.
    model.use_backend(locus::backend::best_backend());
    auto cR = model.make_cache(16);
    auto wR = model.make_workspace();
    Seq sR;
    std::vector<std::vector<float>> want(m, std::vector<float>(V));
    for (std::uint32_t i = 0; i < m; ++i) {
        REQUIRE(cR.ensure_capacity(sR, 1));
        const std::span<const float> e =
            embd[i] != nullptr ? std::span<const float>(embd[i], E)
                               : std::span<const float>();
        model.forward_layers(toks[i], e, 0, hp.n_layers, cR, sR, wR,
                             want[i]);
    }

    model.use_backend(*vk);
    auto cB = model.make_cache(16);
    auto wB = model.make_workspace();
    Seq sB;
    REQUIRE(cB.ensure_capacity(sB, m));
    std::vector<float> got(static_cast<std::size_t>(m) * V);
    model.forward_batch(toks, cB, sB, wB, got, /*all_logits=*/true, embd);
    REQUIRE(sB.n_tokens == m);
    for (std::uint32_t i = 0; i < m; ++i) {
        INFO("position " << i);
        const float* gr = &got[static_cast<std::size_t>(i) * V];
        double dot = 0.0, ng = 0.0, nw = 0.0;
        std::uint32_t ag = 0, aw = 0;
        for (std::uint32_t r = 0; r < V; ++r) {
            dot += double{gr[r]} * want[i][r];
            ng += double{gr[r]} * gr[r];
            nw += double{want[i][r]} * want[i][r];
            if (gr[r] > gr[ag]) ag = r;
            if (want[i][r] > want[i][aw]) aw = r;
        }
        REQUIRE(dot / std::sqrt(ng * nw) > 0.9999);
        REQUIRE(ag == aw);
    }
    // A wrong embd length is refused on this path too.
    auto cX = model.make_cache(16);
    auto wX = model.make_workspace();
    Seq sX;
    REQUIRE(cX.ensure_capacity(sX, m));
    std::vector<float> last(V);
    const std::vector<const float*> short_embd = {nullptr, rows[0].data()};
    REQUIRE_THROWS_AS(
        model.forward_batch(toks, cX, sX, wX, last, false, short_embd),
        std::invalid_argument);
    cR.release(sR);
    cB.release(sB);
    cX.release(sX);
}

// i#671: qwen2-family attention (q / k / v biases, NEOX rope, M-RoPE)
// batches its projections in forward_batch, as llama's does. Without
// M-RoPE positions a chunk equals one forward() per position bit for bit
// (forward() runs qwen2moe_attention per token); with them, a chunk
// equals the same positions fed one token per forward_batch.
TEST_CASE("qwen2 forward_batch equals per-position forward",
          "[model][batch]") {
    for (const char* arch : {"qwen2", "qwen2vl"}) {
        DYNAMIC_SECTION(arch) {
            const auto img = qwen_tiny::lm(arch);
            const auto g = locus::gguf::GgufFile::parse(img);
            auto model = TransformerModel::load(g);
            const std::uint32_t V = model.hparams().n_vocab;
            const std::vector<locus::tok::TokenId> toks = {3, 17, 9, 30, 1,
                                                           22, 5};
            const auto m = static_cast<std::uint32_t>(toks.size());

            auto cR = model.make_cache(16);
            auto wR = model.make_workspace();
            Seq sR;
            std::vector<std::vector<float>> want(m, std::vector<float>(V));
            for (std::uint32_t i = 0; i < m; ++i) {
                REQUIRE(cR.ensure_capacity(sR, 1));
                model.forward(toks[i], cR, sR, wR, want[i]);
            }
            auto cB = model.make_cache(16);
            auto wB = model.make_workspace();
            Seq sB;
            REQUIRE(cB.ensure_capacity(sB, m));
            std::vector<float> got(std::size_t{m} * V);
            model.forward_batch(toks, cB, sB, wB, got, true);
            for (std::uint32_t i = 0; i < m; ++i) {
                INFO("position " << i);
                REQUIRE(std::vector<float>(got.begin() + i * V,
                                           got.begin() + (i + 1) * V) ==
                        want[i]);
            }
            cR.release(sR);
            cB.release(sB);
        }
    }
    SECTION("qwen2vl with M-RoPE positions") {
        const auto img = qwen_tiny::lm("qwen2vl");
        const auto g = locus::gguf::GgufFile::parse(img);
        auto model = TransformerModel::load(g);
        const std::uint32_t V = model.hparams().n_vocab;
        // Two text tokens, a 2 x 2 image span, two text tokens.
        const auto pos = locus::model::mrope_positions(2, 2, 2, 2);
        const std::vector<locus::tok::TokenId> toks(pos.size(), 7);
        const auto m = static_cast<std::uint32_t>(toks.size());
        auto cB = model.make_cache(16);
        auto wB = model.make_workspace();
        Seq sB;
        REQUIRE(cB.ensure_capacity(sB, m));
        std::vector<float> got(std::size_t{m} * V);
        model.forward_batch(toks, cB, sB, wB, got, true, {}, pos);
        auto cO = model.make_cache(16);
        auto wO = model.make_workspace();
        Seq sO;
        REQUIRE(cO.ensure_capacity(sO, m));
        std::vector<float> one(V);
        for (std::uint32_t i = 0; i < m; ++i) {
            INFO("position " << i);
            model.forward_batch(std::span(toks).subspan(i, 1), cO, sO, wO,
                                one, true, {}, std::span(pos).subspan(i, 1));
            REQUIRE(std::vector<float>(got.begin() + i * V,
                                       got.begin() + (i + 1) * V) == one);
        }
        cB.release(sB);
        cO.release(sO);
    }
}
