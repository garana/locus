#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/llama.hpp"

namespace {
std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}
using Seq = locus::kv::PagedKvCache::Seq;
}  // namespace

// forward_batch_layers must be byte-identical to running the same N
// sequences one at a time through forward_layers: same layer range, same
// first/last roles, same per-stage seq-advance. This is the kernel-free
// batched stage step the event-loop executor (i#24) will drive. Exercised
// as a two-stage split [0,k)+[k,L) (hidden handed stage to stage) over
// two positions (so position advance across the split is covered), with
// distinct per-sequence tokens so any cross-contamination shows as a
// wrong logit.
TEST_CASE("forward_batch_layers equals N forward_layers (two-stage)",
          "[batch][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;
    const std::uint32_t E = model.hparams().n_embd;
    const std::uint32_t V = model.hparams().n_vocab;
    REQUIRE(L >= 2);
    const std::uint32_t k = L / 2 == 0 ? 1 : L / 2;
    const int N = 4;
    // Two positions per sequence, distinct ids (all well within vocab).
    const std::vector<std::vector<locus::tok::TokenId>> prompt = {
        {1, 9}, {2, 10}, {3, 11}, {4, 12}};

    auto ws = model.make_workspace();

    // Reference: each sequence alone, two-stage per token; keep the
    // final (pos-1) logits.
    std::vector<std::vector<float>> ref(N, std::vector<float>(V));
    {
        auto cache = model.make_cache(256);
        std::vector<Seq> seq(N);
        for (int p = 0; p < 2; ++p) {
            for (int i = 0; i < N; ++i) {
                REQUIRE(cache.ensure_capacity(seq[i], 1));
                std::vector<float> hid(E);
                model.forward_layers(prompt[i][p], {}, 0, k, cache,
                                     seq[i], ws, hid);
                model.forward_layers(0, hid, k, L, cache, seq[i], ws,
                                     ref[i]);
            }
        }
    }

    // Batched: all N sequences together, two-stage per position.
    std::vector<float> batch(static_cast<std::size_t>(N) * V);
    {
        auto cache = model.make_cache(256);
        std::vector<Seq> seq(N);
        std::vector<Seq*> ptr(N);
        for (int i = 0; i < N; ++i) {
            ptr[i] = &seq[i];
        }
        for (int p = 0; p < 2; ++p) {
            for (int i = 0; i < N; ++i) {
                REQUIRE(cache.ensure_capacity(seq[i], 1));
            }
            std::vector<locus::tok::TokenId> toks(N);
            for (int i = 0; i < N; ++i) {
                toks[i] = prompt[i][p];
            }
            std::vector<float> hid(static_cast<std::size_t>(N) * E);
            model.forward_batch_layers(toks, {}, 0, k, cache, ptr, ws,
                                       hid);
            model.forward_batch_layers({}, hid, k, L, cache, ptr, ws,
                                       batch);
        }
    }

    for (int i = 0; i < N; ++i) {
        for (std::uint32_t v = 0; v < V; ++v) {
            REQUIRE(batch[static_cast<std::size_t>(i) * V + v] ==
                    ref[i][v]);
        }
    }
}

// The single-stage degenerate case (layer_begin==0, layer_end==L): a
// batched full forward must match N forward_layers(0, L) exactly. Covers
// the first-and-last path (embed + logits + advance) without a hidden
// hand-off.
TEST_CASE("forward_batch_layers full stack equals N forward_layers",
          "[batch][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::LlamaModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;
    const std::uint32_t V = model.hparams().n_vocab;
    const int N = 3;
    const std::vector<locus::tok::TokenId> toks = {7, 8, 9};

    auto ws = model.make_workspace();
    std::vector<std::vector<float>> ref(N, std::vector<float>(V));
    {
        auto cache = model.make_cache(64);
        std::vector<Seq> seq(N);
        for (int i = 0; i < N; ++i) {
            REQUIRE(cache.ensure_capacity(seq[i], 1));
            model.forward_layers(toks[i], {}, 0, L, cache, seq[i], ws,
                                 ref[i]);
        }
    }
    std::vector<float> batch(static_cast<std::size_t>(N) * V);
    {
        auto cache = model.make_cache(64);
        std::vector<Seq> seq(N);
        std::vector<Seq*> ptr(N);
        for (int i = 0; i < N; ++i) {
            ptr[i] = &seq[i];
            REQUIRE(cache.ensure_capacity(seq[i], 1));
        }
        model.forward_batch_layers(toks, {}, 0, L, cache, ptr, ws, batch);
    }
    for (int i = 0; i < N; ++i) {
        for (std::uint32_t v = 0; v < V; ++v) {
            REQUIRE(batch[static_cast<std::size_t>(i) * V + v] ==
                    ref[i][v]);
        }
    }
}
