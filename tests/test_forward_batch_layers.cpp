#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"

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
    auto model = locus::model::TransformerModel::load(g);
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
    auto model = locus::model::TransformerModel::load(g);
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

// The property the executor actually depends on: a single batch whose
// members sit at DIFFERENT positions. The executor batches independent
// sessions that arrived at different times, so ragged positions are the
// normal case, not the exception. Every other test here advances all N
// in lockstep (one shared position per batch), so a bug that used one
// position for the whole batch (pos[0], or seqs[0]->n_tokens) would pass
// them all yet make every off-lead sequence attend over the wrong number
// of past tokens -- silent output corruption. This pins per-sequence
// position handling: warm each sequence to a distinct position, then one
// batched step must equal N serial forward_layers from those same states.
TEST_CASE("forward_batch_layers ragged batch (mixed positions)",
          "[batch][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    const std::uint32_t L = model.hparams().n_layers;
    const std::uint32_t V = model.hparams().n_vocab;
    const std::uint32_t k = L / 2 == 0 ? 1 : L / 2;
    const int N = 4;
    const int pre[N] = {3, 1, 0, 2};  // distinct starting positions
    const std::vector<locus::tok::TokenId> step = {5, 6, 7, 8};

    auto ws = model.make_workspace();

    // Warm two caches identically to the same ragged starting states,
    // one for the serial reference and one for the batched run. Full-
    // stack forward_layers(0, L) advances a seq by one each call, so
    // after pre[i] calls seq[i] sits at position pre[i] with its KV
    // populated. The warm tokens are deterministic, so both caches hold
    // byte-identical KV.
    auto warm = [&](locus::kv::PagedKvCache& cache, std::vector<Seq>& seq) {
        std::vector<float> scratch(V);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < pre[i]; ++j) {
                REQUIRE(cache.ensure_capacity(seq[i], 1));
                const locus::tok::TokenId wt =
                    static_cast<locus::tok::TokenId>(1 + ((i + j) % 20));
                model.forward_layers(wt, {}, 0, L, cache, seq[i], ws,
                                     scratch);
            }
        }
    };

    // Reference: each sequence alone, two-stage, from its warmed state.
    std::vector<std::vector<float>> ref(N, std::vector<float>(V));
    {
        auto cache = model.make_cache(256);
        std::vector<Seq> seq(N);
        warm(cache, seq);
        for (int i = 0; i < N; ++i) {
            REQUIRE(cache.ensure_capacity(seq[i], 1));
            std::vector<float> hid(model.hparams().n_embd);
            model.forward_layers(step[i], {}, 0, k, cache, seq[i], ws,
                                 hid);
            model.forward_layers(0, hid, k, L, cache, seq[i], ws, ref[i]);
        }
    }

    // Batched: one step over all N at their mixed positions.
    std::vector<float> batch(static_cast<std::size_t>(N) * V);
    {
        auto cache = model.make_cache(256);
        std::vector<Seq> seq(N);
        warm(cache, seq);
        std::vector<Seq*> ptr(N);
        for (int i = 0; i < N; ++i) {
            ptr[i] = &seq[i];
            REQUIRE(cache.ensure_capacity(seq[i], 1));
        }
        std::vector<locus::tok::TokenId> toks(step.begin(), step.end());
        std::vector<float> hid(static_cast<std::size_t>(N) *
                               model.hparams().n_embd);
        model.forward_batch_layers(toks, {}, 0, k, cache, ptr, ws, hid);
        model.forward_batch_layers({}, hid, k, L, cache, ptr, ws, batch);
    }

    for (int i = 0; i < N; ++i) {
        for (std::uint32_t v = 0; v < V; ++v) {
            REQUIRE(batch[static_cast<std::size_t>(i) * V + v] ==
                    ref[i][v]);
        }
    }
}
