#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"

using locus::model::TransformerModel;
using Seq = locus::kv::PagedKvCache::Seq;

namespace {
std::string stories_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}
std::string q4k_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/llama-3.2-1b-q4_k_m.gguf";
}

// i#52: the last stage amortizes the LM head (out_w_) into ONE
// matvec_batch over the whole batch instead of n per-token matvec
// passes. This asserts the amortized path is bit-for-bit the per-
// sequence forward() path (which still projects out_w_ per token), for
// all three batched sites: forward_batch_decode, forward_batch_layers,
// and forward_batch's all_logits branch.
void check_outw(TransformerModel& model) {
    const auto& hp = model.hparams();
    const std::uint32_t V = hp.n_vocab;
    const std::uint32_t L = hp.n_layers;
    const std::vector<std::vector<locus::tok::TokenId>> ctx = {
        {3, 7}, {1, 5, 9}, {2}, {4, 8, 6, 0}};
    const std::vector<locus::tok::TokenId> next = {5, 2, 7, 1};
    const std::uint32_t N = 4;

    auto prefill = [&](locus::kv::PagedKvCache& cache,
                       std::vector<Seq>& seqs,
                       TransformerModel::Workspace& ws) {
        std::vector<float> logits(V);
        for (std::uint32_t i = 0; i < N; ++i) {
            for (auto t : ctx[i]) {
                REQUIRE(cache.ensure_capacity(seqs[i], 1));
                model.forward(t, cache, seqs[i], ws, logits);
            }
        }
    };

    // Reference: per-sequence forward() (per-token out_w_).
    auto cacheR = model.make_cache(16);
    auto wsR = model.make_workspace();
    std::vector<Seq> seqsR(N);
    prefill(cacheR, seqsR, wsR);
    std::vector<std::vector<float>> ref(N, std::vector<float>(V));
    for (std::uint32_t i = 0; i < N; ++i) {
        REQUIRE(cacheR.ensure_capacity(seqsR[i], 1));
        model.forward(next[i], cacheR, seqsR[i], wsR, ref[i]);
    }

    // forward_batch_decode: amortized LM head over N decode tokens.
    {
        auto cache = model.make_cache(16);
        auto ws = model.make_workspace();
        std::vector<Seq> seqs(N);
        prefill(cache, seqs, ws);
        std::vector<Seq*> ptrs;
        for (std::uint32_t i = 0; i < N; ++i) {
            REQUIRE(cache.ensure_capacity(seqs[i], 1));
            ptrs.push_back(&seqs[i]);
        }
        std::vector<float> got(static_cast<std::size_t>(N) * V);
        model.forward_batch_decode(next, cache, ptrs, ws, got);
        for (std::uint32_t i = 0; i < N; ++i) {
            for (std::uint32_t r = 0; r < V; ++r) {
                REQUIRE(got[static_cast<std::size_t>(i) * V + r] ==
                        ref[i][r]);
            }
        }
        for (auto& s : seqs) {
            cache.release(s);
        }
    }

    // forward_batch_layers over the full stack [0,L): first AND last
    // stage, so the amortized LM head runs in the last-stage path too.
    {
        auto cache = model.make_cache(16);
        auto ws = model.make_workspace();
        std::vector<Seq> seqs(N);
        prefill(cache, seqs, ws);
        std::vector<Seq*> ptrs;
        for (std::uint32_t i = 0; i < N; ++i) {
            REQUIRE(cache.ensure_capacity(seqs[i], 1));
            ptrs.push_back(&seqs[i]);
        }
        std::vector<float> got(static_cast<std::size_t>(N) * V);
        model.forward_batch_layers(next, {}, 0, L, cache, ptrs, ws, got);
        for (std::uint32_t i = 0; i < N; ++i) {
            for (std::uint32_t r = 0; r < V; ++r) {
                REQUIRE(got[static_cast<std::size_t>(i) * V + r] ==
                        ref[i][r]);
            }
        }
        for (auto& s : seqs) {
            cache.release(s);
        }
    }

    // forward_batch all_logits: n tokens of ONE sequence in one chunk
    // (the speculative-verify shape). The amortized head must match
    // per-token forward() at every position.
    {
        const std::vector<locus::tok::TokenId> chunk = {3, 7, 2, 5};
        const std::uint32_t m = static_cast<std::uint32_t>(chunk.size());
        auto cR = model.make_cache(16);
        auto wR = model.make_workspace();
        Seq sR;
        std::vector<std::vector<float>> refc(m, std::vector<float>(V));
        for (std::uint32_t i = 0; i < m; ++i) {
            REQUIRE(cR.ensure_capacity(sR, 1));
            model.forward(chunk[i], cR, sR, wR, refc[i]);
        }
        auto cB = model.make_cache(16);
        auto wB = model.make_workspace();
        Seq sB;
        REQUIRE(cB.ensure_capacity(sB, m));
        std::vector<float> got(static_cast<std::size_t>(m) * V);
        model.forward_batch(chunk, cB, sB, wB, got, /*all_logits=*/true);
        for (std::uint32_t i = 0; i < m; ++i) {
            for (std::uint32_t r = 0; r < V; ++r) {
                REQUIRE(got[static_cast<std::size_t>(i) * V + r] ==
                        refc[i][r]);
            }
        }
        cR.release(sR);
        cB.release(sB);
    }
    for (auto& s : seqsR) {
        cacheR.release(s);
    }
}

// Run check_outw across EVERY backend available on this host, not just
// best_backend(), so the layout guard is checked by the test rather than
// decided by which backend the host happens to pick (the two wrapper
// paths are selected by backend, so a single-backend run only exercises
// one). The reference forward() and the batched paths share the one
// model object, so they always use the section's backend. Vulkan is
// skipped (no batched forward). Convention: test_model_e2e.cpp.
void sweep_backends(const std::string& path) {
    // The wrapper's op.matvec fallback branch is reached only on a
    // backend that leaves Ops::matvec_batch null, which today is avx2
    // alone. The sweep below therefore cannot reach it on any host
    // without avx2 (every arm64 one, and any pre-Haswell x86), leaving
    // the layout path a default modern x86-64 run takes uncovered.
    // Copying a real backend and nulling that one pointer reaches the
    // fallback everywhere, so this stays a property of the test rather
    // than of whoever happens to run it. Declared BEFORE `model` so it
    // outlives it: use_backend() stores a pointer to the backend, and at
    // scope exit `model` must not be left pointing into a destroyed `fb`.
    locus::backend::Backend fb = locus::backend::best_backend();
    fb.ops.matvec_batch = nullptr;
    fb.name = "fallback (matvec_batch nulled)";

    auto g = locus::gguf::GgufFile::open(path);
    auto model = TransformerModel::load(g);
    REQUIRE(model.supports_batch());
    for (const auto& b : locus::backend::backends()) {
        if (b.available && b.selectable && b.name != "vulkan") {
            DYNAMIC_SECTION("backend " << b.name) {
                model.use_backend(b);
                check_outw(model);
            }
        }
    }
    // Finally the synthetic fallback backend prepared above (see the
    // comment at its declaration for why it is hoisted before `model`).
    DYNAMIC_SECTION("backend " << fb.name) {
        model.use_backend(fb);
        check_outw(model);
    }
}
}  // namespace

// F32 head: this is the LAYOUT GUARD, not redundant with the quantized
// case (i#52). matvec_batch's wrapper writes token-major [t*V+r]; for an
// F32 head the backend's matvec_batch falls back to a per-token kernel
// whose output the wrapper transposes. If that fallback ever wrote
// token-major directly (or forward_* added its own transpose), the
// logits would silently PERMUTE for F32 heads only (a quantized-head
// test never reaches that path). Two implementations, one layout
// promise; this asserts they agree. Do not delete this as a dup.
TEST_CASE("amortized LM head byte-exact: F32 head (layout guard)",
          "[batch][outw]") {
    if (!std::filesystem::exists(stories_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    sweep_backends(stories_path());
}

// Quantized head: exercises matvec_batch's register-blocked path (the
// one-read-per-batch kernel that the amortization exists for).
TEST_CASE("amortized LM head byte-exact: quantized head",
          "[batch][outw]") {
    if (!std::filesystem::exists(q4k_path())) {
        SKIP("Q4_K model not present (llama-3.2-1b-q4_k_m.gguf)");
    }
    sweep_backends(q4k_path());
}
