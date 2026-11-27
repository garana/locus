#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/metal/matvec.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"
#include "locus/sys/features.hpp"

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
// passes. This asserts the amortized path is the per-sequence forward()
// path (which still projects out_w_ per token), for all three batched
// sites: forward_batch_decode, forward_batch_layers, and forward_batch's
// all_logits branch: bit for bit at tol 0, else within tol (one token and
// a batch may take kernels of different float orders, DESIGN 16: on AVX
// one-token Q4_K / Q6_K rows are llama.cpp's, i#1244). A permuted layout
// is off by far more than any tol used here.
void check_outw(TransformerModel& model, float tol = 0.0f) {
    float worst = 0.0f;  // the largest |got - want| seen (tol > 0)
    const auto same = [tol, &worst](float got, float want) {
        if (tol == 0.0f) {
            REQUIRE(got == want);
        } else {
            worst = std::max(worst, std::abs(got - want));
            REQUIRE(got == Catch::Approx(want).margin(tol));
        }
    };
    // Per row, at tol > 0: the batched argmax is the per-token one.
    const auto same_top = [tol](std::span<const float> got,
                                std::span<const float> want) {
        if (tol > 0.0f) {
            REQUIRE(locus::model::argmax(got) == locus::model::argmax(want));
        }
    };
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
                same(got[static_cast<std::size_t>(i) * V + r], ref[i][r]);
            }
            same_top({got.data() + std::size_t{i} * V, V}, ref[i]);
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
                same(got[static_cast<std::size_t>(i) * V + r], ref[i][r]);
            }
            same_top({got.data() + std::size_t{i} * V, V}, ref[i]);
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
                same(got[static_cast<std::size_t>(i) * V + r], refc[i][r]);
            }
            same_top({got.data() + std::size_t{i} * V, V}, refc[i]);
        }
        cR.release(sR);
        cB.release(sB);
    }
    for (auto& s : seqsR) {
        cacheR.release(s);
    }
    if (tol > 0.0f) {
        WARN("largest |batched - per-token| logit gap: " << worst);
    }
}

// Run check_outw across EVERY backend available on this host, not just
// best_backend(), so the layout guard is checked by the test rather than
// decided by which backend the host happens to pick (the two wrapper
// paths are selected by backend, so a single-backend run only exercises
// one). The reference forward() and the batched paths share the one
// model object, so they always use the section's backend. Vulkan is
// skipped (no batched forward). Convention: test_model_e2e.cpp.
// tol applies only where one token and a batch take different float
// orders: the sse4 backend on an AVX CPU (i#1244) and the fallback copy
// of it; every other backend, metal's batch-invariant mode among them,
// keeps bit for bit.
void sweep_backends(const std::string& path, float tol = 0.0f) {
    const bool avx = locus::sys::detect().avx;
    const auto tol_for = [&](std::string_view name) {
        return avx && name == "sse4" ? tol : 0.0f;
    };
    const float fb_tol = tol_for(locus::backend::best_backend().name);
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
                check_outw(model, tol_for(b.name));
            }
        }
    }
    // Finally the synthetic fallback backend prepared above (see the
    // comment at its declaration for why it is hoisted before `model`).
    DYNAMIC_SECTION("backend " << fb.name) {
        model.use_backend(fb);
        check_outw(model, fb_tol);
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
        SKIP("model not present; run scripts/tests/fetch-test-model.sh");
    }
    sweep_backends(stories_path());
}

// Quantized head: exercises matvec_batch's register-blocked path (the
// one-read-per-batch kernel that the amortization exists for). On AVX
// one token takes llama.cpp's Q4_K / Q6_K rows and a batch the sse4
// order (i#1244, DESIGN 16): after 16 layers of Llama 3.2 1B Q4_K_M the
// logits differ by up to 0.49 (each layer's Q8_K activations can round
// to the next step), with the same top token on every row. So sse4 on an
// AVX CPU (and its fallback copy) is held within 1 of the per-token
// logits and the same argmax; every other backend, metal's
// batch-invariant mode among them, stays bit for bit.
TEST_CASE("amortized LM head tracks per-token: quantized head",
          "[batch][outw]") {
    if (!std::filesystem::exists(q4k_path())) {
        SKIP("Q4_K model not present (llama-3.2-1b-q4_k_m.gguf)");
    }
    // On Metal a K-quant model decodes alone through metal_forward and
    // batched through other kernels unless batch-invariant mode is set
    // (i#1184); set, the two agree byte for byte.
    struct Mode {
        Mode() { locus::backend::metal::set_batch_invariant(true); }
        ~Mode() { locus::backend::metal::reset_batch_invariant(); }
    } const mode;
    sweep_backends(q4k_path(), 1.0f);
}
