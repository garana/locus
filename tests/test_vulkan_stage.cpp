#include <algorithm>
#include <cmath>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::kv::KvType;
using locus::kv::PagedKvCache;
using locus::model::TransformerModel;

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

}  // namespace

// i#565: a pipeline stage runs its layer slice on the GPU over a cache
// that holds only that slice (layer_base > 0 on the second stage). Two
// stages chained give the full Vulkan forward's logits bit for bit: the
// same kernels run, so any slip in the slice's cache offsets (a row
// stored or read at the absolute layer) shows up as different bits.
TEST_CASE("two vulkan stages give the full vulkan forward's logits",
          "[vulkan][pipeline]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    const KvType kv = GENERATE(KvType::kF32, KvType::kQ8, KvType::kF16);
    CAPTURE(static_cast<int>(kv));
    auto g = locus::gguf::GgufFile::open(model_path());
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const auto& vk = *locus::backend::find_backend("vulkan");
    auto full = TransformerModel::load(g);
    full.use_backend(vk);
    const std::uint32_t L = full.hparams().n_layers;
    const std::uint32_t E = full.hparams().n_embd;
    const std::uint32_t V = full.hparams().n_vocab;
    // Three stages: first [0, k1), middle [k1, k), last [k, L).
    REQUIRE(L >= 3);
    const std::uint32_t k1 = 1;
    const std::uint32_t k = L / 2 + 1;
    auto a = TransformerModel::load(g, 0, k1);
    auto mid = TransformerModel::load(g, k1, k);
    auto b = TransformerModel::load(g, k, L);
    a.use_backend(vk);
    mid.use_backend(vk);
    b.use_backend(vk);
    REQUIRE(locus::backend::vulkan_forward_supported(a));
    REQUIRE(locus::backend::vulkan_forward_supported(mid));
    REQUIRE(locus::backend::vulkan_forward_supported(b));

    PagedKvCache fc = full.make_cache(0, kv);
    PagedKvCache ca = a.make_cache(0, kv);
    PagedKvCache cm = mid.make_cache(0, kv);
    PagedKvCache cb = b.make_cache(0, kv);
    REQUIRE(cm.geometry().layer_base == k1);
    REQUIRE(cb.geometry().layer_base == k);
    REQUIRE(cb.geometry().n_layers == L - k);
    auto wa = a.make_workspace();
    auto wb = b.make_workspace();
    PagedKvCache::Seq fs, sa, sm, sb;

    // More than two 16-token blocks, so the block stride (whose size
    // the slice changes) is crossed, not only the layer offset.
    const auto prompt = tok.encode(
        "Once upon a time, there was a little girl named Lily who loved "
        "to explore the forest near her home every day.",
        true);
    REQUIRE(prompt.size() > 32);
    std::vector<float> want(V), got(V), hidden(E), hidden2(E);
    for (std::size_t p = 0; p < prompt.size(); ++p) {
        CAPTURE(p);
        REQUIRE(fc.ensure_capacity(fs, 1));
        REQUIRE(ca.ensure_capacity(sa, 1));
        REQUIRE(cm.ensure_capacity(sm, 1));
        REQUIRE(cb.ensure_capacity(sb, 1));
        // The reference: the single-sequence kernels, as each stage
        // runs (model.forward may pick the multi path on Apple GPUs).
        REQUIRE(locus::backend::vulkan_forward(full, prompt[p], fc, fs,
                                               want));
        // Through forward_layers (the delegation), then the GPU entry
        // directly, so a decline to the host loop fails here.
        a.forward_layers(prompt[p], {}, 0, k1, ca, sa, wa, hidden);
        REQUIRE(locus::backend::vulkan_forward_layers(
            mid, prompt[p], hidden, k1, k, cm, sm, hidden2));
        REQUIRE(sa.n_tokens == p);  // a non-final stage does not advance
        REQUIRE(sm.n_tokens == p);
        sa.n_tokens += 1;  // their PipelineStage does
        sm.n_tokens += 1;
        b.forward_layers(prompt[p], hidden2, k, L, cb, sb, wb, got);
        REQUIRE(sb.n_tokens == p + 1);
        REQUIRE(got == want);
        // The rows the GPU stored sit where the cache says layer l is
        // (load_row subtracts layer_base): a store at the absolute
        // layer can still give the right logits when it lands in spare
        // rows of the next block, so compare the rows themselves.
        const auto pos = static_cast<std::uint32_t>(p);
        const std::uint32_t kv_dim = cb.geometry().kv_dim;
        std::vector<float> rs(kv_dim), rf(kv_dim);
        for (std::uint32_t l = k1; l < L; ++l) {
            PagedKvCache& c = l < k ? cm : cb;
            const PagedKvCache::Seq& s = l < k ? sm : sb;
            for (const bool value : {false, true}) {
                CAPTURE(l, value);
                c.load_row(s, l, pos, value, rs);
                fc.load_row(fs, l, pos, value, rf);
                REQUIRE(rs == rf);
            }
        }
    }
}

TEST_CASE("a vulkan stage refuses a range outside its cache",
          "[vulkan][pipeline]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    const auto& vk = *locus::backend::find_backend("vulkan");
    auto full = TransformerModel::load(g);
    full.use_backend(vk);
    const std::uint32_t L = full.hparams().n_layers;
    const std::uint32_t E = full.hparams().n_embd;
    const std::uint32_t k = L / 2;
    auto b = TransformerModel::load(g, k, L);
    b.use_backend(vk);
    PagedKvCache cb = b.make_cache();
    PagedKvCache::Seq sb;
    REQUIRE(cb.ensure_capacity(sb, 1));
    std::vector<float> hidden(E, 0.01f), out(E);
    // Layers [0, k) are not in this cache: a throw, not a silent write
    // at the wrong rows.
    REQUIRE_THROWS_AS(locus::backend::vulkan_forward_layers(
                          b, 0, hidden, 0, k, cb, sb, out),
                      std::invalid_argument);
    // A later stage needs the residual stream.
    std::vector<float> logits(full.hparams().n_vocab);
    REQUIRE_THROWS_AS(locus::backend::vulkan_forward_layers(
                          b, 0, {}, k, L, cb, sb, logits),
                      std::invalid_argument);
    // The last stage writes logits, not an n_embd stream.
    REQUIRE_THROWS_AS(locus::backend::vulkan_forward_layers(
                          b, 0, hidden, k, L, cb, sb, out),
                      std::invalid_argument);
    // A full-height cache does not make the slice's missing layers
    // runnable: they are not loaded in this model.
    PagedKvCache whole = full.make_cache();
    PagedKvCache::Seq sw;
    REQUIRE(whole.ensure_capacity(sw, 1));
    REQUIRE_THROWS_AS(locus::backend::vulkan_forward_layers(
                          b, 0, {}, 0, k, whole, sw, out),
                      std::invalid_argument);
    // So does a whole-model call on the slice (as the CPU path does).
    REQUIRE_THROWS_AS(locus::backend::vulkan_forward(b, 0, whole, sw, logits),
                      std::invalid_argument);
}

// i#566: a Vulkan stage batches its sequences in one multi-sequence
// pass. Against the full model's multi-sequence pass over the same
// sequences (the multi kernels may differ from the single ones in the
// last bits, so like is compared with like), bit for bit.
TEST_CASE("two batched vulkan stages give the full multi pass's logits",
          "[vulkan][pipeline]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const auto& vk = *locus::backend::find_backend("vulkan");
    auto full = TransformerModel::load(g);
    full.use_backend(vk);
    const std::uint32_t L = full.hparams().n_layers;
    const std::uint32_t E = full.hparams().n_embd;
    const std::uint32_t V = full.hparams().n_vocab;
    const std::uint32_t k = L / 2;
    auto a = TransformerModel::load(g, 0, k);
    auto b = TransformerModel::load(g, k, L);
    a.use_backend(vk);
    b.use_backend(vk);
    REQUIRE(a.supports_batch_layers());  // the StageExecutor batches
    REQUIRE_FALSE(a.supports_batch());   // engine prefill unchanged

    constexpr std::uint32_t n = 3;
    const char* texts[n] = {
        "Once upon a time, there was a little girl named Lily who loved "
        "to play outside.",
        "The cat sat on the mat and looked at the bird in the sky above "
        "the old tree.",
        "A dog ran in the park with a big red ball and a happy boy who "
        "laughed."};
    std::vector<std::vector<locus::tok::TokenId>> prompts;
    std::size_t T = 1000;
    for (const char* t : texts) {
        prompts.push_back(tok.encode(t, true));
        T = std::min(T, prompts.back().size());
    }
    REQUIRE(T > 16);

    PagedKvCache fc = full.make_cache();
    PagedKvCache ca = a.make_cache();
    PagedKvCache cb = b.make_cache();
    auto wa = a.make_workspace();
    std::vector<PagedKvCache::Seq> fs(n), sa(n), sb(n);
    std::vector<PagedKvCache::Seq*> pa, pb;
    for (std::uint32_t i = 0; i < n; ++i) {
        pa.push_back(&sa[i]);
        pb.push_back(&sb[i]);
    }
    std::vector<float> want(std::size_t{n} * V), got(std::size_t{n} * V);
    std::vector<float> normed(std::size_t{n} * E), normed_b(normed.size());
    std::vector<float> hidden(std::size_t{n} * E);
    std::vector<locus::tok::TokenId> toks(n);
    for (std::size_t p = 0; p < T; ++p) {
        CAPTURE(p);
        std::vector<locus::model::EmbedStep> fsteps(n), bsteps(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            REQUIRE(fc.ensure_capacity(fs[i], 1));
            REQUIRE(ca.ensure_capacity(sa[i], 1));
            REQUIRE(cb.ensure_capacity(sb[i], 1));
            toks[i] = prompts[i][p];
            fsteps[i] = {toks[i], {}, &fs[i],
                         std::span(want).subspan(std::size_t{i} * V, V),
                         std::span(normed).subspan(std::size_t{i} * E, E)};
            bsteps[i] = {0, std::span(hidden).subspan(std::size_t{i} * E, E),
                         &sb[i], std::span(got).subspan(std::size_t{i} * V, V),
                         std::span(normed_b).subspan(std::size_t{i} * E, E)};
        }
        REQUIRE(locus::backend::vulkan_forward_multi(full, fsteps, fc, 0, V));
        a.forward_batch_layers(toks, {}, 0, k, ca, pa, wa, hidden);
        for (auto& s : sa) {
            REQUIRE(s.n_tokens == p);  // a non-final stage does not advance
            s.n_tokens += 1;
        }
        // The last stage through the GPU entry, so a decline fails here.
        REQUIRE(locus::backend::vulkan_forward_multi_layers(b, bsteps, k, L,
                                                            cb, 0, V));
        REQUIRE(got == want);
        const auto pos = static_cast<std::uint32_t>(p);
        std::vector<float> rs(cb.geometry().kv_dim), rf(rs.size());
        for (std::uint32_t i = 0; i < n; ++i) {
            REQUIRE(sb[i].n_tokens == p + 1);
            for (std::uint32_t l = k; l < L; ++l) {
                for (const bool value : {false, true}) {
                    cb.load_row(sb[i], l, pos, value, rs);
                    fc.load_row(fs[i], l, pos, value, rf);
                    REQUIRE(rs == rf);
                }
            }
        }
    }
}

// i#623: a model with q / k / v biases (qwen2) runs the multi pass, and
// each sequence's logits there match its own single-sequence pass (other
// kernels sum in another order, so within a bound, with the same
// greedy pick).
TEST_CASE("a qwen2 multi pass matches its sequences' single passes",
          "[vulkan][pipeline]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    const std::string path = std::string(LOCUS_SOURCE_DIR) +
                             "/tests/models/qwen2.5-0.5b-instruct-q8_0.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("qwen2.5-0.5b-instruct-q8_0.gguf absent");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = TransformerModel::load(g);
    model.use_backend(*locus::backend::find_backend("vulkan"));
    REQUIRE(model.layers()[0].has_qkv_bias());
    const std::uint32_t E = model.hparams().n_embd;
    const std::uint32_t V = model.hparams().n_vocab;
    constexpr std::uint32_t n = 3;
    PagedKvCache cm = model.make_cache(1024);
    PagedKvCache cs = model.make_cache(1024);
    std::vector<PagedKvCache::Seq> sm(n), ss(n);
    std::vector<float> got(std::size_t{n} * V), want(V);
    std::vector<float> normed(std::size_t{n} * E);
    for (std::uint32_t p = 0; p < 12; ++p) {
        CAPTURE(p);
        std::vector<locus::model::EmbedStep> steps(n);
        std::vector<locus::tok::TokenId> toks(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            REQUIRE(cm.ensure_capacity(sm[i], 1));
            toks[i] = static_cast<locus::tok::TokenId>(
                100 + (p * 37 + i * 1009) % 30000);
            steps[i] = {toks[i], {}, &sm[i],
                        std::span(got).subspan(std::size_t{i} * V, V),
                        std::span(normed).subspan(std::size_t{i} * E, E)};
        }
        // Declined before i#623 (q / k / v biases).
        REQUIRE(locus::backend::vulkan_forward_multi(model, steps, cm, 0, V));
        for (std::uint32_t i = 0; i < n; ++i) {
            CAPTURE(i);
            REQUIRE(cs.ensure_capacity(ss[i], 1));
            REQUIRE(locus::backend::vulkan_forward(model, toks[i], cs, ss[i],
                                                   want));
            const auto row = std::span(got).subspan(std::size_t{i} * V, V);
            double err = 0.0, norm = 0.0;
            for (std::uint32_t v = 0; v < V; ++v) {
                const double d = double{row[v]} - want[v];
                err += d * d;
                norm += double{want[v]} * want[v];
            }
            CHECK(std::sqrt(err) <= 1e-3 * std::sqrt(norm));
            CHECK(std::max_element(row.begin(), row.end()) - row.begin() ==
                  std::max_element(want.begin(), want.end()) - want.begin());
        }
    }
}

// i#623: an MoE model with per-head q / k norms (qwen3moe) runs the multi
// pass (routed experts through rec_moe_routed), and each sequence's
// logits match its own single pass, within a bound and with the same
// greedy pick. Heavy: the 30B GGUF.
TEST_CASE("a qwen3moe multi pass matches its sequences' single passes",
          "[vulkan][pipeline][e2e]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    const std::string path =
        std::string(LOCUS_SOURCE_DIR) +
        "/tests/models/qwen3-coder-30b-a3b-instruct-q4_k_m.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("qwen3-coder-30b-a3b-instruct-q4_k_m.gguf absent");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = TransformerModel::load(g);
    model.use_backend(*locus::backend::find_backend("vulkan"));
    REQUIRE(model.layers()[0].is_moe());
    REQUIRE(model.layers()[0].has_qk_norm());
    const std::uint32_t E = model.hparams().n_embd;
    const std::uint32_t V = model.hparams().n_vocab;
    constexpr std::uint32_t n = 3;
    PagedKvCache cm = model.make_cache(256);
    PagedKvCache cs = model.make_cache(256);
    std::vector<PagedKvCache::Seq> sm(n), ss(n);
    std::vector<float> got(std::size_t{n} * V), want(V);
    std::vector<float> normed(std::size_t{n} * E);
    double worst = 0.0;
    for (std::uint32_t p = 0; p < 6; ++p) {
        CAPTURE(p);
        std::vector<locus::model::EmbedStep> steps(n);
        std::vector<locus::tok::TokenId> toks(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            REQUIRE(cm.ensure_capacity(sm[i], 1));
            toks[i] = static_cast<locus::tok::TokenId>(
                100 + (p * 37 + i * 1009) % 30000);
            steps[i] = {toks[i], {}, &sm[i],
                        std::span(got).subspan(std::size_t{i} * V, V),
                        std::span(normed).subspan(std::size_t{i} * E, E)};
        }
        // Declined before i#623 (MoE layers, q / k norms).
        REQUIRE(locus::backend::vulkan_forward_multi(model, steps, cm, 0, V));
        for (std::uint32_t i = 0; i < n; ++i) {
            CAPTURE(i);
            REQUIRE(cs.ensure_capacity(ss[i], 1));
            REQUIRE(locus::backend::vulkan_forward(model, toks[i], cs, ss[i],
                                                   want));
            const auto row = std::span(got).subspan(std::size_t{i} * V, V);
            double err = 0.0, norm = 0.0;
            for (std::uint32_t v = 0; v < V; ++v) {
                const double d = double{row[v]} - want[v];
                err += d * d;
                norm += double{want[v]} * want[v];
            }
            worst = std::max(worst, std::sqrt(err / norm));
            // Measured on an M2 Pro: 1.2e-5.
            CHECK(std::sqrt(err) <= 1e-3 * std::sqrt(norm));
            CHECK(std::max_element(row.begin(), row.end()) - row.begin() ==
                  std::max_element(want.begin(), want.end()) - want.begin());
        }
    }
    WARN("worst relative error " << worst);
}

// forward_batch_layers on both stages: 10 sequences run as multi passes
// of 8 and 2 (the multi path's cap), and one sequence keeps the single
// kernels (outside the batch-invariant mode), as forward() does.
TEST_CASE("batched vulkan stages chunk by 8 and keep a lone sequence "
          "single",
          "[vulkan][pipeline]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    const std::uint32_t n = GENERATE(10u, 1u);
    CAPTURE(n);
    if (n == 1 && locus::backend::vulkan_batch_invariant()) {
        SKIP("batch-invariant mode: a lone sequence takes the multi pass");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    const auto& vk = *locus::backend::find_backend("vulkan");
    auto full = TransformerModel::load(g);
    full.use_backend(vk);
    const std::uint32_t L = full.hparams().n_layers;
    const std::uint32_t E = full.hparams().n_embd;
    const std::uint32_t V = full.hparams().n_vocab;
    const std::uint32_t k = L / 2;
    auto a = TransformerModel::load(g, 0, k);
    auto b = TransformerModel::load(g, k, L);
    a.use_backend(vk);
    b.use_backend(vk);
    PagedKvCache fc = full.make_cache();
    PagedKvCache ca = a.make_cache();
    PagedKvCache cb = b.make_cache();
    auto wa = a.make_workspace();
    auto wb = b.make_workspace();
    std::vector<PagedKvCache::Seq> fs(n), sa(n), sb(n);
    std::vector<PagedKvCache::Seq*> pa, pb;
    for (std::uint32_t i = 0; i < n; ++i) {
        pa.push_back(&sa[i]);
        pb.push_back(&sb[i]);
    }
    std::vector<float> want(std::size_t{n} * V), got(want.size());
    std::vector<float> normed(std::size_t{n} * E);
    std::vector<float> hidden(std::size_t{n} * E);
    std::vector<locus::tok::TokenId> toks(n);
    for (std::uint32_t p = 0; p < 20; ++p) {
        CAPTURE(p);
        std::vector<locus::model::EmbedStep> fsteps(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            REQUIRE(fc.ensure_capacity(fs[i], 1));
            REQUIRE(ca.ensure_capacity(sa[i], 1));
            REQUIRE(cb.ensure_capacity(sb[i], 1));
            // A different token per sequence and position.
            toks[i] = static_cast<locus::tok::TokenId>(1 + (p * 7 + i * 13) %
                                                           (V - 1));
            fsteps[i] = {toks[i], {}, &fs[i],
                         std::span(want).subspan(std::size_t{i} * V, V),
                         std::span(normed).subspan(std::size_t{i} * E, E)};
        }
        // The reference in the same chunks: multi passes of 8 and 2, or
        // the single kernels for a lone sequence.
        if (n == 1) {
            REQUIRE(locus::backend::vulkan_forward(full, toks[0], fc, fs[0],
                                                   want));
        } else {
            for (std::uint32_t c = 0; c < n; c += 8) {
                const std::uint32_t cn = std::min(8u, n - c);
                REQUIRE(locus::backend::vulkan_forward_multi(
                    full, std::span(fsteps).subspan(c, cn), fc, 0, V));
            }
        }
        a.forward_batch_layers(toks, {}, 0, k, ca, pa, wa, hidden);
        for (auto& s : sa) {
            s.n_tokens += 1;
        }
        b.forward_batch_layers({}, hidden, k, L, cb, pb, wb, got);
        for (const auto& s : sb) {
            REQUIRE(s.n_tokens == p + 1);
        }
        REQUIRE(got == want);
    }
}

// i#116: past a cap of blocks, decode attention merges its per-block
// partials in rounds (attn_combine_part), so the scratch stays bounded
// at long context. With the cap forced to 1 and 2 blocks a round, a
// 100-position decode (4 blocks of 32 grouped, or 2 of 64) takes the
// round path from position 33 on; its logits track the one-round path
// within float rounding (the merge order changes) with the same pick.
TEST_CASE("vulkan decode attention merges its partials in rounds",
          "[vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    const KvType kv = GENERATE(KvType::kF32, KvType::kF16);
    CAPTURE(static_cast<int>(kv));
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = TransformerModel::load(g);
    model.use_backend(*locus::backend::find_backend("vulkan"));
    const std::uint32_t V = model.hparams().n_vocab;
    constexpr std::uint32_t n_pos = 100;
    // The logits at every position, with the partials cap at `cap`
    // blocks a round (0: the default, one round here).
    auto run = [&](std::uint32_t cap) {
        locus::backend::vulkan_set_attn_part_blocks(cap);
        PagedKvCache c = model.make_cache(0, kv);
        PagedKvCache::Seq s;
        std::vector<float> all(std::size_t{n_pos} * V);
        for (std::uint32_t p = 0; p < n_pos; ++p) {
            REQUIRE(c.ensure_capacity(s, 1));
            const auto t =
                static_cast<locus::tok::TokenId>(1 + (p * 37) % (V - 1));
            REQUIRE(locus::backend::vulkan_forward(
                model, t, c, s,
                std::span(all).subspan(std::size_t{p} * V, V)));
        }
        c.release(s);
        locus::backend::vulkan_set_attn_part_blocks(0);
        return all;
    };
    const auto want = run(0);
    for (const std::uint32_t cap : {1u, 2u}) {
        CAPTURE(cap);
        const auto got = run(cap);
        for (std::uint32_t p = 0; p < n_pos; ++p) {
            CAPTURE(p);
            const auto w = std::span(want).subspan(std::size_t{p} * V, V);
            const auto o = std::span(got).subspan(std::size_t{p} * V, V);
            double err = 0.0, norm = 0.0;
            for (std::uint32_t v = 0; v < V; ++v) {
                const double d = double{o[v]} - w[v];
                err += d * d;
                norm += double{w[v]} * w[v];
            }
            CHECK(std::sqrt(err) <= 1e-5 * std::sqrt(norm));
            CHECK(std::max_element(o.begin(), o.end()) - o.begin() ==
                  std::max_element(w.begin(), w.end()) - w.begin());
        }
    }
}
