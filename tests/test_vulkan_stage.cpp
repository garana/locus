#include <filesystem>
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
