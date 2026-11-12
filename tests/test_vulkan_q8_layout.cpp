// i#678: Q8_0 weights in GGUF's own block layout (read in place) give
// the same bits as the split GPU copy on every Q8_0 kernel family: only
// the loads differ, not the sums or their order.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/backend/vulkan/q8_layout.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"

using locus::backend::Q8Layout;
using locus::backend::vk::Kernel;
using locus::backend::vk::VulkanContext;
using locus::model::TransformerModel;
using Seq = locus::kv::PagedKvCache::Seq;

namespace {

std::string llama_q8() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/llama-3.2-1b-q8_0.gguf";
}

/** Every output of one model under one layout: a lone sequence's
 *  decode (single-sequence kernels: the matvec_q8_0 family), the same
 *  through the multi-sequence kernels (the _n family), three sequences
 *  at once, and a batched prefill (the Q8_0 matmul). */
std::vector<float> run_all(Q8Layout layout) {
    locus::backend::vulkan_reset_state();
    locus::backend::vulkan_set_q8_layout(layout);
    auto g = locus::gguf::GgufFile::open(llama_q8());
    auto model = TransformerModel::load(g);
    model.use_backend(*locus::backend::find_backend("vulkan"));
    const std::uint32_t V = model.hparams().n_vocab;
    std::vector<float> all;
    const std::vector<locus::tok::TokenId> toks = {128000, 9906, 11, 1917,
                                                   0,      578};
    for (const bool invariant : {false, true}) {
        locus::backend::vulkan_set_batch_invariant(invariant);
        auto cache = model.make_cache(512);
        auto ws = model.make_workspace();
        Seq seq;
        std::vector<float> logits(V);
        for (auto t : toks) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(t, cache, seq, ws, logits);
            all.insert(all.end(), logits.begin(), logits.end());
        }
        cache.release(seq);
    }
    locus::backend::vulkan_reset_batch_invariant();
    {
        // A 70-token prefill: long enough for the GPU batch path.
        std::vector<locus::tok::TokenId> p(70);
        for (std::size_t i = 0; i < p.size(); ++i) {
            p[i] = static_cast<locus::tok::TokenId>(1000 + 37 * i);
        }
        auto cache = model.make_cache(512);
        auto ws = model.make_workspace();
        Seq seq;
        REQUIRE(cache.ensure_capacity(seq, 70));
        std::vector<float> logits(V);
        model.forward_batch(p, cache, seq, ws, logits);
        all.insert(all.end(), logits.begin(), logits.end());
        cache.release(seq);
    }
    locus::backend::vulkan_set_q8_layout(Q8Layout::kAuto);
    locus::backend::vulkan_reset_state();
    return all;
}

}  // namespace

TEST_CASE("vulkan q8_0 in GGUF's layout gives the split layout's bits",
          "[vulkan][e2e]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    if (!std::filesystem::exists(llama_q8())) {
        SKIP("llama-3.2-1b-q8_0.gguf absent");
    }
    const auto split = run_all(Q8Layout::kSplit);
    const auto gguf = run_all(Q8Layout::kGguf);
    REQUIRE(split.size() == gguf.size());
    REQUIRE(split == gguf);
}

// matvec_t_q8 (MLA's weight absorption) on the same weights in both
// layouts, its buffers marked as the layouts they hold.
TEST_CASE("vulkan matvec_t_q8 reads both Q8_0 layouts alike", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(678);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const std::uint32_t rows = 9, cols = 256, heads = 2;
    const std::size_t row_bytes = std::size_t{cols} / 32 * 34;
    std::vector<std::byte> w(std::size_t{rows} * heads * row_bytes);
    for (std::size_t b = 0; b < w.size() / 34; ++b) {
        const std::uint16_t d =
            locus::backend::f32_to_f16(0.01f + 0.001f * (b % 7));
        std::memcpy(&w[b * 34], &d, 2);
        for (int i = 0; i < 32; ++i) {
            w[b * 34 + 2 + i] =
                static_cast<std::byte>(static_cast<std::int8_t>(qd(rng)));
        }
    }
    std::vector<std::byte> split(w.size());
    locus::backend::vk::q8_0_split_rows(w.data(), split.data(),
                                        std::size_t{rows} * heads, cols);
    std::vector<float> x(std::size_t{rows} * heads);
    for (auto& v : x) v = dist(rng);
    auto run = [&](const std::vector<std::byte>& bytes, bool gguf) {
        auto wb = ctx.create_buffer(bytes.size());
        ctx.write_buffer(wb, bytes);
        ctx.set_q8_gguf(wb, gguf);
        auto xb = ctx.create_buffer(x.size() * 4);
        ctx.write_buffer(xb, std::as_bytes(std::span(x)));
        auto ob = ctx.create_buffer(std::size_t{cols} * heads * 4);
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {wb, xb, ob};
        const std::uint32_t push[] = {
            rows,
            cols,
            0,
            0,
            0,
            static_cast<std::uint32_t>(rows * row_bytes),
            rows,
            cols};
        ctx.dispatch(Kernel::kMatvecTQ8, bufs, push, cols / 128, heads);
        ctx.end_batch();
        std::vector<float> out(std::size_t{cols} * heads);
        ctx.read_buffer(ob, std::as_writable_bytes(std::span(out)));
        for (auto b : {wb, xb, ob}) ctx.destroy_buffer(b);
        return out;
    };
    const auto a = run(split, false);
    const auto b = run(w, true);
    REQUIRE(a == b);
    // And not trivially: the GGUF bytes read as split differ.
    CHECK(run(w, false) != a);
}
