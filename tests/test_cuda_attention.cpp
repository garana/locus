// GQA attention on the GPU over the device KV mirror (i#466 step 2).

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/gqa_attend.hpp"
#include "locus/model/transformer.hpp"

using locus::kv::PagedKvCache;
using Seq = PagedKvCache::Seq;

namespace {

bool same_bits(float a, float b) {
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

}  // namespace

TEST_CASE("cuda attention over the KV mirror matches the host",
          "[cuda][attention][mirror]") {
    if (!locus::backend::cuda_backend_usable()) {
        SKIP("no usable CUDA device");
    }
    // 8 query heads over 2 KV heads (group 4), head_dim 32, layer 1 of
    // 2, 16-token blocks; 100 positions over 7 blocks that are not
    // consecutive in the pool (another sequence holds blocks 0 and 1).
    constexpr std::uint32_t n_heads = 8, n_kv = 2, hd = 32, layer = 1;
    PagedKvCache::Geometry g;
    g.n_layers = 2;
    g.kv_dim = n_kv * hd;
    g.block_tokens = 16;
    g.n_blocks = 12;
    PagedKvCache cache(g);
    void* dev = locus::backend::cuda_kv_mirror_alloc(
        PagedKvCache::pool_floats(g));
    REQUIRE(dev != nullptr);
    cache.attach_mirror({dev, &locus::backend::cuda_kv_mirror_store,
                         &locus::backend::cuda_kv_mirror_free});
    Seq other, seq;
    REQUIRE(cache.ensure_capacity(other, 32));
    constexpr std::uint32_t n_pos = 100;
    REQUIRE(cache.ensure_capacity(seq, n_pos));
    std::mt19937 rng(466);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    for (std::uint32_t t = 0; t < n_pos; ++t) {
        for (std::uint32_t i = 0; i < g.kv_dim; ++i) {
            cache.k(seq, layer, t)[i] = ud(rng);
            cache.v(seq, layer, t)[i] = ud(rng);
        }
    }
    cache.sync_rows(seq, layer, 0, n_pos);

    locus::model::Hparams hp;
    hp.n_heads = n_heads;
    hp.n_kv_heads = n_kv;
    hp.head_dim = hd;
    hp.kq_scale = 1.0f / std::sqrt(static_cast<float>(hd));
    // Five queries at positions 95 .. 99, rows 300 floats apart: the gap
    // past each row's 256 floats must stay untouched.
    constexpr std::uint32_t n_q = 5, base = n_pos - n_q;
    constexpr std::size_t width = n_heads * hd, stride = 300;
    std::vector<float> q(n_q * stride), out(n_q * stride, 7.0f);
    for (auto& v : q) {
        v = ud(rng);
    }
    const auto& cuda = *locus::backend::find_backend("cuda");
    REQUIRE(locus::model::device_gqa_attend(cuda.ops, cache, seq, layer, hp,
                                            base, n_q, q.data(), out.data(),
                                            stride));
    for (std::uint32_t i = 0; i < n_q; ++i) {
        INFO("query " << i);
        std::vector<float> ref(width);
        locus::model::gqa_attend(
            q.data() + i * stride, ref.data(), n_heads, hd, n_heads / n_kv,
            base + i, hp.kq_scale,
            [&](std::uint32_t t) -> const float* {
                return cache.k(seq, layer, t);
            },
            [&](std::uint32_t t) -> const float* {
                return cache.v(seq, layer, t);
            });
        for (std::size_t j = 0; j < width; ++j) {
            REQUIRE(out[i * stride + j] ==
                    Catch::Approx(ref[j]).epsilon(1e-5).margin(1e-5));
        }
        for (std::size_t j = width; j < stride; ++j) {
            REQUIRE(out[i * stride + j] == 7.0f);
        }
        // The same query alone gives the same bits as in the batch.
        std::vector<float> one(width);
        REQUIRE(locus::model::device_gqa_attend(
            cuda.ops, cache, seq, layer, hp, base + i, 1,
            q.data() + i * stride, one.data(), width));
        for (std::size_t j = 0; j < width; ++j) {
            REQUIRE(same_bits(one[j], out[i * stride + j]));
        }
    }

    // Two sequences in one launch (batch decode): each query gets the
    // bits of its own single-sequence launch.
    Seq seq2;
    REQUIRE(cache.ensure_capacity(seq2, 40));
    for (std::uint32_t t = 0; t < 40; ++t) {
        for (std::uint32_t i = 0; i < g.kv_dim; ++i) {
            cache.k(seq2, layer, t)[i] = ud(rng);
            cache.v(seq2, layer, t)[i] = ud(rng);
        }
    }
    cache.sync_rows(seq2, layer, 0, 40);
    {
        Seq* both[] = {&seq, &seq2};
        const std::uint32_t pos[] = {n_pos - 1, 39};
        std::vector<float> got(2 * width);
        REQUIRE(locus::model::device_gqa_attend_seqs(
            cuda.ops, cache, both, layer, hp, pos, q.data(), got.data(),
            width));
        for (std::uint32_t t = 0; t < 2; ++t) {
            INFO("sequence " << t);
            std::vector<float> one(width);
            REQUIRE(locus::model::device_gqa_attend(
                cuda.ops, cache, *both[t], layer, hp, pos[t], 1,
                q.data() + t * width, one.data(), width));
            for (std::size_t j = 0; j < width; ++j) {
                REQUIRE(same_bits(one[j], got[t * width + j]));
            }
        }
    }

    // Without a mirror the caller attends on the host.
    PagedKvCache plain(g);
    Seq ps;
    REQUIRE(plain.ensure_capacity(ps, 1));
    REQUIRE_FALSE(locus::model::device_gqa_attend(
        cuda.ops, plain, ps, layer, hp, 0, 1, q.data(), out.data(), width));
}

TEST_CASE("cuda attention: prefill and decode give the same logits",
          "[cuda][attention][mirror]") {
    const auto* cuda = locus::backend::find_backend("cuda");
    if (cuda == nullptr || !cuda->available) {
        SKIP("no usable CUDA backend");
    }
    const std::string path =
        std::string(LOCUS_SOURCE_DIR) + "/tests/models/stories260K.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP(path + " absent");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    const std::uint32_t nv = model.hparams().n_vocab;
    const std::vector<locus::tok::TokenId> prompt{
        1, 403, 407, 261, 89, 17, 23, 42, 300, 12, 77, 5, 401, 260, 33};
    const auto n = static_cast<std::uint32_t>(prompt.size());

    // One prefill with every position's logits (speculative verify's
    // path), then the same tokens one decode step at a time.
    model.use_backend(*cuda);
    auto ws = model.make_workspace();
    std::vector<float> all(std::size_t{n} * nv);
    {
        auto cache = model.make_cache(8);
        REQUIRE(cache.mirror_data() != nullptr);
        Seq s;
        REQUIRE(cache.ensure_capacity(s, n));
        model.forward_batch(prompt, cache, s, ws, all, true);
    }
    std::vector<float> step(std::size_t{n} * nv);
    {
        auto cache = model.make_cache(8);
        Seq s;
        REQUIRE(cache.ensure_capacity(s, n));
        for (std::uint32_t t = 0; t < n; ++t) {
            model.forward(prompt[t], cache, s, ws,
                          {step.data() + std::size_t{t} * nv, nv});
        }
    }
    for (std::size_t i = 0; i < all.size(); ++i) {
        INFO("position " << i / nv << " logit " << i % nv);
        REQUIRE(same_bits(all[i], step[i]));
    }

    // And close to the host attention on the CPU.
    model.use_backend(locus::backend::best_backend());
    std::vector<float> cpu(std::size_t{n} * nv);
    {
        auto cache = model.make_cache(8);
        REQUIRE(cache.mirror_data() == nullptr);
        Seq s;
        REQUIRE(cache.ensure_capacity(s, n));
        model.forward_batch(prompt, cache, s, ws, cpu, true);
    }
    float worst = 0.0f, top = 0.0f;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        worst = std::max(worst, std::fabs(cpu[i] - all[i]));
        top = std::max(top, std::fabs(cpu[i]));
    }
    CHECK(worst <= 1e-3f * top);
}

TEST_CASE("cuda forward with a head row range keeps the GPU path and the "
          "kept rows' bits",
          "[cuda][attention][mirror]") {
    // Speech computes only its own tokens' head rows (i#173); on CUDA
    // that must still run the whole token on the GPU (i#689), with the
    // same bits in the kept rows as the full head.
    const auto* cuda = locus::backend::find_backend("cuda");
    if (cuda == nullptr || !cuda->available) {
        SKIP("no usable CUDA backend");
    }
    const std::string path =
        std::string(LOCUS_SOURCE_DIR) + "/tests/models/stories260K.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP(path + " absent");
    }
    using locus::model::Stage;
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    model.use_backend(*cuda);
    const std::uint32_t nv = model.hparams().n_vocab;
    const locus::model::LogitRows rows{nv / 3, nv - nv / 5};
    const std::vector<locus::tok::TokenId> prompt{1, 403, 407, 261, 89, 17};
    const auto n = static_cast<std::uint32_t>(prompt.size());
    auto ws = model.make_workspace();
    const auto run = [&](locus::model::LogitRows r) {
        std::vector<float> out(std::size_t{n} * nv);
        auto cache = model.make_cache(8);
        REQUIRE(cache.mirror_data() != nullptr);
        Seq s;
        REQUIRE(cache.ensure_capacity(s, n));
        for (std::uint32_t t = 0; t < n; ++t) {
            model.forward(prompt[t], cache, s, ws,
                          {out.data() + std::size_t{t} * nv, nv}, r);
        }
        return out;
    };
    const auto full = run({});
    const auto* sm = model.stage_metrics();
    REQUIRE(sm != nullptr);
    const std::uint64_t head0 = sm->stage_ns(Stage::kHead);
    const std::uint64_t dev0 = sm->stage_ns(Stage::kDevice);
    const auto part = run(rows);
    // The per-op path laps the head; the GPU path laps only the device.
    CHECK(sm->stage_ns(Stage::kHead) == head0);
    CHECK(sm->stage_ns(Stage::kDevice) > dev0);
    for (std::size_t i = 0; i < part.size(); ++i) {
        const std::uint32_t v = static_cast<std::uint32_t>(i % nv);
        INFO("position " << i / nv << " logit " << v);
        if (v >= rows.begin && v < rows.end) {
            REQUIRE(same_bits(part[i], full[i]));
        } else {
            REQUIRE(part[i] == -std::numeric_limits<float>::infinity());
        }
    }

    // A reversed range throws through the public forward_layers, on
    // CUDA and on the host, instead of wrapping the row count.
    for (const auto* b : {cuda, &locus::backend::best_backend()}) {
        model.use_backend(*b);
        INFO(b->name);
        auto cache = model.make_cache(8);
        Seq s;
        REQUIRE(cache.ensure_capacity(s, 1));
        std::vector<float> out(nv);
        REQUIRE_THROWS_AS(
            model.forward_layers(prompt[0], {}, 0, model.hparams().n_layers,
                                 cache, s, ws, out, {10, 5}),
            std::invalid_argument);
    }
}
