// GQA attention on the GPU over the device KV mirror (i#466 step 2).

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <span>
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

    // Every group from 1 to 8 on the same KV (i#1196: Qwen2.5-0.5B's is
    // 7, Llama 3.2 3B's 3): the device matches the host.
    for (std::uint32_t grp = 1; grp <= 8; ++grp) {
        INFO("group " << grp);
        locus::model::Hparams hg = hp;
        hg.n_heads = n_kv * grp;
        const std::size_t w = std::size_t{hg.n_heads} * hd;
        std::vector<float> qg(w), og(w), rg(w);
        for (auto& v : qg) {
            v = ud(rng);
        }
        REQUIRE(locus::model::device_gqa_attend(cuda.ops, cache, seq, layer,
                                                hg, n_pos - 1, 1, qg.data(),
                                                og.data(), w));
        locus::model::gqa_attend(
            qg.data(), rg.data(), hg.n_heads, hd, grp, n_pos - 1,
            hg.kq_scale,
            [&](std::uint32_t t) -> const float* {
                return cache.k(seq, layer, t);
            },
            [&](std::uint32_t t) -> const float* {
                return cache.v(seq, layer, t);
            });
        for (std::size_t j = 0; j < w; ++j) {
            REQUIRE(og[j] == Catch::Approx(rg[j]).epsilon(1e-5).margin(1e-5));
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

// i#1254: a prefill chunk of one sequence at head width 64 or 128 takes
// the flash kernel (K / V tiles in shared memory, an online softmax);
// it tracks the host attention for every group the CUDA path takes, a
// chunk from position 0 and one deep in the sequence (tiles of 32 or 16
// positions cut mid-chunk, a last query tile not full).
TEST_CASE("cuda flash prefill attention tracks the host",
          "[cuda][attention][mirror]") {
    if (!locus::backend::cuda_backend_usable()) {
        SKIP("no usable CUDA device");
    }
    const auto& cuda = *locus::backend::find_backend("cuda");
    std::mt19937 rng(1254);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    for (const std::uint32_t hd : {64u, 128u}) {
        constexpr std::uint32_t n_kv = 2, layer = 0, n_pos = 150;
        PagedKvCache::Geometry g;
        g.n_layers = 1;
        g.kv_dim = n_kv * hd;
        g.block_tokens = 16;
        g.n_blocks = 12;
        PagedKvCache cache(g);
        void* dev = locus::backend::cuda_kv_mirror_alloc(
            PagedKvCache::pool_floats(g));
        REQUIRE(dev != nullptr);
        cache.attach_mirror({dev, &locus::backend::cuda_kv_mirror_store,
                             &locus::backend::cuda_kv_mirror_free});
        // Another sequence first, so seq's blocks are not the pool's
        // first ones and its table can sit at an offset (below).
        Seq other, seq;
        REQUIRE(cache.ensure_capacity(other, 20));
        REQUIRE(cache.ensure_capacity(seq, n_pos));
        for (Seq* s : {&other, &seq}) {
            const std::uint32_t np = s == &seq ? n_pos : 20;
            for (std::uint32_t t = 0; t < np; ++t) {
                for (std::uint32_t i = 0; i < g.kv_dim; ++i) {
                    cache.k(*s, layer, t)[i] = ud(rng);
                    cache.v(*s, layer, t)[i] = ud(rng);
                }
            }
            cache.sync_rows(*s, layer, 0, np);
        }
        const auto host_ref = [&](const locus::model::Hparams& hp,
                                  std::uint32_t grp, const float* qr,
                                  std::uint32_t pos, float* ref) {
            locus::model::gqa_attend(
                qr, ref, hp.n_heads, hd, grp, pos, hp.kq_scale,
                [&](std::uint32_t t) -> const float* {
                    return cache.k(seq, layer, t);
                },
                [&](std::uint32_t t) -> const float* {
                    return cache.v(seq, layer, t);
                });
        };
        // seq's chunk with its table at an offset in a shared blocks
        // array (other's table first), as a CUDA forward's device tables
        // hold several sequences: the flash kernel must start at it.
        {
            locus::model::Hparams hp;
            hp.n_kv_heads = n_kv;
            hp.n_heads = n_kv * 4;
            hp.head_dim = hd;
            hp.kq_scale = 1.0f / std::sqrt(static_cast<float>(hd));
            const std::size_t w = std::size_t{hp.n_heads} * hd;
            constexpr std::uint32_t n_q = 24, base = 100;
            std::vector<std::uint32_t> blocks(other.blocks.begin(),
                                              other.blocks.end());
            const auto off = static_cast<std::uint32_t>(blocks.size());
            blocks.insert(blocks.end(), seq.blocks.begin(), seq.blocks.end());
            std::vector<std::uint32_t> pos(n_q), tab(n_q, off);
            for (std::uint32_t i = 0; i < n_q; ++i) {
                pos[i] = base + i;
            }
            std::vector<float> q(n_q * w), out(n_q * w);
            for (auto& v : q) {
                v = ud(rng);
            }
            locus::backend::KvAttend a;
            a.mirror = cache.mirror_data();
            a.blocks = blocks.data();
            a.n_blocks = static_cast<std::uint32_t>(blocks.size());
            a.block_stride = cache.block_floats();
            a.layer_off = cache.layer_offset(layer);
            a.v_off = cache.layer_floats() / 2;
            a.block_tokens = g.block_tokens;
            a.n_heads = hp.n_heads;
            a.n_kv_heads = n_kv;
            a.head_dim = hd;
            a.kq_scale = hp.kq_scale;
            a.n_q = n_q;
            a.q_pos = pos.data();
            a.q_table = tab.data();
            a.q = q.data();
            a.out = out.data();
            a.q_stride = w;
            INFO("hd " << hd << ", table at offset " << off);
            REQUIRE(locus::backend::cuda_kv_attend(a));
            for (std::uint32_t i = 0; i < n_q; ++i) {
                std::vector<float> ref(w);
                host_ref(hp, 4, q.data() + i * w, base + i, ref.data());
                for (std::size_t j = 0; j < w; ++j) {
                    REQUIRE(out[i * w + j] ==
                            Catch::Approx(ref[j]).epsilon(1e-4).margin(1e-5));
                }
            }
        }
        for (const std::uint32_t grp : {1u, 4u, 7u, 8u}) {
            locus::model::Hparams hp;
            hp.n_kv_heads = n_kv;
            hp.n_heads = n_kv * grp;
            hp.head_dim = hd;
            hp.kq_scale = 1.0f / std::sqrt(static_cast<float>(hd));
            const std::size_t w = std::size_t{hp.n_heads} * hd;
            for (const auto& [base, n_q] :
                 {std::pair<std::uint32_t, std::uint32_t>{0, 40},
                  {113, 37}}) {
                INFO("hd " << hd << " group " << grp << " base " << base
                           << " n_q " << n_q);
                std::vector<float> q(n_q * w), out(n_q * w);
                for (auto& v : q) {
                    v = ud(rng);
                }
                REQUIRE(locus::model::device_gqa_attend(
                    cuda.ops, cache, seq, layer, hp, base, n_q, q.data(),
                    out.data(), w));
                for (std::uint32_t i = 0; i < n_q; ++i) {
                    std::vector<float> ref(w);
                    locus::model::gqa_attend(
                        q.data() + i * w, ref.data(), hp.n_heads, hd, grp,
                        base + i, hp.kq_scale,
                        [&](std::uint32_t t) -> const float* {
                            return cache.k(seq, layer, t);
                        },
                        [&](std::uint32_t t) -> const float* {
                            return cache.v(seq, layer, t);
                        });
                    for (std::size_t j = 0; j < w; ++j) {
                        REQUIRE(out[i * w + j] ==
                                Catch::Approx(ref[j]).epsilon(1e-4).margin(
                                    1e-5));
                    }
                }
            }
        }
    }
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

TEST_CASE("cuda forward runs qwen2 on the GPU path, prefill equal to decode",
          "[cuda][attention][mirror][qwen]") {
    // Qwen2's attention (q / k / v biases, NEOX RoPE) takes the
    // one-submit forward (i#1196): the device stage moves and the per-op
    // attention stage does not; a prefill tracks the decode steps (its
    // attention is the flash kernel, i#1254); both stay close to the
    // CPU's forward.
    const auto* cuda = locus::backend::find_backend("cuda");
    if (cuda == nullptr || !cuda->available) {
        SKIP("no usable CUDA backend");
    }
    const std::string path = std::string(LOCUS_SOURCE_DIR) +
                             "/tests/models/qwen2.5-0.5b-instruct-q8_0.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP(path + " absent");
    }
    using locus::model::Stage;
    auto g = locus::gguf::GgufFile::open(path);
    REQUIRE(g.get_string("general.architecture") == "qwen2");
    auto model = locus::model::TransformerModel::load(g);
    const std::uint32_t nv = model.hparams().n_vocab;
    const std::vector<locus::tok::TokenId> prompt{
        9707, 11, 1246, 525, 498, 3351, 30, 576, 6722, 315, 9625, 374};
    const auto n = static_cast<std::uint32_t>(prompt.size());
    model.use_backend(*cuda);
    auto ws = model.make_workspace();
    const auto* sm = model.stage_metrics();
    REQUIRE(sm != nullptr);
    const std::uint64_t dev0 = sm->stage_ns(Stage::kDevice);
    const std::uint64_t att0 = sm->stage_ns(Stage::kAttention);
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
    CHECK(sm->stage_ns(Stage::kDevice) > dev0);
    CHECK(sm->stage_ns(Stage::kAttention) == att0);
    // The prefill's attention is the flash kernel (i#1254), decode's the
    // two-pass one: close, not bit for bit (DESIGN 16), and the same top
    // token at every position.
    {
        float gap = 0.0f, big = 0.0f;
        for (std::size_t i = 0; i < all.size(); ++i) {
            gap = std::max(gap, std::fabs(all[i] - step[i]));
            big = std::max(big, std::fabs(step[i]));
        }
        WARN("prefill against decode, largest logit gap " << gap << " of "
                                                          << big);
        // On Q8_0 activations (i#1254) the attention's float gap can
        // move a quant by one step: measured 0.40 of 23 (3.2e-5 on f32).
        CHECK(gap <= (locus::backend::cuda_q8_int8_on() ? 0.03f : 1e-4f) *
                         big);
        for (std::uint32_t t = 0; t < n; ++t) {
            INFO("position " << t);
            CHECK(locus::model::argmax(std::span<const float>(
                      all.data() + std::size_t{t} * nv, nv)) ==
                  locus::model::argmax(std::span<const float>(
                      step.data() + std::size_t{t} * nv, nv)));
        }
    }

    // The CPU on the activations the CUDA forward uses: Q8_0 ones for
    // its Q8_0 matvecs (i#1254), else f32 (the two put the logits about
    // 1.3 apart, of 23).
    model.use_backend(locus::backend::best_backend());
    model.use_q8k_activations(locus::backend::cuda_q8_int8_on());
    std::vector<float> cpu(std::size_t{n} * nv);
    {
        auto cache = model.make_cache(8);
        Seq s;
        REQUIRE(cache.ensure_capacity(s, n));
        model.forward_batch(prompt, cache, s, ws, cpu, true);
    }
    float worst = 0.0f, top = 0.0f;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        worst = std::max(worst, std::fabs(cpu[i] - all[i]));
        top = std::max(top, std::fabs(cpu[i]));
    }
    if (!locus::backend::cuda_q8_int8_on()) {
        CHECK(worst <= 1e-4f * top);  // measured 4.7e-5 of 23
    } else {
        // Both quantize their activations, so a float rounding apart
        // upstream (sum orders) can move a quant by one step and the
        // logits by more than the rounding: measured 0.40 to 0.42 of 23.
        // Each position's top token must agree.
        WARN("logit gap against the CPU on Q8 activations: " << worst
                                                             << " of " << top);
        CHECK(worst <= 0.03f * top);
        for (std::uint32_t t = 0; t < n; ++t) {
            const auto row = [&](const std::vector<float>& v) {
                const auto b = v.begin() + std::ptrdiff_t{t} * nv;
                return std::max_element(b, b + nv) - b;
            };
            INFO("position " << t);
            CHECK(row(cpu) == row(all));
        }
    }
    // Its 640 MB of weights would stay in the 2 GB card's pool and
    // leave the next test's KV mirror no room.
    locus::backend::cuda_pool_reset();
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
