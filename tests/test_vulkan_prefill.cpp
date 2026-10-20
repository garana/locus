#include <algorithm>
#include <bit>
#include <cmath>
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
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"

using locus::backend::vk::Kernel;
using locus::backend::vk::VulkanContext;

namespace {

std::uint32_t fbits(float v) { return std::bit_cast<std::uint32_t>(v); }

template <typename T>
VulkanContext::Buffer upload(VulkanContext& ctx, const std::vector<T>& v) {
    auto b = ctx.create_buffer(
        std::max<std::size_t>((v.size() * sizeof(T) + 3) & ~std::size_t{3},
                              4));
    ctx.write_buffer(b, std::as_bytes(std::span<const T>(v)));
    return b;
}

template <typename T>
std::vector<T> download(VulkanContext& ctx, VulkanContext::Buffer b,
                        std::size_t n) {
    std::vector<T> out(n);
    ctx.read_buffer(b, std::as_writable_bytes(std::span<T>(out)));
    return out;
}

/** Random Q8_0 bytes for rows x cols (34-byte blocks). */
std::vector<std::byte> random_q8(std::uint32_t rows, std::uint32_t cols,
                                 std::mt19937& rng) {
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.001f, 0.02f);
    std::vector<std::byte> w(static_cast<std::size_t>(rows) * (cols / 32) *
                             34);
    for (std::size_t b = 0; b < w.size(); b += 34) {
        const std::uint16_t h = locus::backend::f32_to_f16(sd(rng));
        std::memcpy(&w[b], &h, 2);
        for (int i = 0; i < 32; ++i) {
            w[b + 2 + i] =
                static_cast<std::byte>(static_cast<std::int8_t>(qd(rng)));
        }
    }
    return w;
}

std::vector<float> random_f32(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = d(rng);
    }
    return v;
}

/** dot(row r of w, x[0 .. cols)) on the CPU. */
float cpu_dot(const locus::backend::Mat& w, std::uint32_t r,
              const float* x) {
    std::vector<float> row(w.cols);
    locus::backend::dequant_row(w, r, row);
    double acc = 0.0;
    for (std::uint32_t c = 0; c < w.cols; ++c) {
        acc += static_cast<double>(row[c]) * x[c];
    }
    return static_cast<float>(acc);
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

std::size_t argmax(const std::vector<float>& v) {
    return static_cast<std::size_t>(
        std::max_element(v.begin(), v.end()) - v.begin());
}

}  // namespace

TEST_CASE("vulkan matmul matches the CPU reference", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(41);

    // Ragged on purpose: rows and tokens are not tile multiples, the
    // F32 cols are not a K-step multiple, and 3 batches shift every
    // operand by its batch stride.
    struct Case {
        const char* name;
        locus::gguf::TensorType type;
        Kernel kernel;
        std::uint32_t cols;
    };
    const Case cases[] = {
        {"q8_0", locus::gguf::TensorType::kQ8_0, Kernel::kMatmulQ8_0, 96},
        {"f32", locus::gguf::TensorType::kF32, Kernel::kMatmulF32, 80},
    };
    const std::uint32_t rows = 100, n_tok = 70, n_batch = 3;
    const std::uint32_t x_stride = 300, out_stride = 333;
    const std::uint32_t x_bstride = 7, out_bstride = 101;
    const float scale = 0.75f;
    for (const Case& c : cases) {
        DYNAMIC_SECTION(c.name) {
            const bool q8 = c.type == locus::gguf::TensorType::kQ8_0;
            const std::size_t mat_bytes =
                q8 ? std::size_t{rows} * (c.cols / 32) * 34
                   : std::size_t{rows} * c.cols * 4;
            std::vector<std::byte> w;
            if (q8) {
                w = random_q8(rows * n_batch, c.cols, rng);
            } else {
                auto f = random_f32(std::size_t{rows} * c.cols * n_batch,
                                    rng);
                w.resize(f.size() * 4);
                std::memcpy(w.data(), f.data(), w.size());
            }
            const auto x = random_f32(
                std::size_t{n_tok} * x_stride + n_batch * x_bstride, rng);
            const auto init = random_f32(
                std::size_t{n_tok} * out_stride + n_batch * out_bstride,
                rng);

            auto wb = upload(ctx, w);
            auto xb = upload(ctx, x);
            for (const bool acc : {false, true}) {
                auto ob = upload(ctx, init);
                const std::uint32_t tiles = (n_tok + 63) / 64;
                const std::uint32_t w_bstride = static_cast<std::uint32_t>(
                    q8 ? mat_bytes : mat_bytes / 4);
                ctx.begin_batch();
                const VulkanContext::Buffer bufs[] = {wb, xb, ob};
                const std::uint32_t push[] = {
                    rows,  c.cols, n_tok, acc ? 1u : 0u, 0,
                    0,     x_stride, 0,   out_stride,    fbits(scale),
                    w_bstride, x_bstride, out_bstride, tiles, 1, 0,
                    1,     0,      0,     0,         1,           0,
                    0,     0};
                ctx.dispatch(c.kernel, bufs, push, (rows + 63) / 64,
                             tiles * n_batch);
                ctx.end_batch();
                const auto got = download<float>(ctx, ob, init.size());
                ctx.destroy_buffer(ob);
                for (std::uint32_t b = 0; b < n_batch; ++b) {
                    locus::backend::Mat m{c.type, w.data() + b * mat_bytes,
                                          rows, c.cols};
                    for (std::uint32_t t = 0; t < n_tok; ++t) {
                        const float* xt =
                            x.data() + b * x_bstride + t * x_stride;
                        for (std::uint32_t r = 0; r < rows; ++r) {
                            const std::size_t at =
                                b * out_bstride + t * out_stride + r;
                            const float want =
                                (acc ? init[at] : 0.0f) +
                                scale * cpu_dot(m, r, xt);
                            INFO("acc " << acc << " batch " << b
                                        << " token " << t << " row "
                                        << r);
                            REQUIRE(got[at] ==
                                    Catch::Approx(want).margin(1e-4));
                        }
                    }
                }
                // Rows between token outputs are left untouched.
                REQUIRE(got[rows] == init[rows]);
            }
            ctx.destroy_buffer(wb);
            ctx.destroy_buffer(xb);
        }
    }
}

TEST_CASE("vulkan prefill MoE: route, group, expert matmul, combine",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(43);

    // 37 tokens, top-3 of 8 experts; a gate-like projection (E -> ff,
    // X row = token) then a down-like one (ff -> E, X row = pair),
    // combined with the route weights into a residual.
    const std::uint32_t n_tok = 37, k = 3, n_exp = 8;
    const std::uint32_t E = 64, ff = 96;
    const auto wg = random_q8(ff * n_exp, E, rng);
    const auto wd = random_q8(E * n_exp, ff, rng);
    const std::size_t g_bytes = std::size_t{ff} * (E / 32) * 34;
    const std::size_t d_bytes = std::size_t{E} * (ff / 32) * 34;
    const auto x = random_f32(std::size_t{n_tok} * E, rng);
    const auto resid = random_f32(std::size_t{n_tok} * E, rng);

    // Per token: k distinct ids, then k weights (moe_route's layout).
    std::vector<std::uint32_t> route(std::size_t{n_tok} * 2 * k);
    std::vector<float> wts(std::size_t{n_tok} * k);
    for (std::uint32_t t = 0; t < n_tok; ++t) {
        std::vector<std::uint32_t> ids(n_exp);
        for (std::uint32_t e = 0; e < n_exp; ++e) {
            ids[e] = e;
        }
        std::shuffle(ids.begin(), ids.end(), rng);
        for (std::uint32_t j = 0; j < k; ++j) {
            const float wj =
                std::uniform_real_distribution<float>(0.1f, 1.0f)(rng);
            route[t * 2 * k + j] = ids[j];
            route[t * 2 * k + k + j] = fbits(wj);
            wts[t * k + j] = wj;
        }
    }

    auto wgb = upload(ctx, wg);
    auto wdb = upload(ctx, wd);
    auto xb = upload(ctx, x);
    auto rb = upload(ctx, route);
    auto idb = ctx.create_buffer((n_exp + std::size_t{n_exp} * n_tok) * 4);
    auto gb = ctx.create_buffer(std::size_t{n_tok} * k * ff * 4);
    auto db = ctx.create_buffer(std::size_t{n_tok} * k * E * 4);
    auto residb = upload(ctx, resid);
    const std::uint32_t tiles = (n_tok + 15) / 16;

    ctx.begin_batch();
    {
        const VulkanContext::Buffer bufs[] = {rb, idb};
        const std::uint32_t push[] = {n_tok, k, n_exp, n_tok};
        ctx.dispatch(Kernel::kMoeGroup, bufs, push, 1);
    }
    {
        const VulkanContext::Buffer bufs[] = {wgb, xb, gb, idb};
        const std::uint32_t push[] = {
            ff, E, n_tok, 0, 0, 0, E, 0, ff, fbits(1.0f),
            static_cast<std::uint32_t>(g_bytes), 0, 0, tiles, k, n_tok,
            1, 0, 0, 0, 1, 0, 0, 0};
        ctx.dispatch(Kernel::kMatmulQ8_0Id, bufs, push, (ff + 63) / 64,
                     tiles * n_exp);
    }
    {
        const VulkanContext::Buffer bufs[] = {wdb, gb, db, idb};
        const std::uint32_t push[] = {
            E, ff, n_tok, 0, 0, 0, ff, 0, E, fbits(1.0f),
            static_cast<std::uint32_t>(d_bytes), 0, 0, tiles, 1, n_tok,
            1, 0, 0, 0, 1, 0, 0, 0};
        ctx.dispatch(Kernel::kMatmulQ8_0Id, bufs, push, (E + 63) / 64,
                     tiles * n_exp);
    }
    {
        const VulkanContext::Buffer bufs[] = {db, rb, residb};
        const std::uint32_t push[] = {E, k};
        ctx.dispatch(Kernel::kMoeCombine, bufs, push, (E + 255) / 256,
                     n_tok);
    }
    ctx.end_batch();

    const auto counts = download<std::uint32_t>(ctx, idb, n_exp);
    std::uint32_t total = 0;
    for (auto c : counts) {
        total += c;
    }
    REQUIRE(total == n_tok * k);

    const auto g = download<float>(ctx, gb, std::size_t{n_tok} * k * ff);
    const auto got = download<float>(ctx, residb, resid.size());
    for (std::uint32_t t = 0; t < n_tok; ++t) {
        std::vector<double> want(resid.begin() + t * E,
                                 resid.begin() + (t + 1) * E);
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint32_t e = route[t * 2 * k + j];
            locus::backend::Mat mg{locus::gguf::TensorType::kQ8_0,
                                   wg.data() + e * g_bytes, ff, E};
            locus::backend::Mat md{locus::gguf::TensorType::kQ8_0,
                                   wd.data() + e * d_bytes, E, ff};
            const std::size_t p = std::size_t{t} * k + j;
            std::vector<float> gp(ff);
            for (std::uint32_t r = 0; r < ff; ++r) {
                gp[r] = cpu_dot(mg, r, x.data() + t * E);
                INFO("token " << t << " slot " << j << " row " << r);
                REQUIRE(g[p * ff + r] == Catch::Approx(gp[r]).margin(1e-4));
            }
            // The down projection reads the GPU's gate rows, so the
            // check isolates the down matmul and the combine.
            for (std::uint32_t r = 0; r < E; ++r) {
                want[r] += wts[p] * cpu_dot(md, r, g.data() + p * ff);
            }
        }
        for (std::uint32_t r = 0; r < E; ++r) {
            REQUIRE(got[t * E + r] == Catch::Approx(want[r]).margin(1e-4));
        }
    }
    for (auto b : {wgb, wdb, xb, rb, idb, gb, db, residb}) {
        ctx.destroy_buffer(b);
    }
}

namespace {

std::string test_model(const char* name) {
    return std::string(LOCUS_SOURCE_DIR) + "/tests/models/" + name;
}

/**
 * Prefills `prompt` on the Vulkan backend either in chunks of
 * `chunk` tokens through forward_batch, or one forward() per token
 * (chunk 0), then greedily decodes `n_gen` tokens. Returns the last
 * prompt position's logits followed by the generated ids.
 */
std::pair<std::vector<float>, std::vector<locus::tok::TokenId>> prefill_run(
    const locus::model::TransformerModel& model,
    const std::vector<locus::tok::TokenId>& prompt, std::uint32_t chunk,
    std::uint32_t n_gen) {
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    std::size_t at = 0;
    while (at < prompt.size()) {
        const std::size_t nb =
            chunk == 0 ? 1 : std::min<std::size_t>(chunk, prompt.size() - at);
        REQUIRE(cache.ensure_capacity(seq, static_cast<std::uint32_t>(nb)));
        if (chunk == 0) {
            model.forward(prompt[at], cache, seq, ws, logits);
        } else {
            model.forward_batch({prompt.data() + at, nb}, cache, seq, ws,
                                logits);
        }
        at += nb;
    }
    const std::vector<float> last = logits;
    std::vector<locus::tok::TokenId> gen;
    for (std::uint32_t i = 0; i < n_gen; ++i) {
        const auto next = static_cast<locus::tok::TokenId>(argmax(logits));
        gen.push_back(next);
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(next, cache, seq, ws, logits);
    }
    cache.release(seq);
    return {last, gen};
}

}  // namespace

TEST_CASE("vulkan batched prefill tracks per-token prefill",
          "[e2e][vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    // llama-1b exercises the dense GPU path; DeepSeek-Coder-V2-Lite
    // Q8_0 (optional, 16GB) the MLA + GPU-routed MoE path.
    const char* const models[] = {"llama-3.2-1b-q8_0.gguf",
                                  "deepseek-coder-v2-lite-q8_0.gguf"};
    std::size_t ran = 0;
    for (const char* name : models) {
        const std::string path = test_model(name);
        if (!std::filesystem::exists(path)) {
            continue;
        }
        DYNAMIC_SECTION(name) {
            ++ran;
            auto g = locus::gguf::GgufFile::open(path);
            auto model = locus::model::TransformerModel::load(g);
            model.use_backend(*locus::backend::find_backend("vulkan"));
            REQUIRE(model.supports_batch_prefill());

            // 150 pseudo-random ids: three chunks of 64 (the last
            // one ragged), so later chunks attend over earlier ones.
            std::mt19937 rng(47);
            std::uniform_int_distribution<int> id(
                100, static_cast<int>(model.hparams().n_vocab) - 100);
            std::vector<locus::tok::TokenId> prompt(150);
            for (auto& t : prompt) {
                t = id(rng);
            }
            const auto [want_logits, want_gen] =
                prefill_run(model, prompt, 0, 8);
            const auto [got_logits, got_gen] =
                prefill_run(model, prompt, 64, 8);
            REQUIRE(cosine(got_logits, want_logits) > 0.9999);
            REQUIRE(argmax(got_logits) == argmax(want_logits));
            REQUIRE(got_gen == want_gen);

            // all_logits (speculative verify): one row per position.
            auto cache = model.make_cache();
            auto ws = model.make_workspace();
            locus::kv::PagedKvCache::Seq seq;
            const std::uint32_t V = model.hparams().n_vocab;
            const std::uint32_t n = 20;
            std::vector<float> rows(std::size_t{n} * V);
            REQUIRE(cache.ensure_capacity(seq, n));
            model.forward_batch({prompt.data(), n}, cache, seq, ws, rows,
                                true);
            REQUIRE(seq.n_tokens == n);
            auto ref_cache = model.make_cache();
            locus::kv::PagedKvCache::Seq ref_seq;
            std::vector<float> one(V);
            for (std::uint32_t t = 0; t < n; ++t) {
                REQUIRE(ref_cache.ensure_capacity(ref_seq, 1));
                model.forward(prompt[t], ref_cache, ref_seq, ws, one);
                const std::vector<float> row(rows.begin() + t * V,
                                             rows.begin() + (t + 1) * V);
                INFO("position " << t);
                REQUIRE(cosine(row, one) > 0.9999);
            }
            cache.release(seq);
            ref_cache.release(ref_seq);
        }
    }
    if (ran == 0) {
        SKIP("no prefill test model present (llama-3.2-1b-q8_0.gguf)");
    }
}
