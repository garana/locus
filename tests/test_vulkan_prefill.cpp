#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
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

/** GGUF Q8_0 rows in the split GPU layout the shaders read. */
std::vector<std::byte> gpu_q8(const std::vector<std::byte>& w,
                              std::uint32_t cols) {
    std::vector<std::byte> out(w.size());
    locus::backend::vk::q8_0_split_rows(w.data(), out.data(),
                                        w.size() / ((cols / 32) * 34), cols);
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

/**
 * Random K-quant bytes for rows x cols: random quants and scales,
 * the f16 d (at d_off in each block, dmin after it for Q4_K) pinned
 * to a sane magnitude.
 */
std::vector<std::byte> random_kq(locus::gguf::TensorType type,
                                 std::uint32_t rows, std::uint32_t cols,
                                 std::mt19937& rng) {
    // Q4_K / Q6_K super-blocks of 256, or Q5_0 blocks of 32 (d first).
    const bool q4 = type == locus::gguf::TensorType::kQ4_K;
    const bool q5 = type == locus::gguf::TensorType::kQ5_0;
    const std::size_t block = q5 ? 22 : q4 ? 144 : 210;
    const std::size_t d_off = q5 || q4 ? 0 : 208;
    const std::uint32_t per = q5 ? 32 : 256;
    std::uniform_int_distribution<int> bd(0, 255);
    std::vector<std::byte> w(std::size_t{rows} * (cols / per) * block);
    for (auto& b : w) {
        b = static_cast<std::byte>(bd(rng));
    }
    for (std::size_t at = d_off; at < w.size(); at += block) {
        const std::uint16_t d = locus::backend::f32_to_f16(0.01f);
        std::memcpy(&w[at], &d, 2);
        if (q4) {
            const std::uint16_t dmin = locus::backend::f32_to_f16(0.005f);
            std::memcpy(&w[at + 2], &dmin, 2);
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
        float margin;  // f16 staging rounds both operands
    };
    const Case cases[] = {
        {"q8_0", locus::gguf::TensorType::kQ8_0, Kernel::kMatmulQ8_0, 96,
         1e-4f},
        {"f32", locus::gguf::TensorType::kF32, Kernel::kMatmulF32, 80,
         1e-4f},
        {"q8_0 f16 staging", locus::gguf::TensorType::kQ8_0,
         Kernel::kMatmulQ8_0H, 96, 2e-2f},
        {"f32 f16 staging", locus::gguf::TensorType::kF32,
         Kernel::kMatmulF32H, 80, 2e-2f},
        {"q4_k", locus::gguf::TensorType::kQ4_K, Kernel::kMatmulQ4_K, 512,
         1e-3f},
        {"q6_k", locus::gguf::TensorType::kQ6_K, Kernel::kMatmulQ6_K, 512,
         1e-3f},
        // 22-byte blocks: rows and halves start off word alignment.
        {"q5_0", locus::gguf::TensorType::kQ5_0, Kernel::kMatmulQ5_0, 96,
         1e-3f},
    };
    const std::uint32_t rows = 100, n_tok = 70, n_batch = 3;
    const std::uint32_t x_stride = 300, out_stride = 333;
    const std::uint32_t x_bstride = 7, out_bstride = 101;
    const float scale = 0.75f;
    for (const Case& c : cases) {
        if (VulkanContext::kernel_needs_f16(c.kernel) && !ctx.has_f16()) {
            continue;  // f16 kernels exist only with shaderFloat16
        }
        DYNAMIC_SECTION(c.name) {
            const bool q8 = c.type == locus::gguf::TensorType::kQ8_0;
            const bool kq = c.type == locus::gguf::TensorType::kQ4_K ||
                            c.type == locus::gguf::TensorType::kQ6_K ||
                            c.type == locus::gguf::TensorType::kQ5_0;
            std::vector<std::byte> w;
            if (q8) {
                w = random_q8(rows * n_batch, c.cols, rng);
            } else if (kq) {
                w = random_kq(c.type, rows * n_batch, c.cols, rng);
            } else {
                auto f = random_f32(std::size_t{rows} * c.cols * n_batch,
                                    rng);
                w.resize(f.size() * 4);
                std::memcpy(w.data(), f.data(), w.size());
            }
            // Q6_K's odd 210-byte super-blocks start off word
            // alignment, exercising the shader's funnel reads.
            const std::size_t mat_bytes = w.size() / n_batch;
            // Rows overlap when cols > x_stride (the K-quants); the
            // last one still ends inside x.
            const auto x = random_f32(
                std::size_t{n_tok} * x_stride + n_batch * x_bstride +
                    c.cols,
                rng);
            const auto init = random_f32(
                std::size_t{n_tok} * out_stride + n_batch * out_bstride,
                rng);

            auto wb = upload(ctx, q8 ? gpu_q8(w, c.cols) : w);
            auto xb = upload(ctx, x);
            for (const bool acc : {false, true}) {
                auto ob = upload(ctx, init);
                const std::uint32_t tiles = (n_tok + 63) / 64;
                const std::uint32_t w_bstride = static_cast<std::uint32_t>(
                    q8 || kq ? mat_bytes : mat_bytes / 4);
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
                            REQUIRE(got[at] == Catch::Approx(want)
                                                   .margin(c.margin));
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

    auto wgb = upload(ctx, gpu_q8(wg, E));
    auto wdb = upload(ctx, gpu_q8(wd, ff));
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
 * (chunk 0), then greedily decodes `n_gen` tokens, with a `kv_type`
 * cache. Returns the last prompt position's logits followed by the
 * generated ids.
 */
std::pair<std::vector<float>, std::vector<locus::tok::TokenId>> prefill_run(
    const locus::model::TransformerModel& model,
    const std::vector<locus::tok::TokenId>& prompt, std::uint32_t chunk,
    std::uint32_t n_gen,
    locus::kv::KvType kv_type = locus::kv::KvType::kF32) {
    auto cache = model.make_cache(0, kv_type);
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
    // Q8_0 (optional, 16GB) the MLA + GPU-routed MoE path;
    // Qwen1.5-MoE Q4_K_M (optional, 9GB) q / k / v biases, the gated
    // shared expert and the Q4_K expert matmuls.
    const char* const models[] = {"llama-3.2-1b-q8_0.gguf",
                                  "deepseek-coder-v2-lite-q8_0.gguf",
                                  "Qwen1.5-MoE-A2.7B.Q4_K_M.gguf"};
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
            // MoE models prefill in bigger chunks (fuller expert tiles).
            REQUIRE(model.prefill_chunk() ==
                    (model.hparams().n_expert > 0 ? 1024u : 512u));

            // 150 pseudo-random ids: three chunks of 64 (the last
            // one ragged), so later chunks attend over earlier ones.
            std::mt19937 rng(47);
            std::uniform_int_distribution<int> id(
                100, static_cast<int>(model.hparams().n_vocab) - 100);
            std::vector<locus::tok::TokenId> prompt(150);
            for (auto& t : prompt) {
                t = id(rng);
            }
            {
                // A short prompt on freshly reset GPU state: one
                // tiny chunk splits its matmuls' K (partials in a
                // scratch buffer that grows mid-batch -- the old
                // buffer must outlive the dispatches recorded on it).
                locus::backend::vulkan_reset_state();
                const std::vector<locus::tok::TokenId> short_p(
                    prompt.begin(), prompt.begin() + 8);
                const auto [one_l, one_gen] =
                    prefill_run(model, short_p, 0, 8);
                locus::backend::vulkan_reset_state();
                const auto [batch_l, batch_gen] =
                    prefill_run(model, short_p, 8, 8);
                REQUIRE(cosine(batch_l, one_l) > 0.9999);
                REQUIRE(batch_gen == one_gen);
            }
            const auto [want_logits, want_gen] =
                prefill_run(model, prompt, 0, 8);
            const auto [got_logits, got_gen] =
                prefill_run(model, prompt, 64, 8);
            REQUIRE(cosine(got_logits, want_logits) > 0.9999);
            REQUIRE(argmax(got_logits) == argmax(want_logits));
            REQUIRE(got_gen == want_gen);

            // all_logits (speculative verify): one row per position.
            // 20 rows take the matmul path, 5 the short-batch
            // matvecs (<= 8 rows; MoE pair matvecs).
            for (const std::uint32_t n : {20u, 5u}) {
                auto cache = model.make_cache();
                auto ws = model.make_workspace();
                locus::kv::PagedKvCache::Seq seq;
                const std::uint32_t V = model.hparams().n_vocab;
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
                    const std::vector<float> row(
                        rows.begin() + t * V, rows.begin() + (t + 1) * V);
                    INFO("batch " << n << " position " << t);
                    REQUIRE(cosine(row, one) > 0.9999);
                    REQUIRE(argmax(row) == argmax(one));
                }
                cache.release(seq);
                ref_cache.release(ref_seq);
            }
        }
    }
    if (ran == 0) {
        SKIP("no prefill test model present (llama-3.2-1b-q8_0.gguf)");
    }
}

TEST_CASE("vulkan f16 prefill staging tracks the f32 prefill",
          "[e2e][vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    if (!VulkanContext().has_f16()) {
        SKIP("device has no shaderFloat16: --prefill-f16 is a no-op");
    }
    // Opt-in (--prefill-f16): batched prefill with f16-staged matmul
    // operands stays close to the exact f32 path (not bit-exact).
    struct Restore {
        ~Restore() { locus::backend::vulkan_set_prefill_f16(false); }
    } restore;
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
            std::mt19937 rng(67);
            std::uniform_int_distribution<int> id(
                100, static_cast<int>(model.hparams().n_vocab) - 100);
            std::vector<locus::tok::TokenId> prompt(150);
            for (auto& t : prompt) {
                t = id(rng);
            }
            locus::backend::vulkan_set_prefill_f16(false);
            const auto [f32_l, f32_gen] = prefill_run(model, prompt, 64, 0);
            locus::backend::vulkan_set_prefill_f16(true);
            const auto [f16_l, f16_gen] = prefill_run(model, prompt, 64, 0);
            locus::backend::vulkan_set_prefill_f16(false);
            REQUIRE(cosine(f16_l, f32_l) > 0.999);
            REQUIRE(f16_l != f32_l);  // the staging really ran
        }
    }
    if (ran == 0) {
        SKIP("no prefill test model present (llama-3.2-1b-q8_0.gguf)");
    }
}

TEST_CASE("vulkan f16 KV cache tracks the f32 cache", "[e2e][vulkan][kv]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    // GQA (llama-1b) and MLA (DeepSeek-Coder-V2-Lite, optional) both
    // keep an F16 pool on the GPU: decode stores and reads halves,
    // batched prefill scatters and reads them.
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
            using locus::kv::KvType;
            {
                // The F16 pool really is half the F32 one.
                const auto c32 = model.make_cache(0, KvType::kF32);
                const auto c16 = model.make_cache(0, KvType::kF16);
                REQUIRE(c16.q_row_bytes() * 2 ==
                        c32.geometry().kv_dim * sizeof(float));
            }
            std::mt19937 rng(59);
            std::uniform_int_distribution<int> id(
                100, static_cast<int>(model.hparams().n_vocab) - 100);
            std::vector<locus::tok::TokenId> prompt(150);
            for (auto& t : prompt) {
                t = id(rng);
            }
            // Per-token decode: F16 against F32 (precision only).
            const auto [f32_l, f32_gen] =
                prefill_run(model, prompt, 0, 4, KvType::kF32);
            const auto [f16_l, f16_gen] =
                prefill_run(model, prompt, 0, 4, KvType::kF16);
            REQUIRE(cosine(f16_l, f32_l) > 0.999);
            // Batched prefill into an F16 pool against per-token F16.
            const auto [b16_l, b16_gen] =
                prefill_run(model, prompt, 64, 4, KvType::kF16);
            REQUIRE(cosine(b16_l, f16_l) > 0.9999);
            REQUIRE(argmax(b16_l) == argmax(f16_l));
            REQUIRE(b16_gen == f16_gen);
        }
    }
    if (ran == 0) {
        SKIP("no prefill test model present (llama-3.2-1b-q8_0.gguf)");
    }
}

// Hidden micro-benchmark for matmul.comp tuning; run explicitly:
//   ./build/tests/locus_tests "bench: vulkan matmul"
TEST_CASE("bench: vulkan matmul throughput", "[.][bench][vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(53);
    const std::uint32_t n_tok = 512, reps = 20;
    struct Shape {
        std::uint32_t rows, cols;
    };
    for (const Shape sh : {Shape{2048, 2048}, Shape{8192, 2048},
                           Shape{2048, 8192}}) {
        const auto w = random_q8(sh.rows, sh.cols, rng);
        const auto x = random_f32(std::size_t{n_tok} * sh.cols, rng);
        auto wb = upload(ctx, gpu_q8(w, sh.cols));
        auto xb = upload(ctx, x);
        auto ob = ctx.create_buffer(std::size_t{n_tok} * sh.rows * 4);
        const std::uint32_t TR = VulkanContext::kMatmulTileRows;
        const std::uint32_t TT = VulkanContext::kMatmulTileToks;
        const std::uint32_t tiles = (n_tok + TT - 1) / TT;
        const std::uint32_t push[] = {
            sh.rows, sh.cols, n_tok, 0, 0, 0, sh.cols, 0, sh.rows,
            fbits(1.0f), 0, 0, 0, tiles, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0};
        const VulkanContext::Buffer bufs[] = {wb, xb, ob};
        auto run = [&](std::uint32_t n) {
            ctx.begin_batch();
            for (std::uint32_t i = 0; i < n; ++i) {
                ctx.dispatch(Kernel::kMatmulQ8_0, bufs, push,
                             (sh.rows + TR - 1) / TR, tiles);
            }
            ctx.end_batch();
        };
        run(2);  // warm-up
        const auto t0 = std::chrono::steady_clock::now();
        run(reps);
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
        const double flop =
            2.0 * sh.rows * sh.cols * n_tok * static_cast<double>(reps);
        WARN("q8_0 " << sh.rows << "x" << sh.cols << " x " << n_tok
                     << " tok: " << s / reps * 1e3 << " ms, "
                     << flop / s / 1e12 << " TFLOPS");
        for (auto b : {wb, xb, ob}) {
            ctx.destroy_buffer(b);
        }
    }
}

// Hidden micro-benchmark: decode matvec bandwidth at real shapes.
//   ./build/tests/locus_tests "bench: vulkan matvec"
TEST_CASE("bench: vulkan matvec bandwidth", "[.][bench][vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(59);
    const std::uint32_t reps = 2000;
    struct Shape {
        std::uint32_t rows, cols;
    };
    for (const Shape sh : {Shape{8, 32}, Shape{512, 2048},
                           Shape{576, 2048}, Shape{2048, 2048},
                           Shape{3072, 2048}, Shape{2816, 2048},
                           Shape{2048, 2816}, Shape{10944, 2048},
                           Shape{2048, 8192}, Shape{8192, 2048},
                           Shape{128256, 2048}}) {
        const auto w = random_q8(sh.rows, sh.cols, rng);
        const auto x = random_f32(sh.cols, rng);
        auto wb = upload(ctx, gpu_q8(w, sh.cols));
        auto xb = upload(ctx, x);
        auto ob = ctx.create_buffer(std::size_t{sh.rows} * 4);
        const std::uint32_t push[] = {sh.rows, sh.cols, 0, 0, 0, 0,
                                      fbits(1.0f)};
        const VulkanContext::Buffer bufs[] = {wb, xb, ob};
        const std::uint32_t groups =
            VulkanContext::matvec_groups(Kernel::kMatvecQ8_0, sh.rows);
        auto run = [&](std::uint32_t n) {
            ctx.begin_batch();
            for (std::uint32_t i = 0; i < n; ++i) {
                ctx.dispatch(Kernel::kMatvecQ8_0, bufs, push, groups);
            }
            ctx.end_batch();
        };
        run(4);
        const auto t0 = std::chrono::steady_clock::now();
        run(reps);
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
        WARN("q8_0 matvec " << sh.rows << "x" << sh.cols << ": "
                            << s / reps * 1e6 << " us, "
                            << w.size() * double(reps) / s / 1e9 << " GB/s");
        for (auto b : {wb, xb, ob}) {
            ctx.destroy_buffer(b);
        }
    }
}

TEST_CASE("bench: vulkan k-quant matvec bandwidth", "[.][bench][vulkan]") {
    // Qwen3-Coder-30B-A3B's decode shapes: an expert's gate / up and
    // down, q, o, and the Q6_K output head.
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(61);
    const std::uint32_t reps = 1000;
    struct Case {
        locus::gguf::TensorType type;
        Kernel kernel;
        const char* name;
        std::uint32_t rows, cols;
    };
    const auto q4 = locus::gguf::TensorType::kQ4_K;
    const auto q6 = locus::gguf::TensorType::kQ6_K;
    for (const Case c :
         {Case{q4, Kernel::kMatvecQ4_K, "q4_k", 768, 2048},
          Case{q6, Kernel::kMatvecQ6_K, "q6_k", 2048, 768},
          Case{q4, Kernel::kMatvecQ4_K, "q4_k", 4096, 2048},
          Case{q4, Kernel::kMatvecQ4_K, "q4_k", 2048, 4096},
          Case{q4, Kernel::kMatvecQ4_K, "q4_k", 16384, 4096},
          Case{q6, Kernel::kMatvecQ6_K, "q6_k", 151936, 2048}}) {
        const auto w = random_kq(c.type, c.rows, c.cols, rng);
        const auto x = random_f32(c.cols, rng);
        auto wb = upload(ctx, w);
        auto xb = upload(ctx, x);
        auto ob = ctx.create_buffer(std::size_t{c.rows} * 4);
        const std::uint32_t push[] = {c.rows, c.cols, 0, 0, 0, 0,
                                      fbits(1.0f)};
        const VulkanContext::Buffer bufs[] = {wb, xb, ob};
        const std::uint32_t groups =
            VulkanContext::matvec_groups(c.kernel, c.rows);
        auto run = [&](std::uint32_t n) {
            ctx.begin_batch();
            for (std::uint32_t i = 0; i < n; ++i) {
                ctx.dispatch(c.kernel, bufs, push, groups);
            }
            ctx.end_batch();
        };
        run(4);
        const auto t0 = std::chrono::steady_clock::now();
        run(reps);
        const double sec = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
        WARN(c.name << " matvec " << c.rows << "x" << c.cols << ": "
                                   << sec / reps * 1e6 << " us, "
                                   << w.size() * double(reps) / sec / 1e9
                                   << " GB/s");
        for (auto b : {wb, xb, ob}) {
            ctx.destroy_buffer(b);
        }
    }
}

TEST_CASE("bench: vulkan f32 matvec bandwidth", "[.][bench][vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(61);
    const std::uint32_t reps = 500;
    for (const auto& [rows, cols] :
         {std::pair{2048u, 2048u}, std::pair{8192u, 2048u},
          std::pair{32768u, 2048u}}) {
        const auto w = random_f32(std::size_t{rows} * cols, rng);
        const auto x = random_f32(cols, rng);
        auto wb = upload(ctx, w);
        auto xb = upload(ctx, x);
        auto ob = ctx.create_buffer(std::size_t{rows} * 4);
        const std::uint32_t push[] = {rows, cols, 0, 0, 0, 0, fbits(1.0f)};
        const VulkanContext::Buffer bufs[] = {wb, xb, ob};
        const std::uint32_t groups =
            VulkanContext::matvec_groups(Kernel::kMatvecF32, rows);
        auto run = [&](std::uint32_t n) {
            ctx.begin_batch();
            for (std::uint32_t i = 0; i < n; ++i) {
                ctx.dispatch(Kernel::kMatvecF32, bufs, push, groups);
            }
            ctx.end_batch();
        };
        run(4);
        const auto t0 = std::chrono::steady_clock::now();
        run(reps);
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
        WARN("f32 matvec " << rows << "x" << cols << ": "
                           << s / reps * 1e6 << " us, "
                           << w.size() * 4.0 * reps / s / 1e9 << " GB/s");
        for (auto b : {wb, xb, ob}) {
            ctx.destroy_buffer(b);
        }
    }
}

TEST_CASE("only the f16-staging matmuls need shaderFloat16", "[vulkan]") {
    // Devices without shaderFloat16 skip exactly these pipelines, so
    // the rest of the backend still comes up.
    for (int k = 0; k < static_cast<int>(Kernel::kCount_); ++k) {
        const auto kk = static_cast<Kernel>(k);
        const bool h = kk == Kernel::kMatmulQ8_0H ||
                       kk == Kernel::kMatmulF32H ||
                       kk == Kernel::kMatmulQ8_0IdH ||
                       kk == Kernel::kMatmulPagedF32H ||
                       kk == Kernel::kMatmulPagedTF32H;
        INFO("kernel " << k);
        REQUIRE(VulkanContext::kernel_needs_f16(kk) == h);
    }
}

TEST_CASE("vulkan comes up without shaderFloat16, f16 kernels skipped",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    // LOCUS_VK_NO_F16 forces the path a device without shaderFloat16
    // takes: the context still builds, only the f16 kernels are absent.
    setenv("LOCUS_VK_NO_F16", "1", 1);
    struct Unset {
        ~Unset() { unsetenv("LOCUS_VK_NO_F16"); }
    } unset;
    VulkanContext ctx;
    REQUIRE_FALSE(ctx.has_f16());
    // A non-f16 kernel still runs: 4 rows of an f32 matvec.
    const std::vector<float> w = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> x = {1, -1};
    auto wb = upload(ctx, w);
    auto xb = upload(ctx, x);
    auto ob = ctx.create_buffer(4 * 4);
    ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {wb, xb, ob};
    const std::uint32_t push[] = {4, 2, 0, 0, 0, 0, fbits(1.0f)};
    ctx.dispatch(Kernel::kMatvecF32, bufs, push,
                 VulkanContext::matvec_groups(Kernel::kMatvecF32, 4));
    ctx.end_batch();
    const auto got = download<float>(ctx, ob, 4);
    REQUIRE(got == std::vector<float>{-1, -1, -1, -1});
    for (auto b : {wb, xb, ob}) {
        ctx.destroy_buffer(b);
    }
}
