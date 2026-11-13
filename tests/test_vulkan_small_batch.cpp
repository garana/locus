// Short-batch kernels for the Vulkan batched forward (speculative
// verify): matvec_q8_0_n / matvec_f32_n over <= 8 token rows, and the
// routed-MoE (token, expert) pair matvecs matvec_q8_0_glu_p /
// matvec_q8_0_p; matvec_q4_k_n / matvec_q6_k_n (#436).

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/backend/vulkan/q8_layout.hpp"

using locus::backend::vk::Kernel;
using locus::backend::vk::VulkanContext;

namespace {

std::uint32_t fbits(float v) { return std::bit_cast<std::uint32_t>(v); }

template <typename T>
VulkanContext::Buffer upload(VulkanContext& ctx, const std::vector<T>& v) {
    auto b = ctx.create_buffer(std::max<std::size_t>(v.size() * sizeof(T),
                                                     4));
    ctx.write_buffer(b, std::as_bytes(std::span<const T>(v)));
    return b;
}

std::vector<float> download(VulkanContext& ctx, VulkanContext::Buffer b,
                            std::size_t n) {
    std::vector<float> out(n);
    ctx.read_buffer(b, std::as_writable_bytes(std::span<float>(out)));
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

/** Uploads Q8_0 rows in the split GPU layout. */
VulkanContext::Buffer upload_q8(VulkanContext& ctx,
                                const std::vector<std::byte>& w,
                                std::uint32_t rows, std::uint32_t cols) {
    std::vector<std::byte> g((w.size() + 3) & ~std::size_t{3});
    locus::backend::vk::q8_0_split_rows(w.data(), g.data(), rows, cols);
    auto b = ctx.create_buffer(g.size());
    ctx.write_buffer(b, g);
    return b;
}

/** CPU W x for rows [r0, r0 + rows) of a Q8_0 or F32 matrix. */
std::vector<float> cpu_mv(locus::gguf::TensorType type, const void* w,
                          std::uint32_t r0, std::uint32_t rows,
                          std::uint32_t cols, std::span<const float> x) {
    using locus::gguf::TensorType;
    const std::size_t row_bytes =
        type == TensorType::kQ8_0   ? (cols / 32) * 34
        : type == TensorType::kQ4_K ? (cols / 256) * 144
        : type == TensorType::kQ6_K ? (cols / 256) * 210
                                    : std::size_t{cols} * 4;
    locus::backend::Mat m{type,
                          static_cast<const std::byte*>(w) + r0 * row_bytes,
                          rows, cols};
    std::vector<float> out(rows);
    locus::backend::matvec(m, x, out);
    return out;
}

std::vector<float> random_floats(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& f : v) {
        f = dist(rng);
    }
    return v;
}

}  // namespace

TEST_CASE("vulkan short-batch matvecs match the CPU per token",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(71);
    using locus::gguf::TensorType;

    // 2 batches of 13 x 96 weights (odd nb: unaligned Q8_0 rows).
    const std::uint32_t rows = 13, cols = 96, n_batch = 2;
    for (const auto type : {TensorType::kQ8_0, TensorType::kF32}) {
        const bool q8 = type == TensorType::kQ8_0;
        const std::size_t mat_bytes =
            q8 ? rows * (cols / 32) * 34 : std::size_t{rows} * cols * 4;
        std::vector<std::byte> w;
        if (q8) {
            w = random_q8(rows * n_batch, cols, rng);
        } else {
            const auto f = random_floats(rows * n_batch * cols, rng);
            w.resize(f.size() * 4);
            std::memcpy(w.data(), f.data(), w.size());
        }
        auto wb = q8 ? upload_q8(ctx, w, rows * n_batch, cols)
                     : upload(ctx, w);
        for (const std::uint32_t n_tok : {1u, 5u, 8u}) {
            for (const std::uint32_t x_off : {0u, 3u}) {
                DYNAMIC_SECTION((q8 ? "q8_0" : "f32")
                                << " n_tok " << n_tok << " x_off "
                                << x_off) {
                    // Token rows x_stride apart, batches x_bstride apart;
                    // x_off 3 takes the unaligned (scalar) x path.
                    const std::uint32_t x_stride = cols + 4;
                    const std::uint32_t x_bstride = n_tok * x_stride + 8;
                    const auto x = random_floats(
                        x_off + n_batch * x_bstride, rng);
                    const std::uint32_t out_stride = rows + 2;
                    const std::uint32_t out_bstride = 8 * out_stride;
                    const std::uint32_t out_off = 1;
                    auto xb = upload(ctx, x);
                    auto ob = upload(
                        ctx, std::vector<float>(
                                 out_off + n_batch * out_bstride, 0.5f));
                    ctx.begin_batch();
                    const VulkanContext::Buffer bufs[] = {wb, xb, ob};
                    const std::uint32_t push[] = {
                        rows,
                        cols,
                        out_off,
                        1,  // accumulate onto the 0.5 fill
                        0,
                        x_off,
                        fbits(1.0f),
                        n_tok,
                        x_stride,
                        out_stride,
                        static_cast<std::uint32_t>(q8 ? mat_bytes
                                                      : rows * cols),
                        x_bstride,
                        out_bstride};
                    const Kernel k = q8 ? Kernel::kMatvecQ8_0N
                                        : Kernel::kMatvecF32N;
                    ctx.dispatch(k, bufs, push,
                                 VulkanContext::matvec_groups(k, rows),
                                 n_batch);
                    ctx.end_batch();
                    const auto got =
                        download(ctx, ob, out_off + n_batch * out_bstride);
                    for (std::uint32_t b = 0; b < n_batch; ++b) {
                        for (std::uint32_t t = 0; t < n_tok; ++t) {
                            const auto want = cpu_mv(
                                type, w.data(), b * rows, rows, cols,
                                std::span<const float>(x).subspan(
                                    x_off + b * x_bstride + t * x_stride,
                                    cols));
                            for (std::uint32_t r = 0; r < rows; ++r) {
                                REQUIRE(got[out_off + b * out_bstride +
                                            t * out_stride + r] ==
                                        Catch::Approx(0.5f + want[r])
                                            .margin(1e-4));
                            }
                        }
                        // Rows past n_tok keep the fill.
                        if (n_tok < 8) {
                            REQUIRE(got[out_off + b * out_bstride +
                                        n_tok * out_stride] == 0.5f);
                        }
                    }
                    ctx.destroy_buffer(xb);
                    ctx.destroy_buffer(ob);
                }
            }
        }
        ctx.destroy_buffer(wb);
    }
}

TEST_CASE("vulkan MoE pair matvecs match the CPU routed experts",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(73);

    // 5 experts (ff x cols gate / up, cols x ff down); 3 tokens, 2
    // experts each, routed per token in moe_route_b's layout.
    const std::uint32_t ff = 64, cols = 96, n_exp = 5, k = 2, n_tok = 3;
    const std::size_t up_bytes = ff * (cols / 32) * 34;
    const std::size_t down_bytes = cols * (ff / 32) * 34;
    const auto wg = random_q8(ff * n_exp, cols, rng);
    const auto wu = random_q8(ff * n_exp, cols, rng);
    const auto wd = random_q8(cols * n_exp, ff, rng);
    const auto x = random_floats(n_tok * cols, rng);
    const std::uint32_t ids[n_tok][k] = {{4, 1}, {0, 4}, {2, 3}};
    const float wts[n_tok][k] = {{0.6f, 0.4f}, {-0.3f, 1.2f},
                                 {0.25f, 0.75f}};
    std::vector<std::uint32_t> route;
    route.reserve(std::size_t{n_tok} * 2 * k);
    for (std::uint32_t t = 0; t < n_tok; ++t) {
        route.insert(route.end(), {ids[t][0], ids[t][1], fbits(wts[t][0]),
                                   fbits(wts[t][1])});
    }
    const auto resid = random_floats(n_tok * cols, rng);

    auto gb = upload_q8(ctx, wg, ff * n_exp, cols);
    auto ub = upload_q8(ctx, wu, ff * n_exp, cols);
    auto db = upload_q8(ctx, wd, cols * n_exp, ff);
    auto xb = upload(ctx, x);
    auto rb = upload(ctx, route);
    auto hb = ctx.create_buffer(n_tok * k * ff * 4);
    auto ob = upload(ctx, resid);
    ctx.begin_batch();
    {
        const VulkanContext::Buffer bufs[] = {gb, xb, hb, rb, ub};
        const std::uint32_t push[] = {
            ff, cols, 0, 0, 0, 0, fbits(1.0f),
            static_cast<std::uint32_t>(up_bytes), cols, ff, 0, 0, 0,
            2 * k, k};
        ctx.dispatch(Kernel::kMatvecQ8_0GluP, bufs, push,
                     VulkanContext::matvec_groups(Kernel::kMatvecQ8_0GluP,
                                                  ff),
                     n_tok * k);
    }
    {
        const VulkanContext::Buffer bufs[] = {db, hb, ob, rb};
        const std::uint32_t push[] = {
            cols, ff, 0, 1, 0, 0, fbits(1.0f),
            static_cast<std::uint32_t>(down_bytes), ff, cols, 0, 0, 0,
            2 * k, k};
        ctx.dispatch(Kernel::kMatvecQ8_0P, bufs, push,
                     VulkanContext::matvec_groups(Kernel::kMatvecQ8_0P,
                                                  cols),
                     n_tok);
    }
    ctx.end_batch();
    const auto h = download(ctx, hb, n_tok * k * ff);
    const auto out = download(ctx, ob, n_tok * cols);

    using locus::gguf::TensorType;
    for (std::uint32_t t = 0; t < n_tok; ++t) {
        const auto xt = std::span<const float>(x).subspan(t * cols, cols);
        std::vector<float> want = {resid.begin() + t * cols,
                                   resid.begin() + (t + 1) * cols};
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint32_t p = t * k + j;
            const auto a =
                cpu_mv(TensorType::kQ8_0, wg.data(), ids[t][j] * ff, ff,
                       cols, xt);
            const auto b =
                cpu_mv(TensorType::kQ8_0, wu.data(), ids[t][j] * ff, ff,
                       cols, xt);
            for (std::uint32_t r = 0; r < ff; ++r) {
                const float sv = a[r] / (1.0f + std::exp(-a[r])) * b[r];
                REQUIRE(h[p * ff + r] == Catch::Approx(sv).margin(1e-4));
            }
            // The down check reads the GPU's own intermediate.
            const auto d = cpu_mv(
                TensorType::kQ8_0, wd.data(), ids[t][j] * cols, cols, ff,
                std::span<const float>(h).subspan(p * ff, ff));
            for (std::uint32_t r = 0; r < cols; ++r) {
                want[r] += wts[t][j] * d[r];
            }
        }
        for (std::uint32_t r = 0; r < cols; ++r) {
            REQUIRE(out[t * cols + r] == Catch::Approx(want[r]).margin(1e-4));
        }
    }
    for (auto b : {gb, ub, db, xb, rb, hb, ob}) {
        ctx.destroy_buffer(b);
    }
}

// i#623: the K-quant pair kernels (Q4_K gate / up with silu_mul, then a
// Q4_K or Q6_K weighted down sum per token) against the CPU routed
// experts, as the Q8_0 case above.
TEST_CASE("vulkan K-quant MoE pair matvecs match the CPU routed experts",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(97);
    std::uniform_int_distribution<int> byte_d(0, 255);
    using locus::gguf::TensorType;
    // Random K-quant bytes with small super-block scales (f16 d / dmin
    // at bytes 0..3 for Q4_K, d at bytes 208..209 for Q6_K).
    auto random_k = [&](TensorType t, std::uint32_t rows,
                        std::uint32_t cols) {
        const std::size_t bb = t == TensorType::kQ4_K ? 144 : 210;
        std::vector<std::byte> w(std::size_t{rows} * (cols / 256) * bb);
        for (auto& b : w) b = static_cast<std::byte>(byte_d(rng));
        for (std::size_t at = 0; at < w.size(); at += bb) {
            if (t == TensorType::kQ4_K) {
                const std::uint16_t dd[] = {
                    locus::backend::f32_to_f16(0.01f),
                    locus::backend::f32_to_f16(0.005f)};
                std::memcpy(&w[at], dd, 4);
            } else {
                const std::uint16_t d = locus::backend::f32_to_f16(0.002f);
                std::memcpy(&w[at + 208], &d, 2);
            }
        }
        return w;
    };
    auto upload_bytes = [&](const std::vector<std::byte>& w) {
        std::vector<std::byte> g((w.size() + 3) & ~std::size_t{3});
        std::copy(w.begin(), w.end(), g.begin());
        auto b = ctx.create_buffer(g.size());
        ctx.write_buffer(b, g);
        return b;
    };
    const std::uint32_t ff = 256, cols = 512, n_exp = 5, k = 2, n_tok = 3;
    const std::uint32_t ids[n_tok][k] = {{4, 1}, {0, 4}, {2, 3}};
    const float wts[n_tok][k] = {{0.6f, 0.4f}, {-0.3f, 1.2f},
                                 {0.25f, 0.75f}};
    std::vector<std::uint32_t> route;
    for (std::uint32_t t = 0; t < n_tok; ++t) {
        route.insert(route.end(), {ids[t][0], ids[t][1], fbits(wts[t][0]),
                                   fbits(wts[t][1])});
    }
    for (const TensorType down_t : {TensorType::kQ4_K, TensorType::kQ6_K}) {
        DYNAMIC_SECTION("down " << (down_t == TensorType::kQ4_K ? "Q4_K"
                                                                : "Q6_K")) {
            const auto wg = random_k(TensorType::kQ4_K, ff * n_exp, cols);
            const auto wu = random_k(TensorType::kQ4_K, ff * n_exp, cols);
            const auto wd = random_k(down_t, cols * n_exp, ff);
            const std::size_t up_bytes = std::size_t{ff} * (cols / 256) * 144;
            const std::size_t down_bytes =
                std::size_t{cols} * (ff / 256) *
                (down_t == TensorType::kQ4_K ? 144 : 210);
            const auto x = random_floats(n_tok * cols, rng);
            const auto resid = random_floats(n_tok * cols, rng);
            auto gb = upload_bytes(wg);
            auto ub = upload_bytes(wu);
            auto db = upload_bytes(wd);
            auto xb = upload(ctx, x);
            auto rb = upload(ctx, route);
            auto hb = ctx.create_buffer(n_tok * k * ff * 4);
            auto ob = upload(ctx, resid);
            const Kernel down_k = down_t == TensorType::kQ4_K
                                      ? Kernel::kMatvecQ4_KP
                                      : Kernel::kMatvecQ6_KP;
            ctx.begin_batch();
            {
                const VulkanContext::Buffer bufs[] = {gb, xb, hb, rb, ub};
                const std::uint32_t push[] = {
                    ff, cols, 0, 0, 0, 0, fbits(1.0f),
                    static_cast<std::uint32_t>(up_bytes), cols, ff, 0, 0, 0,
                    2 * k, k};
                ctx.dispatch(Kernel::kMatvecQ4_KGluP, bufs, push,
                             VulkanContext::matvec_groups(
                                 Kernel::kMatvecQ4_KGluP, ff),
                             n_tok * k);
            }
            {
                const VulkanContext::Buffer bufs[] = {db, hb, ob, rb};
                const std::uint32_t push[] = {
                    cols, ff, 0, 1, 0, 0, fbits(1.0f),
                    static_cast<std::uint32_t>(down_bytes), ff, cols, 0, 0, 0,
                    2 * k, k};
                ctx.dispatch(down_k, bufs, push,
                             VulkanContext::matvec_groups(down_k, cols),
                             n_tok);
            }
            ctx.end_batch();
            const auto h = download(ctx, hb, n_tok * k * ff);
            const auto out = download(ctx, ob, n_tok * cols);
            for (std::uint32_t t = 0; t < n_tok; ++t) {
                const auto xt =
                    std::span<const float>(x).subspan(t * cols, cols);
                std::vector<float> want = {resid.begin() + t * cols,
                                           resid.begin() + (t + 1) * cols};
                for (std::uint32_t j = 0; j < k; ++j) {
                    const std::uint32_t p = t * k + j;
                    const auto a = cpu_mv(TensorType::kQ4_K, wg.data(),
                                          ids[t][j] * ff, ff, cols, xt);
                    const auto b = cpu_mv(TensorType::kQ4_K, wu.data(),
                                          ids[t][j] * ff, ff, cols, xt);
                    for (std::uint32_t r = 0; r < ff; ++r) {
                        const float sv =
                            a[r] / (1.0f + std::exp(-a[r])) * b[r];
                        REQUIRE(h[p * ff + r] ==
                                Catch::Approx(sv).epsilon(1e-3).margin(1e-3));
                    }
                    // The down check reads the GPU's own intermediate.
                    const auto d = cpu_mv(
                        down_t, wd.data(), ids[t][j] * cols, cols, ff,
                        std::span<const float>(h).subspan(p * ff, ff));
                    for (std::uint32_t r = 0; r < cols; ++r) {
                        want[r] += wts[t][j] * d[r];
                    }
                }
                for (std::uint32_t r = 0; r < cols; ++r) {
                    REQUIRE(out[t * cols + r] ==
                            Catch::Approx(want[r]).epsilon(1e-3).margin(1e-3));
                }
            }
            for (auto buf : {gb, ub, db, xb, rb, hb, ob}) {
                ctx.destroy_buffer(buf);
            }
        }
    }
}

// i#432: matvec_q8_0_n_glu (gate, up and silu_mul in one short-batch
// dispatch) against the CPU, and a token's result from matvec_q8_0_n
// and matvec_q8_0_n_glu being the same bits whatever the batch size
// (the property batch-invariant decode rests on).
TEST_CASE("vulkan short-batch SwiGLU matches the CPU and keeps a "
          "token's bits at every batch size",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(79);
    using locus::gguf::TensorType;
    const std::uint32_t rows = 13, cols = 96;
    const auto wg = random_q8(rows, cols, rng);
    const auto wu = random_q8(rows, cols, rng);
    auto gb = upload_q8(ctx, wg, rows, cols);
    auto ub = upload_q8(ctx, wu, rows, cols);
    const auto x = random_floats(std::size_t{8} * cols, rng);
    auto xb = upload(ctx, x);
    // The guarded builds and the guard-free ones (i#471), each pair on
    // its own: token 0's outputs at every batch size.
    for (const bool ng : {false, true}) {
        const Kernel kglu =
            ng ? Kernel::kMatvecQ8_0NGluNg : Kernel::kMatvecQ8_0NGlu;
        const Kernel kplain =
            ng ? Kernel::kMatvecQ8_0NNg : Kernel::kMatvecQ8_0N;
        std::vector<float> first_glu, first_plain;
        for (const std::uint32_t n_tok : {1u, 2u, 5u, 8u}) {
            INFO((ng ? "guard-free" : "guarded") << " n_tok " << n_tok);
            auto og = upload(ctx, std::vector<float>(8 * rows, 0.0f));
            auto op = upload(ctx, std::vector<float>(8 * rows, 0.0f));
            ctx.begin_batch();
            const std::uint32_t push[] = {rows,  cols, 0, 0, 0, 0, fbits(1.0f),
                                          n_tok, cols, rows, 0, 0, 0};
            const VulkanContext::Buffer gbufs[] = {gb, xb, og, ub};
            ctx.dispatch(kglu, gbufs, push,
                         VulkanContext::matvec_groups(kglu, rows));
            const VulkanContext::Buffer pbufs[] = {gb, xb, op};
            ctx.dispatch(kplain, pbufs, push,
                         VulkanContext::matvec_groups(kplain, rows));
            ctx.end_batch();
            const auto got = download(ctx, og, 8 * rows);
            const auto plain = download(ctx, op, 8 * rows);
            for (std::uint32_t t = 0; t < n_tok; ++t) {
                const auto xs = std::span<const float>(x).subspan(
                    std::size_t{t} * cols, cols);
                const auto g = cpu_mv(TensorType::kQ8_0, wg.data(), 0, rows,
                                      cols, xs);
                const auto u = cpu_mv(TensorType::kQ8_0, wu.data(), 0, rows,
                                      cols, xs);
                for (std::uint32_t r = 0; r < rows; ++r) {
                    const float want = g[r] / (1.0f + std::exp(-g[r])) * u[r];
                    REQUIRE(got[t * rows + r] ==
                            Catch::Approx(want).epsilon(1e-4).margin(1e-5));
                }
            }
            const std::vector<float> g0(got.begin(), got.begin() + rows);
            const std::vector<float> p0(plain.begin(), plain.begin() + rows);
            if (first_glu.empty()) {
                first_glu = g0;
                first_plain = p0;
            } else {
                REQUIRE(g0 == first_glu);
                REQUIRE(p0 == first_plain);
            }
            ctx.destroy_buffer(og);
            ctx.destroy_buffer(op);
        }
    }
    for (auto b : {gb, ub, xb}) ctx.destroy_buffer(b);
}

// #432: the NT 8, guard-free, NT 1 (i#511), NT 2 and NT 4 builds of
// matvec_q8_0_n (and _glu) run the same per-token arithmetic, so
// batch-invariant decode may pick one by batch size; this checks a
// token's output is the same bits from each build on this device.
TEST_CASE("vulkan short-batch builds give the same bits", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(83);
    const std::uint32_t rows = 37, cols = 4096;
    const auto wg = random_q8(rows, cols, rng);
    const auto wu = random_q8(rows, cols, rng);
    auto gb = upload_q8(ctx, wg, rows, cols);
    auto ub = upload_q8(ctx, wu, rows, cols);
    const auto x = random_floats(std::size_t{4} * cols, rng);
    auto xb = upload(ctx, x);
    const auto run = [&](Kernel k, bool glu, std::uint32_t n_tok) {
        auto ob = upload(ctx, std::vector<float>(4 * rows, 0.0f));
        ctx.begin_batch();
        const std::uint32_t push[] = {rows,  cols, 0, 0, 0, 0, fbits(1.0f),
                                      n_tok, cols, rows, 0, 0, 0};
        const VulkanContext::Buffer b3[] = {gb, xb, ob};
        const VulkanContext::Buffer b4[] = {gb, xb, ob, ub};
        if (glu) {
            ctx.dispatch(k, b4, push, VulkanContext::matvec_groups(k, rows));
        } else {
            ctx.dispatch(k, b3, push, VulkanContext::matvec_groups(k, rows));
        }
        ctx.end_batch();
        auto out = download(ctx, ob, 4 * rows);
        ctx.destroy_buffer(ob);
        return out;
    };
    for (const bool glu : {false, true}) {
        INFO((glu ? "glu" : "plain"));
        const Kernel k8 = glu ? Kernel::kMatvecQ8_0NGlu : Kernel::kMatvecQ8_0N;
        const Kernel kng =
            glu ? Kernel::kMatvecQ8_0NGluNg : Kernel::kMatvecQ8_0NNg;
        const Kernel k2 =
            glu ? Kernel::kMatvecQ8_0N2Glu : Kernel::kMatvecQ8_0N2;
        const Kernel k4 =
            glu ? Kernel::kMatvecQ8_0N4Glu : Kernel::kMatvecQ8_0N4;
        const Kernel k1 =
            glu ? Kernel::kMatvecQ8_0N1Glu : Kernel::kMatvecQ8_0N1;
        for (const std::uint32_t n_tok : {1u, 2u, 3u, 4u}) {
            INFO("n_tok " << n_tok);
            const auto a = run(k8, glu, n_tok);
            REQUIRE(run(kng, glu, n_tok) == a);
            REQUIRE(run(k4, glu, n_tok) == a);
            if (n_tok <= 2) {
                REQUIRE(run(k2, glu, n_tok) == a);
            }
            if (n_tok == 1) {
                REQUIRE(run(k1, glu, n_tok) == a);
            }
        }
    }
    for (auto b : {gb, ub, xb}) ctx.destroy_buffer(b);
}

namespace {

/**
 * Random K-quant bytes for rows x cols (as test_vulkan_prefill.cpp):
 * random quants and scales, the f16 d (and Q4_K's dmin) pinned to a
 * sane magnitude.
 */
std::vector<std::byte> random_kq(locus::gguf::TensorType type,
                                 std::uint32_t rows, std::uint32_t cols,
                                 std::mt19937& rng) {
    const bool q4 = type == locus::gguf::TensorType::kQ4_K;
    const std::size_t block = q4 ? 144 : 210;
    const std::size_t d_off = q4 ? 0 : 208;
    std::uniform_int_distribution<int> bd(0, 255);
    std::vector<std::byte> w(std::size_t{rows} * (cols / 256) * block);
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

}  // namespace

// #436: matvec_q4_k_n / matvec_q6_k_n (and their NT 2 builds) against
// the CPU per token, over two batches (w_bstride) with accumulate, and
// a token's bits being the same whatever the batch size and build.
TEST_CASE("vulkan K-quant short-batch matvecs match the CPU per token",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(97);
    using locus::gguf::TensorType;
    struct Case {
        TensorType type;
        Kernel k8, k2, k4;
        const char* name;
        std::size_t block;
    };
    const std::uint32_t rows = 21, cols = 768, n_batch = 2;
    for (const auto& c :
         {Case{TensorType::kQ4_K, Kernel::kMatvecQ4_KN, Kernel::kMatvecQ4_KN2,
               Kernel::kMatvecQ4_KN4, "q4_k", 144},
          Case{TensorType::kQ6_K, Kernel::kMatvecQ6_KN, Kernel::kMatvecQ6_KN2,
               Kernel::kMatvecQ6_KN4, "q6_k", 210}}) {
        const std::size_t mat_bytes =
            std::size_t{rows} * (cols / 256) * c.block;
        const auto w = random_kq(c.type, rows * n_batch, cols, rng);
        auto wb = upload(ctx, w);
        // Token rows x_stride apart, batches x_bstride apart; x_off 3
        // takes the unaligned (scalar) x path.
        const std::uint32_t x_stride = cols + 4;
        const std::uint32_t out_stride = rows + 2, out_off = 1;
        const auto run = [&](Kernel k, std::uint32_t n_tok,
                             std::uint32_t x_off,
                             const std::vector<float>& x) {
            const std::uint32_t x_bstride = 8 * x_stride + 8;
            const std::uint32_t out_bstride = 8 * out_stride;
            auto xb = upload(ctx, x);
            auto ob = upload(ctx, std::vector<float>(
                                      out_off + n_batch * out_bstride, 0.5f));
            ctx.begin_batch();
            const VulkanContext::Buffer bufs[] = {wb, xb, ob};
            const std::uint32_t push[] = {
                rows,       cols,
                out_off,    1u,
                0u,         x_off,
                fbits(1.0f), n_tok,
                x_stride,   out_stride,
                static_cast<std::uint32_t>(mat_bytes), x_bstride,
                out_bstride};
            ctx.dispatch(k, bufs, push, VulkanContext::matvec_groups(k, rows),
                         n_batch);
            ctx.end_batch();
            auto out = download(ctx, ob, out_off + n_batch * out_bstride);
            ctx.destroy_buffer(xb);
            ctx.destroy_buffer(ob);
            return out;
        };
        for (const std::uint32_t x_off : {0u, 3u}) {
            DYNAMIC_SECTION(c.name << " x_off " << x_off) {
                const std::uint32_t x_bstride = 8 * x_stride + 8;
                const auto x = random_floats(x_off + n_batch * x_bstride, rng);
                const auto all8 = run(c.k8, 8, x_off, x);
                for (std::uint32_t b = 0; b < n_batch; ++b) {
                    for (std::uint32_t t = 0; t < 8; ++t) {
                        const std::span<const float> xt(
                            x.data() + x_off + b * x_bstride + t * x_stride,
                            cols);
                        const auto ref = cpu_mv(c.type, w.data(), b * rows,
                                                rows, cols, xt);
                        for (std::uint32_t r = 0; r < rows; ++r) {
                            const float got =
                                all8[out_off + b * 8 * out_stride +
                                     t * out_stride + r];
                            INFO("b " << b << " t " << t << " r " << r);
                            CHECK(std::abs(got - (0.5f + ref[r])) <=
                                  1e-3f * (1.0f + std::abs(ref[r])));
                        }
                    }
                }
                // A token's bits the same through NT 8 at n = 1, 2 and
                // 8, NT 2 at n = 1 and 2, and NT 4 at n = 3 and 4.
                for (const auto& [k, n] :
                     {std::pair{c.k8, 1u}, std::pair{c.k8, 2u},
                      std::pair{c.k2, 1u}, std::pair{c.k2, 2u},
                      std::pair{c.k4, 3u}, std::pair{c.k4, 4u}}) {
                    INFO("n_tok " << n
                                  << (k == c.k2   ? " nt2"
                                      : k == c.k4 ? " nt4"
                                                  : " nt8"));
                    const auto got = run(k, n, x_off, x);
                    for (std::uint32_t b = 0; b < n_batch; ++b) {
                        for (std::uint32_t t = 0; t < n; ++t) {
                            for (std::uint32_t r = 0; r < rows; ++r) {
                                const std::size_t i = out_off +
                                                      b * 8 * out_stride +
                                                      t * out_stride + r;
                                REQUIRE(got[i] == all8[i]);
                            }
                        }
                    }
                }
            }
        }
        ctx.destroy_buffer(wb);
    }
}

// #436: matvec_q4_k_n_glu (gate, up and silu_mul in one) against the
// CPU, and its NT 8, NT 2 and NT 4 builds giving a token the same bits.
TEST_CASE("vulkan Q4_K short-batch SwiGLU matches the CPU", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(101);
    using locus::gguf::TensorType;
    const std::uint32_t rows = 19, cols = 512, n_tok = 8;
    const auto wg = random_kq(TensorType::kQ4_K, rows, cols, rng);
    const auto wu = random_kq(TensorType::kQ4_K, rows, cols, rng);
    const auto x = random_floats(std::size_t{n_tok} * cols, rng);
    auto gb = upload(ctx, wg);
    auto ub = upload(ctx, wu);
    auto xb = upload(ctx, x);
    const auto run = [&](Kernel k, std::uint32_t n) {
        auto ob = upload(ctx, std::vector<float>(n_tok * rows, 0.0f));
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {gb, xb, ob, ub};
        const std::uint32_t push[] = {rows, cols, 0, 0, 0, 0, fbits(1.0f),
                                      n,    cols, rows, 0, 0, 0};
        ctx.dispatch(k, bufs, push, VulkanContext::matvec_groups(k, rows));
        ctx.end_batch();
        auto out = download(ctx, ob, n_tok * rows);
        ctx.destroy_buffer(ob);
        return out;
    };
    const auto all8 = run(Kernel::kMatvecQ4_KNGlu, n_tok);
    for (std::uint32_t t = 0; t < n_tok; ++t) {
        const std::span<const float> xt(x.data() + t * cols, cols);
        const auto g = cpu_mv(TensorType::kQ4_K, wg.data(), 0, rows, cols, xt);
        const auto u = cpu_mv(TensorType::kQ4_K, wu.data(), 0, rows, cols, xt);
        for (std::uint32_t r = 0; r < rows; ++r) {
            const float want = g[r] / (1.0f + std::exp(-g[r])) * u[r];
            INFO("t " << t << " r " << r);
            CHECK(std::abs(all8[t * rows + r] - want) <=
                  2e-3f * (1.0f + std::abs(want)));
        }
    }
    for (const auto& [k, n] : {std::pair{Kernel::kMatvecQ4_KNGlu, 1u},
                               std::pair{Kernel::kMatvecQ4_KN2Glu, 1u},
                               std::pair{Kernel::kMatvecQ4_KN2Glu, 2u},
                               std::pair{Kernel::kMatvecQ4_KN4Glu, 3u},
                               std::pair{Kernel::kMatvecQ4_KN4Glu, 4u}}) {
        INFO("n_tok " << n);
        const auto got = run(k, n);
        for (std::uint32_t i = 0; i < n * rows; ++i) {
            REQUIRE(got[i] == all8[i]);
        }
    }
    for (auto b : {gb, ub, xb}) ctx.destroy_buffer(b);
}
