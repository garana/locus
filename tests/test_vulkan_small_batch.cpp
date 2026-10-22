// Short-batch kernels for the Vulkan batched forward (speculative
// verify): matvec_q8_0_n / matvec_f32_n over <= 8 token rows, and the
// routed-MoE (token, expert) pair matvecs matvec_q8_0_glu_p /
// matvec_q8_0_p.

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
    const std::size_t row_bytes = type == locus::gguf::TensorType::kQ8_0
                                      ? (cols / 32) * 34
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
