// Q8_0 matvec variants that fold several dispatches into one: stacked
// projections (matvec_q8_0_seg) and MoE shared experts riding the
// routed ones (matvec_q8_0_glu_sh_b, matvec_q8_0_sh_b).

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

/** CPU W x for rows [r0, r0 + rows) of the Q8_0 matrix w. */
std::vector<float> cpu_mv(const std::vector<std::byte>& w, std::uint32_t r0,
                          std::uint32_t rows, std::uint32_t cols,
                          std::span<const float> x) {
    locus::backend::Mat m{locus::gguf::TensorType::kQ8_0,
                          w.data() + std::size_t{r0} * (cols / 32) * 34,
                          rows, cols};
    std::vector<float> out(rows);
    locus::backend::matvec(m, x, out);
    return out;
}

float silu(float v) { return v / (1.0f + std::exp(-v)); }

}  // namespace

TEST_CASE("vulkan segmented q8_0 matvec routes stacked rows to two "
          "outputs",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(43);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // q | k | v stacked: 9 + 5 + 5 rows of 96 (odd nb: unaligned rows).
    const std::uint32_t nq = 9, nk = 5, nv = 5, cols = 96;
    const std::uint32_t rows = nq + nk + nv;
    const auto w = random_q8(rows, cols, rng);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    auto wb = upload_q8(ctx, w, rows, cols);
    auto xb = upload(ctx, x);
    // O gets q at 3; O2 gets k at 20 and v at 2 (out of row order).
    auto ob = upload(ctx, std::vector<float>(16, -7.0f));
    auto o2b = upload(ctx, std::vector<float>(32, -7.0f));

    ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {wb, xb, ob, o2b};
    const std::uint32_t push[] = {rows, cols, 3, 0, 0, 0, fbits(1.0f),
                                  nq, 20, nq + nk, 2};
    ctx.dispatch(Kernel::kMatvecQ8_0Seg, bufs, push,
                 VulkanContext::matvec_groups(Kernel::kMatvecQ8_0Seg, rows));
    ctx.end_batch();
    const auto o = download<float>(ctx, ob, 16);
    const auto o2 = download<float>(ctx, o2b, 32);

    const auto q = cpu_mv(w, 0, nq, cols, x);
    const auto k = cpu_mv(w, nq, nk, cols, x);
    const auto v = cpu_mv(w, nq + nk, nv, cols, x);
    for (std::uint32_t r = 0; r < nq; ++r) {
        REQUIRE(o[3 + r] == Catch::Approx(q[r]).margin(1e-4));
    }
    for (std::uint32_t r = 0; r < nk; ++r) {
        REQUIRE(o2[20 + r] == Catch::Approx(k[r]).margin(1e-4));
    }
    for (std::uint32_t r = 0; r < nv; ++r) {
        REQUIRE(o2[2 + r] == Catch::Approx(v[r]).margin(1e-4));
    }
    // Untouched elements keep their sentinel.
    REQUIRE(o[2] == -7.0f);
    REQUIRE(o[3 + nq] == -7.0f);
    REQUIRE(o2[1] == -7.0f);
    REQUIRE(o2[2 + nv] == -7.0f);
    REQUIRE(o2[19] == -7.0f);
    REQUIRE(o2[20 + nk] == -7.0f);
    for (auto b : {wb, xb, ob, o2b}) {
        ctx.destroy_buffer(b);
    }
}

TEST_CASE("vulkan MoE matvecs fold the shared experts into the routed "
          "dispatch",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 4 routed experts (ff x cols gate / up, cols x ff down), 2 of
    // them picked, plus 2 shared experts' worth (2 * ff rows) of
    // shared gate / up and a cols x 2 * ff shared down.
    const std::uint32_t ff = 64, cols = 96, n_exp = 4, k_used = 2,
                        n_sh = 2;
    const std::size_t up_bytes = ff * (cols / 32) * 34;
    const std::size_t down_bytes = cols * (ff / 32) * 34;
    const auto wg = random_q8(ff * n_exp, cols, rng);
    const auto wu = random_q8(ff * n_exp, cols, rng);
    const auto wd = random_q8(cols * n_exp, ff, rng);
    const auto sg = random_q8(ff * n_sh, cols, rng);
    const auto su = random_q8(ff * n_sh, cols, rng);
    const auto sd = random_q8(cols, ff * n_sh, rng);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    const std::uint32_t ids[] = {3, 1};
    const float wts[] = {0.75f, -0.5f};
    const std::vector<std::uint32_t> route = {ids[0], ids[1],
                                              fbits(wts[0]),
                                              fbits(wts[1])};

    auto gb = upload_q8(ctx, wg, ff * n_exp, cols);
    auto ub = upload_q8(ctx, wu, ff * n_exp, cols);
    auto db = upload_q8(ctx, wd, cols * n_exp, ff);
    auto sgb = upload_q8(ctx, sg, ff * n_sh, cols);
    auto sub = upload_q8(ctx, su, ff * n_sh, cols);
    auto sdb = upload_q8(ctx, sd, cols, ff * n_sh);
    auto xb = upload(ctx, x);
    auto rb = upload(ctx, route);
    const std::uint32_t inner = (k_used + n_sh) * ff;
    auto hb = ctx.create_buffer(inner * 4);
    std::vector<float> resid(cols);
    for (auto& v : resid) {
        v = dist(rng);
    }
    auto ob = upload(ctx, resid);

    ctx.begin_batch();
    {
        const VulkanContext::Buffer bufs[] = {gb, xb, hb, rb, ub, sgb, sub};
        const std::uint32_t push[] = {
            ff, cols, 0, 0, 0, 0, fbits(1.0f),
            static_cast<std::uint32_t>(up_bytes), 0, ff, k_used, 1u, 0u,
            static_cast<std::uint32_t>(up_bytes), 0u};
        ctx.dispatch(Kernel::kMatvecQ8_0GluShB, bufs, push,
                     VulkanContext::matvec_groups(Kernel::kMatvecQ8_0GluShB,
                                                  ff),
                     k_used + n_sh);
    }
    {
        const VulkanContext::Buffer bufs[] = {db, hb, ob, rb, sdb};
        const std::uint32_t push[] = {
            cols, ff, 0, 1, 0, 0, fbits(1.0f),
            static_cast<std::uint32_t>(down_bytes), ff, 0, k_used, 3u, 0u,
            ff * n_sh, k_used * ff};
        ctx.dispatch(Kernel::kMatvecQ8_0ShB, bufs, push,
                     VulkanContext::matvec_groups(Kernel::kMatvecQ8_0ShB,
                                                  cols));
    }
    ctx.end_batch();
    const auto h = download<float>(ctx, hb, inner);
    const auto out = download<float>(ctx, ob, cols);

    // CPU reference: routed slot k = expert ids[k]; slots k_used..
    // are the shared experts' row blocks, in order.
    std::vector<float> want_h(inner);
    for (std::uint32_t k = 0; k < k_used + n_sh; ++k) {
        const bool shared = k >= k_used;
        const std::uint32_t r0 = shared ? (k - k_used) * ff : ids[k] * ff;
        const auto a = cpu_mv(shared ? sg : wg, r0, ff, cols, x);
        const auto b = cpu_mv(shared ? su : wu, r0, ff, cols, x);
        for (std::uint32_t r = 0; r < ff; ++r) {
            want_h[k * ff + r] = silu(a[r]) * b[r];
        }
    }
    for (std::uint32_t i = 0; i < inner; ++i) {
        REQUIRE(h[i] == Catch::Approx(want_h[i]).margin(1e-4));
    }
    // Use the GPU's own intermediate so the down check isolates the
    // down kernel.
    std::vector<float> want = resid;
    for (std::uint32_t k = 0; k < k_used; ++k) {
        const auto d = cpu_mv(
            wd, ids[k] * cols, cols, ff,
            std::span<const float>(h).subspan(k * ff, ff));
        for (std::uint32_t r = 0; r < cols; ++r) {
            want[r] += wts[k] * d[r];
        }
    }
    const auto ds = cpu_mv(sd, 0, cols, ff * n_sh,
                           std::span<const float>(h).subspan(k_used * ff,
                                                             n_sh * ff));
    for (std::uint32_t r = 0; r < cols; ++r) {
        want[r] += ds[r];
        REQUIRE(out[r] == Catch::Approx(want[r]).margin(1e-4));
    }
    for (auto b : {gb, ub, db, sgb, sub, sdb, xb, rb, hb, ob}) {
        ctx.destroy_buffer(b);
    }
}
