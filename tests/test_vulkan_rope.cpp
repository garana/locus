// rope.comp variants against the CPU references: interleaved pairs
// (rope / rope_b, rope_norm_yarn) and NEOX pairs (rope_neox /
// rope_neox_b, rope_neox_yarn).

#include <algorithm>
#include <bit>
#include <random>
#include <tuple>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/vulkan/context.hpp"

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

}  // namespace

TEST_CASE("vulkan rope kernels match the CPU interleaved / NEOX rope",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(83);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 3 heads of 16 at offset 5, heads 16 apart; 2 token rows (the
    // batched variants) 64 floats apart at positions pos, pos + 1.
    const std::uint32_t heads = 3, hd = 16, off = 5, rows = 2;
    const std::uint32_t row_stride = 64, pos = 41;
    const float base = 10000.0f;
    std::vector<float> x(off + rows * row_stride);
    for (auto& v : x) {
        v = dist(rng);
    }
    const std::vector<float> ones(hd / 2, 1.0f);

    for (const bool neox : {false, true}) {
        for (const bool batched : {false, true}) {
            DYNAMIC_SECTION((neox ? "neox" : "interleaved")
                            << (batched ? " batched" : "")) {
                auto xb = upload(ctx, x);
                auto fb = upload(ctx, ones);
                ctx.begin_batch();
                const VulkanContext::Buffer bufs[] = {xb, fb};
                const Kernel k = neox ? (batched ? Kernel::kRopeNeoxB
                                                 : Kernel::kRopeNeox)
                                      : (batched ? Kernel::kRopeB
                                                 : Kernel::kRope);
                if (batched) {
                    const std::uint32_t push[] = {
                        heads, hd, pos, off, hd, fbits(base),
                        fbits(1.0f), 0, 0, row_stride};
                    ctx.dispatch(k, bufs, push, heads, rows);
                } else {
                    const std::uint32_t push[] = {
                        heads, hd, pos, off, hd, fbits(base),
                        fbits(1.0f), 0, 0};
                    ctx.dispatch(k, bufs, push, heads);
                }
                ctx.end_batch();
                const auto got = download(ctx, xb, x.size());

                std::vector<float> want = x;
                for (std::uint32_t r = 0; r < (batched ? rows : 1u); ++r) {
                    std::span<float> span(want.data() + off + r * row_stride,
                                          heads * hd);
                    if (neox) {
                        locus::backend::rope_neox_yarn(
                            span, heads, hd, pos + r, base, {});
                    } else {
                        locus::backend::rope_norm_yarn(
                            span, heads, hd, pos + r, base, {});
                    }
                }
                for (std::size_t i = 0; i < x.size(); ++i) {
                    INFO("element " << i);
                    REQUIRE(got[i] == Catch::Approx(want[i]).margin(1e-5));
                }
                ctx.destroy_buffer(xb);
                ctx.destroy_buffer(fb);
            }
        }
    }
}

TEST_CASE("vulkan rope matches the CPU across the pair stride",
          "[vulkan]") {
    // rope.comp gives each head 64 invocations striding over its pairs
    // by 128 elements; head_dim 160 (80 pairs) makes lanes 0..15 take
    // a second, partial stride, which head_dim <= 128 never reaches.
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(61);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const std::uint32_t heads = 3, hd = 160, pos = 37, off = 5;
    const float base = 10000.0f;
    for (const bool neox : {false, true}) {
        INFO((neox ? "neox" : "interleaved"));
        std::vector<float> x(off + heads * hd);
        for (auto& v : x) {
            v = dist(rng);
        }
        std::vector<float> want(x.begin() + off, x.end());
        if (neox) {
            locus::backend::rope_neox_yarn(want, heads, hd, pos, base,
                                           locus::backend::Yarn{});
        } else {
            locus::backend::rope_norm(want, heads, hd, pos, base);
        }
        auto xb = upload(ctx, x);
        auto fb = upload(ctx, std::vector<float>(hd / 2, 1.0f));
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {xb, fb};
        const std::uint32_t push[] = {heads, hd, pos, off, hd,
                                      fbits(base), fbits(1.0f), 0, 0};
        ctx.dispatch(neox ? Kernel::kRopeNeox : Kernel::kRope, bufs, push,
                     heads);
        ctx.end_batch();
        std::vector<float> got(x.size());
        ctx.read_buffer(xb, std::as_writable_bytes(std::span<float>(got)));
        for (std::uint32_t i = 0; i < off; ++i) {
            REQUIRE(got[i] == x[i]);  // before the first head: untouched
        }
        for (std::size_t i = 0; i < want.size(); ++i) {
            INFO("element " << i);
            REQUIRE(got[off + i] ==
                    Catch::Approx(want[i]).margin(1e-4).epsilon(1e-3));
        }
        ctx.destroy_buffer(xb);
        ctx.destroy_buffer(fb);
    }
}

TEST_CASE("vulkan qk_norm_rope matches rmsnorm_b then rope bit for bit",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(89);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> fd(0.5f, 4.0f);
    const float eps = 1e-6f, base = 1000000.0f;
    const std::uint32_t pos = 1234;

    // 5 q heads at offset 3 of Q, 2 k heads at offset 7 of K; head
    // dims from one pair per thread up to a strided second pass.
    for (const std::uint32_t hd : {64u, 128u, 512u}) {
        const std::uint32_t nq = 5, nk = 2, q_off = 3, k_off = 7;
        std::vector<float> q(q_off + nq * hd), k(k_off + nk * hd + 5);
        std::vector<float> qw(hd), kw(hd), div(hd / 2);
        for (auto* v : {&q, &k, &qw, &kw}) {
            for (auto& e : *v) {
                e = dist(rng);
            }
        }
        for (auto& e : div) {
            e = fd(rng);
        }
        for (const bool neox : {false, true}) {
            DYNAMIC_SECTION("head dim " << hd
                                        << (neox ? " neox" : " interleaved")) {
                auto fb = upload(ctx, div);
                auto qwb = upload(ctx, qw);
                auto kwb = upload(ctx, kw);
                // Reference: rmsnorm_b over the heads, then rope.
                auto qr = upload(ctx, q);
                auto kr = upload(ctx, k);
                const Kernel rk = neox ? Kernel::kRopeNeox : Kernel::kRope;
                ctx.begin_batch();
                for (const auto& [b, wb, off, n] :
                     {std::tuple{qr, qwb, q_off, nq},
                      std::tuple{kr, kwb, k_off, nk}}) {
                    const VulkanContext::Buffer nb[] = {b, wb, b};
                    const std::uint32_t np[] = {hd, fbits(eps), off,
                                                off, hd, hd};
                    ctx.dispatch(Kernel::kRmsNormB, nb, np, 1, n);
                    const VulkanContext::Buffer rb[] = {b, fb};
                    const std::uint32_t rp[] = {n,           hd, pos, off,
                                                hd, fbits(base),
                                                fbits(1.0f), 0,   0};
                    ctx.dispatch(rk, rb, rp, n);
                }
                ctx.end_batch();
                // Fused.
                auto qf = upload(ctx, q);
                auto kf = upload(ctx, k);
                ctx.begin_batch();
                const VulkanContext::Buffer fbufs[] = {qf, kf, qwb, kwb,
                                                       fb};
                const std::uint32_t fp[] = {nq,    hd,          fbits(eps),
                                            q_off, k_off,       pos,
                                            fbits(base), neox ? 1u : 0u};
                ctx.dispatch(Kernel::kQkNormRope, fbufs, fp, nq + nk);
                ctx.end_batch();
                const auto qwant = download(ctx, qr, q.size());
                const auto kwant = download(ctx, kr, k.size());
                const auto qgot = download(ctx, qf, q.size());
                const auto kgot = download(ctx, kf, k.size());
                for (std::size_t i = 0; i < q.size(); ++i) {
                    INFO("q " << i);
                    REQUIRE(fbits(qgot[i]) == fbits(qwant[i]));
                }
                for (std::size_t i = 0; i < k.size(); ++i) {
                    INFO("k " << i);
                    REQUIRE(fbits(kgot[i]) == fbits(kwant[i]));
                }
                for (auto b : {fb, qwb, kwb, qr, kr, qf, kf}) {
                    ctx.destroy_buffer(b);
                }
            }
        }
    }
}
