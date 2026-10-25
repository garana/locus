// The Vulkan conv1d / conv1d_transpose / snake1d / conv2d shaders
// against the CPU scalar reference (cpu_ops), the oracle for every
// backend.

#include <bit>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/vulkan/context.hpp"

using locus::backend::vk::Kernel;
using locus::backend::vk::VulkanContext;

namespace {

std::uint32_t fbits(float v) { return std::bit_cast<std::uint32_t>(v); }

VulkanContext::Buffer upload(VulkanContext& ctx,
                             const std::vector<float>& v) {
    auto b = ctx.create_buffer(
        std::max<std::size_t>(v.size() * sizeof(float), 4));
    ctx.write_buffer(b, std::as_bytes(std::span<const float>(v)));
    return b;
}

std::vector<float> download(VulkanContext& ctx, VulkanContext::Buffer b,
                            std::size_t n) {
    std::vector<float> out(n);
    ctx.read_buffer(b, std::as_writable_bytes(std::span<float>(out)));
    return out;
}

std::vector<float> randvec(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

}  // namespace

TEST_CASE("vulkan conv1d matches the CPU reference", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(91);

    struct Case {
        const char* name;
        std::uint32_t in_ch, out_ch, in_len, k, stride, dilation, pad,
            groups, has_bias;
    };
    const Case cases[] = {
        {"dense k3 same-pad", 3, 5, 16, 3, 1, 1, 1, 1, 1},
        {"depthwise k7 dil3", 8, 8, 32, 7, 1, 3, 9, 8, 1},
        {"depthwise k7 dil9", 8, 8, 48, 7, 1, 9, 27, 8, 1},
        {"dense stride2 dil2", 2, 4, 20, 3, 2, 2, 0, 1, 1},
        {"pointwise k1", 6, 7, 10, 1, 1, 1, 0, 1, 1},
        {"grouped 2", 4, 6, 16, 3, 1, 1, 1, 2, 1},
        // SNAC's NoiseBlock is a bias-free 1x1 conv: the no-bias path
        // #155 will use. Bias buffer is bound but must not be read.
        {"pointwise k1 no bias", 6, 7, 10, 1, 1, 1, 0, 1, 0},
    };
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.name) {
            const std::uint32_t icg = c.in_ch / c.groups;
            const std::uint32_t out_len = locus::backend::conv1d_out_len(
                c.in_len, c.k, c.stride, c.dilation, c.pad);
            auto in = randvec(std::size_t{c.in_ch} * c.in_len, rng);
            auto w = randvec(std::size_t{c.out_ch} * icg * c.k, rng);
            auto bias = randvec(c.out_ch, rng);
            std::vector<float> want(std::size_t{c.out_ch} * out_len);
            locus::backend::conv1d(
                in, c.in_ch, c.in_len, w, c.out_ch, c.k,
                c.has_bias ? std::span<const float>(bias)
                           : std::span<const float>{},
                c.stride, c.dilation, c.pad, c.groups, want);
            auto ib = upload(ctx, in), wb = upload(ctx, w),
                 bb = upload(ctx, bias);
            auto ob = upload(ctx, std::vector<float>(want.size(), 0.0f));
            ctx.begin_batch();
            const VulkanContext::Buffer bufs[] = {ib, wb, bb, ob};
            const std::uint32_t push[] = {c.in_ch,  c.in_len,  c.out_ch,
                                          out_len,  c.k,       c.stride,
                                          c.dilation, c.pad,   c.groups,
                                          c.has_bias};
            ctx.dispatch(Kernel::kConv1d, bufs, push,
                         (static_cast<std::uint32_t>(want.size()) + 63) /
                             64);
            ctx.end_batch();
            auto got = download(ctx, ob, want.size());
            for (std::size_t i = 0; i < want.size(); ++i) {
                INFO("element " << i);
                REQUIRE(got[i] == Catch::Approx(want[i]).margin(1e-4));
            }
            for (auto b : {ib, wb, bb, ob}) ctx.destroy_buffer(b);
        }
    }
}

TEST_CASE("vulkan conv1d_transpose matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(92);

    struct Case {
        const char* name;
        std::uint32_t in_ch, out_ch, in_len, k, stride, pad, groups;
    };
    const Case cases[] = {
        {"dense upsample x2", 4, 3, 8, 4, 2, 1, 1},
        {"dense k2 stride1", 1, 1, 6, 2, 1, 0, 1},
        {"grouped 2", 4, 6, 8, 4, 2, 1, 2},
    };
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.name) {
            const std::uint32_t ocg = c.out_ch / c.groups;
            const std::uint32_t out_len =
                locus::backend::conv1d_transpose_out_len(c.in_len, c.k,
                                                         c.stride, 1, c.pad);
            auto in = randvec(std::size_t{c.in_ch} * c.in_len, rng);
            auto w = randvec(std::size_t{c.in_ch} * ocg * c.k, rng);
            auto bias = randvec(c.out_ch, rng);
            std::vector<float> want(std::size_t{c.out_ch} * out_len);
            locus::backend::conv1d_transpose(in, c.in_ch, c.in_len, w,
                                             c.out_ch, c.k, bias, c.stride,
                                             1, c.pad, c.groups, want);
            auto ib = upload(ctx, in), wb = upload(ctx, w),
                 bb = upload(ctx, bias);
            auto ob = upload(ctx, std::vector<float>(want.size(), 0.0f));
            ctx.begin_batch();
            const VulkanContext::Buffer bufs[] = {ib, wb, bb, ob};
            const std::uint32_t push[] = {c.in_ch, c.in_len, c.out_ch,
                                          out_len, c.k,      c.stride,
                                          1u,      c.pad,    c.groups,
                                          1u};
            ctx.dispatch(Kernel::kConv1dTranspose, bufs, push,
                         (static_cast<std::uint32_t>(want.size()) + 63) /
                             64);
            ctx.end_batch();
            auto got = download(ctx, ob, want.size());
            for (std::size_t i = 0; i < want.size(); ++i) {
                INFO("element " << i);
                REQUIRE(got[i] == Catch::Approx(want[i]).margin(1e-4));
            }
            for (auto b : {ib, wb, bb, ob}) ctx.destroy_buffer(b);
        }
    }
}

TEST_CASE("vulkan snake1d matches the CPU reference", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(93);
    const std::uint32_t ch = 5, len = 37;
    auto x = randvec(std::size_t{ch} * len, rng);
    auto alpha = randvec(ch, rng);
    for (auto& a : alpha) a += 1.5f;  // keep alpha away from -eps
    std::vector<float> want = x;
    locus::backend::snake1d(want, ch, len, alpha);
    auto xb = upload(ctx, x), ab = upload(ctx, alpha);
    ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {xb, ab};
    const std::uint32_t push[] = {ch, len, fbits(1e-9f)};
    ctx.dispatch(Kernel::kSnake1d, bufs, push, (ch * len + 63) / 64);
    ctx.end_batch();
    auto got = download(ctx, xb, x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        INFO("element " << i);
        REQUIRE(got[i] == Catch::Approx(want[i]).margin(1e-4));
    }
    for (auto b : {xb, ab}) ctx.destroy_buffer(b);
}

// i#222: conv2d.comp against the scalar reference, on image-decoder
// shapes (3x3 same and 1x1, a channel count past one workgroup, groups,
// no bias, stride 2).
TEST_CASE("vulkan conv2d matches the CPU reference", "[vulkan][conv2d]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(93);

    struct Case {
        const char* name;
        std::uint32_t in_ch, in_h, in_w, out_ch, k, stride, pad, groups,
            has_bias;
    };
    const Case cases[] = {
        {"3x3 same", 8, 17, 23, 12, 3, 1, 1, 1, 1},
        {"1x1", 70, 9, 11, 5, 1, 1, 0, 1, 1},
        {"grouped no bias", 6, 12, 12, 6, 3, 1, 1, 3, 0},
        {"stride 2", 4, 11, 13, 6, 3, 2, 1, 1, 1},
        // 3 x 1200 x 1200 = 4.32M outputs: a flat grid of 64-wide
        // groups would need 67,500 > 65535 (Vulkan's guaranteed x
        // count), as the decoder's 384 x 384 x 128 convs would.
        {"past the 1-D group limit", 1, 1200, 1200, 3, 3, 1, 1, 1, 1},
    };
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.name) {
            const std::uint32_t icg = c.in_ch / c.groups;
            const std::uint32_t oh = locus::backend::conv1d_out_len(
                c.in_h, c.k, c.stride, 1, c.pad);
            const std::uint32_t ow = locus::backend::conv1d_out_len(
                c.in_w, c.k, c.stride, 1, c.pad);
            auto in = randvec(std::size_t{c.in_ch} * c.in_h * c.in_w, rng);
            auto w = randvec(std::size_t{c.out_ch} * icg * c.k * c.k, rng);
            auto bias = randvec(c.out_ch, rng);
            std::vector<float> want(std::size_t{c.out_ch} * oh * ow);
            locus::backend::conv2d(
                in, c.in_ch, c.in_h, c.in_w, w, c.out_ch, c.k, c.k,
                c.has_bias ? std::span<const float>(bias)
                           : std::span<const float>{},
                c.stride, c.pad, c.groups, want);
            auto ib = upload(ctx, in), wb = upload(ctx, w),
                 bb = upload(ctx, bias);
            auto ob = upload(ctx, std::vector<float>(want.size(), 0.0f));
            ctx.begin_batch();
            const VulkanContext::Buffer bufs[] = {ib, wb, bb, ob};
            const std::uint32_t push[] = {c.in_ch,  c.in_h,    c.in_w,
                                          c.out_ch, oh,        ow,
                                          c.k,      c.k,       c.stride,
                                          c.pad,    c.groups,  c.has_bias};
            // x tiles one output plane in 64s; y is the out channel.
            ctx.dispatch(Kernel::kConv2d, bufs, push, (oh * ow + 63) / 64,
                         c.out_ch);
            ctx.end_batch();
            auto got = download(ctx, ob, want.size());
            for (std::size_t i = 0; i < want.size(); ++i) {
                INFO("element " << i);
                REQUIRE(got[i] == Catch::Approx(want[i]).margin(1e-4));
            }
            for (auto b : {ib, wb, bb, ob}) ctx.destroy_buffer(b);
        }
    }
}
