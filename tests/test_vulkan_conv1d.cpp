// The Vulkan conv1d / conv1d_transpose / snake1d / conv2d (direct and
// implicit GEMM) shaders against the CPU scalar reference (cpu_ops),
// the oracle for every backend.

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/variants.hpp"
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

// conv2d_gemm.comp (the groups == 1 path) against the scalar
// reference: a K that is not a multiple of the 16-column step, more
// than one 128-channel tile, partial pixel tiles, stride 2, no bias.
TEST_CASE("vulkan conv2d_gemm matches the CPU reference",
          "[vulkan][conv2d]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(94);

    struct Case {
        const char* name;
        std::uint32_t in_ch, in_h, in_w, out_ch, k, stride, pad, has_bias;
    };
    const Case cases[] = {
        {"3x3 same", 5, 17, 23, 12, 3, 1, 1, 1},
        {"1x1", 70, 9, 11, 5, 1, 1, 0, 1},
        {"two channel tiles", 16, 10, 14, 200, 3, 1, 1, 1},
        {"stride 2 no bias", 4, 11, 13, 6, 3, 2, 1, 0},
        {"one in channel", 1, 40, 40, 3, 3, 1, 1, 1},
    };
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.name) {
            const std::uint32_t oh = locus::backend::conv1d_out_len(
                c.in_h, c.k, c.stride, 1, c.pad);
            const std::uint32_t ow = locus::backend::conv1d_out_len(
                c.in_w, c.k, c.stride, 1, c.pad);
            auto in = randvec(std::size_t{c.in_ch} * c.in_h * c.in_w, rng);
            auto w = randvec(std::size_t{c.out_ch} * c.in_ch * c.k * c.k,
                             rng);
            auto bias = randvec(c.out_ch, rng);
            std::vector<float> want(std::size_t{c.out_ch} * oh * ow);
            locus::backend::conv2d(
                in, c.in_ch, c.in_h, c.in_w, w, c.out_ch, c.k, c.k,
                c.has_bias ? std::span<const float>(bias)
                           : std::span<const float>{},
                c.stride, c.pad, 1, want);
            auto ib = upload(ctx, in), wb = upload(ctx, w),
                 bb = upload(ctx, bias);
            auto ob = upload(ctx, std::vector<float>(want.size(), 0.0f));
            ctx.begin_batch();
            const VulkanContext::Buffer bufs[] = {ib, wb, bb, ob};
            const std::uint32_t push[] = {c.in_ch,  c.in_h, c.in_w,
                                          c.out_ch, oh,     ow,
                                          c.k,      c.k,    c.stride,
                                          c.pad,    1,      c.has_bias};
            // x tiles the output plane in 64 pixels, y the output
            // channels in 128.
            ctx.dispatch(Kernel::kConv2dGemm, bufs, push,
                         (oh * ow + 63) / 64, (c.out_ch + 127) / 128);
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

namespace {

/** Sets LOCUS_VK_SPLIT for the next VulkanContext; restores the old
 *  value (or its absence) on destruction. */
struct SplitEnv {
    explicit SplitEnv(const char* v) {
        if (const char* o = std::getenv("LOCUS_VK_SPLIT")) {
            old = o;
            had = true;
        }
        ::setenv("LOCUS_VK_SPLIT", v, 1);
    }
    ~SplitEnv() {
        if (had) {
            ::setenv("LOCUS_VK_SPLIT", old.c_str(), 1);
        } else {
            ::unsetenv("LOCUS_VK_SPLIT");
        }
    }
    std::string old;
    bool had = false;
};

}  // namespace

// i#682: on V3D a large grid goes as several vkCmdDispatchBase pieces
// (LOCUS_VK_SPLIT sets the budget). A split grid must give the whole
// one's bits: here on any device, with the split forced, over a grid of
// 2 x 2 output tiles; a budget of 2 splits x only, one of 1 splits y
// too.
TEST_CASE("vulkan split dispatches give the bits of a whole one",
          "[vulkan][conv2d]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    std::mt19937 rng(682);
    const std::uint32_t in_ch = 8, hw = 12, out_ch = 200, k = 3;
    const std::uint32_t o = hw - k + 1;  // stride 1, no padding
    const auto in = randvec(std::size_t{in_ch} * hw * hw, rng);
    const auto w = randvec(std::size_t{out_ch} * in_ch * k * k, rng);
    const auto bias = randvec(out_ch, rng);
    const auto run = [&](const char* split) {
        const SplitEnv env(split);
        VulkanContext ctx;
        auto ib = upload(ctx, in), wb = upload(ctx, w),
             bb = upload(ctx, bias);
        auto ob = upload(ctx, std::vector<float>(out_ch * o * o, 0.0f));
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {ib, wb, bb, ob};
        const std::uint32_t push[] = {in_ch, hw, hw, out_ch, o, o, k,
                                      k,     1,  0,  1,      1};
        // 100 output pixels in 64s and 200 channels in 128s: 2 x 2.
        ctx.dispatch(Kernel::kConv2dGemm, bufs, push, (o * o + 63) / 64,
                     (out_ch + 127) / 128);
        ctx.end_batch();
        auto got = download(ctx, ob, std::size_t{out_ch} * o * o);
        for (auto b : {ib, wb, bb, ob}) ctx.destroy_buffer(b);
        return got;
    };
    const auto whole = run("0");
    for (const char* split : {"2", "1"}) {
        INFO("LOCUS_VK_SPLIT=" << split);
        const auto got = run(split);
        REQUIRE(got.size() == whole.size());
        for (std::size_t i = 0; i < got.size(); ++i) {
            REQUIRE(fbits(got[i]) == fbits(whole[i]));
        }
    }
}

// i#682: on V3D, unsplit, about 4% of these rounds came back wrong: the
// kernel reset the GPU in the middle of the 338-workgroup 3 x 3 conv, and
// its last workgroups never ran. Inception's shapes run in turn through
// conv2d_vulkan (the path inception takes) on fixed inputs, and every run
// must give the first one's bits; a wrong one prints how many outputs
// moved and the channels and pixels they span. Hidden (minutes on V3D):
// LOCUS_VK_STRESS_ITERS sets the rounds (default 30). A failure is worth
// a look at the kernel log ("Resetting GPU for hang").
TEST_CASE("vulkan conv2d repeats its bits", "[.][vulkan][stress]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device");
    }
    int iters = 30;
    if (const char* e = std::getenv("LOCUS_VK_STRESS_ITERS")) {
        iters = std::max(1, std::atoi(e));
    }
    struct Shape {
        std::uint32_t in_ch, hw, out_ch, k, stride, pad;
    };
    // The stem's 3 x 3, a 1 x 1 at 35 x 35, the stride 2 reduction, a 1 x
    // 1 at 17 x 17 and one at 8 x 8.
    const Shape shapes[] = {{32, 149, 32, 3, 1, 0},
                            {288, 35, 64, 1, 1, 0},
                            {288, 35, 384, 3, 2, 0},
                            {768, 17, 192, 1, 1, 0},
                            {2048, 8, 320, 1, 1, 0}};
    struct Data {
        std::vector<float> in, w, b, first;
        std::size_t plane;
    };
    std::mt19937 rng(682);
    std::vector<Data> data;
    for (const Shape& s : shapes) {
        Data d;
        d.in = randvec(std::size_t{s.in_ch} * s.hw * s.hw, rng);
        d.w = randvec(std::size_t{s.out_ch} * s.in_ch * s.k * s.k, rng);
        d.b = randvec(s.out_ch, rng);
        const std::uint32_t o = (s.hw + 2 * s.pad - s.k) / s.stride + 1;
        d.plane = std::size_t{o} * o;
        d.first.resize(s.out_ch * d.plane);
        locus::backend::conv2d_vulkan(d.in, s.in_ch, s.hw, s.hw, d.w,
                                      s.out_ch, s.k, s.k, d.b, s.stride,
                                      s.pad, 1, d.first);
        data.push_back(std::move(d));
    }
    int bad_runs = 0;
    for (int it = 1; it < iters; ++it) {
        for (std::size_t si = 0; si < std::size(shapes); ++si) {
            const Shape& s = shapes[si];
            const Data& d = data[si];
            std::vector<float> got(d.first.size());
            locus::backend::conv2d_vulkan(d.in, s.in_ch, s.hw, s.hw, d.w,
                                          s.out_ch, s.k, s.k, d.b, s.stride,
                                          s.pad, 1, got);
            std::size_t n = 0, lo = got.size(), hi = 0;
            for (std::size_t i = 0; i < got.size(); ++i) {
                if (fbits(got[i]) != fbits(d.first[i])) {
                    ++n;
                    lo = std::min(lo, i);
                    hi = i;
                }
            }
            if (n > 0) {
                ++bad_runs;
                WARN("conv " << s.in_ch << " -> " << s.out_ch << " k "
                             << s.k << " at " << s.hw << ": round " << it
                             << ": " << n << " of " << got.size()
                             << " outputs differ, channels "
                             << lo / d.plane << " .. " << hi / d.plane
                             << ", pixels " << lo % d.plane << " .. "
                             << hi % d.plane);
            }
        }
    }
    CHECK(bad_runs == 0);
}
