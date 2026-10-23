// rope.comp variants against the CPU references: interleaved pairs
// (rope / rope_b, rope_norm_yarn) and NEOX pairs (rope_neox /
// rope_neox_b, rope_neox_yarn).

#include <algorithm>
#include <bit>
#include <random>
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
