#include <bit>
#include <cstdint>
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
    auto b = ctx.create_buffer(v.size() * sizeof(T));
    ctx.write_buffer(b, std::as_bytes(std::span<const T>(v)));
    return b;
}

}  // namespace

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
