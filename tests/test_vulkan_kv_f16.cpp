// i#133: the CPU f16 KV encoder (kv_quantize_row, KvType::kF16) against
// the GPU's packHalf2x16 (kv_scatter_f16), bit for bit, on values that
// stress round-to-nearest-even: every exact subnormal tie, normal ties
// across the exponent range, the subnormal and max boundaries, and
// both zeros. A mismatch is either a driver that does not round to
// nearest even or a regression in the CPU encoder.

#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/kv/kv_quant.hpp"

using locus::backend::vk::Kernel;
using locus::backend::vk::VulkanContext;

namespace {

/** The f32 values the comparison covers (an even count). */
std::vector<float> stress_values() {
    std::vector<float> v;
    // Every exact subnormal tie: (k + 0.5) * 2^-24, k = 0 .. 1023
    // (k = 1023 is the tie between the largest subnormal and the
    // smallest normal).
    for (int k = 0; k < 1024; ++k) {
        v.push_back(std::ldexp(static_cast<float>(k) + 0.5f, -24));
    }
    // Normal ties, (1024 + m + 0.5) * 2^(e - 10): odd and even m, so
    // half round up and half down, at every f16 exponent.
    for (int e = -14; e <= 15; ++e) {
        for (int m : {0, 1, 2, 511, 512, 1021, 1022}) {
            v.push_back(std::ldexp(1024.0f + m + 0.5f, e - 10));
        }
    }
    // The subnormal boundaries: the smallest subnormal and its half
    // (a tie to zero), just above that half, the smallest normal and
    // the float just below it.
    const float min_sub = std::ldexp(1.0f, -24);
    const float min_norm = std::ldexp(1.0f, -14);
    for (float x : {min_sub, min_sub / 2, std::nextafter(min_sub / 2, 1.0f),
                    min_norm, std::nextafter(min_norm, 0.0f)}) {
        v.push_back(x);
    }
    // The max: 65504, the largest float that still rounds to it, and
    // the float above that.
    for (float x : {65504.0f, 65519.0f, std::nextafter(65519.0f, 0.0f)}) {
        v.push_back(x);
    }
    v.push_back(0.0f);
    v.push_back(-0.0f);
    // The negatives of everything so far.
    const std::size_t n = v.size();
    for (std::size_t i = 0; i < n; ++i) {
        v.push_back(-v[i]);
    }
    if (v.size() % 2 != 0) {
        v.push_back(1.0f);
    }
    return v;
}

}  // namespace

TEST_CASE("CPU f16 KV encoding matches the GPU's packHalf2x16 bit for bit",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    const std::vector<float> x = stress_values();
    const auto n = static_cast<std::uint32_t>(x.size());

    std::vector<std::uint16_t> cpu(n);
    locus::kv::kv_quantize_row(x.data(), n, cpu.data(),
                               locus::kv::KvType::kF16);

    VulkanContext ctx;
    auto src = ctx.create_buffer(std::size_t{n} * 4);
    ctx.write_buffer(src, std::as_bytes(std::span(x)));
    auto pool = ctx.create_buffer(std::size_t{n} * 2);
    const std::uint32_t table0 = 0;
    auto table = ctx.create_buffer(4);
    ctx.write_buffer(table, std::as_bytes(std::span(&table0, 1)));
    // One position of one block, n halves wide.
    const std::uint32_t push[] = {n, 0, 1, n, n, 0, 0, n, 0, 0};
    const VulkanContext::Buffer bufs[] = {src, pool, table};
    ctx.begin_batch();
    ctx.dispatch(Kernel::kKvScatterF16, bufs, push, 1, 1, 1);
    ctx.end_batch();
    std::vector<std::uint16_t> gpu(n);
    ctx.read_buffer(pool, std::as_writable_bytes(std::span(gpu)));
    for (auto b : {src, pool, table}) {
        ctx.destroy_buffer(b);
    }

    std::uint32_t bad = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (cpu[i] != gpu[i]) {
            if (++bad <= 8) {
                UNSCOPED_INFO("x = " << x[i] << " (bits 0x" << std::hex
                                     << std::bit_cast<std::uint32_t>(x[i])
                                     << "): CPU 0x" << cpu[i] << ", GPU 0x"
                                     << gpu[i] << std::dec);
            }
        }
    }
    INFO(bad << " of " << n << " values differ on " << ctx.device_name());
    CHECK(bad == 0);
}
