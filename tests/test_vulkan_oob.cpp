// On-device out-of-bounds proof for the F2 fix (exact-byte Q8_0
// weight uploads rounded up to 16 with a zeroed tail). The Khronos
// validation layer with GPU-assisted validation bounds-checks every
// shader buffer access on the device, so an out-of-range SSBO read
// (a whole-word Q8_0 read past an under-sized weight buffer) is
// reported as a validation error.
//
// These cases need the layer enabled at instance creation, so they
// run only when LOCUS_VK_VALIDATION is set (the ctest target
// "vulkan_oob_validation" sets it); otherwise they skip. They are
// hidden ([.]) so the default suite does not pull them in.

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/backend/vulkan/limits.hpp"

namespace vk = locus::backend::vk;
using locus::backend::vk::Kernel;
using locus::backend::vk::VulkanContext;

namespace {

/**
 * Random GGUF Q8_0 weight (34-byte blocks: f16 scale then 32 int8
 * quants) for rows x cols. rows x (cols / 32) odd makes the total
 * byte count 2 (mod 4), so the last block's final whole-word read
 * ends past the tensor: the F2 boundary shape.
 */
std::vector<std::byte> random_q8(std::uint32_t rows, std::uint32_t cols,
                                 std::mt19937& rng) {
    const std::size_t blocks =
        std::size_t{rows} * (cols / 32);
    std::vector<std::byte> w(blocks * 34);
    std::uniform_int_distribution<int> qd(-127, 127);
    for (std::size_t b = 0; b < blocks; ++b) {
        std::byte* blk = w.data() + b * 34;
        const std::uint16_t d = locus::backend::f32_to_f16(0.05f);
        std::memcpy(blk, &d, 2);
        for (int i = 0; i < 32; ++i) {
            blk[2 + i] = static_cast<std::byte>(
                static_cast<std::int8_t>(qd(rng)));
        }
    }
    return w;
}

bool validation_on() {
    return std::getenv("LOCUS_VK_VALIDATION") != nullptr;
}

}  // namespace

// Negative control: proves the harness has teeth on this device. A
// weight buffer far smaller than rows x cols forces the matvec_q8_0
// kernel to read past it; GPU-assisted validation must report it.
TEST_CASE("vulkan GPU-assisted validation catches an out-of-bounds "
          "buffer read",
          "[.][validation]") {
    if (!validation_on()) {
        SKIP("set LOCUS_VK_VALIDATION (ctest target "
             "vulkan_oob_validation) to run");
    }
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;

    const std::uint32_t rows = 7, cols = 96;  // wants 720 bytes
    auto wb = ctx.create_buffer(64);          // far too small
    auto xb = ctx.create_buffer(std::size_t{cols} * 4);
    auto ob = ctx.create_buffer(std::size_t{rows} * 4);
    std::vector<float> x(cols, 1.0f);
    ctx.write_buffer(xb, std::as_bytes(std::span<const float>(x)));

    const std::uint32_t before = VulkanContext::validation_errors();
    ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {wb, xb, ob};
    const std::uint32_t push[] = {
        rows, cols, 0, 0, 0, 0, std::bit_cast<std::uint32_t>(1.0f)};
    ctx.dispatch(Kernel::kMatvecQ8_0, bufs, push,
                 VulkanContext::matvec_groups(Kernel::kMatvecQ8_0, rows));
    ctx.end_batch();
    ctx.destroy_buffer(wb);
    ctx.destroy_buffer(xb);
    ctx.destroy_buffer(ob);

    // If this fails, GPU-assisted validation is not catching OOB SSBO
    // reads on this device, so the positive case below proves nothing.
    REQUIRE(VulkanContext::validation_errors() > before);
}

// F2 proof: a boundary-shape Q8_0 weight pushed through the
// production upload path (matvec_vulkan -> State::upload_rows,
// vk::gpu_alloc_bytes) and dispatched for real reports no
// out-of-bounds access, and the result matches the CPU reference.
TEST_CASE("vulkan Q8_0 matvec at a 2-mod-4 boundary shape has no "
          "out-of-bounds access (F2)",
          "[.][validation]") {
    if (!validation_on()) {
        SKIP("set LOCUS_VK_VALIDATION (ctest target "
             "vulkan_oob_validation) to run");
    }
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    // 7 x 96 = 21 Q8_0 blocks = 714 bytes; 714 = 2 (mod 4), the shape
    // whose final whole-word read ends past the tensor. gpu_alloc_bytes
    // rounds the buffer to 720 with a zeroed tail, so the read is in
    // bounds.
    const std::uint32_t rows = 7, cols = 96;
    REQUIRE(vk::gpu_alloc_bytes(std::size_t{rows} * (cols / 32) * 34) ==
            720);

    std::mt19937 rng(23);
    const auto w = random_q8(rows, cols, rng);
    std::vector<float> x(cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) {
        v = dist(rng);
    }

    locus::backend::Mat m{locus::gguf::TensorType::kQ8_0, w.data(), rows,
                          cols};
    std::vector<float> cpu(rows), gpu(rows);
    locus::backend::matvec(m, x, cpu);

    locus::backend::vulkan_reset_state();
    const std::uint32_t before = VulkanContext::validation_errors();
    locus::backend::matvec_vulkan(m, x, gpu);  // production path
    REQUIRE(VulkanContext::validation_errors() == before);

    for (std::uint32_t r = 0; r < rows; ++r) {
        REQUIRE(gpu[r] == Catch::Approx(cpu[r]).margin(1e-3));
    }
}
