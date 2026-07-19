#include <chrono>
#include <cstring>
#include <random>
#include <vector>

#include "catch_amalgamated.hpp"
#include "cppllm/backend/cpu_ops.hpp"
#include "cppllm/backend/variants.hpp"
#include "cppllm/backend/vulkan/context.hpp"

using cppllm::backend::vk::Kernel;
using cppllm::backend::vk::VulkanContext;

TEST_CASE("vulkan matvec matches the CPU reference", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;

    const std::uint32_t rows = 96;
    const std::uint32_t cols = 192;
    std::vector<float> w(static_cast<std::size_t>(rows) * cols);
    std::vector<float> x(cols);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : w) {
        v = dist(rng);
    }
    for (auto& v : x) {
        v = dist(rng);
    }

    std::vector<float> gpu(rows), cpu(rows);
    ctx.matvec_f32(w, rows, cols, x, gpu);

    cppllm::backend::Mat m{
        cppllm::gguf::TensorType::kF32,
        reinterpret_cast<const std::byte*>(w.data()), rows, cols};
    cppllm::backend::matvec(m, x, cpu);

    for (std::uint32_t r = 0; r < rows; ++r) {
        REQUIRE(gpu[r] ==
                Catch::Approx(cpu[r]).margin(1e-4).epsilon(1e-4));
    }
}

TEST_CASE("vulkan q8_0 matvec matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;

    const std::uint32_t rows = 7, cols = 96;  // 3 blocks per row
    std::mt19937 rng(23);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::byte> w(rows * (cols / 32) * 34);
    for (std::size_t b = 0; b < rows * (cols / 32); ++b) {
        std::byte* blk = w.data() + b * 34;
        const std::uint16_t d =
            cppllm::backend::f32_to_f16(0.05f);
        std::memcpy(blk, &d, 2);
        for (int i = 0; i < 32; ++i) {
            blk[2 + i] = static_cast<std::byte>(
                static_cast<std::int8_t>(qd(rng)));
        }
    }
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }

    std::vector<float> cpu(rows), gpu(rows);
    cppllm::backend::Mat m{cppllm::gguf::TensorType::kQ8_0,
                           w.data(), rows, cols};
    cppllm::backend::matvec(m, x, cpu);

    auto wb = ctx.create_buffer((w.size() + 3) & ~std::size_t{3});
    auto xb = ctx.create_buffer(x.size() * 4);
    auto ob = ctx.create_buffer(gpu.size() * 4);
    ctx.write_buffer(wb, w);
    ctx.write_buffer(xb, std::as_bytes(std::span(x)));
    ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {wb, xb, ob};
    const std::uint32_t push[] = {rows, cols, 0, 0};
    ctx.dispatch(Kernel::kMatvecQ8_0, bufs, push,
                 (rows + 63) / 64);
    ctx.end_batch();
    ctx.read_buffer(ob, std::as_writable_bytes(std::span(gpu)));

    for (std::uint32_t r = 0; r < rows; ++r) {
        REQUIRE(gpu[r] == Catch::Approx(cpu[r]).margin(1e-3));
    }
    ctx.destroy_buffer(wb);
    ctx.destroy_buffer(xb);
    ctx.destroy_buffer(ob);
}

TEST_CASE("vulkan f16 and q4_0 matvec match the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::uint32_t rows = 5, cols = 64;
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    auto run_kernel = [&](Kernel k, std::span<const std::byte> w,
                          std::span<float> out) {
        auto wb =
            ctx.create_buffer((w.size() + 3) & ~std::size_t{3});
        auto xb = ctx.create_buffer(x.size() * 4);
        auto ob = ctx.create_buffer(out.size() * 4);
        ctx.write_buffer(wb, w);
        ctx.write_buffer(xb, std::as_bytes(std::span(x)));
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {wb, xb, ob};
        const std::uint32_t push[] = {rows, cols, 0, 0};
        ctx.dispatch(k, bufs, push, (rows + 63) / 64);
        ctx.end_batch();
        ctx.read_buffer(ob, std::as_writable_bytes(out));
        ctx.destroy_buffer(wb);
        ctx.destroy_buffer(xb);
        ctx.destroy_buffer(ob);
    };

    SECTION("f16") {
        std::vector<std::uint16_t> w(rows * cols);
        for (auto& v : w) {
            v = cppllm::backend::f32_to_f16(dist(rng));
        }
        std::vector<float> cpu(rows), gpu(rows);
        cppllm::backend::Mat m{
            cppllm::gguf::TensorType::kF16,
            reinterpret_cast<const std::byte*>(w.data()), rows,
            cols};
        cppllm::backend::matvec(m, x, cpu);
        run_kernel(Kernel::kMatvecF16,
                   std::as_bytes(std::span(w)), gpu);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(gpu[r] ==
                    Catch::Approx(cpu[r]).margin(1e-3));
        }
    }

    SECTION("q4_0") {
        std::vector<std::byte> w(rows * (cols / 32) * 18);
        std::uniform_int_distribution<int> nib(0, 255);
        for (std::size_t b = 0; b < rows * (cols / 32); ++b) {
            std::byte* blk = w.data() + b * 18;
            const std::uint16_t d =
                cppllm::backend::f32_to_f16(0.25f);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 16; ++i) {
                blk[2 + i] =
                    static_cast<std::byte>(nib(rng));
            }
        }
        std::vector<float> cpu(rows), gpu(rows);
        cppllm::backend::Mat m{cppllm::gguf::TensorType::kQ4_0,
                               w.data(), rows, cols};
        cppllm::backend::matvec(m, x, cpu);
        run_kernel(Kernel::kMatvecQ4_0, w, gpu);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(gpu[r] ==
                    Catch::Approx(cpu[r]).margin(1e-3));
        }
    }
}

TEST_CASE("gpu matvec beats scalar cpu at real-model sizes",
          "[vulkan][benchmark]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;

    const std::uint32_t n = 2048;  // a 7B-class weight matrix dim
    std::vector<float> w(static_cast<std::size_t>(n) * n);
    std::vector<float> x(n);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& v : w) {
        v = dist(rng);
    }
    for (auto& v : x) {
        v = dist(rng);
    }

    auto wb = ctx.create_buffer(w.size() * 4);
    auto xb = ctx.create_buffer(x.size() * 4);
    auto ob = ctx.create_buffer(x.size() * 4);
    ctx.write_buffer(wb, std::as_bytes(std::span(w)));
    ctx.write_buffer(xb, std::as_bytes(std::span(x)));

    using clock = std::chrono::steady_clock;
    const int iters = 32;

    std::vector<float> gpu(n), cpu(n);
    ctx.matvec_f32(wb, n, n, xb, ob);  // warm up
    auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        ctx.matvec_f32(wb, n, n, xb, ob);
    }
    auto gpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now() - t0)
            .count() /
        iters;
    ctx.read_buffer(ob, std::as_writable_bytes(std::span(gpu)));

    cppllm::backend::Mat m{
        cppllm::gguf::TensorType::kF32,
        reinterpret_cast<const std::byte*>(w.data()), n, n};
    t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        cppllm::backend::matvec(m, x, cpu);
    }
    auto cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now() - t0)
            .count() /
        iters;

    WARN("matvec 2048x2048: gpu " << gpu_us << "us, scalar cpu "
                                  << cpu_us << "us");
    for (std::uint32_t r = 0; r < n; r += 97) {
        REQUIRE(gpu[r] == Catch::Approx(cpu[r]).margin(1e-3));
    }
    // The exit criterion: GPU ahead of scalar CPU at sizes that
    // matter (checked loosely; machines vary).
    REQUIRE(gpu_us < cpu_us);
    ctx.destroy_buffer(wb);
    ctx.destroy_buffer(xb);
    ctx.destroy_buffer(ob);
}

TEST_CASE("vulkan matvec validates sizes", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::vector<float> w(6), x(3), out(1);  // out too small
    REQUIRE_THROWS_AS(ctx.matvec_f32(w, 2, 3, x, out),
                      std::runtime_error);
}
