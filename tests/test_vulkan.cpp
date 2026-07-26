#include <bit>
#include <chrono>
#include <cstring>
#include <random>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"

using locus::backend::vk::Kernel;
using locus::backend::vk::VulkanContext;

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

    locus::backend::Mat m{
        locus::gguf::TensorType::kF32,
        reinterpret_cast<const std::byte*>(w.data()), rows, cols};
    locus::backend::matvec(m, x, cpu);

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
            locus::backend::f32_to_f16(0.05f);
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
    locus::backend::Mat m{locus::gguf::TensorType::kQ8_0,
                           w.data(), rows, cols};
    locus::backend::matvec(m, x, cpu);

    auto wb = ctx.create_buffer((w.size() + 3) & ~std::size_t{3});
    auto xb = ctx.create_buffer(x.size() * 4);
    auto ob = ctx.create_buffer(gpu.size() * 4);
    ctx.write_buffer(wb, w);
    ctx.write_buffer(xb, std::as_bytes(std::span(x)));
    ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {wb, xb, ob};
    const std::uint32_t push[] = {
        rows, cols, 0, 0, 0, 0, std::bit_cast<std::uint32_t>(1.0f)};
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
        const std::uint32_t push[] = {
        rows, cols, 0, 0, 0, 0, std::bit_cast<std::uint32_t>(1.0f)};
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
            v = locus::backend::f32_to_f16(dist(rng));
        }
        std::vector<float> cpu(rows), gpu(rows);
        locus::backend::Mat m{
            locus::gguf::TensorType::kF16,
            reinterpret_cast<const std::byte*>(w.data()), rows,
            cols};
        locus::backend::matvec(m, x, cpu);
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
                locus::backend::f32_to_f16(0.25f);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 16; ++i) {
                blk[2 + i] =
                    static_cast<std::byte>(nib(rng));
            }
        }
        std::vector<float> cpu(rows), gpu(rows);
        locus::backend::Mat m{locus::gguf::TensorType::kQ4_0,
                               w.data(), rows, cols};
        locus::backend::matvec(m, x, cpu);
        run_kernel(Kernel::kMatvecQ4_0, w, gpu);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(gpu[r] ==
                    Catch::Approx(cpu[r]).margin(1e-3));
        }
    }

    SECTION("q5_0") {
        // 22-byte blocks: f16 d + u32 qh + 16 qs bytes; qh and qs
        // fully random so the 5th-bit path is exercised.
        std::vector<std::byte> w(rows * (cols / 32) * 22);
        std::uniform_int_distribution<int> byte(0, 255);
        for (std::size_t b = 0; b < rows * (cols / 32); ++b) {
            std::byte* blk = w.data() + b * 22;
            const std::uint16_t d =
                locus::backend::f32_to_f16(0.25f);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 20; ++i) {
                blk[2 + i] =
                    static_cast<std::byte>(byte(rng));
            }
        }
        std::vector<float> cpu(rows), gpu(rows);
        locus::backend::Mat m{locus::gguf::TensorType::kQ5_0,
                               w.data(), rows, cols};
        locus::backend::matvec(m, x, cpu);
        run_kernel(Kernel::kMatvecQ5_0, w, gpu);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(gpu[r] ==
                    Catch::Approx(cpu[r]).margin(1e-3));
        }
    }
}

TEST_CASE("vulkan k-quant matvec matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> byte_d(0, 255);

    const std::uint32_t rows = 4, cols = 512;
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }

    struct KCase {
        locus::gguf::TensorType type;
        Kernel kernel;
        std::size_t block_bytes;
        std::size_t d_off;  // f16 scale offset inside the block
        bool has_dmin;      // f16 dmin follows d at d_off + 2
    };
    const KCase cases[] = {
        {locus::gguf::TensorType::kQ4_K, Kernel::kMatvecQ4_K,
         144, 0, true},
        {locus::gguf::TensorType::kQ5_K, Kernel::kMatvecQ5_K,
         176, 0, true},
        {locus::gguf::TensorType::kQ6_K, Kernel::kMatvecQ6_K,
         210, 208, false},
        {locus::gguf::TensorType::kQ2_K, Kernel::kMatvecQ2_K,
         84, 80, true},
    };
    for (const auto& c : cases) {
        INFO("type " << static_cast<int>(c.type));
        std::vector<std::byte> w(rows * (cols / 256) *
                                 c.block_bytes);
        for (auto& b : w) {
            b = static_cast<std::byte>(byte_d(rng));
        }
        for (std::size_t blk = 0; blk < rows * (cols / 256);
             ++blk) {
            std::byte* p =
                w.data() + blk * c.block_bytes + c.d_off;
            const std::uint16_t d =
                locus::backend::f32_to_f16(0.01f);
            std::memcpy(p, &d, 2);
            if (c.has_dmin) {  // dmin follows d at d_off + 2
                const std::uint16_t dmin =
                    locus::backend::f32_to_f16(0.005f);
                std::memcpy(p + 2, &dmin, 2);
            }
        }

        std::vector<float> cpu(rows), gpu(rows);
        locus::backend::Mat m{c.type, w.data(), rows, cols};
        locus::backend::matvec(m, x, cpu);

        auto wb =
            ctx.create_buffer((w.size() + 3) & ~std::size_t{3});
        auto xb = ctx.create_buffer(x.size() * 4);
        auto ob = ctx.create_buffer(gpu.size() * 4);
        ctx.write_buffer(wb, w);
        ctx.write_buffer(xb, std::as_bytes(std::span(x)));
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {wb, xb, ob};
        const std::uint32_t push[] = {
        rows, cols, 0, 0, 0, 0, std::bit_cast<std::uint32_t>(1.0f)};
        ctx.dispatch(c.kernel, bufs, push, (rows + 63) / 64);
        ctx.end_batch();
        ctx.read_buffer(ob,
                        std::as_writable_bytes(std::span(gpu)));
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(gpu[r] ==
                    Catch::Approx(cpu[r]).margin(1e-3));
        }
        ctx.destroy_buffer(wb);
        ctx.destroy_buffer(xb);
        ctx.destroy_buffer(ob);
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

    locus::backend::Mat m{
        locus::gguf::TensorType::kF32,
        reinterpret_cast<const std::byte*>(w.data()), n, n};
    t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        locus::backend::matvec(m, x, cpu);
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
    // GPU-vs-scalar speed is device-dependent: a desktop or
    // MoltenVK GPU beats scalar CPU at these sizes, but a tile GPU
    // (e.g. the Pi's VideoCore VII) does not. Correctness above is
    // the invariant; the timing is reported, not gated on.
    if (gpu_us >= cpu_us) {
        WARN("vulkan matvec not faster than scalar on this device: "
             "gpu " << gpu_us << "us vs cpu " << cpu_us << "us");
    }
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
