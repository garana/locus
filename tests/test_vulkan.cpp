#include <random>
#include <vector>

#include "catch_amalgamated.hpp"
#include "cppllm/backend/cpu_ops.hpp"
#include "cppllm/backend/vulkan/context.hpp"

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

TEST_CASE("vulkan matvec validates sizes", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::vector<float> w(6), x(3), out(1);  // out too small
    REQUIRE_THROWS_AS(ctx.matvec_f32(w, 2, 3, x, out),
                      std::runtime_error);
}
