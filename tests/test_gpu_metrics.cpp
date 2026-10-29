#include <cstdint>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/gpu_metrics.hpp"
#include "locus/backend/vulkan/context.hpp"

using locus::backend::GpuApi;
using locus::backend::GpuMetrics;
using locus::backend::MemKind;
using locus::backend::gpu_metrics;
using locus::backend::vk::VulkanContext;

TEST_CASE("gpu metrics names and byte gauges", "[metrics]") {
    REQUIRE(std::string(locus::backend::mem_kind_name(MemKind::kKv)) ==
            "kv");
    REQUIRE(std::string(locus::backend::mem_kind_name(MemKind::kMapped)) ==
            "mapped");
    REQUIRE(std::string(locus::backend::gpu_api_name(GpuApi::kCuda)) ==
            "cuda");
    // The counters are process-wide, so check deltas.
    GpuMetrics& gm = gpu_metrics(GpuApi::kCuda);
    const std::int64_t before = gm.held(MemKind::kTable);
    gm.add_bytes(MemKind::kTable, 4096);
    REQUIRE(gm.held(MemKind::kTable) == before + 4096);
    gm.add_bytes(MemKind::kTable, -4096);
    REQUIRE(gm.held(MemKind::kTable) == before);
}

// A Vulkan buffer counts toward its kind until destroyed, and a batch
// adds one submit, its kernels, its host times and (with timestamps)
// device time.
TEST_CASE("vulkan buffers and batches feed the gpu metrics",
          "[metrics][vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    GpuMetrics& gm = gpu_metrics(GpuApi::kVulkan);

    const std::int64_t kv0 = gm.held(MemKind::kKv);
    auto b = ctx.create_buffer(1 << 20, MemKind::kKv);
    REQUIRE(gm.held(MemKind::kKv) >= kv0 + (1 << 20));
    ctx.destroy_buffer(b);
    REQUIRE(gm.held(MemKind::kKv) == kv0);

    const std::uint64_t submits0 = gm.submits.load();
    const std::uint64_t dispatches0 = gm.dispatches.load();
    const std::uint64_t wait0 = gm.wait_ns.load();
    const std::uint64_t device0 = gm.device_ns.load();
    const std::int64_t scratch0 = gm.held(MemKind::kScratch);
    const std::uint32_t rows = 256, cols = 1024;
    std::vector<float> w(static_cast<std::size_t>(rows) * cols, 0.5f);
    std::vector<float> x(cols, 1.0f), out(rows);
    ctx.matvec_f32(w, rows, cols, x, out);
    REQUIRE(out[0] == 512.0f);
    REQUIRE(gm.submits.load() == submits0 + 1);
    REQUIRE(gm.dispatches.load() >= dispatches0 + 1);
    REQUIRE(gm.wait_ns.load() > wait0);
    if (ctx.timestamps()) {
        REQUIRE(gm.device_ns.load() > device0);
    }
    // matvec_f32's buffers are scratch and freed on return.
    REQUIRE(gm.held(MemKind::kScratch) == scratch0);
}
