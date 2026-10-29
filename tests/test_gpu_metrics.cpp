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
// adds one submit, its kernels and its host times. Device time comes
// from about 1 batch in 16 (the pick is seeded, so this is
// deterministic per context).
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
    const std::uint64_t samples0 = gm.device_samples.load();
    const std::int64_t scratch0 = gm.held(MemKind::kScratch);
    const std::uint32_t rows = 256, cols = 1024;
    std::vector<float> w(static_cast<std::size_t>(rows) * cols, 0.5f);
    std::vector<float> x(cols, 1.0f), out(rows);
    ctx.matvec_f32(w, rows, cols, x, out);
    REQUIRE(out[0] == 512.0f);
    REQUIRE(gm.submits.load() == submits0 + 1);
    REQUIRE(gm.dispatches.load() >= dispatches0 + 1);
    REQUIRE(gm.wait_ns.load() > wait0);
    // matvec_f32's buffers are scratch and freed on return.
    REQUIRE(gm.held(MemKind::kScratch) == scratch0);
    if (ctx.timestamps()) {
        int n = 1;
        while (gm.device_ns.load() == device0 && n < 256) {
            ctx.matvec_f32(w, rows, cols, x, out);
            ++n;
        }
        INFO(n << " batches");
        REQUIRE(gm.device_ns.load() > device0);
        REQUIRE(gm.device_samples.load() == samples0 + 1);
        // Sampled, not every batch: under LOCUS_VK_PROFILE every one.
        if (!VulkanContext::profiling()) {
            REQUIRE(n > 1);
        }
    }
}
