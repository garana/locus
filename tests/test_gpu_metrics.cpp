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

// Stage marks split a staged batch's device time (i#391): each mark
// takes the time since the previous one, the rest goes to unmarked_ns,
// and the parts add up to that batch's device time, scaled by
// kStageEvery (a staged batch stands for that many timed ones). Stage
// 1 runs 8 large kernels,
// stage 2 one tiny kernel, and the unmarked rest 4 large ones, so a
// mark that does not wait for the kernels before it (MoltenVK without
// a barrier) shows as stage 1 taking the rest's time.
TEST_CASE("vulkan stage marks split the device time", "[metrics][vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    if (!ctx.timestamps()) {
        SKIP("the queue has no GPU timestamps");
    }
    GpuMetrics& gm = gpu_metrics(GpuApi::kVulkan);
    const std::uint32_t big = 1u << 22;
    auto a = ctx.create_buffer(std::size_t{big} * 4, MemKind::kScratch);
    auto b = ctx.create_buffer(std::size_t{big} * 4, MemKind::kScratch);
    const VulkanContext::Buffer bufs[] = {a, b, a};
    const auto silu = [&](std::uint32_t n, int times) {
        const std::uint32_t push[] = {n};
        for (int i = 0; i < times; ++i) {
            ctx.dispatch(locus::backend::vk::Kernel::kSiluMul, bufs, push,
                         (n + 63) / 64);
        }
    };
    const auto split = [&] {
        std::uint64_t sum = gm.unmarked_ns.load();
        for (const auto& s : gm.stage_ns) sum += s.load();
        return sum;
    };
    const std::uint64_t split0 = split();
    const std::uint64_t one0 = gm.stage_ns[1].load();
    const std::uint64_t two0 = gm.stage_ns[2].load();
    const std::uint64_t rest0 = gm.unmarked_ns.load();
    const std::uint64_t samples0 = gm.stage_samples.load();
    std::uint64_t device0 = 0;  // before the last batch
    int tries = 0;
    while (gm.stage_samples.load() == samples0 && tries < 1024) {
        device0 = gm.device_ns.load();
        ctx.begin_batch();
        silu(big, 8);
        ctx.mark_stage(1);
        silu(64, 1);
        ctx.mark_stage(2);
        silu(big, 4);  // after the last mark: unmarked
        ctx.end_batch();
        ++tries;
    }
    ctx.destroy_buffer(a);
    ctx.destroy_buffer(b);
    INFO(tries << " batches");
    REQUIRE(gm.stage_samples.load() == samples0 + 1);
    const std::uint64_t one = gm.stage_ns[1].load() - one0;
    const std::uint64_t two = gm.stage_ns[2].load() - two0;
    const std::uint64_t rest = gm.unmarked_ns.load() - rest0;
    INFO("stage 1 " << one << " ns, stage 2 " << two << " ns, rest " << rest
                    << " ns");
    REQUIRE(two > 0);
    REQUIRE(one > rest);
    REQUIRE(rest > two);
    // Each sum truncates to whole nanoseconds.
    const std::int64_t ratio =
        VulkanContext::profiling() ? 1 : VulkanContext::kStageEvery;
    const auto device = ratio * static_cast<std::int64_t>(
                                    gm.device_ns.load() - device0);
    const auto parts = static_cast<std::int64_t>(split() - split0);
    REQUIRE(parts > 0);
    REQUIRE(device - parts <= 4 * ratio);
    REQUIRE(parts - device <= 4 * ratio);
}
