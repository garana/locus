#pragma once

#include <atomic>
#include <cstdint>

namespace locus::backend {

/** What a GPU allocation holds, for the memory-by-kind metrics. */
enum class MemKind : std::uint8_t {
    kWeights,  ///< weights and other model data copied to the device
    kMapped,   ///< weights the device reads in place from the file map
    kKv,       ///< the KV cache
    kScratch,  ///< activations and other per-step buffers
    kTable,    ///< constant lookup tables (quant grids, ones)
    kCount
};

/** @returns The kind's metric label ("weights", "mapped", ...). */
const char* mem_kind_name(MemKind k);

/** The GPU APIs that keep counters. */
enum class GpuApi : std::uint8_t { kVulkan, kCuda, kCount };

/** @returns The API's metric label ("vulkan" or "cuda"). */
const char* gpu_api_name(GpuApi a);

/**
 * Process-wide GPU counters for one API (i#285), always on. The backend
 * writes them as relaxed atomics where it already reads the clock or
 * allocates, so /metrics reads them without the GPU lock.
 *
 * - submits: command batches (Vulkan) or synchronizing calls (CUDA).
 * - dispatches: kernels recorded (Vulkan only).
 * - record_ns: host time spent recording a batch, from its begin to its
 *   submit (Vulkan only).
 * - wait_ns: host time from the submit to the result being ready.
 * - device_ns: device time between a batch's first and last command, by
 *   GPU timestamps (Vulkan, when the queue has them); 0 elsewhere.
 * - bytes: device memory this process holds, per MemKind.
 */
struct GpuMetrics {
    std::atomic<std::uint64_t> submits{0};
    std::atomic<std::uint64_t> dispatches{0};
    std::atomic<std::uint64_t> record_ns{0};
    std::atomic<std::uint64_t> wait_ns{0};
    std::atomic<std::uint64_t> device_ns{0};
    std::atomic<std::int64_t> bytes[static_cast<int>(MemKind::kCount)] = {};

    /** Adds `n` (negative to free) to the bytes held for kind k. */
    void add_bytes(MemKind k, std::int64_t n) {
        bytes[static_cast<int>(k)].fetch_add(n, std::memory_order_relaxed);
    }

    /** @returns The bytes held for kind k. */
    std::int64_t held(MemKind k) const {
        return bytes[static_cast<int>(k)].load(std::memory_order_relaxed);
    }
};

/** @returns The process's counters for API a. */
GpuMetrics& gpu_metrics(GpuApi a);

}  // namespace locus::backend
