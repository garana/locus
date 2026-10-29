#include "locus/backend/gpu_metrics.hpp"

namespace locus::backend {

const char* mem_kind_name(MemKind k) {
    switch (k) {
        case MemKind::kWeights: return "weights";
        case MemKind::kMapped: return "mapped";
        case MemKind::kKv: return "kv";
        case MemKind::kScratch: return "scratch";
        case MemKind::kTable: return "table";
        case MemKind::kCount: break;
    }
    return "other";
}

const char* gpu_api_name(GpuApi a) {
    switch (a) {
        case GpuApi::kVulkan: return "vulkan";
        case GpuApi::kCuda: return "cuda";
        case GpuApi::kCount: break;
    }
    return "other";
}

GpuMetrics& gpu_metrics(GpuApi a) {
    static GpuMetrics m[static_cast<int>(GpuApi::kCount)];
    return m[static_cast<int>(a)];
}

}  // namespace locus::backend
