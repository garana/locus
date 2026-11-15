#pragma once

// Host helpers the CUDA translation units share: cuda_matvec.cu defines
// the functions, cuda_conv.cu and cuda_attend.cu use them.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "locus/backend/gpu_metrics.hpp"

namespace locus::backend::cuda_detail {

/** @returns Whether a CUDA call succeeded; callers fall back to the CPU
 *  path when it did not. */
bool cuda_ok(cudaError_t e);

/**
 * Binds the calling thread to the process-selected CUDA device if one
 * was chosen (cuda_set_device). Cheap (a thread-local set) and
 * idempotent; called at the top of every CUDA entry so whichever thread
 * runs the forward lands on the selected device, and so the device set
 * precedes the first weight-pool cudaMalloc by construction.
 */
void ensure_device();

/**
 * cudaDeviceSynchronize after a launch, charging the wait to the CUDA
 * metrics (i#285): one submit and the host time until the device is
 * done. A call may launch more than one kernel, so dispatches stays 0.
 *
 * @returns Whether the sync succeeded.
 */
bool sync_counted();

/** RAII-ish scoped device buffer; frees on scope exit. */
struct DevBuf {
    void* p = nullptr;
    std::size_t bytes_ = 0;
    bool alloc(std::size_t bytes) {
        if (!cuda_ok(cudaMalloc(&p, bytes))) {
            return false;
        }
        bytes_ = bytes;
        gpu_metrics(GpuApi::kCuda)
            .add_bytes(MemKind::kScratch, static_cast<std::int64_t>(bytes));
        return true;
    }
    ~DevBuf() {
        if (p != nullptr) {
            cudaFree(p);
            gpu_metrics(GpuApi::kCuda)
                .add_bytes(MemKind::kScratch,
                           -static_cast<std::int64_t>(bytes_));
        }
    }
};

}  // namespace locus::backend::cuda_detail
