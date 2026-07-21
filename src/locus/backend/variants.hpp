#pragma once

#include <span>

#include "locus/backend/cpu_ops.hpp"

namespace locus::backend {

#if defined(__aarch64__)
/**
 * NEON-vectorized matvec: F32 and Q8_0 rows use NEON inner loops;
 * other weight types delegate to the scalar reference.
 */
void matvec_neon(const Mat& w, std::span<const float> x,
                 std::span<float> out);
#endif

#if defined(__x86_64__)
/**
 * SSE4-vectorized matvec: F32 and Q8_0 rows use 128-bit SSE4.1
 * inner loops; other weight types delegate to the scalar
 * reference. Compiled only on x86-64 hosts (per-source -msse4.1).
 * The baseline x86 vector path for CPUs without AVX2.
 */
void matvec_sse4(const Mat& w, std::span<const float> x,
                 std::span<float> out);

/**
 * AVX2-vectorized matvec: F32 and Q8_0 rows use AVX2 inner
 * loops; other weight types delegate to the scalar reference.
 * Compiled only on x86-64 hosts (per-source -mavx2).
 */
void matvec_avx2(const Mat& w, std::span<const float> x,
                 std::span<float> out);
#endif

/**
 * Vulkan matvec: F32 weights run on the GPU (uploaded once,
 * resident); other types delegate to the scalar reference.
 * Attention and norms stay on the CPU (hybrid backend).
 */
void matvec_vulkan(const Mat& w, std::span<const float> x,
                   std::span<float> out);

/** @returns true when a usable Vulkan device + kernels exist. */
bool vulkan_backend_usable();

/** KV-pool allocator returning a GPU-mapped pointer (Ops hook). */
float* vulkan_alloc_kv(std::size_t n_floats);

/**
 * CUDA matvec: F32 and Q8_0 weights run on an NVIDIA GPU (streamed
 * per call for now); other types delegate to the scalar reference.
 * When the build has no CUDA toolkit a stub delegates entirely to
 * scalar and reports the backend unusable.
 */
void matvec_cuda(const Mat& w, std::span<const float> x,
                 std::span<float> out);

/** @returns true when a usable CUDA device is present. */
bool cuda_backend_usable();

}  // namespace locus::backend
