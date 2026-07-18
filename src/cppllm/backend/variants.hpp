#pragma once

#include <span>

#include "cppllm/backend/cpu_ops.hpp"

namespace cppllm::backend {

#if defined(__aarch64__)
/**
 * NEON-vectorized matvec: F32 and Q8_0 rows use NEON inner loops;
 * other weight types delegate to the scalar reference.
 */
void matvec_neon(const Mat& w, std::span<const float> x,
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

}  // namespace cppllm::backend
