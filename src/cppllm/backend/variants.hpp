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

}  // namespace cppllm::backend
