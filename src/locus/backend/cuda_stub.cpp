// Fallback definitions of the CUDA backend entry points, compiled
// only when the build has no CUDA toolkit (LOCUS_HAS_CUDA unset).
// Keeps registry.cpp linking on non-CUDA hosts: matvec delegates to
// the scalar reference and the backend reports itself unusable.

#include "locus/backend/variants.hpp"

#ifndef LOCUS_HAS_CUDA

#include <stdexcept>
#include <string>

namespace locus::backend {

void matvec_cuda(const Mat& w, std::span<const float> x,
                 std::span<float> out) {
    matvec(w, x, out);
}

void matvec_cuda_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out) {
    matvec_q8k(w, x, out);
}

void cuda_prefetch(const Mat&) {}

void matvec_batch_cuda(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch, std::uint32_t n) {
    matvec_batch_scalar(w, x_batch, out_batch, n);
}

bool cuda_backend_usable() { return false; }

void cuda_set_device(int ordinal) {
    // Fail loud: a GPU executor was requested on a build with no CUDA
    // toolkit. Never silently fall back to the CPU (i#24 inc 4).
    throw std::runtime_error(
        "cannot select CUDA device " + std::to_string(ordinal) +
        ": this is a non-CUDA build");
}

void cuda_pool_reset() {}

}  // namespace locus::backend

#endif  // !LOCUS_HAS_CUDA
