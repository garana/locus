// Fallback definitions of the CUDA backend entry points, compiled
// only when the build has no CUDA toolkit (LOCUS_HAS_CUDA unset).
// Keeps registry.cpp linking on non-CUDA hosts: matvec delegates to
// the scalar reference and the backend reports itself unusable.

#include "locus/backend/variants.hpp"

#ifndef LOCUS_HAS_CUDA

namespace locus::backend {

void matvec_cuda(const Mat& w, std::span<const float> x,
                 std::span<float> out) {
    matvec(w, x, out);
}

void cuda_prefetch(const Mat&) {}

void matvec_batch_cuda(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch, std::uint32_t n) {
    matvec_batch_scalar(w, x_batch, out_batch, n);
}

bool cuda_backend_usable() { return false; }

}  // namespace locus::backend

#endif  // !LOCUS_HAS_CUDA
