// Fallback definitions of the CUDA backend entry points, compiled
// only when the build has no CUDA toolkit (LOCUS_HAS_CUDA unset).
// Keeps registry.cpp linking on non-CUDA hosts: matvec delegates to
// the scalar reference and the backend reports itself unusable.

#include "locus/backend/registry.hpp"
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

void cuda_evict(const Mat&) {}

std::size_t cuda_pool_budget() { return 0; }

void* cuda_kv_mirror_alloc(std::size_t) { return nullptr; }

bool cuda_kv_mirror_store(void*, std::size_t, const float*, std::size_t) {
    return false;
}

bool cuda_kv_mirror_load(const void*, std::size_t, float*, std::size_t) {
    return false;
}

void cuda_kv_mirror_free(void*) {}

bool cuda_kv_attend(const KvAttend&) { return false; }

void matvec_batch_cuda(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch, std::uint32_t n) {
    matvec_batch_scalar(w, x_batch, out_batch, n);
}

bool cuda_backend_usable() { return false; }

std::string cuda_device_name() { return ""; }

void cuda_set_device(int ordinal) {
    // Fail loud: a GPU executor was requested on a build with no CUDA
    // toolkit. Never silently fall back to the CPU (i#24 inc 4).
    throw std::runtime_error(
        "cannot select CUDA device " + std::to_string(ordinal) +
        ": this is a non-CUDA build");
}

void cuda_pool_reset() {}

void conv1d_cuda(std::span<const float> in, std::uint32_t in_ch,
                 std::uint32_t in_len, std::span<const float> weight,
                 std::uint32_t out_ch, std::uint32_t k,
                 std::span<const float> bias, std::uint32_t stride,
                 std::uint32_t dilation, std::uint32_t pad,
                 std::uint32_t groups, std::span<float> out) {
    conv1d(in, in_ch, in_len, weight, out_ch, k, bias, stride, dilation,
           pad, groups, out);
}

void conv1d_transpose_cuda(std::span<const float> in,
                           std::uint32_t in_ch, std::uint32_t in_len,
                           std::span<const float> weight,
                           std::uint32_t out_ch, std::uint32_t k,
                           std::span<const float> bias,
                           std::uint32_t stride, std::uint32_t dilation,
                           std::uint32_t pad, std::uint32_t groups,
                           std::span<float> out) {
    conv1d_transpose(in, in_ch, in_len, weight, out_ch, k, bias, stride,
                     dilation, pad, groups, out);
}

void snake1d_cuda(std::span<float> x, std::uint32_t ch,
                  std::uint32_t len, std::span<const float> alpha,
                  float eps) {
    snake1d(x, ch, len, alpha, eps);
}

void conv2d_cuda(std::span<const float> in, std::uint32_t in_ch,
                 std::uint32_t in_h, std::uint32_t in_w,
                 std::span<const float> weight, std::uint32_t out_ch,
                 std::uint32_t kh, std::uint32_t kw,
                 std::span<const float> bias, std::uint32_t stride,
                 std::uint32_t pad, std::uint32_t groups,
                 std::span<float> out) {
    conv2d_check_args(in, in_ch, in_h, in_w, weight, out_ch, kh, kw, bias,
                      stride, pad, groups, out);
    conv2d(in, in_ch, in_h, in_w, weight, out_ch, kh, kw, bias, stride, pad,
           groups, out);
}

}  // namespace locus::backend

#endif  // !LOCUS_HAS_CUDA
