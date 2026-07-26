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

/**
 * NEON register-blocked batched matvec (R11, Ops::matvec_batch):
 * applies w to all n token vectors in x_batch (token-major, token t
 * at t*w.cols) into out_batch (row-major, out[r*n + t]). Q4_K and
 * Q6_K rows are weight-stationary -- each weight block is read and
 * dequantized ONCE, then FMA'd into all n token accumulators, so
 * weight DRAM traffic drops ~n-fold vs n matvec() calls. Other
 * types fall back to per-token matvec_neon. Byte-identical to n
 * matvec_neon() calls (same per-block accumulation order).
 * Single-threaded: the model row-slices and calls this per slice.
 */
void matvec_batch_neon(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch,
                       std::uint32_t n);
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
 * SSE4 batched matvec (R11 Ops::matvec_batch): Q4_K/Q6_K use a
 * register-blocked kernel (weight block dequanted once, FMA'd into n
 * token accumulators -- byte-identical to n matvec_sse4() calls but
 * ~n-fold less weight traffic); other types delegate to
 * matvec_batch_scalar. out_batch is row-major (out[r*n + t]).
 */
void matvec_batch_sse4(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch, std::uint32_t n);

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
 * CUDA matvec: F32, Q8_0, Q4_K, Q5_K, Q6_K and IQ1_S weights run on
 * an NVIDIA GPU (pooled/paged); other types delegate to scalar.
 * When the build has no CUDA toolkit a stub delegates entirely to
 * scalar and reports the backend unusable.
 */
void matvec_cuda(const Mat& w, std::span<const float> x,
                 std::span<float> out);

/**
 * CUDA batched matvec (R11 Ops::matvec_batch, batch_self_parallel):
 * one kernel launch over all rows x n tokens. Q4_K/Q6_K read each
 * weight block from VRAM once and FMA it into n token accumulators
 * (bit-identical to n matvec_cuda() calls, ~n-fold less VRAM weight
 * traffic, weight shared from the pool); other types fall back to n
 * matvec_cuda() calls. out_batch is row-major (out[r*n + t]); x_batch
 * token-major. Scalar-delegating stub on non-CUDA builds.
 */
void matvec_batch_cuda(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch, std::uint32_t n);

/** @returns true when a usable CUDA device is present. */
bool cuda_backend_usable();

/**
 * Async weight prefetch (Ops.prefetch): begins a cudaMemcpyAsync of
 * w onto a dedicated stream into the CUDA weight pool, so a later
 * matvec_cuda(w) finds it resident. No-op for weight types the GPU
 * kernels do not handle, and a no-op stub on non-CUDA builds.
 */
void cuda_prefetch(const Mat& w);

}  // namespace locus::backend
