#pragma once

#include <cstdint>
#include <span>

#include "locus/backend/cpu_ops.hpp"

namespace locus::backend {

/** @returns Whether this build has the Accelerate path (Apple only). */
bool accelerate_available();

/** @returns Whether the CPU's batched matmuls go to sgemm_batch: this
 *  build has Accelerate and LOCUS_CPU_FAST is set. */
bool sgemm_prefill_on();

/**
 * The CPU prefill chunk when sgemm_batch takes the matmuls (i#782).
 * Each call dequantizes its weights, so a chunk of 64 tokens (the CPU
 * default) spends three times as long dequantizing as in sgemm; a big
 * chunk spreads that over its tokens. M2 Pro, Qwen2.5-0.5B Q8_0, a
 * 1680-token prompt: 249 tok/s at 64, 615 at 512, 827 at 2048 (the
 * CPU kernels: 653).
 */
inline constexpr std::uint32_t kSgemmPrefillChunk = 2048;

/**
 * out[t * w.rows + r] = row r of w . x[t] for n token-major tokens,
 * through Accelerate's cblas_sgemm (i#782), which runs on Apple's
 * matrix coprocessor. F32 weights go to sgemm as they are; any other
 * type is dequantized to F32 a block of rows at a time (a scratch of at
 * most 16 MB a thread), each block one sgemm.
 *
 * The sums run in sgemm's order on F32 activations, so the result is
 * not bit-identical to the CPU kernels (which quantize the activations
 * for the K-quants and Q8_0): callers use it only under
 * LOCUS_CPU_FAST. Call it from outside the thread pool (the dequant
 * runs on it).
 *
 * @returns false, doing nothing, when the build has no Accelerate.
 */
bool sgemm_batch(const Mat& w, std::span<const float> x,
                 std::span<float> out, std::uint32_t n);

}  // namespace locus::backend
