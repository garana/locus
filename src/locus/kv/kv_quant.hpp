#pragma once

#include <cstddef>
#include <cstdint>

namespace locus::kv {

/**
 * Storage precision for a paged KV cache (DESIGN.md R14). Quantizing
 * K/V shrinks the resident cache -- the footprint lever for long
 * context on streamed, bigger-than-RAM models -- at a small,
 * bounded accuracy cost. F32 is the exact default; Q8/Q4 use
 * block-of-32 codecs modelled on ggml's Q8_0 / Q4_0 (one f16 scale
 * per block), so a per-head slice of a row stays block-aligned when
 * head_dim is a multiple of 32.
 */
enum class KvType : std::uint8_t {
    kF32 = 0,
    kQ8 = 1,
    kQ4 = 2,
};

/** Elements per quantization block (both Q8 and Q4). */
inline constexpr std::size_t kKvBlock = 32;

/**
 * @returns Bytes one row of `n` elements occupies in `type`. `n`
 *     must be a multiple of kKvBlock for the quantized types.
 */
std::size_t kv_row_bytes(std::size_t n, KvType type);

/**
 * Quantizes `n` floats from `src` into `dst` (kv_row_bytes(n, type)
 * bytes). No-op copy for kF32. `n % kKvBlock == 0` for kQ8/kQ4.
 */
void kv_quantize_row(const float* src, std::size_t n, void* dst,
                     KvType type);

/**
 * Dequantizes a `type` row of `n` elements from `src` into `dst`.
 * Straight copy for kF32.
 */
void kv_dequantize_row(const void* src, std::size_t n, float* dst,
                       KvType type);

}  // namespace locus::kv
