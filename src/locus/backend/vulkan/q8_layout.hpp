#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace locus::backend::vk {

/**
 * GPU layout of a Q8_0 weight row. GGUF stores a row as nb blocks of
 * 34 bytes (f16 scale + 32 int8), so quant words straddle 4-byte
 * boundaries and every shader load needs two reads and shifts. The
 * GPU copy splits each row into [nb f16 scales | nb x 32 quants]:
 * still 34 * nb bytes per row, so row slices, per-head offsets and
 * expert strides keep their byte offsets, while the quants become
 * word-aligned (16-byte aligned when cols % 256 == 0).
 *
 * @param src rows x (cols / 32) GGUF Q8_0 blocks.
 * @param dst Same size; receives the split rows.
 */
inline void q8_0_split_rows(const std::byte* src, std::byte* dst,
                            std::size_t rows, std::uint32_t cols) {
    const std::size_t nb = cols / 32;
    const std::size_t row_bytes = nb * 34;
    for (std::size_t r = 0; r < rows; ++r) {
        const std::byte* in = src + r * row_bytes;
        std::byte* scales = dst + r * row_bytes;
        std::byte* quants = scales + 2 * nb;
        for (std::size_t b = 0; b < nb; ++b) {
            std::memcpy(scales + 2 * b, in + 34 * b, 2);
            std::memcpy(quants + 32 * b, in + 34 * b + 2, 32);
        }
    }
}

}  // namespace locus::backend::vk
