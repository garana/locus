#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace locus::image {

/**
 * Inflates a zlib stream (RFC 1950 around RFC 1951 deflate): stored,
 * fixed-Huffman and dynamic-Huffman blocks. The input is untrusted, so
 * every read is bounds-checked and the output never grows past
 * `max_out` bytes. The header must name deflate with no preset
 * dictionary, and the Adler-32 after the last block must match; bytes
 * after it are ignored.
 *
 * @param in The zlib stream.
 * @param max_out The most bytes the stream may produce.
 * @returns The inflated bytes.
 * @throws std::invalid_argument on a malformed or truncated stream, a
 *     checksum mismatch, or output over max_out.
 */
std::vector<std::uint8_t> zlib_inflate(std::span<const std::uint8_t> in,
                                       std::size_t max_out);

/** @returns The Adler-32 of d (as zlib computes it). */
std::uint32_t adler32(std::span<const std::uint8_t> d);

}  // namespace locus::image
