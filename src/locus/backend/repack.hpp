#pragma once

// Load-time weight repack (i#1243): a Q8_0 matrix copied once, four
// rows at a time, into llama.cpp's block_q8_0x4 layout, so the four-row
// sdot kernel (matvec_neon_dotprod_fast.cpp) reads each quad of rows as
// it computes instead of transposing it per token tile.
//
// The copy lives beside the mapped weights: Mat::data still points at
// the file's rows, which everything else reads (embedding rows, a tied
// output head, GPU uploads keyed by host pointer, odd row slices). A
// kernel finds the copy from the Mat it is given (find_packed_q8_0), so
// a slice of a repacked matrix that starts on a quad finds its quads,
// and any other Mat falls back to the rows as stored.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "locus/backend/cpu_ops.hpp"

namespace locus::backend {

/**
 * Bytes of one packed quad block: four f16 scales, then the four rows'
 * 32 quants, word k of row j at byte 16 k + 4 j (word k is quants 4 k
 * .. 4 k + 3). The same size as the four source blocks, so quad q of a
 * matrix starts at q * 4 * mat_row_bytes(w), as its rows do.
 */
inline constexpr std::size_t kPackedQ8_0Block = 136;

/**
 * A Q8_0 matrix repacked four rows at a time, registered for
 * find_packed_q8_0 for as long as it lives. A last quad short of rows
 * repeats the last row (its extra lanes are never stored).
 */
class PackedQ8_0 {
public:
    /**
     * Copies w (Q8_0, cols a multiple of 32, rows >= 1) into the
     * packed layout and registers it.
     *
     * @param w The matrix as mapped; it must outlive this object.
     */
    explicit PackedQ8_0(const Mat& w);
    ~PackedQ8_0();
    PackedQ8_0(const PackedQ8_0&) = delete;
    PackedQ8_0& operator=(const PackedQ8_0&) = delete;

    /** @returns The packed bytes (a whole number of quads). */
    std::size_t bytes() const { return bytes_.size(); }

private:
    const std::byte* src_;
    std::size_t src_bytes_;
    std::vector<std::byte> bytes_;
};

/**
 * @returns Whether w can be repacked: Q8_0, cols a positive multiple
 *     of 32, rows >= 1 and data not null.
 */
bool packs_q8_0(const Mat& w);

/**
 * @returns The packed quads of w's rows when w starts on a quad of a
 *     registered matrix with the same row width, else nullptr. Quad q
 *     of w (rows 4 q .. 4 q + 3) is at result + q * nb * 136, nb =
 *     cols / 32. Safe to call from any thread.
 */
const std::byte* find_packed_q8_0(const Mat& w);

/**
 * Writes rows r0 .. r0 + 3 of w (a row past the last repeats the last)
 * as nb packed quad blocks at out (nb * 136 bytes). Exposed for tests.
 */
void pack_q8_0_quad(const Mat& w, std::uint32_t r0, std::byte* out);

}  // namespace locus::backend
