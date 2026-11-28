#pragma once

// Load-time weight repack (i#1243): a block-32 matrix (Q8_0 or Q4_0)
// copied once, four rows at a time, into llama.cpp's block_q8_0x4 or
// block_q4_0x4 layout, so the four-row sdot kernel
// (matvec_neon_dotprod_fast.cpp) reads each quad of rows as it
// computes instead of transposing it per token tile.
//
// The copy lives beside the mapped weights: Mat::data still points at
// the file's rows, which everything else reads (embedding rows, a tied
// output head, GPU uploads keyed by host pointer, odd row slices). A
// kernel finds the copy from the Mat it is given (find_packed), so a
// slice of a repacked matrix that starts on a quad finds its quads,
// and any other Mat falls back to the rows as stored.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"

namespace locus::backend {

/**
 * Bytes of one packed Q8_0 quad block: four f16 scales, then the four
 * rows' 32 quants, word k of row j at byte 8 + 16 k + 4 j (word k is
 * quants 4 k .. 4 k + 3). The same size as the four source blocks, so
 * quad q of a matrix starts at q * 4 * mat_row_bytes(w), as its rows
 * do.
 */
inline constexpr std::size_t kPackedQ8_0Block = 136;

/**
 * Bytes of one packed Q4_0 quad block (llama.cpp's block_q4_0x4 with
 * interleave 4): four f16 scales, then byte 8 + 16 k + 4 j + i is byte
 * 4 k + i of row j's 16 quant bytes, xor 0x88 (k 0 .. 3): its low
 * nibble is element 4 k + i, its high one element 16 + 4 k + i, each
 * now a signed 4-bit value (q - 8) in the nibble's top bits once
 * shifted or masked. The same size as the four source blocks (4 x 18).
 */
inline constexpr std::size_t kPackedQ4_0Block = 72;

/**
 * A block-32 matrix (Q8_0 or Q4_0) repacked four rows at a time.
 * find_packed sees it once it is registered (register_packed); a
 * registered copy that is destroyed unregisters itself. A last quad
 * short of rows repeats the last row (its extra lanes are never
 * stored).
 */
class PackedBlock32 {
public:
    /**
     * Copies w (packs_block32(w)) into the packed layout of its type.
     * Not registered yet.
     *
     * @param w The matrix as mapped; it must outlive this object.
     */
    explicit PackedBlock32(const Mat& w);
    ~PackedBlock32();
    PackedBlock32(const PackedBlock32&) = delete;
    PackedBlock32& operator=(const PackedBlock32&) = delete;

    /** @returns The packed bytes (a whole number of quads). */
    std::size_t bytes() const { return bytes_.size(); }

private:
    friend void register_packed(std::span<PackedBlock32* const>);
    friend void unregister_packed(std::span<PackedBlock32* const>);
    gguf::TensorType type_;
    const std::byte* src_;
    std::size_t src_bytes_;
    std::size_t row_bytes_;
    std::vector<std::byte> bytes_;
    bool registered_ = false;
};

/**
 * Registers a set of copies in one step (one new table snapshot), so a
 * model's load publishes once, not once per matrix. A copy already
 * registered is skipped. Not for concurrent use with a matvec on these
 * copies' matrices.
 */
void register_packed(std::span<PackedBlock32* const> set);

/** Unregisters a set of copies in one step; one not registered is
 *  skipped. */
void unregister_packed(std::span<PackedBlock32* const> set);

/**
 * @returns Whether w can be repacked: Q8_0 or Q4_0, cols a positive
 *     multiple of 32, rows >= 1 and data not null.
 */
bool packs_block32(const Mat& w);

/**
 * @returns The packed quads of w's rows when w starts on a quad of a
 *     registered matrix of the same type and row width, else nullptr.
 *     Quad q of w (rows 4 q .. 4 q + 3) is at result + q * nb * block,
 *     nb = cols / 32, block kPackedQ8_0Block or kPackedQ4_0Block. Safe
 *     to call from any thread.
 */
const std::byte* find_packed(const Mat& w);

/**
 * Writes rows r0 .. r0 + 3 of a Q8_0 w (a row past the last repeats the
 * last) as nb packed quad blocks at out (nb * 136 bytes). Exposed for
 * tests.
 */
void pack_q8_0_quad(const Mat& w, std::uint32_t r0, std::byte* out);

/** As pack_q8_0_quad for a Q4_0 w: nb * 72 bytes. */
void pack_q4_0_quad(const Mat& w, std::uint32_t r0, std::byte* out);

}  // namespace locus::backend
