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
 * Bytes of one packed Q4_K quad block (256 columns of 4 rows), its 6-bit
 * scales and mins decoded at load so the kernel does not unpack them:
 * d[4] and dmin[4] as f32 (from f16, exact); sc[8][4] and mn[8][4],
 * sub-block s's scale and min of row j at 32 + 4 s + j and 64 + 4 s +
 * j; then 512 bytes of quants, byte 96 + 16 v + 4 j + i being byte 4 v
 * + i of row j's 128 nibble bytes (v 0 .. 31): its low nibble is element
 * 64 (v / 8) + 4 (v % 8) + i, its high one 32 more. Larger than the
 * four source blocks (576), so the lookup scales the offset.
 */
inline constexpr std::size_t kPackedQ4KBlock = 608;

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
    std::size_t quad_bytes_;  // packed bytes per quad of rows
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
 * @returns Whether w can be repacked: Q8_0 or Q4_0 with cols a positive
 *     multiple of 32, or Q4_K with cols a positive multiple of 256;
 *     rows >= 1 and data not null.
 */
bool packs_block32(const Mat& w);

/**
 * @returns The bytes of w's packed copy: one packed block per 32 (Q8_0,
 *     Q4_0) or 256 (Q4_K) columns per quad of rows, a last short quad
 *     counted whole. packs_block32(w) must hold.
 */
std::size_t packed_size(const Mat& w);

/**
 * @returns The packed quads of w's rows when w starts on a quad of a
 *     registered matrix of the same type and row width, else nullptr.
 *     Quad q of w (rows 4 q .. 4 q + 3) is at result + q * nb * block,
 *     nb = cols / 32 with block kPackedQ8_0Block or kPackedQ4_0Block,
 *     or cols / 256 with kPackedQ4KBlock. Safe to call from any thread.
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

/** As pack_q8_0_quad for a Q4_K w: cols / 256 * 608 bytes. */
void pack_q4_k_quad(const Mat& w, std::uint32_t r0, std::byte* out);

}  // namespace locus::backend
