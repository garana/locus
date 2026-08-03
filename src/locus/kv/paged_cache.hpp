#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/kv/block_allocator.hpp"
#include "locus/kv/kv_quant.hpp"

namespace locus::kv {

/**
 * Paged KV storage (DESIGN.md 4.1): a pool of fixed-size blocks of
 * `block_tokens` positions each. A block covers all layers for its
 * token range, so a sequence has one block table shared by every
 * layer. Prefix sharing forks a sequence by ref-counting its full
 * blocks; a partial tail block is deep-copied at fork time, so
 * shared blocks are always full and therefore never written --
 * appends land in private blocks by construction.
 *
 * Bookkeeping lives in BlockAllocator; this class owns the float
 * storage and the (seq, layer, pos) -> address mapping.
 */
class PagedKvCache {
  public:
    /** Geometry of the cached tensors. */
    struct Geometry {
        std::uint32_t n_layers = 0;
        /** n_kv_heads * head_dim floats per position. */
        std::uint32_t kv_dim = 0;
        /** Positions per block (16 default per DESIGN.md). */
        std::uint32_t block_tokens = 16;
        /** Total blocks in the pool. */
        std::uint32_t n_blocks = 0;
        /**
         * Storage precision (DESIGN.md R14). kF32 is the exact
         * default and the only mode the GPU backends and the MLA/DSA
         * paths use. Quantized modes (kQ8/kQ4) shrink the resident
         * cache and are read on the CPU GQA attention path via
         * store_row/load_row; kv_dim must be a multiple of kKvBlock.
         */
        KvType kv_type = KvType::kF32;
    };

    /** Per-sequence cache handle: block table + committed length. */
    struct Seq {
        std::vector<BlockId> blocks;
        /** Positions written and readable so far. */
        std::uint32_t n_tokens = 0;
    };

    explicit PagedKvCache(const Geometry& geom);

    /**
     * Uses caller-provided storage (e.g. a GPU-mapped buffer on
     * unified memory) instead of allocating. `storage` must hold
     * pool_floats(geom) floats and outlive the cache. F32 only.
     */
    PagedKvCache(const Geometry& geom, float* storage);

    /**
     * Quantized counterpart: wraps caller-provided GPU-mapped byte
     * storage for a quantized geometry (so a Vulkan/CUDA backend can
     * bind the KV pool and read/write it in-shader). `storage` must
     * hold pool_bytes(geom) bytes and outlive the cache. Requires a
     * quantized kv_type. The CPU codec (store_row/load_row) still
     * works against it, so it is unit-testable without a GPU.
     */
    PagedKvCache(const Geometry& geom, std::uint8_t* storage);

    /** @returns Floats an F32 pool for this geometry occupies. */
    static std::size_t pool_floats(const Geometry& geom) {
        return static_cast<std::size_t>(geom.n_blocks) *
               geom.n_layers * geom.block_tokens * geom.kv_dim * 2;
    }

    /** @returns Bytes a quantized pool for this geometry occupies. */
    static std::size_t pool_bytes(const Geometry& geom) {
        return static_cast<std::size_t>(geom.n_blocks) *
               geom.n_layers * geom.block_tokens * 2 *
               kv_row_bytes(geom.kv_dim, geom.kv_type);
    }

    /** @returns true when K/V are stored quantized (not kF32). */
    bool quantized() const { return geom_.kv_type != KvType::kF32; }

    /** @returns The F32 pool base address (for backend lookup). */
    const float* pool_data() const { return pool_ptr_; }

    /** @returns The quantized pool base (for backend binding). */
    const std::uint8_t* qpool_data() const { return qpool_ptr_; }

    /** Byte strides of the quantized pool (for shader indexing). */
    std::size_t q_row_bytes() const { return q_row_bytes_; }
    std::size_t q_layer_stride() const { return q_layer_stride_; }
    std::size_t q_block_stride() const { return q_block_stride_; }

    /**
     * Ensures seq can hold n_more positions past seq.n_tokens,
     * allocating blocks as needed. All-or-nothing: on false the
     * seq is unchanged (caller queues or preempts).
     */
    bool ensure_capacity(Seq& seq, std::uint32_t n_more);

    /**
     * Releases every block of seq back to the pool (ref-counted:
     * blocks shared with a fork survive) and resets it.
     */
    void release(Seq& seq);

    /**
     * Forks parent into child, sharing parent's full blocks by
     * ref-count; a partially-filled tail block is deep-copied so
     * both sides can append independently.
     *
     * @param child Must be empty (no blocks).
     * @returns true on success; false and child untouched when
     *     the pool cannot cover the tail-block copy.
     */
    bool fork(const Seq& parent, Seq& child);

    /**
     * Adopts `blocks` (full, shared prefix blocks from a prefix
     * cache) into the empty `child` by ref-count, setting its
     * committed length to cover them. Shared blocks are full and
     * never written, so the child appends only in private blocks.
     *
     * @param child Must be empty.
     */
    void retain_prefix(Seq& child,
                       const std::vector<BlockId>& blocks);

    /** Increments a live block's ref count (prefix-cache pin). */
    void retain_block(BlockId id) { alloc_.retain(id); }

    /** Decrements a block's ref count; @returns true if freed. */
    bool release_block(BlockId id) { return alloc_.release(id); }

    /** Writable K row for a position < capacity of seq. F32 only. */
    float* k(const Seq& seq, std::uint32_t layer, std::uint32_t pos);

    /** Writable V row for a position < capacity of seq. F32 only. */
    float* v(const Seq& seq, std::uint32_t layer, std::uint32_t pos);

    /**
     * Stores a kv_dim-length K (value=false) or V (value=true) row
     * at `pos`, quantizing per geom.kv_type. Works in every mode
     * (F32 is a straight copy), so the CPU attention path can write
     * uniformly. `src` must have kv_dim floats.
     */
    void store_row(const Seq& seq, std::uint32_t layer,
                   std::uint32_t pos, bool value,
                   std::span<const float> src);

    /**
     * Loads (dequantizing when needed) a kv_dim-length K/V row at
     * `pos` into `dst` (kv_dim floats). Companion to store_row.
     */
    void load_row(const Seq& seq, std::uint32_t layer,
                  std::uint32_t pos, bool value,
                  std::span<float> dst) const;

    const Geometry& geometry() const { return geom_; }
    std::uint32_t free_blocks() const { return alloc_.free_blocks(); }
    std::uint32_t total_blocks() const {
        return alloc_.total_blocks();
    }

    /** @returns Positions currently allocatable for seq. */
    std::uint32_t capacity(const Seq& seq) const {
        return static_cast<std::uint32_t>(seq.blocks.size()) *
               geom_.block_tokens;
    }

  private:
    float* row(const Seq& seq, std::uint32_t layer,
               std::uint32_t pos, bool value);
    /** Byte address of a K/V row in the quantized pool. */
    std::uint8_t* qrow(const Seq& seq, std::uint32_t layer,
                       std::uint32_t pos, bool value) const;

    Geometry geom_;
    BlockAllocator alloc_;
    /** Floats per (block, layer): block_tokens * kv_dim * 2. */
    std::size_t layer_stride_;
    /** Floats per block: n_layers * layer_stride_. */
    std::size_t block_stride_;
    /** Owned F32 storage (empty when external or quantized). */
    std::vector<float> pool_;
    /** Points at pool_.data() or the external storage (F32 mode). */
    float* pool_ptr_ = nullptr;

    // Quantized mode (kQ8/kQ4): a parallel byte pool replaces the
    // float pool. Strides are in bytes.
    std::size_t q_row_bytes_ = 0;     // one K or V row
    std::size_t q_layer_stride_ = 0;  // block_tokens * 2 * row_bytes
    std::size_t q_block_stride_ = 0;  // n_layers * q_layer_stride_
    /** Owned quant storage (empty when external). */
    std::vector<std::uint8_t> qpool_;
    /** Points at qpool_.data() or the external byte storage. */
    std::uint8_t* qpool_ptr_ = nullptr;
};

}  // namespace locus::kv
