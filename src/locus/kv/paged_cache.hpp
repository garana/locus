#pragma once

#include <cstdint>
#include <vector>

#include "locus/kv/block_allocator.hpp"

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
     * pool_floats(geom) floats and outlive the cache.
     */
    PagedKvCache(const Geometry& geom, float* storage);

    /** @returns Floats a pool for this geometry occupies. */
    static std::size_t pool_floats(const Geometry& geom) {
        return static_cast<std::size_t>(geom.n_blocks) *
               geom.n_layers * geom.block_tokens * geom.kv_dim * 2;
    }

    /** @returns The pool base address (for backend lookup). */
    const float* pool_data() const { return pool_ptr_; }

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

    /** Writable K row for a position < capacity of seq. */
    float* k(const Seq& seq, std::uint32_t layer, std::uint32_t pos);

    /** Writable V row for a position < capacity of seq. */
    float* v(const Seq& seq, std::uint32_t layer, std::uint32_t pos);

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

    Geometry geom_;
    BlockAllocator alloc_;
    /** Floats per (block, layer): block_tokens * kv_dim * 2. */
    std::size_t layer_stride_;
    /** Floats per block: n_layers * layer_stride_. */
    std::size_t block_stride_;
    /** Owned storage (empty when external). */
    std::vector<float> pool_;
    /** Points at pool_.data() or the external storage. */
    float* pool_ptr_ = nullptr;
};

}  // namespace locus::kv
