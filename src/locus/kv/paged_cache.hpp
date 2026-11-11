#pragma once

#include <cstdint>
#include <memory>
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
        /**
         * Absolute index of the first layer this cache stores, for a
         * pipeline stage that owns only a slice of the model's layers
         * (multi-server, DESIGN.md R15+). The cache holds `n_layers`
         * layers starting here; callers still pass absolute layer
         * indices and the cache subtracts this base internally. 0 (the
         * default) means a full-model cache indexed 0..n_layers.
         */
        std::uint32_t layer_base = 0;
        /** n_kv_heads * head_dim floats per position. */
        std::uint32_t kv_dim = 0;
        /** Positions per block (16 default per DESIGN.md). */
        std::uint32_t block_tokens = 16;
        /** Total blocks in the pool. */
        std::uint32_t n_blocks = 0;
        /**
         * Storage precision (DESIGN.md R14). kF32 is the exact
         * default. The other types ("quantized": a byte pool) shrink
         * the resident cache and are read on the CPU GQA attention
         * path via store_row/load_row; kv_dim must be a multiple of
         * kKvBlock. kF16 keeps the F32 layout in halves, which the
         * Vulkan path reads for GQA and MLA (#105).
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
     * pool_floats(geom) floats. F32 only.
     *
     * @param release When not null, called with `storage` when the
     *     cache is destroyed (the cache then owns the storage);
     *     otherwise the storage must outlive the cache.
     */
    PagedKvCache(const Geometry& geom, float* storage,
                 void (*release)(float*) = nullptr);

    /**
     * Quantized counterpart: wraps caller-provided GPU-mapped byte
     * storage for a quantized geometry (so a Vulkan/CUDA backend can
     * bind the KV pool and read/write it in-shader). `storage` must
     * hold pool_bytes(geom) bytes and outlive the cache. Requires a
     * quantized kv_type. The CPU codec (store_row/load_row) still
     * works against it, so it is unit-testable without a GPU.
     *
     * @param release As for the float overload; called with `storage`
     *     reinterpreted as the float* an allocator handed out.
     */
    PagedKvCache(const Geometry& geom, std::uint8_t* storage,
                 void (*release)(float*) = nullptr);

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
     * Number of blocks ensure_capacity(seq, n_more) would have to
     * allocate to succeed (0 when the growth already fits). Lets a
     * caller size a reclaim before retrying ensure_capacity.
     * @returns additional blocks the growth needs.
     */
    std::uint32_t blocks_needed(const Seq& seq,
                                std::uint32_t n_more) const;

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

    /**
     * A device copy of the F32 pool (i#466), with the same layout, so a
     * row's float offset is the same on both sides. The host pool stays
     * the source of truth; the device copy is only as fresh as the
     * sync_rows calls made after each write.
     */
    struct Mirror {
        /** Device allocation of pool_floats(geom) floats. */
        void* dev = nullptr;
        /**
         * Copies `n` floats from host `src` to `dev` at float offset
         * `off`. @returns false on a device error.
         */
        bool (*store)(void* dev, std::size_t off, const float* src,
                      std::size_t n) = nullptr;
        /** Frees `dev`; called when the cache drops the mirror. */
        void (*release)(void* dev) = nullptr;
    };

    /**
     * Adopts `m` as this cache's device mirror (F32 only). The cache
     * owns it from here and releases it when destroyed. fork() keeps
     * the copied tail block in step on its own; every other row write
     * needs a sync_rows call.
     * @throws std::invalid_argument on a quantized cache or an
     *     incomplete Mirror.
     */
    void attach_mirror(const Mirror& m);

    /** @returns The mirror's device pointer, or nullptr without one. */
    const void* mirror_data() const { return mirror_.get(); }

    /** F32 pool layout, for a device reader of the mirror: floats per
     *  block, and per (block, layer) with K rows in the first half and
     *  V rows in the second. */
    std::size_t block_floats() const { return block_stride_; }
    std::size_t layer_floats() const { return layer_stride_; }

    /** @returns Float offset of absolute `layer` inside a block. */
    std::size_t layer_offset(std::uint32_t layer) const {
        return std::size_t{layer - geom_.layer_base} * layer_stride_;
    }

    /**
     * Pushes the K and V rows of positions [pos, pos + n) of `layer`
     * from the host pool to the mirror. No-op without a mirror. A
     * device error drops the mirror (mirror_data() turns nullptr), so a
     * device reader falls back to the host pool instead of reading
     * stale rows.
     */
    void sync_rows(const Seq& seq, std::uint32_t layer,
                   std::uint32_t pos, std::uint32_t n);

    /** @returns Bytes of one block (all layers, K and V) in either
     *  pool: a block is one contiguous range, so it can be read from
     *  or written to a file as is. */
    std::size_t block_bytes() const {
        return quantized() ? q_block_stride_ : block_stride_ * sizeof(float);
    }

    /** @returns The first byte of block `id` in the pool. */
    std::uint8_t* block_data(BlockId id) {
        return quantized() ? qpool_ptr_ + id * q_block_stride_
                           : reinterpret_cast<std::uint8_t*>(
                                 pool_ptr_ + id * block_stride_);
    }

    /**
     * Pushes a whole block from the host pool to the mirror, after
     * the host bytes were replaced in bulk (a disk read). No-op without
     * a mirror; a device error drops it, as in sync_rows.
     */
    void sync_block(BlockId id);

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
    /** Pushes rows [in_block, in_block + n) of one (block, slice
     *  layer) to the mirror, K then V; drops the mirror on error. */
    void push_rows(BlockId b, std::uint32_t rel, std::uint32_t in_block,
                   std::uint32_t n);

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
    /** External storage the cache releases when destroyed (null when
     *  owned here or when the caller keeps it). */
    std::unique_ptr<float, void (*)(float*)> external_{nullptr, nullptr};
    /** Device mirror (null when none), released with the cache. */
    std::unique_ptr<void, void (*)(void*)> mirror_{nullptr, nullptr};
    /** Mirror::store of the attached mirror. */
    bool (*mirror_store_)(void*, std::size_t, const float*,
                          std::size_t) = nullptr;
};

}  // namespace locus::kv
