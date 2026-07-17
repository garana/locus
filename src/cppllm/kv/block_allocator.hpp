#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace cppllm::kv {

/** Identifier of a KV-cache block inside a BlockAllocator pool. */
using BlockId = std::uint32_t;

/**
 * Ref-counted allocator for a fixed pool of KV-cache blocks.
 *
 * Pure bookkeeping: it tracks which block ids are free and how many
 * sequences reference each live block, but owns no tensor memory.
 * Backends map a BlockId to their own storage. Ref counts > 1 model
 * copy-on-write prefix sharing between forked sequences.
 *
 * Not thread-safe; the scheduler serializes access.
 */
class BlockAllocator {
  public:
    /**
     * @param num_blocks Total number of blocks in the pool.
     */
    explicit BlockAllocator(std::uint32_t num_blocks);

    /**
     * Takes a free block, setting its ref count to 1.
     *
     * @returns The block id, or std::nullopt if the pool is
     *     exhausted (caller decides whether to queue or preempt).
     */
    std::optional<BlockId> allocate();

    /**
     * Increments the ref count of a live block (prefix sharing).
     *
     * @param id A block previously returned by allocate().
     * @throws std::logic_error if the block is not live.
     */
    void retain(BlockId id);

    /**
     * Decrements the ref count; frees the block when it reaches 0.
     *
     * @param id A block previously returned by allocate().
     * @returns true if the block was freed by this call.
     * @throws std::logic_error if the block is not live.
     */
    bool release(BlockId id);

    /**
     * @param id Any id inside the pool.
     * @returns Current ref count (0 means the block is free).
     * @throws std::out_of_range if id is outside the pool.
     */
    std::uint32_t ref_count(BlockId id) const;

    /** @returns Number of blocks currently free. */
    std::uint32_t free_blocks() const;

    /** @returns Total pool size. */
    std::uint32_t total_blocks() const;

  private:
    std::vector<std::uint32_t> ref_counts_;
    std::vector<BlockId> free_list_;
};

}  // namespace cppllm::kv
