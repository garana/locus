#pragma once

#include <cstdint>
#include <list>
#include <span>
#include <vector>

#include "locus/kv/paged_cache.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::engine {

/**
 * Exact prompt-prefix KV cache. Registers the block-aligned prefix
 * of finished prompts (pinning their full KV blocks by ref-count)
 * so a later request sharing that prefix can adopt the blocks and
 * skip re-prefilling them. Reuse is byte-exact: a token prefix at
 * positions 0..k always produces the same KV, so the shared blocks
 * match what recomputation would write.
 *
 * Keyed on the exact prefix tokens (block granularity); a small LRU
 * bounds it, and evict_until_free() drops entries under pool
 * pressure. Single-threaded, owned by the engine.
 */
class PrefixCache {
  public:
    PrefixCache(kv::PagedKvCache& cache, std::uint32_t slots);

    /** @returns the blocks of the longest cached prefix of `prompt`
     * (empty if none), i.e. the blocks to adopt. */
    std::vector<kv::BlockId> match(
        std::span<const tok::TokenId> prompt) const;

    /** Registers `prompt`'s full block-aligned prefix, mapping it to
     * the leading blocks of `seq_blocks` and pinning them. No-op if
     * the prefix is already cached (just refreshes LRU). */
    void insert(std::span<const tok::TokenId> prompt,
                const std::vector<kv::BlockId>& seq_blocks);

    /** Evicts LRU entries until the pool has `need` free blocks or
     * the cache is empty. @returns blocks freed. */
    std::uint32_t evict_until_free(std::uint32_t need);

  private:
    struct Entry {
        std::vector<tok::TokenId> key;    // block-aligned prefix
        std::vector<kv::BlockId> blocks;  // one per block_tokens
    };
    void evict_lru();

    kv::PagedKvCache& cache_;
    std::uint32_t block_tokens_;
    std::uint32_t slots_;
    std::list<Entry> entries_;  // front = most recently used
};

}  // namespace locus::engine
