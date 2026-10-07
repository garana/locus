#pragma once

#include <cstdint>
#include <list>
#include <optional>
#include <span>
#include <unordered_map>
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
 *
 * Lookup is digest-indexed (issue 65). Every key is block-aligned, so
 * a prompt can only match at a block boundary: the cache indexes
 * entries by a 64-bit digest of their key and match() walks the
 * prompt's block boundaries from longest to shortest, probing the
 * index once per boundary. That makes a lookup O(prompt) instead of
 * O(entries x prompt).
 *
 * The digest is ONLY an index. It narrows the candidate set; the
 * authoritative key is still the token sequence, which match() and
 * insert() compare in full before accepting a candidate. A digest
 * collision therefore costs one extra comparison and can never return
 * a wrong hit, so reuse stays token-exact.
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
     * the prefix is already cached (just refreshes LRU).
     * @returns prompt tokens newly written to the cache (0 if the
     *     prefix was already cached, or nothing was cacheable). */
    std::uint32_t insert(std::span<const tok::TokenId> prompt,
                         const std::vector<kv::BlockId>& seq_blocks);

    /** Evicts LRU entries until the pool has `need` free blocks or
     * the cache is empty. @returns blocks freed. */
    std::uint32_t evict_until_free(std::uint32_t need);

  private:
    struct Entry {
        std::vector<tok::TokenId> key;    // block-aligned prefix
        std::vector<kv::BlockId> blocks;  // one per block_tokens
        std::uint64_t digest = 0;         // index key (see class note)
    };
    using EntryIt = std::list<Entry>::iterator;

    void evict_lru();
    /** Removes `it` from its digest bucket, dropping the bucket when
     * it empties. Called with `it` still valid. */
    void unindex(EntryIt it);
    /** @returns the entry whose key is exactly `prompt`'s first
     * `blocks * block_tokens_` tokens, or nullopt. `digest` must be
     * that prefix's digest; candidates sharing it are verified
     * token-by-token before being accepted. */
    std::optional<EntryIt> find_exact(std::span<const tok::TokenId> prompt,
                                      std::size_t blocks,
                                      std::uint64_t digest) const;

    kv::PagedKvCache& cache_;
    std::uint32_t block_tokens_;
    std::uint32_t slots_;
    // front = most recently used. Iterators are stable across splice
    // and across other entries' erasure, which is what lets the index
    // hold them.
    std::list<Entry> entries_;
    // digest -> the entries carrying it. A vector because a collision
    // is possible (and harmless: see the class note).
    std::unordered_map<std::uint64_t, std::vector<EntryIt>> index_;
};

}  // namespace locus::engine
