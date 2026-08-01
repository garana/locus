#include "locus/engine/prefix_cache.hpp"

#include <algorithm>

namespace locus::engine {

PrefixCache::PrefixCache(kv::PagedKvCache& cache,
                         std::uint32_t slots)
    : cache_(cache),
      block_tokens_(cache.geometry().block_tokens),
      slots_(slots) {}

std::vector<kv::BlockId> PrefixCache::match(
    std::span<const tok::TokenId> prompt) const {
    std::vector<kv::BlockId> best;
    std::size_t best_len = 0;
    for (const Entry& e : entries_) {
        if (e.key.size() <= prompt.size() &&
            e.key.size() > best_len &&
            std::equal(e.key.begin(), e.key.end(),
                       prompt.begin())) {
            best = e.blocks;
            best_len = e.key.size();
        }
    }
    return best;
}

void PrefixCache::insert(
    std::span<const tok::TokenId> prompt,
    const std::vector<kv::BlockId>& seq_blocks) {
    const std::size_t full = prompt.size() / block_tokens_;
    if (full == 0 || full > seq_blocks.size()) {
        return;
    }
    const std::size_t klen = full * block_tokens_;
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->key.size() == klen &&
            std::equal(it->key.begin(), it->key.end(),
                       prompt.begin())) {
            entries_.splice(entries_.begin(), entries_, it);
            return;  // already cached; refresh LRU only
        }
    }
    Entry e;
    e.key.assign(prompt.begin(), prompt.begin() + klen);
    e.blocks.assign(seq_blocks.begin(), seq_blocks.begin() + full);
    for (kv::BlockId b : e.blocks) {
        cache_.retain_block(b);
    }
    entries_.push_front(std::move(e));
    while (entries_.size() > slots_) {
        evict_lru();
    }
}

void PrefixCache::evict_lru() {
    if (entries_.empty()) {
        return;
    }
    for (kv::BlockId b : entries_.back().blocks) {
        cache_.release_block(b);
    }
    entries_.pop_back();
}

std::uint32_t PrefixCache::evict_until_free(std::uint32_t need) {
    std::uint32_t freed = 0;
    while (cache_.free_blocks() < need && !entries_.empty()) {
        const std::uint32_t before = cache_.free_blocks();
        evict_lru();
        freed += cache_.free_blocks() - before;
    }
    return freed;
}

}  // namespace locus::engine
