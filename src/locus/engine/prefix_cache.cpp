#include "locus/engine/prefix_cache.hpp"

#include <algorithm>
#include <iterator>
#include <optional>

namespace locus::engine {

namespace {

/**
 * 64-bit avalanche (splitmix64's finalizer), applied once per BLOCK
 * boundary rather than once per token. Per-token work stays at the
 * two-operation FNV step below, which matters because every lookup
 * hashes the prompt once and that cost is the floor on a lookup: a
 * full avalanche per token made a small-cache lookup slower than the
 * linear scan it replaces.
 */
std::uint64_t mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

constexpr std::uint64_t kDigestSeed = 0x6c6f637573707266ull;  // "locusprf"
constexpr std::uint64_t kFnvPrime = 0x100000001b3ull;

/**
 * Folds one token into the running accumulator (FNV-1a over token
 * VALUES, not object bytes, so no padding or endianness is in play and
 * the digest is a pure function of the token sequence).
 */
std::uint64_t accum_push(std::uint64_t h, tok::TokenId t) {
    return (h ^ static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(t))) * kFnvPrime;
}

}  // namespace

PrefixCache::PrefixCache(kv::PagedKvCache& cache,
                         std::uint32_t slots)
    : cache_(cache),
      block_tokens_(cache.geometry().block_tokens),
      slots_(slots) {}

std::optional<PrefixCache::EntryIt> PrefixCache::find_exact(
    std::span<const tok::TokenId> prompt, std::size_t blocks,
    std::uint64_t digest) const {
    const auto bucket = index_.find(digest);
    if (bucket == index_.end()) {
        return std::nullopt;
    }
    const std::size_t klen = blocks * block_tokens_;
    for (const EntryIt& it : bucket->second) {
        // The digest only narrowed the candidates; the token sequence
        // is the key. Verify it in full, so a collision costs this
        // comparison and never yields a wrong hit.
        if (it->key.size() == klen &&
            std::equal(it->key.begin(), it->key.end(), prompt.begin())) {
            return it;
        }
    }
    return std::nullopt;
}

std::vector<kv::BlockId> PrefixCache::match(
    std::span<const tok::TokenId> prompt) const {
    const std::size_t prompt_blocks = prompt.size() / block_tokens_;
    if (prompt_blocks == 0 || index_.empty()) {
        return {};
    }
    // One forward pass: digest[j] is the digest of the prompt's first
    // j blocks, so the array is 1-based and slot 0 is unused padding
    // (the empty prefix is never a key).
    std::vector<std::uint64_t> digest(prompt_blocks + 1);
    std::uint64_t acc = kDigestSeed;
    for (std::size_t j = 1; j <= prompt_blocks; ++j) {
        const std::size_t base = (j - 1) * block_tokens_;
        for (std::uint32_t i = 0; i < block_tokens_; ++i) {
            acc = accum_push(acc, prompt[base + i]);
        }
        digest[j] = mix64(acc);
    }
    // Longest first: the first verified boundary IS the longest cached
    // prefix, which is what the linear scan used to compute. Two
    // entries cannot tie, because two keys of equal length that are
    // both prefixes of this prompt are the same key, and insert()
    // keeps keys unique.
    for (std::size_t j = prompt_blocks; j >= 1; --j) {
        if (const auto it = find_exact(prompt, j, digest[j])) {
            return (*it)->blocks;
        }
    }
    return {};
}

std::uint32_t PrefixCache::insert(
    std::span<const tok::TokenId> prompt,
    const std::vector<kv::BlockId>& seq_blocks) {
    const std::size_t full = prompt.size() / block_tokens_;
    if (full == 0 || full > seq_blocks.size()) {
        return 0;
    }
    const std::size_t klen = full * block_tokens_;
    std::uint64_t acc = kDigestSeed;
    for (std::size_t i = 0; i < klen; ++i) {
        acc = accum_push(acc, prompt[i]);
    }
    const std::uint64_t h = mix64(acc);
    if (const auto it = find_exact(prompt, full, h)) {
        entries_.splice(entries_.begin(), entries_, *it);
        return 0;  // already cached; refresh LRU only
    }
    Entry e;
    e.key.assign(prompt.begin(), prompt.begin() + klen);
    e.blocks.assign(seq_blocks.begin(), seq_blocks.begin() + full);
    e.digest = h;
    for (kv::BlockId b : e.blocks) {
        cache_.retain_block(b);
    }
    entries_.push_front(std::move(e));
    index_[h].push_back(entries_.begin());
    while (entries_.size() > slots_) {
        evict_lru();
    }
    return static_cast<std::uint32_t>(klen);
}

void PrefixCache::unindex(EntryIt it) {
    const auto bucket = index_.find(it->digest);
    if (bucket == index_.end()) {
        return;
    }
    std::vector<EntryIt>& v = bucket->second;
    v.erase(std::remove(v.begin(), v.end(), it), v.end());
    if (v.empty()) {
        index_.erase(bucket);
    }
}

void PrefixCache::evict_lru() {
    if (entries_.empty()) {
        return;
    }
    const EntryIt last = std::prev(entries_.end());
    for (kv::BlockId b : last->blocks) {
        cache_.release_block(b);
    }
    unindex(last);  // while `last` is still valid
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
