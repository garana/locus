#include <cstdint>
#include <list>
#include <random>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/engine/prefix_cache.hpp"
#include "locus/kv/paged_cache.hpp"

using locus::engine::PrefixCache;
using locus::kv::PagedKvCache;
using locus::tok::TokenId;

namespace {
PagedKvCache::Geometry geom() {
    PagedKvCache::Geometry g;
    g.n_layers = 1;
    g.kv_dim = 4;
    g.block_tokens = 4;
    g.n_blocks = 8;
    return g;
}
std::vector<locus::kv::BlockId> alloc_blocks(PagedKvCache& c,
                                             std::uint32_t n) {
    PagedKvCache::Seq s;
    REQUIRE(c.ensure_capacity(s, n * 4));
    return s.blocks;
}
}  // namespace

TEST_CASE("PrefixCache matches the longest cached prefix",
          "[prefix]") {
    PagedKvCache cache(geom());
    PrefixCache pc(cache, /*slots=*/4);
    auto blocks = alloc_blocks(cache, 2);  // 2 blocks = 8 tokens

    // 10-token prompt -> 2 full blocks cached (8 tokens).
    std::vector<TokenId> prompt{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    pc.insert(prompt, blocks);

    SECTION("a superset prompt matches the cached prefix") {
        std::vector<TokenId> longer = prompt;
        longer.push_back(99);
        auto m = pc.match(longer);
        REQUIRE(m.size() == 2);
        REQUIRE(m[0] == blocks[0]);
        REQUIRE(m[1] == blocks[1]);
    }
    SECTION("a divergent prompt does not match") {
        std::vector<TokenId> other{42, 42, 42, 42, 42};
        REQUIRE(pc.match(other).empty());
    }
    SECTION("a prompt shorter than the cached prefix misses") {
        std::vector<TokenId> shortp{1, 2, 3};
        REQUIRE(pc.match(shortp).empty());
    }
}

TEST_CASE("PrefixCache evicts LRU beyond its slot budget",
          "[prefix]") {
    PagedKvCache cache(geom());
    PrefixCache pc(cache, /*slots=*/1);
    auto ba = alloc_blocks(cache, 2);
    auto bb = alloc_blocks(cache, 2);
    std::vector<TokenId> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<TokenId> b{9, 10, 11, 12, 13, 14, 15, 16};

    pc.insert(a, ba);
    REQUIRE(pc.match(a).size() == 2);
    pc.insert(b, bb);  // slots=1 -> evicts a
    REQUIRE(pc.match(a).empty());
    REQUIRE(pc.match(b).size() == 2);
}

// ---- issue 65: the digest index must not change behaviour ----

namespace {

// Reference implementation: the linear longest-prefix scan the digest
// index replaces, kept deliberately dumb. The differential test below
// drives both and requires identical answers, which is the real proof
// that indexing changed lookup cost and nothing else.
struct Oracle {
    struct Ent {
        std::vector<TokenId> key;
        std::vector<locus::kv::BlockId> blocks;
    };
    std::uint32_t block_tokens;
    std::uint32_t slots;
    std::list<Ent> entries;  // front = MRU

    std::vector<locus::kv::BlockId> match(
        const std::vector<TokenId>& prompt) const {
        std::vector<locus::kv::BlockId> best;
        std::size_t best_len = 0;
        for (const Ent& e : entries) {
            if (e.key.size() <= prompt.size() && e.key.size() > best_len &&
                std::equal(e.key.begin(), e.key.end(), prompt.begin())) {
                best = e.blocks;
                best_len = e.key.size();
            }
        }
        return best;
    }
    std::uint32_t insert(const std::vector<TokenId>& prompt,
                         const std::vector<locus::kv::BlockId>& blocks) {
        const std::size_t full = prompt.size() / block_tokens;
        if (full == 0 || full > blocks.size()) {
            return 0;
        }
        const std::size_t klen = full * block_tokens;
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->key.size() == klen &&
                std::equal(it->key.begin(), it->key.end(), prompt.begin())) {
                entries.splice(entries.begin(), entries, it);
                return 0;
            }
        }
        Ent e;
        e.key.assign(prompt.begin(), prompt.begin() + klen);
        e.blocks.assign(blocks.begin(), blocks.begin() + full);
        entries.push_front(std::move(e));
        while (entries.size() > slots) {
            entries.pop_back();
        }
        return static_cast<std::uint32_t>(klen);
    }
};

}  // namespace

// A tiny token alphabet makes shared prefixes and near-misses the common
// case rather than a rarity, which is what stresses longest-prefix and
// the digest's candidate narrowing.
TEST_CASE("PrefixCache digest index matches the linear scan exactly",
          "[prefix]") {
    PagedKvCache::Geometry g;
    g.n_layers = 1;
    g.kv_dim = 4;
    g.block_tokens = 2;
    // Big enough that 4000 steps of inserts cannot exhaust the pool:
    // each test seq is allocated and never released, so the comparison
    // stays about lookup answers rather than pool accounting.
    g.n_blocks = 8192;
    PagedKvCache cache(g);
    PrefixCache pc(cache, /*slots=*/6);
    Oracle oracle{g.block_tokens, 6, {}};

    std::mt19937 rng(20251007);
    std::uint32_t inserts = 0, hits = 0, misses = 0;
    for (int step = 0; step < 4000; ++step) {
        // Prompts over {1,2,3} so prefixes collide often.
        std::vector<TokenId> prompt;
        const std::size_t n = 1 + rng() % 9;
        for (std::size_t i = 0; i < n; ++i) {
            prompt.push_back(static_cast<TokenId>(1 + rng() % 3));
        }
        if (rng() % 2) {
            // Insert: give both the SAME block ids so the comparison is
            // about which entry is chosen, not about allocation.
            const std::size_t full = prompt.size() / g.block_tokens;
            std::vector<locus::kv::BlockId> blocks;
            for (std::size_t i = 0; i < full; ++i) {
                blocks.push_back(static_cast<locus::kv::BlockId>(
                    1000 + step * 10 + i));
            }
            if (!blocks.empty()) {
                // PrefixCache retains/releases real blocks, so hand it
                // ids it owns; the oracle only needs the same vector.
                PagedKvCache::Seq s;
                REQUIRE(cache.ensure_capacity(
                    s, static_cast<std::uint32_t>(full * g.block_tokens)));
                const std::uint32_t a = pc.insert(prompt, s.blocks);
                const std::uint32_t b = oracle.insert(prompt, s.blocks);
                REQUIRE(a == b);  // the usage-cache number must agree
                inserts += (a > 0) ? 1 : 0;
            }
        } else {
            const auto got = pc.match(prompt);
            const auto want = oracle.match(prompt);
            REQUIRE(got == want);
            (got.empty() ? misses : hits)++;
        }
    }
    // The run must actually exercise both outcomes, or it proves little.
    REQUIRE(inserts > 50);
    REQUIRE(hits > 50);
    REQUIRE(misses > 50);
}

// The digest narrows candidates; the token sequence decides. Two keys of
// the SAME length that differ must never be confused, which is the
// property that makes a collision harmless.
TEST_CASE("PrefixCache verifies tokens, not just the digest",
          "[prefix]") {
    PagedKvCache cache(geom());
    PrefixCache pc(cache, /*slots=*/4);
    auto ba = alloc_blocks(cache, 1);
    auto bb = alloc_blocks(cache, 1);

    std::vector<TokenId> a{1, 2, 3, 4};
    std::vector<TokenId> b{1, 2, 3, 5};  // same length, last token differs
    pc.insert(a, ba);
    pc.insert(b, bb);

    REQUIRE(pc.match(a) == ba);
    REQUIRE(pc.match(b) == bb);
    // A third key of the same length that was never inserted must miss,
    // however similar it looks.
    std::vector<TokenId> c{1, 2, 3, 6};
    REQUIRE(pc.match(c).empty());
}

// Eviction must drop the index entry too: a re-inserted key has to come
// back as a fresh entry rather than resurrect a stale iterator.
TEST_CASE("PrefixCache index survives evict and re-insert",
          "[prefix]") {
    PagedKvCache cache(geom());
    PrefixCache pc(cache, /*slots=*/1);
    auto ba = alloc_blocks(cache, 1);
    auto bb = alloc_blocks(cache, 1);
    std::vector<TokenId> a{7, 7, 7, 7};
    std::vector<TokenId> b{8, 8, 8, 8};

    pc.insert(a, ba);
    REQUIRE(pc.match(a) == ba);
    pc.insert(b, bb);          // evicts a, and must unindex it
    REQUIRE(pc.match(a).empty());
    pc.insert(a, ba);          // evicts b, re-indexes a
    REQUIRE(pc.match(a) == ba);
    REQUIRE(pc.match(b).empty());
    // And evict_until_free must also keep the index consistent.
    pc.evict_until_free(cache.geometry().n_blocks);
    REQUIRE(pc.match(a).empty());
}
