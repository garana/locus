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
