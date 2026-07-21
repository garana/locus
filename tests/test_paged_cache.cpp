#include <set>
#include <stdexcept>

#include "catch_amalgamated.hpp"
#include "locus/kv/paged_cache.hpp"

using locus::kv::PagedKvCache;

namespace {

PagedKvCache::Geometry small_geom() {
    PagedKvCache::Geometry g;
    g.n_layers = 2;
    g.kv_dim = 4;
    g.block_tokens = 4;
    g.n_blocks = 8;
    return g;
}

}  // namespace

TEST_CASE("capacity grows in whole blocks", "[kv]") {
    PagedKvCache cache(small_geom());
    PagedKvCache::Seq seq;

    REQUIRE(cache.capacity(seq) == 0);
    REQUIRE(cache.ensure_capacity(seq, 1));
    REQUIRE(cache.capacity(seq) == 4);
    REQUIRE(cache.free_blocks() == 7);

    seq.n_tokens = 4;
    REQUIRE(cache.ensure_capacity(seq, 3));
    REQUIRE(cache.capacity(seq) == 8);

    // All-or-nothing: 6 blocks free, 32 more tokens past the 8
    // already covered needs 7 -- must not partially allocate.
    REQUIRE_FALSE(cache.ensure_capacity(seq, 32));
    REQUIRE(cache.free_blocks() == 6);
    REQUIRE(cache.capacity(seq) == 8);

    cache.release(seq);
    REQUIRE(cache.free_blocks() == 8);
    REQUIRE(seq.blocks.empty());
    REQUIRE(seq.n_tokens == 0);
}

TEST_CASE("rows are distinct and stable across blocks", "[kv]") {
    PagedKvCache cache(small_geom());
    PagedKvCache::Seq seq;
    REQUIRE(cache.ensure_capacity(seq, 8));  // 2 blocks

    std::set<float*> rows;
    for (std::uint32_t l = 0; l < 2; ++l) {
        for (std::uint32_t p = 0; p < 8; ++p) {
            rows.insert(cache.k(seq, l, p));
            rows.insert(cache.v(seq, l, p));
        }
    }
    REQUIRE(rows.size() == 2 * 8 * 2);  // no aliasing

    // Values written across a block boundary survive.
    cache.k(seq, 1, 3)[0] = 42.0f;
    cache.k(seq, 1, 4)[0] = 43.0f;
    REQUIRE(cache.k(seq, 1, 3)[0] == 42.0f);
    REQUIRE(cache.k(seq, 1, 4)[0] == 43.0f);
}

TEST_CASE("fork shares full blocks and copies the tail", "[kv]") {
    PagedKvCache cache(small_geom());
    PagedKvCache::Seq parent;
    REQUIRE(cache.ensure_capacity(parent, 6));  // 2 blocks
    parent.n_tokens = 6;                        // 1 full + tail(2)
    cache.k(parent, 0, 1)[0] = 7.0f;            // in shared block
    cache.k(parent, 0, 5)[0] = 9.0f;            // in tail block

    PagedKvCache::Seq child;
    REQUIRE(cache.fork(parent, child));
    REQUIRE(child.n_tokens == 6);
    // 8 - 2 (parent) - 1 (tail copy) = 5; full block is shared.
    REQUIRE(cache.free_blocks() == 5);
    REQUIRE(cache.k(child, 0, 1) == cache.k(parent, 0, 1));
    REQUIRE(cache.k(child, 0, 5) != cache.k(parent, 0, 5));
    REQUIRE(cache.k(child, 0, 5)[0] == 9.0f);  // copied content

    // Diverge: child's tail is private.
    cache.k(child, 0, 5)[0] = 100.0f;
    REQUIRE(cache.k(parent, 0, 5)[0] == 9.0f);

    // Shared block survives one release, dies with the second.
    cache.release(parent);
    REQUIRE(cache.k(child, 0, 1)[0] == 7.0f);
    REQUIRE(cache.free_blocks() == 6);
    cache.release(child);
    REQUIRE(cache.free_blocks() == 8);
}

TEST_CASE("fork fails cleanly when the pool is dry", "[kv]") {
    auto geom = small_geom();
    geom.n_blocks = 2;
    PagedKvCache cache(geom);
    PagedKvCache::Seq a, b;
    REQUIRE(cache.ensure_capacity(a, 8));  // both blocks
    a.n_tokens = 6;                        // partial tail

    REQUIRE_FALSE(cache.fork(a, b));  // no block for the tail copy
    REQUIRE(b.blocks.empty());
    REQUIRE(cache.free_blocks() == 0);
    cache.release(a);
    REQUIRE(cache.free_blocks() == 2);
}

TEST_CASE("rejects empty geometry", "[kv]") {
    PagedKvCache::Geometry g;  // all zeros
    REQUIRE_THROWS_AS(PagedKvCache(g), std::invalid_argument);
}
