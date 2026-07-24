#include <cstdint>

#include "catch_amalgamated.hpp"
#include "locus/backend/vulkan/weight_pool.hpp"

using locus::backend::WeightPoolT;

namespace {

/** A stand-in for a device buffer: just a serial id. */
struct FakeBuf {
    int id = 0;
};

/** A pool wired to a counting allocator (no GPU involved). */
struct Harness {
    int next_id = 0;
    int live = 0;
    WeightPoolT<FakeBuf> pool;

    Harness() {
        pool.create = [this](std::size_t) {
            ++live;
            return FakeBuf{++next_id};
        };
        pool.destroy = [this](FakeBuf) { --live; };
    }
    FakeBuf get(const void* key, std::size_t bytes) {
        return pool.acquire(key, bytes, [](FakeBuf) {});
    }
};

const char* A = "a";
const char* B = "b";
const char* C = "c";
const char* D = "d";

}  // namespace

TEST_CASE("weight pool caches and reports hits", "[vulkan-pool]") {
    Harness h;
    h.pool.budget_ = 100;  // bytes; skip env init
    auto a1 = h.get(A, 10);
    auto a2 = h.get(A, 10);  // hit: same key
    REQUIRE(a1.id == a2.id);
    REQUIRE(h.pool.hits_ == 1);
    REQUIRE(h.pool.misses_ == 1);
    REQUIRE(h.live == 1);
}

TEST_CASE("weight pool evicts the LRU unpinned buffer",
          "[vulkan-pool]") {
    Harness h;
    h.pool.budget_ = 20;  // holds two 10-byte buffers
    h.get(A, 10);          // used 10, A pinned
    h.get(B, 10);          // used 20, B pinned
    h.pool.on_batch_end();  // unpin A, B
    // C needs 10 but 20+10 > 20 -> evict LRU unpinned: A (older).
    h.get(C, 10);
    REQUIRE(h.pool.evictions_ == 1);
    REQUIRE(h.pool.map_.count(A) == 0);
    REQUIRE(h.pool.map_.count(B) == 1);
    REQUIRE(h.pool.map_.count(C) == 1);
    REQUIRE(h.live == 2);  // A destroyed, B and C live

    // Touch B so C becomes the LRU, then D evicts C not B.
    h.get(B, 10);
    h.pool.on_batch_end();
    h.get(D, 10);
    REQUIRE(h.pool.map_.count(C) == 0);
    REQUIRE(h.pool.map_.count(B) == 1);
    REQUIRE(h.pool.map_.count(D) == 1);
}

TEST_CASE("weight pool never evicts a pinned buffer",
          "[vulkan-pool]") {
    Harness h;
    h.pool.budget_ = 20;
    h.get(A, 10);  // pinned (this batch)
    h.get(B, 10);  // pinned
    // C cannot fit: both resident are pinned -> transient (uncached).
    auto c = h.get(C, 10);
    REQUIRE(h.pool.evictions_ == 0);
    REQUIRE(h.pool.map_.count(C) == 0);  // not cached
    REQUIRE(h.pool.map_.size() == 2);
    REQUIRE(h.live == 3);  // A, B, plus the transient C
    REQUIRE(c.id != 0);
    // Batch end drops the transient; A and B remain.
    h.pool.on_batch_end();
    REQUIRE(h.live == 2);
}

TEST_CASE("weight pool serves an oversized weight transiently",
          "[vulkan-pool]") {
    Harness h;
    h.pool.budget_ = 20;
    auto big = h.get(A, 50);  // larger than the whole budget
    REQUIRE(big.id != 0);
    REQUIRE(h.pool.map_.empty());  // never cached
    REQUIRE(h.live == 1);
    h.pool.on_batch_end();
    REQUIRE(h.live == 0);  // transient freed
}

TEST_CASE("unbounded budget never evicts", "[vulkan-pool]") {
    Harness h;
    h.pool.budget_ = 0;
    h.pool.init_budget();  // no env set in tests -> unbounded
    for (int i = 0; i < 50; ++i) {
        // distinct keys; a real run would reuse the mmap pointers
        h.get(reinterpret_cast<const void*>(
                  static_cast<std::uintptr_t>(0x1000 + i)),
              1u << 20);
    }
    REQUIRE(h.pool.evictions_ == 0);
    REQUIRE(h.pool.map_.size() == 50);
}
