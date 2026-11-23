#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

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

TEST_CASE("weight pool keeps imported buffers out of the budget",
          "[vulkan-pool]") {
    Harness h;
    h.pool.budget_ = 100;
    int made = 0;
    auto import = [&]() -> std::optional<FakeBuf> {
        ++made;
        return FakeBuf{1000 + made};
    };
    // A miss builds it; a hit returns it without building again.
    auto a1 = h.pool.acquire_external(A, A, import);
    auto a2 = h.pool.acquire_external(A, A, import);
    REQUIRE(a1);
    REQUIRE(a2);
    REQUIRE(a1->id == a2->id);
    REQUIRE(made == 1);
    // In-place buffers cost nothing: a full-budget copy still fits.
    REQUIRE(h.pool.used_ == 0);
    h.pool.on_batch_end();
    h.get(B, 100);
    REQUIRE(h.pool.used_ == 100);
    REQUIRE(h.pool.transient_.empty());
    // When the import can't be made, nothing is cached.
    auto none = h.pool.acquire_external(
        C, C, []() -> std::optional<FakeBuf> { return std::nullopt; });
    REQUIRE_FALSE(none);
    REQUIRE(h.pool.map_.count(C) == 0);
}

TEST_CASE("weight pool releases the entries over a freed range",
          "[vulkan-pool]") {
    Harness h;
    h.pool.budget_ = 1000;
    char mapping[64];  // stands in for a model file's mapping
    char other[8];
    int made = 0;
    auto import = [&]() -> std::optional<FakeBuf> {
        ++made;
        ++h.live;
        return FakeBuf{2000 + made};
    };
    // An import keyed elsewhere (as upload_experts keys by its
    // ExpertMat) but reading the mapping; a copy keyed elsewhere too
    // but made from the mapping (a Q8_0 expert copy); a copy keyed
    // inside it; one outside it.
    h.pool.acquire_external(other + 1, mapping + 8, import);
    h.pool.acquire(other + 2, 30, [](FakeBuf) {}, mapping + 16);
    h.get(mapping + 32, 10);
    h.get(other, 20);
    REQUIRE(h.pool.map_.size() == 4);
    REQUIRE(h.live == 4);
    h.pool.on_batch_end();  // released between batches, as in use
    h.pool.release_range(mapping, sizeof mapping);
    REQUIRE(h.pool.map_.size() == 1);
    REQUIRE(h.pool.map_.count(other) == 1);
    REQUIRE(h.live == 1);
    REQUIRE(h.pool.used_ == 20);
    // The import is made again for a later model, not served stale.
    h.pool.acquire_external(other + 1, mapping + 8, import);
    REQUIRE(made == 2);
}

// i#1164: on a discrete GPU, a weight that missed the device heap
// (pinned system memory instead) is a spill. Spills are cached under
// their own cap, so a model a little larger than the device stays
// resident, and many distinct weights never pin more than the cap.
TEST_CASE("weight pool caps the weights that spill off the device",
          "[vulkan-pool]") {
    Harness h;
    h.pool.budget_ = 0;
    h.pool.init_budget();  // unbounded, as with LOCUS_GPU_POOL_MB unset
    h.pool.spill_cap_ = 20;
    // The device holds 20 bytes of weights; any more spills.
    int device_bytes = 0;
    std::vector<int> spilled_ids;
    h.pool.create = [&](std::size_t n) {
        ++h.live;
        FakeBuf b{++h.next_id};
        if (device_bytes + static_cast<int>(n) > 20) {
            spilled_ids.push_back(b.id);
        } else {
            device_bytes += static_cast<int>(n);
        }
        return b;
    };
    h.pool.destroy = [&](FakeBuf b) {
        --h.live;
        if (std::find(spilled_ids.begin(), spilled_ids.end(), b.id) ==
            spilled_ids.end()) {
            device_bytes -= 10;
        }
    };
    h.pool.spilled = [&](FakeBuf b) {
        return std::find(spilled_ids.begin(), spilled_ids.end(), b.id) !=
               spilled_ids.end();
    };
    // A dense model of four weights, read every token: two fit the
    // device, two spill and stay cached.
    const auto token = [&] {
        for (const char* k : {A, B, C, D}) {
            h.get(k, 10);
        }
        h.pool.on_batch_end();
    };
    token();
    REQUIRE(h.pool.spills_ == 2);
    REQUIRE(h.pool.budget_ == 20);  // what the device held when it filled
    REQUIRE(h.pool.used_ == 20);
    REQUIRE(h.pool.spill_used_ == 20);
    REQUIRE(h.live == 4);
    const auto uploaded = h.pool.uploaded_;
    for (int i = 0; i < 5; ++i) {
        token();
    }
    // Later tokens: all hits, nothing evicted or uploaded again.
    REQUIRE(h.pool.uploaded_ == uploaded);
    REQUIRE(h.pool.evictions_ == 0);
    REQUIRE(h.pool.hits_ == 20);
    // Many distinct weights (an MoE's experts): spills evict each other
    // within the cap; system memory never holds more than the cap.
    for (int i = 0; i < 50; ++i) {
        h.get(reinterpret_cast<const void*>(
                  static_cast<std::uintptr_t>(0x1000 + i)),
              10);
        h.pool.on_batch_end();
        REQUIRE(h.pool.spill_used_ <= 20);
        REQUIRE(h.pool.used_ <= 20);
    }
    REQUIRE(h.live == 4);
    // A weight read while every spill is pinned: served for the batch.
    h.pool.on_batch_end();
    for (int i = 0; i < 3; ++i) {
        h.get(reinterpret_cast<const void*>(
                  static_cast<std::uintptr_t>(0x2000 + i)),
              10);
    }
    REQUIRE(h.pool.spill_used_ <= 20);
    h.pool.on_batch_end();
    REQUIRE(h.live == 4);

    // A reset between models frees everything and the spill state: the
    // next model's spills are kept again, not served one batch each.
    h.pool.reset();
    REQUIRE(h.live == 0);
    REQUIRE(h.pool.used_ == 0);
    REQUIRE(h.pool.spill_used_ == 0);
    REQUIRE_FALSE(h.pool.device_full_);
    const char* keys[] = {"w", "x", "y", "z"};
    for (const char* k : keys) {
        h.get(k, 10);
    }
    h.pool.on_batch_end();
    REQUIRE(h.pool.used_ == 20);
    REQUIRE(h.pool.spill_used_ == 20);
    REQUIRE(h.live == 4);  // nothing served as a transient
    REQUIRE(h.pool.transient_.empty());
}
