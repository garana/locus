#include <stdexcept>
#include <vector>

#include "catch_amalgamated.hpp"
#include "cppllm/kv/block_allocator.hpp"

using cppllm::kv::BlockAllocator;
using cppllm::kv::BlockId;

TEST_CASE("allocates until exhaustion", "[kv]") {
    BlockAllocator alloc(4);
    REQUIRE(alloc.total_blocks() == 4);
    REQUIRE(alloc.free_blocks() == 4);

    std::vector<BlockId> ids;
    for (int i = 0; i < 4; ++i) {
        auto id = alloc.allocate();
        REQUIRE(id.has_value());
        ids.push_back(*id);
    }
    REQUIRE(alloc.free_blocks() == 0);
    REQUIRE_FALSE(alloc.allocate().has_value());

    REQUIRE(alloc.release(ids[2]));
    REQUIRE(alloc.free_blocks() == 1);
    REQUIRE(alloc.allocate() == ids[2]);
}

TEST_CASE("ref counting models prefix sharing", "[kv]") {
    BlockAllocator alloc(2);
    BlockId id = alloc.allocate().value();
    REQUIRE(alloc.ref_count(id) == 1);

    alloc.retain(id);  // second sequence forks the prefix
    REQUIRE(alloc.ref_count(id) == 2);

    REQUIRE_FALSE(alloc.release(id));  // still held by one sequence
    REQUIRE(alloc.free_blocks() == 1);
    REQUIRE(alloc.release(id));  // last holder frees it
    REQUIRE(alloc.free_blocks() == 2);
    REQUIRE(alloc.ref_count(id) == 0);
}

TEST_CASE("misuse is rejected", "[kv]") {
    BlockAllocator alloc(1);
    BlockId id = alloc.allocate().value();
    REQUIRE(alloc.release(id));

    REQUIRE_THROWS_AS(alloc.release(id), std::logic_error);
    REQUIRE_THROWS_AS(alloc.retain(id), std::logic_error);
    REQUIRE_THROWS_AS(alloc.ref_count(99), std::out_of_range);
}

TEST_CASE("random workload leaks nothing", "[kv]") {
    constexpr std::uint32_t kPool = 64;
    BlockAllocator alloc(kPool);
    std::vector<BlockId> live;

    // Deterministic pseudo-random interleaving of alloc/retain/
    // release; every block acquired here is eventually released.
    std::uint32_t state = 12345;
    auto next = [&state] {
        state = state * 1664525u + 1013904223u;
        return state >> 16;
    };

    for (int step = 0; step < 10000; ++step) {
        switch (next() % 3) {
            case 0: {
                if (auto id = alloc.allocate()) {
                    live.push_back(*id);
                }
                break;
            }
            case 1: {
                if (!live.empty()) {
                    BlockId id = live[next() % live.size()];
                    alloc.retain(id);
                    live.push_back(id);
                }
                break;
            }
            default: {
                if (!live.empty()) {
                    std::size_t at = next() % live.size();
                    alloc.release(live[at]);
                    live.erase(live.begin() + at);
                }
                break;
            }
        }
    }
    for (BlockId id : live) {
        alloc.release(id);
    }
    REQUIRE(alloc.free_blocks() == kPool);
}
