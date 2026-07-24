#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

namespace locus::backend {

/**
 * R8-GPU weight pager (backend-agnostic core). Device buffers are
 * keyed by host (mmap) pointer, uploaded once and reused across
 * tokens under a byte budget with LRU eviction. Buffers touched
 * since the last on_batch_end() are pinned, so eviction never
 * frees a buffer the currently-recording command batch still
 * references; a weight that cannot fit even after evicting every
 * unpinned buffer is served from a transient buffer destroyed at
 * batch end (correct, uncached). Buffer allocation is injected via
 * `create`/`destroy` so the logic is unit-testable without a GPU.
 *
 * Default budget is unbounded (LOCUS_GPU_POOL_MB unset), so the
 * cache never evicts and behaviour matches a plain resident map.
 *
 * @tparam Buffer An opaque handle type (the backend's buffer).
 */
template <class Buffer>
struct WeightPoolT {
    struct Entry {
        Buffer buf;
        std::size_t bytes;
        std::uint64_t last_use;
        bool pinned;
    };

    /** Allocates a device buffer of `bytes`. */
    std::function<Buffer(std::size_t)> create;
    /** Frees a buffer from `create`. */
    std::function<void(Buffer)> destroy;

    std::unordered_map<const void*, Entry> map_;
    std::vector<Buffer> transient_;
    std::size_t budget_ = 0;  // 0 until init; SIZE_MAX = unbounded
    std::size_t used_ = 0;
    std::uint64_t clock_ = 0;
    std::uint64_t hits_ = 0, misses_ = 0, evictions_ = 0;
    std::uint64_t uploaded_ = 0;

    /** Reads LOCUS_GPU_POOL_MB; unbounded when unset/zero. */
    void init_budget() {
        if (const char* mb = std::getenv("LOCUS_GPU_POOL_MB")) {
            budget_ = static_cast<std::size_t>(std::atoll(mb))
                      << 20;
        }
        if (budget_ == 0) {
            budget_ = std::numeric_limits<std::size_t>::max();
        }
    }

    /** Frees LRU unpinned buffers until `bytes` fits, or fails. */
    bool reserve(std::size_t bytes) {
        if (bytes > budget_) {
            return false;
        }
        while (used_ + bytes > budget_) {
            auto victim = map_.end();
            for (auto it = map_.begin(); it != map_.end(); ++it) {
                if (it->second.pinned) {
                    continue;
                }
                if (victim == map_.end() ||
                    it->second.last_use <
                        victim->second.last_use) {
                    victim = it;
                }
            }
            if (victim == map_.end()) {
                return false;  // everything left is pinned
            }
            destroy(victim->second.buf);
            used_ -= victim->second.bytes;
            ++evictions_;
            map_.erase(victim);
        }
        return true;
    }

    /**
     * Returns a buffer holding `bytes` for host pointer `ptr`,
     * resident and pinned for the current batch. `fill` populates
     * a freshly created buffer on a miss.
     */
    Buffer acquire(const void* ptr, std::size_t bytes,
                   const std::function<void(Buffer)>& fill) {
        if (budget_ == 0) {
            init_budget();
        }
        if (auto it = map_.find(ptr); it != map_.end()) {
            it->second.last_use = ++clock_;
            it->second.pinned = true;
            ++hits_;
            return it->second.buf;
        }
        ++misses_;
        uploaded_ += bytes;
        if (!reserve(bytes)) {
            Buffer b = create(bytes);
            fill(b);
            transient_.push_back(b);
            return b;
        }
        Buffer b = create(bytes);
        fill(b);
        map_.emplace(ptr, Entry{b, bytes, ++clock_, true});
        used_ += bytes;
        return b;
    }

    /** End-of-batch: unpin all and drop transient buffers. */
    void on_batch_end() {
        for (auto& kv : map_) {
            kv.second.pinned = false;
        }
        for (auto& b : transient_) {
            destroy(b);
        }
        transient_.clear();
    }

    /** Prints [vulkan-pool] telemetry when a budget was engaged. */
    void report() const {
        if (budget_ == 0 ||
            budget_ == std::numeric_limits<std::size_t>::max()) {
            return;  // never paged (unbounded / unused)
        }
        std::fprintf(
            stderr,
            "[vulkan-pool] budget=%zuMB resident=%zuMB hits=%llu "
            "misses=%llu evictions=%llu uploaded=%lluMB\n",
            budget_ >> 20, used_ >> 20,
            static_cast<unsigned long long>(hits_),
            static_cast<unsigned long long>(misses_),
            static_cast<unsigned long long>(evictions_),
            static_cast<unsigned long long>(uploaded_ >> 20));
    }
};

}  // namespace locus::backend
