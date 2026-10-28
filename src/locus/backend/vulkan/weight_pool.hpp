#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
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
        /** Host memory the buffer was made from (read in place for an
         *  external entry, copied otherwise); null when unknown, and
         *  then release_range matches the key. */
        const void* src = nullptr;
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
     * a freshly created buffer on a miss. `src`, when given, is the
     * host memory `fill` copies from (release_range matches it).
     */
    Buffer acquire(const void* ptr, std::size_t bytes,
                   const std::function<void(Buffer)>& fill,
                   const void* src = nullptr) {
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
        map_.emplace(ptr, Entry{b, bytes, ++clock_, true, src});
        used_ += bytes;
        return b;
    }

    /**
     * Like acquire, for a buffer over host memory used in place (the
     * mmap'd weights imported, not copied): on a miss `make` builds it,
     * and the entry counts 0 bytes against the budget.
     * @param src The host memory the buffer reads (release_range).
     * @returns The buffer, or nullopt when make can't (the caller
     *          then copies through acquire).
     */
    std::optional<Buffer> acquire_external(
        const void* ptr, const void* src,
        const std::function<std::optional<Buffer>()>& make) {
        if (auto it = map_.find(ptr); it != map_.end()) {
            it->second.last_use = ++clock_;
            it->second.pinned = true;
            ++hits_;
            return it->second.buf;
        }
        std::optional<Buffer> b = make();
        if (b) {
            ++misses_;
            map_.emplace(ptr, Entry{*b, 0, ++clock_, true, src});
        }
        return b;
    }

    /**
     * Destroys the entries made from host memory in [p, p + n): those
     * whose src lies there, or, without a src, whose key does. Called
     * before that memory goes away (a model file unmapped), so no
     * buffer outlives the pages it imports, and a later mapping (or a
     * later model's heap) at the same address never hits a stale
     * entry. No released entry may be pinned: an open batch could
     * still use it.
     */
    void release_range(const void* p, std::size_t n) {
        const auto b = reinterpret_cast<std::uintptr_t>(p);
        for (auto it = map_.begin(); it != map_.end();) {
            const auto at = reinterpret_cast<std::uintptr_t>(
                it->second.src != nullptr ? it->second.src : it->first);
            if (at >= b && at - b < n) {
                assert(!it->second.pinned &&
                       "release_range under an open batch");
                destroy(it->second.buf);
                used_ -= it->second.bytes;
                it = map_.erase(it);
            } else {
                ++it;
            }
        }
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
