#pragma once

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
 * With `spilled` set (a discrete GPU, i#1164), a buffer that missed
 * the device heap (the backend fell back to pinned system memory)
 * is a spill. The device budget then drops to what the pool holds on
 * the device, and spills are cached apart, under `spill_cap_` bytes
 * with their own LRU; past it a spill serves one batch. So pinned
 * system memory stays bounded, while a model a little larger than
 * the device keeps its spilled weights resident.
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
        /** In system memory (a spill), counted in spill_used_. */
        bool spill = false;
    };

    /** Allocates a device buffer of `bytes`. */
    std::function<Buffer(std::size_t)> create;
    /** Frees a buffer from `create`. */
    std::function<void(Buffer)> destroy;
    /** Optional: whether a buffer from `create` missed the device
     *  heap (it lives in system memory instead). */
    std::function<bool(Buffer)> spilled;

    std::unordered_map<const void*, Entry> map_;
    std::vector<Buffer> transient_;
    std::size_t budget_ = 0;  // 0 until init; SIZE_MAX = unbounded
    std::size_t used_ = 0;    // device bytes (spills not counted)
    /** Most bytes of spills kept across batches. */
    std::size_t spill_cap_ = 0;
    std::size_t spill_used_ = 0;
    std::uint64_t clock_ = 0;
    std::uint64_t hits_ = 0, misses_ = 0, evictions_ = 0;
    std::uint64_t uploaded_ = 0;
    /** Buffers that missed the device heap. */
    std::uint64_t spills_ = 0;

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

    /** Frees LRU unpinned entries of one kind (spills or device ones)
     *  until `used + bytes <= cap`, or fails. */
    bool evict_until(bool spill, std::size_t& used, std::size_t cap,
                     std::size_t bytes) {
        if (bytes > cap) {
            return false;
        }
        while (used + bytes > cap) {
            auto victim = map_.end();
            for (auto it = map_.begin(); it != map_.end(); ++it) {
                if (it->second.pinned || it->second.spill != spill) {
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
            used -= victim->second.bytes;
            ++evictions_;
            map_.erase(victim);
        }
        return true;
    }

    /** Frees LRU unpinned device buffers until `bytes` fits, or fails. */
    bool reserve(std::size_t bytes) {
        return evict_until(false, used_, budget_, bytes);
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
        // Once the device is full, a miss goes to the spill cache
        // first: evicting device entries for it would cycle a model
        // whose every weight is read each token (each eviction is the
        // next miss). When the spill cache is at its cap with this
        // batch's own pinned spills, the miss falls through to
        // reserve() and evicts device entries instead.
        if (device_full_ && used_ + bytes > budget_ &&
            evict_until(true, spill_used_, spill_cap_, bytes)) {
            Buffer b = create(bytes);
            fill(b);
            keep(ptr, b, bytes, src);
            return b;
        }
        if (!reserve(bytes)) {
            Buffer b = create(bytes);
            fill(b);
            transient_.push_back(b);
            return b;
        }
        Buffer b = create(bytes);
        fill(b);
        if (spilled && spilled(b)) {
            // The device is full: hold no more there than now (1, not
            // 0: 0 means "not set"); keep the spill within spill_cap_.
            device_full_ = true;
            budget_ = used_ > 0 ? used_ : 1;
            if (evict_until(true, spill_used_, spill_cap_, bytes)) {
                keep(ptr, b, bytes, src);
            } else {
                ++spills_;
                transient_.push_back(b);
            }
            return b;
        }
        keep(ptr, b, bytes, src);
        return b;
    }

    /** Whether a buffer missed the device heap since this pool began. */
    bool device_full_ = false;

    /** Caches a new buffer as a spill or a device entry, by where it
     *  landed (a device entry past the budget raises the budget: the
     *  device had room after all). */
    void keep(const void* ptr, Buffer b, std::size_t bytes,
              const void* src) {
        const bool spill = spilled && spilled(b);
        map_.emplace(ptr, Entry{b, bytes, ++clock_, true, src, spill});
        if (spill) {
            ++spills_;
            spill_used_ += bytes;
        } else {
            used_ += bytes;
            if (used_ > budget_) {
                budget_ = used_;
            }
        }
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
     * entry. The caller makes sure no open batch still uses them.
     */
    void release_range(const void* p, std::size_t n) {
        const auto b = reinterpret_cast<std::uintptr_t>(p);
        for (auto it = map_.begin(); it != map_.end();) {
            const auto at = reinterpret_cast<std::uintptr_t>(
                it->second.src != nullptr ? it->second.src : it->first);
            if (at >= b && at - b < n) {
                destroy(it->second.buf);
                (it->second.spill ? spill_used_ : used_) -=
                    it->second.bytes;
                it = map_.erase(it);
            } else {
                ++it;
            }
        }
    }

    /**
     * Destroys every buffer and starts over as a new pool (between
     * models): no device or spill bytes held, the device not known to
     * be full, and the budget read again on the next acquire. The
     * hit, miss, eviction, spill and upload counts keep adding up.
     */
    void reset() {
        for (auto& [ptr, e] : map_) {
            destroy(e.buf);
        }
        map_.clear();
        for (auto& b : transient_) {
            destroy(b);
        }
        transient_.clear();
        used_ = 0;
        spill_used_ = 0;
        device_full_ = false;
        budget_ = 0;
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
            "misses=%llu evictions=%llu spills=%llu spilled=%zuMB of "
            "%zuMB uploaded=%lluMB\n",
            budget_ >> 20, used_ >> 20,
            static_cast<unsigned long long>(hits_),
            static_cast<unsigned long long>(misses_),
            static_cast<unsigned long long>(evictions_),
            static_cast<unsigned long long>(spills_), spill_used_ >> 20,
            spill_cap_ >> 20,
            static_cast<unsigned long long>(uploaded_ >> 20));
    }
};

}  // namespace locus::backend
