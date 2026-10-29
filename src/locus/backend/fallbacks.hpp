#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace locus::backend {

/**
 * One fallback counter (i#285): how often a faster path handed its work
 * to a slower one, for example a GPU forward to the host's per-op loop.
 * The registry copies the three names, so any string works.
 */
struct Fallback {
    std::string from;    ///< the path that declined ("vulkan", "cuda")
    std::string to;      ///< where the work went ("cpu", "per_op", ...)
    std::string reason;  ///< why, in a few words
    std::atomic<std::uint64_t> n{0};

    Fallback(const char* f, const char* t, const char* r)
        : from(f), to(t), reason(r) {}

    /** Counts one fallback (a relaxed add). */
    void add() { n.fetch_add(1, std::memory_order_relaxed); }
};

/**
 * Finds or registers the counter for (from, to, reason). The reference
 * stays valid for the process (the registry is never destroyed, so a
 * thread still running at exit cannot reach freed memory), so a call
 * site keeps it in a static:
 *
 *     static Fallback& f = fallback("cuda", "cpu", "device error");
 *     f.add();
 *
 * Equal text shares one counter. Takes a mutex; the add on the
 * returned counter does not.
 *
 * @returns The counter.
 */
Fallback& fallback(const char* from, const char* to, const char* reason);

/** Calls fn on each registered counter, in registration order. */
void for_each_fallback(const std::function<void(const Fallback&)>& fn);

}  // namespace locus::backend
