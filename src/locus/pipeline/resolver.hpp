#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace locus::pipeline {

/**
 * Cached, background-refreshed hostname resolver for downstream dials
 * (multi-server, issue 40). resolve() returns an IP for a name from an
 * in-memory cache without a DNS round-trip on the hot path; a background
 * thread re-resolves each cached name at ~refresh_frac of its TTL, so
 * the cache tracks DNS changes and the serve loop never blocks on a
 * lookup after the first.
 *
 * TTL note: the real record TTL is not available through getaddrinfo, so
 * this component uses a CONFIGURED TTL (Options.default_ttl_s, capped by
 * max_ttl_s) and applies refresh_frac to it. A later change can feed the
 * actual record TTL in without changing this API (the refresh math is
 * the same). Until then default_ttl_s IS the effective TTL.
 *
 * Thread-safety: resolve() and the background refresh run concurrently;
 * a mutex guards the cache and blocking lookups run outside it. The
 * lookup and clock are injectable so tests are deterministic without
 * real DNS or real time; with the defaults it uses getaddrinfo and a
 * steady clock.
 */
class Resolver {
  public:
    struct Options {
        int default_ttl_s = 30;      /**< TTL when no record TTL known. */
        int max_ttl_s = 300;         /**< Cap on any TTL. */
        double refresh_frac = 0.75;  /**< Refresh at this fraction of TTL. */
    };

    /** Resolves a hostname to zero or more numeric IP strings. Empty on
     * failure. Default: getaddrinfo. Injectable for tests. */
    using LookupFn =
        std::function<std::vector<std::string>(const std::string&)>;
    /** Monotonic clock in milliseconds. Default: steady_clock.
     * Injectable so tests drive TTL/refresh timing deterministically. */
    using ClockFn = std::function<std::int64_t()>;

    /**
     * @param opt TTL / refresh policy.
     * @param lookup Resolution function (default getaddrinfo).
     * @param clock Monotonic-ms clock (default steady_clock).
     * @param start_thread Start the background refresh thread (default
     *     true). Tests pass false and drive refresh_now() by hand.
     */
    explicit Resolver(Options opt, LookupFn lookup = {}, ClockFn clock = {},
                      bool start_thread = true);
    ~Resolver();
    Resolver(const Resolver&) = delete;
    Resolver& operator=(const Resolver&) = delete;

    /**
     * Returns an IP for `host`. A numeric IP is returned unchanged (no
     * caching). On a cache hit, returns one of the cached IPs
     * round-robin without a lookup; on a miss, resolves once
     * (synchronously, blocking) and caches the result. A FAILURE is
     * cached as a negative entry for a short negative TTL (min(TTL, 5 s))
     * and then returns empty on subsequent calls without re-asking, so a
     * bad or dead name costs one lookup per negative TTL rather than one
     * per call; the background refresh clears it when the name resolves
     * again.
     * @returns the IP, or an empty string if resolution failed (or is
     *     within a cached negative entry).
     */
    std::string resolve(const std::string& host);

    /**
     * Registers `host` to be resolved WITHOUT doing the lookup now:
     * inserts a due-now placeholder (if the name is not already cached)
     * that the background refresh thread fills on its next tick. Unlike
     * resolve(), this never blocks on DNS, so it is safe to call from a
     * latency-sensitive context (e.g. a config reload applied on the
     * serve loop): it records intent and the warm arrives off-thread a
     * tick later. A name already cached (good or negative) is left
     * untouched. No-op on a numeric IP.
     */
    void prime(const std::string& host);

    /** Runs one refresh pass now: re-resolves every cached name whose
     * refresh time has passed. The background thread calls this on a
     * tick; tests call it directly (with start_thread=false). */
    void refresh_now();

    /** @returns the number of names currently cached (test aid). */
    std::size_t cached_count();

  private:
    struct Entry {
        std::vector<std::string> ips;
        std::size_t cursor = 0;        // round-robin position
        std::int64_t refresh_due_ms = 0;
    };

    std::int64_t now_ms();
    std::int64_t ttl_ms() const;  // clamped configured TTL
    /** Delay until the next refresh of an entry just (re)resolved.
     * Success: refresh_frac of the TTL. Failure: a short negative TTL
     * (min(TTL, 5 s)) so a bad name is retried soon without re-asking
     * every call -- the same policy for a failed initial resolve and a
     * failed refresh. */
    std::int64_t refresh_delay_ms(bool failed) const;
    void run();                   // background thread body

    Options opt_;
    LookupFn lookup_;
    ClockFn clock_;

    std::mutex mu_;
    std::unordered_map<std::string, Entry> cache_;

    std::condition_variable cv_;
    bool stop_ = false;
    bool threaded_ = false;
    std::thread thread_;
};

}  // namespace locus::pipeline
