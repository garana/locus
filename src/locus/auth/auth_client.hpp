#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "locus/auth/helper_connection.hpp"
#include "locus/auth/helper_message.hpp"
#include "locus/auth/token_cache.hpp"

namespace locus::auth {

/** AuthClient tuning (derived from server config by the caller). */
struct AuthConfig {
    /** Per-helper-request timeout. */
    std::chrono::milliseconds request_timeout{1000};
    /** Positive (allowed) cache: entry cap, byte budget, and a cap
     * on the helper's not_after lifetime (0 = honor not_after). */
    std::size_t positive_capacity = 4096;
    std::size_t positive_max_bytes = 0;
    Nanos positive_ttl_ns = 0;
    /** Negative (denied) cache: entry cap, byte budget, lifetime. */
    std::size_t negative_capacity = 4096;
    std::size_t negative_max_bytes = 0;
    Nanos negative_ttl_ns = 2'000'000'000;  // 2s
    /** Expired-tail purge cap per cache insert. */
    std::size_t max_purge_per_insert = 8;
};

/**
 * Resolves an opaque credential to an allow/deny + identity by asking
 * a pool of helper connections, with a positive+negative TokenCache
 * and single-flight coalescing (concurrent resolves of the SAME
 * credential issue one helper request and share the answer). Also
 * pushes locus's request-lifecycle events to the helper over the same
 * sockets. Thread-safe: many httplib request threads share one
 * client. Fails CLOSED -- a timeout/broken helper resolves to deny
 * and is not cached (so a transient hiccup is not remembered).
 */
class AuthClient {
  public:
    struct Result {
        bool ok = false;
        std::string identity;
    };

    /** @param conns Helper connections (>=1); injected so the core
     *      is testable over socketpairs. @param clock For cache
     *      expiry (wall clock in production; must outlive this). */
    AuthClient(std::vector<std::unique_ptr<HelperConnection>> conns,
               const Clock& clock, AuthConfig cfg);

    AuthClient(const AuthClient&) = delete;
    AuthClient& operator=(const AuthClient&) = delete;

    /** Resolves `credential`. Cache hit is immediate; a miss issues
     * (or joins) one helper request. Blocking (safe on an httplib
     * request thread). */
    Result resolve(const std::string& credential);

    /** Fire-and-forget push of a request-lifecycle event (create /
     * update / cancel) to a helper. */
    void report_event(HelperMessage ev);

  private:
    struct Outcome {
        bool ok = false;
        std::string identity;
        Nanos not_after = 0;
        bool transport_ok = false;  // false = timeout/broken helper
    };
    /** In-flight single-flight state shared by coalesced resolvers. */
    struct InFlight {
        std::mutex m;
        std::condition_variable cv;
        bool done = false;
        Result r;
    };

    HelperConnection& pick();
    Outcome query_helper(const std::string& credential);

    const Clock& clock_;
    AuthConfig cfg_;
    std::vector<std::unique_ptr<HelperConnection>> conns_;
    std::atomic<std::size_t> rr_{0};

    std::mutex mu_;  // guards caches + inflight_
    TokenCache positive_;
    TokenCache negative_;
    std::unordered_map<std::string, std::shared_ptr<InFlight>>
        inflight_;
};

}  // namespace locus::auth
