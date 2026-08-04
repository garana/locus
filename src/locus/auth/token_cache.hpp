#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace locus::auth {

/** Monotonic-or-wall nanoseconds (the domain the expiries live in). */
using Nanos = std::int64_t;

/**
 * Injectable clock so tests can drive expiry deterministically. Use a
 * wall clock in production when expiries are absolute wall times (an
 * auth helper's `not_after`).
 */
class Clock {
  public:
    virtual ~Clock() = default;
    /** @returns The current time in nanoseconds. */
    virtual Nanos now() const = 0;
};

/** A Clock whose value is set explicitly (tests). */
class ManualClock final : public Clock {
  public:
    explicit ManualClock(Nanos t = 0) : t_(t) {}
    Nanos now() const override { return t_; }
    void set(Nanos t) { t_ = t; }
    void advance(Nanos dt) { t_ += dt; }

  private:
    Nanos t_;
};

/** A Clock backed by the system wall clock (ns since the Unix epoch). */
class WallClock final : public Clock {
  public:
    Nanos now() const override;
};

/**
 * Time-bounded credential -> value cache with approximate, "weighted"
 * LRU eviction (ported from codicis auth/token_cache). Maps an opaque
 * credential to a resolved value (an identity string, or "" for a
 * negative/"denied" entry) with a per-entry absolute expiry supplied
 * by the caller. Built on an intrusive doubly linked list:
 *
 *   - a new entry is pushed at the TAIL (the cold, probationary end);
 *   - a lookup promotes its entry one step toward the HEAD (the hot
 *     end), so a repeatedly-read entry drifts away from eviction --
 *     its position is weighted by how often it is used;
 *   - when over the entry-count OR byte budget, the TAIL entry is
 *     evicted (repeatedly) to make room.
 *
 * Admitting at the cold end means a one-off credential cannot push the
 * hot set out on a single miss. Expiry is lazy (an expired entry is
 * dropped on lookup) plus a capped opportunistic tail purge per
 * insert, so one insert never walks the whole list.
 *
 * Single-threaded: the caller serializes access (in locus, the auth
 * layer holds a mutex around it, since httplib is thread-per-request).
 */
class TokenCache {
  public:
    /**
     * @param clock     Read to test entry expiry; must outlive this.
     * @param capacity  Max resident entries (0 disables the cache).
     * @param max_bytes Max total payload bytes (key+value sums); 0 =
     *     no byte budget. An entry larger than the whole budget is
     *     never cached.
     * @param max_purge Max expired tail entries reclaimed per insert;
     *     0 = unbounded. Bounds per-insert purge work.
     */
    TokenCache(const Clock& clock, std::size_t capacity,
               std::size_t max_bytes = 0, std::size_t max_purge = 0);
    ~TokenCache();

    TokenCache(const TokenCache&) = delete;
    TokenCache& operator=(const TokenCache&) = delete;

    /**
     * Inserts or replaces an entry, refreshing its expiry and
     * position (re-admitted at the cold tail). A no-op if disabled or
     * already expired. Evicts from the tail to satisfy the budgets.
     *
     * @param expiry_ns Absolute time (cache clock domain) after which
     *     the entry is invalid.
     */
    void insert(const std::string& key, const std::string& value,
                Nanos expiry_ns);

    /**
     * Looks up a live entry, promoting it one step toward the head.
     * An expired entry is erased and reported as a miss.
     *
     * @returns Pointer to the stored value (possibly ""), or nullptr
     *     on a miss. Valid until the next mutating call.
     */
    const std::string* lookup(const std::string& key);

    /** @returns Resident entry count. */
    std::size_t size() const { return index_.size(); }
    /** @returns Total resident payload bytes (key+value sums). */
    std::size_t bytes() const { return bytes_; }

  private:
    struct Node {
        std::string key;
        std::string value;
        Nanos expiry = 0;
        Node* prev = nullptr;
        Node* next = nullptr;
    };

    void unlink(Node* n);
    void push_tail(Node* n);
    void erase(Node* n);
    void purge_expired_tail();

    const Clock& clock_;
    std::size_t capacity_;
    std::size_t max_bytes_;
    std::size_t max_purge_;
    std::size_t bytes_ = 0;
    Node* head_ = nullptr;  // hot end
    Node* tail_ = nullptr;  // cold end
    std::unordered_map<std::string, std::unique_ptr<Node>> index_;
};

}  // namespace locus::auth
