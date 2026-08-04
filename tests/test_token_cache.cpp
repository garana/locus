#include "catch_amalgamated.hpp"
#include "locus/auth/token_cache.hpp"

using locus::auth::ManualClock;
using locus::auth::TokenCache;

namespace {
constexpr locus::auth::Nanos kFar = 1'000'000'000;  // 1s from t0
}

TEST_CASE("token cache hit, miss, and negative entries", "[auth]") {
    ManualClock clk;
    TokenCache c(clk, /*capacity=*/8);
    REQUIRE(c.lookup("k") == nullptr);  // miss

    c.insert("tok", "user-1", kFar);
    const std::string* v = c.lookup("tok");
    REQUIRE(v != nullptr);
    REQUIRE(*v == "user-1");

    // A negative ("denied") entry caches the empty string.
    c.insert("bad", "", kFar);
    const std::string* n = c.lookup("bad");
    REQUIRE(n != nullptr);
    REQUIRE(n->empty());
    REQUIRE(c.size() == 2);
}

TEST_CASE("token cache expires lazily on lookup", "[auth]") {
    ManualClock clk;
    TokenCache c(clk, 8);
    c.insert("tok", "u", /*expiry=*/100);
    clk.set(100);  // now == expiry -> expired (>= expiry)
    REQUIRE(c.lookup("tok") == nullptr);
    REQUIRE(c.size() == 0);
    // An already-expired insert is a no-op.
    c.insert("old", "u", 50);
    REQUIRE(c.size() == 0);
}

TEST_CASE("token cache evicts the cold tail at capacity", "[auth]") {
    ManualClock clk;
    TokenCache c(clk, /*capacity=*/2);
    c.insert("a", "1", kFar);  // -> [a]     (a is head+tail)
    c.insert("b", "2", kFar);  // -> [a, b]  (a head, b cold tail)
    // Over capacity: the cold tail (b, the previous newest, unread)
    // is evicted, then c is admitted at the tail. The un-read head
    // entry a is protected -- the probationary-tail semantics.
    c.insert("c", "3", kFar);
    REQUIRE(c.size() == 2);
    REQUIRE(c.lookup("a") != nullptr);  // head survives
    REQUIRE(c.lookup("b") == nullptr);  // cold tail evicted
    REQUIRE(c.lookup("c") != nullptr);  // newly admitted
}

TEST_CASE("weighted LRU: a read entry survives a cold one", "[auth]") {
    ManualClock clk;
    TokenCache c(clk, /*capacity=*/2);
    c.insert("a", "1", kFar);  // tail (cold)
    c.insert("b", "2", kFar);  // a is now tail, b is head-ish
    // Read a -> promotes it toward the head, past b.
    REQUIRE(c.lookup("a") != nullptr);
    c.insert("c", "3", kFar);  // evicts the cold tail, which is b
    REQUIRE(c.lookup("a") != nullptr);  // survived by being read
    REQUIRE(c.lookup("b") == nullptr);  // the cold one went
    REQUIRE(c.lookup("c") != nullptr);
}

TEST_CASE("token cache honors the byte budget", "[auth]") {
    ManualClock clk;
    // Budget fits ~2 small entries (key+value bytes).
    TokenCache c(clk, /*capacity=*/100, /*max_bytes=*/12);
    c.insert("aa", "11", kFar);  // 4 bytes
    c.insert("bb", "22", kFar);  // 8 total
    c.insert("cc", "33", kFar);  // 12 -> ok; next forces eviction
    REQUIRE(c.bytes() <= 12);
    c.insert("dd", "44", kFar);
    REQUIRE(c.bytes() <= 12);
    // An entry bigger than the whole budget is never cached.
    c.insert("huge", "way-too-many-bytes-here", kFar);
    REQUIRE(c.lookup("huge") == nullptr);
}

TEST_CASE("token cache disabled at capacity 0", "[auth]") {
    ManualClock clk;
    TokenCache c(clk, /*capacity=*/0);
    c.insert("k", "v", kFar);
    REQUIRE(c.lookup("k") == nullptr);
    REQUIRE(c.size() == 0);
}

TEST_CASE("token cache replace refreshes value and expiry", "[auth]") {
    ManualClock clk;
    TokenCache c(clk, 8);
    c.insert("k", "old", 100);
    c.insert("k", "new", kFar);  // replace
    REQUIRE(c.size() == 1);
    const std::string* v = c.lookup("k");
    REQUIRE(v != nullptr);
    REQUIRE(*v == "new");
    clk.set(100);  // past the old expiry, before the new one
    REQUIRE(c.lookup("k") != nullptr);  // uses the refreshed expiry
}
