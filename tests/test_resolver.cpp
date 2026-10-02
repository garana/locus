#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/pipeline/resolver.hpp"

using locus::pipeline::Resolver;

namespace {

// A controllable clock: tests advance `now_ms` and the resolver reads it
// through this. Milliseconds, monotonic.
struct FakeClock {
    std::int64_t now_ms = 0;
    Resolver::ClockFn fn() {
        return [this] { return now_ms; };
    }
};

}  // namespace

TEST_CASE("Resolver returns a numeric IP unchanged without a lookup",
          "[resolver]") {
    int calls = 0;
    Resolver::LookupFn lk = [&](const std::string&) {
        ++calls;
        return std::vector<std::string>{"9.9.9.9"};
    };
    Resolver r(Resolver::Options{}, lk, {}, /*start_thread=*/false);
    REQUIRE(r.resolve("127.0.0.1") == "127.0.0.1");
    REQUIRE(r.resolve("::1") == "::1");
    REQUIRE(calls == 0);            // numeric literals skip resolution
    REQUIRE(r.cached_count() == 0);  // and are not cached
}

TEST_CASE("Resolver caches a name: one lookup, then cache hits",
          "[resolver]") {
    int calls = 0;
    Resolver::LookupFn lk = [&](const std::string& h) {
        ++calls;
        REQUIRE(h == "stage.local");
        return std::vector<std::string>{"10.0.0.5"};
    };
    Resolver r(Resolver::Options{}, lk, {}, false);
    REQUIRE(r.resolve("stage.local") == "10.0.0.5");
    REQUIRE(r.resolve("stage.local") == "10.0.0.5");
    REQUIRE(r.resolve("stage.local") == "10.0.0.5");
    REQUIRE(calls == 1);  // resolved once, then served from cache
    REQUIRE(r.cached_count() == 1);
}

TEST_CASE("Resolver round-robins across a name's IP set", "[resolver]") {
    Resolver::LookupFn lk = [](const std::string&) {
        return std::vector<std::string>{"10.0.0.1", "10.0.0.2"};
    };
    Resolver r(Resolver::Options{}, lk, {}, false);
    REQUIRE(r.resolve("s") == "10.0.0.1");
    REQUIRE(r.resolve("s") == "10.0.0.2");
    REQUIRE(r.resolve("s") == "10.0.0.1");  // wraps
    REQUIRE(r.resolve("s") == "10.0.0.2");
}

TEST_CASE("Resolver does not cache a resolution failure", "[resolver]") {
    int calls = 0;
    std::vector<std::string> answer;  // starts empty (failure)
    Resolver::LookupFn lk = [&](const std::string&) {
        ++calls;
        return answer;
    };
    Resolver r(Resolver::Options{}, lk, {}, false);
    REQUIRE(r.resolve("s").empty());   // failed
    REQUIRE(r.cached_count() == 0);
    answer = {"10.0.0.9"};             // now DNS works
    REQUIRE(r.resolve("s") == "10.0.0.9");  // retried, not a cached empty
    REQUIRE(calls == 2);
}

TEST_CASE("Resolver refreshes at ~75% of the TTL", "[resolver]") {
    FakeClock clk;
    int calls = 0;
    std::string ip = "10.0.0.1";
    Resolver::LookupFn lk = [&](const std::string&) {
        ++calls;
        return std::vector<std::string>{ip};
    };
    // TTL 100 s, refresh at 75% -> due 75 s after the resolve.
    Resolver::Options opt;
    opt.default_ttl_s = 100;
    opt.max_ttl_s = 1000;
    opt.refresh_frac = 0.75;
    Resolver r(opt, lk, clk.fn(), false);

    REQUIRE(r.resolve("s") == "10.0.0.1");  // t=0, lookup #1
    REQUIRE(calls == 1);

    clk.now_ms = 74'000;  // before 75% of the TTL
    r.refresh_now();
    REQUIRE(calls == 1);  // not due yet

    ip = "10.0.0.2";      // DNS record changed
    clk.now_ms = 76'000;  // past 75% of the TTL
    r.refresh_now();
    REQUIRE(calls == 2);  // re-resolved
    REQUIRE(r.resolve("s") == "10.0.0.2");  // cache now holds the new IP
}

TEST_CASE("Resolver caps the TTL at max_ttl_s", "[resolver]") {
    FakeClock clk;
    int calls = 0;
    Resolver::LookupFn lk = [&](const std::string&) {
        ++calls;
        return std::vector<std::string>{"10.0.0.1"};
    };
    // default 1000 s but capped to 10 s -> refresh due at 7.5 s.
    Resolver::Options opt;
    opt.default_ttl_s = 1000;
    opt.max_ttl_s = 10;
    opt.refresh_frac = 0.75;
    Resolver r(opt, lk, clk.fn(), false);

    REQUIRE(r.resolve("s") == "10.0.0.1");
    clk.now_ms = 7'000;  // before 7.5 s
    r.refresh_now();
    REQUIRE(calls == 1);
    clk.now_ms = 8'000;  // past 7.5 s (cap made the TTL 10 s)
    r.refresh_now();
    REQUIRE(calls == 2);
}

TEST_CASE("Resolver keeps the old IPs when a refresh lookup fails",
          "[resolver]") {
    FakeClock clk;
    std::vector<std::string> answer{"10.0.0.1"};
    Resolver::LookupFn lk = [&](const std::string&) { return answer; };
    Resolver::Options opt;
    opt.default_ttl_s = 100;
    opt.refresh_frac = 0.75;
    Resolver r(opt, lk, clk.fn(), false);

    REQUIRE(r.resolve("s") == "10.0.0.1");
    answer.clear();        // next lookup fails
    clk.now_ms = 80'000;   // past 75%
    r.refresh_now();
    // Stale set retained, not dropped.
    REQUIRE(r.resolve("s") == "10.0.0.1");
    REQUIRE(r.cached_count() == 1);
}
