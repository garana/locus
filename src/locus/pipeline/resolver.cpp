#include "locus/pipeline/resolver.hpp"

#include <arpa/inet.h>
#include <arpa/nameser.h>
#include <netdb.h>
#include <netinet/in.h>
#include <resolv.h>
#include <sys/socket.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <optional>

namespace locus::pipeline {

namespace {

// @returns true if `s` is already a numeric IPv4 or IPv6 literal, so it
// needs no name resolution.
bool is_numeric_ip(const std::string& s) {
    unsigned char buf[sizeof(in6_addr)];
    return ::inet_pton(AF_INET, s.c_str(), buf) == 1 ||
           ::inet_pton(AF_INET6, s.c_str(), buf) == 1;
}

// Default lookup: getaddrinfo(host) -> the set of numeric IP strings it
// resolves to (v4 and v6). Empty on failure. TCP hints keep it to one
// entry per address rather than one per socket type.
std::vector<std::string> getaddrinfo_lookup(const std::string& host) {
    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (::getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0) {
        return {};
    }
    std::vector<std::string> out;
    for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
        char buf[INET6_ADDRSTRLEN] = {0};
        const char* p = nullptr;
        if (a->ai_family == AF_INET) {
            auto* s4 = reinterpret_cast<sockaddr_in*>(a->ai_addr);
            p = ::inet_ntop(AF_INET, &s4->sin_addr, buf, sizeof(buf));
        } else if (a->ai_family == AF_INET6) {
            auto* s6 = reinterpret_cast<sockaddr_in6*>(a->ai_addr);
            p = ::inet_ntop(AF_INET6, &s6->sin6_addr, buf, sizeof(buf));
        }
        if (p != nullptr) {
            std::string ip(p);
            // getaddrinfo can list the same address twice (once per
            // socktype on some systems); keep the set unique.
            if (std::find(out.begin(), out.end(), ip) == out.end()) {
                out.push_back(std::move(ip));
            }
        }
    }
    ::freeaddrinfo(res);
    return out;
}

// Default record-TTL query: a plain DNS query for host's A and AAAA
// records (res_nquery, with its own resolver state, so concurrent calls
// do not share one), the smallest TTL among the answer records, CNAMEs
// included (the chain is only valid as long as its shortest link).
// nullopt when DNS has no answer: a name only in /etc/hosts, no server,
// or a failed query.
std::optional<int> dns_record_ttl(const std::string& host) {
    struct __res_state st;
    std::memset(&st, 0, sizeof(st));
    if (::res_ninit(&st) != 0) {
        return std::nullopt;
    }
    std::optional<int> best;
    for (const int type : {ns_t_a, ns_t_aaaa}) {
        unsigned char buf[4096];
        const int n = ::res_nquery(&st, host.c_str(), ns_c_in, type, buf,
                                   sizeof(buf));
        if (n <= 0) {
            continue;
        }
        ns_msg msg;
        if (::ns_initparse(buf, n, &msg) != 0) {
            continue;
        }
        const int count = ns_msg_count(msg, ns_s_an);
        for (int i = 0; i < count; ++i) {
            ns_rr rr;
            if (::ns_parserr(&msg, ns_s_an, i, &rr) != 0) {
                continue;
            }
            const int ttl = static_cast<int>(ns_rr_ttl(rr));
            best = best ? std::min(*best, ttl) : ttl;
        }
    }
    ::res_nclose(&st);
    return best;
}

}  // namespace

Resolver::Resolver(Options opt, LookupFn lookup, ClockFn clock,
                   bool start_thread, TtlFn ttl)
    : opt_(opt),
      clock_(std::move(clock)) {
    // The DNS TTL query goes with the real lookup only: a test that
    // injects its lookup must not reach the network for a TTL.
    if (ttl) {
        ttl_ = std::move(ttl);
    } else if (!lookup) {
        ttl_ = &dns_record_ttl;
    }
    lookup_ = lookup ? std::move(lookup) : &getaddrinfo_lookup;
    if (opt_.default_ttl_s < 1) {
        opt_.default_ttl_s = 1;
    }
    if (opt_.max_ttl_s < 1) {
        opt_.max_ttl_s = 1;
    }
    if (opt_.refresh_frac <= 0.0 || opt_.refresh_frac > 1.0) {
        opt_.refresh_frac = 0.75;
    }
    if (opt_.min_refresh_s < 0) {
        opt_.min_refresh_s = 0;
    }
    if (start_thread) {
        threaded_ = true;
        thread_ = std::thread(&Resolver::run, this);
    }
}

Resolver::~Resolver() {
    if (threaded_) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }
}

std::int64_t Resolver::now_ms() {
    if (clock_) {
        return clock_();
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::int64_t Resolver::ttl_ms(std::optional<int> record_s) const {
    const int ttl = record_s
                        ? std::clamp(*record_s, 1, opt_.max_ttl_s)
                        : std::min(opt_.default_ttl_s, opt_.max_ttl_s);
    return static_cast<std::int64_t>(ttl) * 1000;
}

std::int64_t Resolver::refresh_delay_ms(bool failed,
                                        std::optional<int> record_s) const {
    if (failed) {
        // Short negative TTL: retry a bad name soon, but not every call.
        return std::min<std::int64_t>(ttl_ms(std::nullopt), 5000);
    }
    const auto delay = static_cast<std::int64_t>(
        static_cast<double>(ttl_ms(record_s)) * opt_.refresh_frac);
    return std::max<std::int64_t>(
        delay, static_cast<std::int64_t>(opt_.min_refresh_s) * 1000);
}

std::string Resolver::resolve(const std::string& host) {
    if (is_numeric_ip(host)) {
        return host;  // already an address; nothing to cache
    }
    std::unique_lock<std::mutex> lk(mu_);
    if (auto it = cache_.find(host); it != cache_.end()) {
        Entry& e = it->second;
        if (!e.ips.empty()) {
            return e.ips[e.cursor++ % e.ips.size()];  // positive hit
        }
        // Negative hit: a recent lookup failed and is cached for a short
        // negative TTL. Return empty WITHOUT re-asking -- the background
        // refresh retries at refresh_due_ms. This bounds a bad or dead
        // name to one lookup per negative TTL instead of one per call
        // (which, with real getaddrinfo on the serve thread, would be a
        // DNS query -- possibly a multi-second timeout -- every call).
        return {};
    }
    // Miss: resolve once, synchronously, with the lock released so a slow
    // lookup does not stall other resolve() callers.
    lk.unlock();
    std::vector<std::string> ips = lookup_(host);
    const std::optional<int> record =
        ttl_ && !ips.empty() ? ttl_(host) : std::nullopt;
    lk.lock();
    Entry& e = cache_[host];
    e.ips = std::move(ips);  // empty == a cached negative entry
    e.cursor = 0;
    e.refresh_due_ms = now_ms() + refresh_delay_ms(e.ips.empty(), record);
    return e.ips.empty() ? std::string{}
                         : e.ips[e.cursor++ % e.ips.size()];
}

void Resolver::prime(const std::string& host) {
    if (is_numeric_ip(host)) {
        return;  // nothing to resolve
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (cache_.find(host) != cache_.end()) {
        return;  // already cached (good or negative): leave it
    }
    // Insert a due-now placeholder (empty ip set). resolve() treats it as
    // a negative hit (returns empty) until the background thread's next
    // refresh_now() fills it -- so this records intent without a blocking
    // lookup on the caller's thread.
    Entry& e = cache_[host];
    e.refresh_due_ms = now_ms();
}

void Resolver::refresh_now() {
    // Snapshot the names that are due, under the lock.
    std::vector<std::string> due;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const std::int64_t now = now_ms();
        for (const auto& [host, e] : cache_) {
            if (now >= e.refresh_due_ms) {
                due.push_back(host);
            }
        }
    }
    // Re-resolve each, with the lock released during the blocking lookup.
    for (const std::string& host : due) {
        std::vector<std::string> ips = lookup_(host);
        const std::optional<int> record =
            ttl_ && !ips.empty() ? ttl_(host) : std::nullopt;
        std::lock_guard<std::mutex> lk(mu_);
        auto it = cache_.find(host);
        if (it == cache_.end()) {
            continue;  // evicted meanwhile
        }
        Entry& e = it->second;
        // Read before the move below: a moved-from vector is empty, which
        // scheduled every successful refresh at the 5 s negative TTL.
        const bool failed = ips.empty();
        if (!failed) {
            e.ips = std::move(ips);  // refresh (or a negative entry
            e.cursor = 0;            // recovering) -> now positive
        }
        // On failure keep whatever is cached (a stale positive set, or
        // an empty negative entry) and reschedule; refresh_delay_ms
        // gives the short negative TTL so a failing name is retried soon
        // without being dropped. Same policy for both failure paths.
        e.refresh_due_ms = now_ms() + refresh_delay_ms(failed, record);
    }
}

std::size_t Resolver::cached_count() {
    std::lock_guard<std::mutex> lk(mu_);
    return cache_.size();
}

void Resolver::run() {
    for (;;) {
        std::unique_lock<std::mutex> lk(mu_);
        // Wake on stop, or every second to check for due refreshes. The
        // tick is the refresh granularity; DNS TTLs are seconds-scale, so
        // a 1 s tick is fine and keeps the thread cheap.
        if (cv_.wait_for(lk, std::chrono::seconds(1),
                         [this] { return stop_; })) {
            return;  // stop requested
        }
        lk.unlock();
        refresh_now();
    }
}

std::vector<SrvTarget> dns_srv_lookup(const std::string& name) {
    std::vector<SrvTarget> out;
    struct __res_state st;
    std::memset(&st, 0, sizeof(st));
    if (::res_ninit(&st) != 0) {
        return out;
    }
    unsigned char buf[8192];
    const int n = ::res_nquery(&st, name.c_str(), ns_c_in, ns_t_srv, buf,
                               sizeof(buf));
    ns_msg msg;
    if (n > 0 && ::ns_initparse(buf, n, &msg) == 0) {
        const int count = ns_msg_count(msg, ns_s_an);
        for (int i = 0; i < count; ++i) {
            ns_rr rr;
            if (::ns_parserr(&msg, ns_s_an, i, &rr) != 0 ||
                ns_rr_type(rr) != ns_t_srv || ns_rr_rdlen(rr) < 7) {
                continue;  // a CNAME in the chain, or a short record
            }
            // RDATA: priority, weight, port (16 bits each, big-endian),
            // then the target name, maybe compressed.
            const unsigned char* r = ns_rr_rdata(rr);
            char target[NS_MAXDNAME];
            if (::dn_expand(ns_msg_base(msg), ns_msg_end(msg), r + 6, target,
                            sizeof(target)) < 0 ||
                target[0] == '\0') {
                continue;  // bad, or "." (no service at this name)
            }
            SrvTarget t;
            t.priority = (r[0] << 8) | r[1];
            t.weight = (r[2] << 8) | r[3];
            t.port = (r[4] << 8) | r[5];
            t.host = target;
            out.push_back(std::move(t));
        }
    }
    ::res_nclose(&st);
    return out;
}

}  // namespace locus::pipeline
