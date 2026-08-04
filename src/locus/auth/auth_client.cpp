#include "locus/auth/auth_client.hpp"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cstdint>
#include <utility>

namespace locus::auth {

AuthClient::AuthClient(
    std::vector<std::unique_ptr<HelperConnection>> conns,
    const Clock& clock, AuthConfig cfg)
    : clock_(clock),
      cfg_(cfg),
      conns_(std::move(conns)),
      positive_(clock, cfg.positive_capacity, cfg.positive_max_bytes,
                cfg.max_purge_per_insert),
      negative_(clock, cfg.negative_capacity, cfg.negative_max_bytes,
                cfg.max_purge_per_insert) {
    assert(!conns_.empty() && "AuthClient needs >= 1 connection");
}

HelperConnection& AuthClient::pick() {
    // Round-robin across helpers; each connection is itself
    // pipelined + thread-safe, so this is only for parallelism.
    const std::size_t i = rr_.fetch_add(1) % conns_.size();
    return *conns_[i];
}

AuthClient::Outcome AuthClient::query_helper(
    const std::string& credential) {
    HelperMessage req;
    req.type = "auth";
    req.set("credential", credential);
    auto resp = pick().request(std::move(req), cfg_.request_timeout);
    Outcome o;
    if (!resp) {
        return o;  // transport_ok stays false -> fail closed, no cache
    }
    o.transport_ok = true;
    const std::string* ok = resp->get("ok");
    o.ok = (ok != nullptr && *ok == "1");
    if (o.ok) {
        if (const std::string* id = resp->get("identity")) {
            o.identity = *id;
        }
        if (const std::string* na = resp->get("not_after")) {
            std::int64_t v = 0;
            const char* b = na->data();
            const char* e = b + na->size();
            if (std::from_chars(b, e, v).ec == std::errc()) {
                o.not_after = v;
            }
        }
    }
    return o;
}

AuthClient::Result AuthClient::resolve(
    const std::string& credential) {
    std::shared_ptr<InFlight> shared;
    bool leader = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (const std::string* id = positive_.lookup(credential)) {
            return {true, *id};
        }
        if (negative_.lookup(credential) != nullptr) {
            return {false, ""};
        }
        auto it = inflight_.find(credential);
        if (it != inflight_.end()) {
            shared = it->second;  // join the in-flight resolve
        } else {
            shared = std::make_shared<InFlight>();
            inflight_.emplace(credential, shared);
            leader = true;
        }
    }

    if (!leader) {
        std::unique_lock<std::mutex> lk(shared->m);
        shared->cv.wait(lk, [&] { return shared->done; });
        return shared->r;
    }

    // Leader: issue the one helper request (no locks held), then
    // cache the answer and wake any coalesced followers.
    const Outcome o = query_helper(credential);
    const Result res{o.ok, o.identity};
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (o.transport_ok) {
            if (o.ok) {
                const Nanos now = clock_.now();
                Nanos exp = 0;
                if (o.not_after > 0 && cfg_.positive_ttl_ns > 0) {
                    exp = std::min(o.not_after,
                                   now + cfg_.positive_ttl_ns);
                } else if (o.not_after > 0) {
                    exp = o.not_after;
                } else if (cfg_.positive_ttl_ns > 0) {
                    exp = now + cfg_.positive_ttl_ns;
                }
                if (exp > now) {
                    positive_.insert(credential, o.identity, exp);
                }
            } else if (cfg_.negative_ttl_ns > 0) {
                negative_.insert(credential, "",
                                 clock_.now() + cfg_.negative_ttl_ns);
            }
        }
        inflight_.erase(credential);
    }
    {
        std::lock_guard<std::mutex> lk(shared->m);
        shared->r = res;
        shared->done = true;
    }
    shared->cv.notify_all();
    return res;
}

void AuthClient::report_event(HelperMessage ev) {
    pick().notify(std::move(ev));
}

}  // namespace locus::auth
