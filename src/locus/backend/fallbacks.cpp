#include "locus/backend/fallbacks.hpp"

#include <deque>
#include <mutex>

namespace locus::backend {

namespace {

// Both are leaked on purpose: a GPU thread still running during static
// destruction at exit must not lock or append to destroyed objects.
std::mutex& registry_mu() {
    static auto* mu = new std::mutex;
    return *mu;
}

/** A deque: registering never moves the counters already handed out. */
std::deque<Fallback>& registry() {
    static auto* r = new std::deque<Fallback>;
    return *r;
}

}  // namespace

Fallback& fallback(const char* from, const char* to, const char* reason) {
    std::lock_guard lk(registry_mu());
    for (Fallback& f : registry()) {
        if (f.from == from && f.to == to && f.reason == reason) {
            return f;
        }
    }
    return registry().emplace_back(from, to, reason);
}

void for_each_fallback(const std::function<void(const Fallback&)>& fn) {
    std::lock_guard lk(registry_mu());
    for (const Fallback& f : registry()) {
        fn(f);
    }
}

}  // namespace locus::backend
