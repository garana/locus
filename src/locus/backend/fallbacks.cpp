#include "locus/backend/fallbacks.hpp"

#include <cstring>
#include <deque>
#include <mutex>

namespace locus::backend {

namespace {

std::mutex& registry_mu() {
    static std::mutex mu;
    return mu;
}

/** A deque: registering never moves the counters already handed out. */
std::deque<Fallback>& registry() {
    static std::deque<Fallback> r;
    return r;
}

}  // namespace

Fallback& fallback(const char* from, const char* to, const char* reason) {
    std::lock_guard lk(registry_mu());
    for (Fallback& f : registry()) {
        if (std::strcmp(f.from, from) == 0 && std::strcmp(f.to, to) == 0 &&
            std::strcmp(f.reason, reason) == 0) {
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
