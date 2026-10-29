#include "locus/http/metrics.hpp"

namespace locus::http {

namespace {

constexpr auto kRelaxed = std::memory_order_relaxed;

// Body sizes in bytes: a short JSON request at the bottom, a large
// agent prompt or an image response near the top.
constexpr double kByteBounds[] = {256,       1024,      4096,
                                  16384,     65536,     262144,
                                  1048576,   4194304,   16777216,
                                  67108864};
// Head received to response written, in seconds: /health at the bottom,
// a long CPU generation at the top.
constexpr double kSecondBounds[] = {0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
                                    0.25,  0.5,   1,    2.5,   5,    10,
                                    30,    60,    300,  1800};

}  // namespace

HttpMetrics::Route::Route()
    : codes{},
      request_bytes(kByteBounds, 1.0),
      response_bytes(kByteBounds, 1.0),
      seconds(kSecondBounds) {}

HttpMetrics::HttpMetrics(std::vector<std::string> routes)
    : names_(std::move(routes)) {
    for (std::size_t i = 0; i <= names_.size(); ++i) {
        routes_.push_back(std::make_unique<Route>());
    }
}

std::size_t HttpMetrics::route(std::string_view path) const {
    for (std::size_t i = 0; i < names_.size(); ++i) {
        const std::string_view n = names_[i];
        if (!n.empty() && n.back() == '/') {
            if (path.size() > n.size() && path.substr(0, n.size()) == n) {
                return i;
            }
        } else if (n == path) {
            return i;
        }
    }
    return other();
}

std::vector<std::string> HttpMetrics::names() const {
    auto out = names_;
    out.emplace_back("other");
    return out;
}

void HttpMetrics::record(std::size_t route, int status,
                         std::uint64_t request_bytes,
                         std::uint64_t response_bytes, std::uint64_t ns) {
    if (route > other()) route = other();
    Route& r = *routes_[route];
    if (status >= kMinCode && status <= kMaxCode) {
        r.codes[static_cast<std::size_t>(status - kMinCode)].fetch_add(
            1, kRelaxed);
    } else {
        r.out_of_range.fetch_add(1, kRelaxed);
    }
    r.request_bytes.observe(request_bytes);
    r.response_bytes.observe(response_bytes);
    r.seconds.observe(ns);
}

}  // namespace locus::http
