#include "locus/engine/metrics.hpp"

#include <stdexcept>

namespace locus::engine {

namespace {

// Bucket bounds in seconds, wide enough for a CPU prefill of a long
// agent prompt (minutes) at the top and a fast GPU token at the bottom.
constexpr double kPhaseBounds[] = {0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                                   1,    2.5,   5,    10,  30,   60,
                                   120,  300,   600};
constexpr double kTokenBounds[] = {0.001, 0.0025, 0.005, 0.01, 0.02,
                                   0.035, 0.05,   0.075, 0.1,  0.15,
                                   0.25,  0.5,    1,     2.5};

constexpr auto kRelaxed = std::memory_order_relaxed;

}  // namespace

Histogram::Histogram(std::span<const double> bounds) {
    if (bounds.size() > kMaxBounds) {
        throw std::invalid_argument("histogram: too many bounds");
    }
    n_bounds_ = bounds.size();
    for (std::size_t i = 0; i < n_bounds_; ++i) {
        bounds_ns_[i] = static_cast<std::uint64_t>(bounds[i] * 1e9);
    }
}

void Histogram::observe(std::uint64_t ns) {
    std::size_t b = 0;
    while (b < n_bounds_ && ns > bounds_ns_[b]) {
        ++b;
    }
    buckets_[b].fetch_add(1, kRelaxed);
    sum_ns_.fetch_add(ns, kRelaxed);
}

Histogram::Snapshot Histogram::snapshot() const {
    Snapshot s;
    std::uint64_t run = 0;
    for (std::size_t i = 0; i <= n_bounds_; ++i) {
        run += buckets_[i].load(kRelaxed);
        s.cumulative.push_back(run);
        if (i < n_bounds_) {
            s.bounds.push_back(static_cast<double>(bounds_ns_[i]) / 1e9);
        }
    }
    // The +Inf bucket is the count by definition; using the bucket
    // sum keeps the series monotonic even mid-update.
    s.count = run;
    s.sum_seconds = static_cast<double>(sum_ns_.load(kRelaxed)) / 1e9;
    return s;
}

EngineMetrics::EngineMetrics()
    : queue(kPhaseBounds),
      prefill(kPhaseBounds),
      ttft(kPhaseBounds),
      decode_token(kTokenBounds),
      request(kPhaseBounds) {}

}  // namespace locus::engine
