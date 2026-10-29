#include "locus/sys/histogram.hpp"

#include <stdexcept>

namespace locus::sys {

namespace {

constexpr auto kRelaxed = std::memory_order_relaxed;

}  // namespace

Histogram::Histogram(std::span<const double> bounds, double per_unit)
    : per_unit_(per_unit) {
    if (bounds.size() > kMaxBounds) {
        throw std::invalid_argument("histogram: too many bounds");
    }
    n_bounds_ = bounds.size();
    for (std::size_t i = 0; i < n_bounds_; ++i) {
        bounds_raw_[i] = static_cast<std::uint64_t>(bounds[i] * per_unit);
    }
}

void Histogram::observe(std::uint64_t v) {
    std::size_t b = 0;
    while (b < n_bounds_ && v > bounds_raw_[b]) {
        ++b;
    }
    buckets_[b].fetch_add(1, kRelaxed);
    sum_.fetch_add(v, kRelaxed);
}

Histogram::Snapshot Histogram::snapshot() const {
    Snapshot s;
    std::uint64_t run = 0;
    for (std::size_t i = 0; i <= n_bounds_; ++i) {
        run += buckets_[i].load(kRelaxed);
        s.cumulative.push_back(run);
        if (i < n_bounds_) {
            s.bounds.push_back(static_cast<double>(bounds_raw_[i]) /
                               per_unit_);
        }
    }
    // The +Inf bucket is the count by definition; using the bucket
    // sum keeps the series monotonic even mid-update.
    s.count = run;
    s.sum = static_cast<double>(sum_.load(kRelaxed)) / per_unit_;
    return s;
}

}  // namespace locus::sys
