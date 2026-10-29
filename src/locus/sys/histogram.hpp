#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace locus::sys {

/**
 * A Prometheus-style histogram with fixed bucket bounds over integer
 * observations (nanoseconds, bytes, ...). Any number of threads may
 * observe and read without a lock: every field is a relaxed atomic, so
 * a reader may see the sum one observation apart from the buckets,
 * never a torn value (the count is the bucket total). observe() costs a
 * short bucket scan and two relaxed adds.
 */
class Histogram {
  public:
    /** Upper bounds (ascending), plus the implicit +Inf. */
    static constexpr std::size_t kMaxBounds = 16;

    /**
     * @param bounds Ascending upper bounds in the rendered unit (at
     *     most kMaxBounds).
     * @param per_unit Observations per rendered unit: 1e9 for
     *     nanoseconds rendered as seconds (the default), 1 for bytes.
     * @throws std::invalid_argument on too many bounds.
     */
    explicit Histogram(std::span<const double> bounds,
                       double per_unit = 1e9);

    /** Records one observation (in the observed unit). */
    void observe(std::uint64_t v);

    /** A consistent-enough copy for rendering, in the rendered unit. */
    struct Snapshot {
        std::vector<double> bounds;
        std::vector<std::uint64_t> cumulative;  ///< per bound, then +Inf
        std::uint64_t count = 0;
        double sum = 0;
    };
    Snapshot snapshot() const;

  private:
    double per_unit_ = 1e9;
    std::array<std::uint64_t, kMaxBounds> bounds_raw_{};
    std::size_t n_bounds_ = 0;
    /** Per-bucket (not cumulative) counts; the last one is +Inf. */
    std::array<std::atomic<std::uint64_t>, kMaxBounds + 1> buckets_{};
    std::atomic<std::uint64_t> sum_{0};
};

}  // namespace locus::sys
