#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace locus::engine {

/**
 * A Prometheus-style histogram of durations with fixed bucket bounds.
 * One thread writes (the engine worker); any thread reads, without a
 * lock: every field is a relaxed atomic, so a reader may see a count
 * one observation ahead of a bucket, never a torn value. observe()
 * costs a short bucket scan and three relaxed adds.
 */
class Histogram {
  public:
    /** Upper bounds in seconds (ascending), plus the implicit +Inf. */
    static constexpr std::size_t kMaxBounds = 16;

    /** @param bounds Ascending upper bounds in seconds (at most
     *     kMaxBounds). */
    explicit Histogram(std::span<const double> bounds);

    /** Records one duration of `ns` nanoseconds. */
    void observe(std::uint64_t ns);

    /** A consistent-enough copy for rendering. */
    struct Snapshot {
        std::vector<double> bounds;          ///< seconds
        std::vector<std::uint64_t> cumulative;  ///< per bound, then +Inf
        std::uint64_t count = 0;
        double sum_seconds = 0;
    };
    Snapshot snapshot() const;

  private:
    std::array<std::uint64_t, kMaxBounds> bounds_ns_{};
    std::size_t n_bounds_ = 0;
    /** Per-bucket (not cumulative) counts; the last one is +Inf. */
    std::array<std::atomic<std::uint64_t>, kMaxBounds + 1> buckets_{};
    std::atomic<std::uint64_t> count_{0};
    std::atomic<std::uint64_t> sum_ns_{0};
};

/**
 * Timing and scheduler metrics of one engine (i#285), written by the
 * engine worker and read by /metrics without the engine lock. Times
 * are wall time per request, so with concurrent requests their sums
 * exceed the elapsed time; busy_ns is the engine's own step time.
 */
struct EngineMetrics {
    EngineMetrics();

    /** Submit to first admission. */
    Histogram queue;
    /** First admission to the first generated token (the prefill of
     * the prompt tokens not served from the prefix cache). */
    Histogram prefill;
    /** Submit to the first generated token. */
    Histogram ttft;
    /** Per generated token after the first: (done - first) / (n - 1). */
    Histogram decode_token;
    /** Submit to done. */
    Histogram request;

    /** Completed requests' prompt tokens that were computed (not
     * reused from the prefix cache), and their prefill wall time. */
    std::atomic<std::uint64_t> prefill_tokens{0};
    std::atomic<std::uint64_t> prefill_ns{0};
    /** Completed requests' tokens after the first, and their decode
     * wall time. */
    std::atomic<std::uint64_t> decode_tokens{0};
    std::atomic<std::uint64_t> decode_ns{0};
    /** Time spent inside Engine::step(), and the step count. */
    std::atomic<std::uint64_t> busy_ns{0};
    std::atomic<std::uint64_t> steps{0};
    /** After the last step: sequences running and requests waiting. */
    std::atomic<std::uint32_t> running{0};
    std::atomic<std::uint32_t> waiting{0};
};

}  // namespace locus::engine
