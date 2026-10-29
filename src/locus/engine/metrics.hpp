#pragma once

#include <atomic>
#include <cstdint>

#include "locus/sys/histogram.hpp"

namespace locus::engine {

/** Durations in nanoseconds, rendered in seconds (sys::Histogram's
 * default unit). */
using Histogram = sys::Histogram;

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
     * reused from the prefix cache); their prefill wall time is the
     * prefill histogram's sum. */
    std::atomic<std::uint64_t> prefill_tokens{0};
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
