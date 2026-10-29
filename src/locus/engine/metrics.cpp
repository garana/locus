#include "locus/engine/metrics.hpp"

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

}  // namespace

EngineMetrics::EngineMetrics()
    : queue(kPhaseBounds),
      prefill(kPhaseBounds),
      ttft(kPhaseBounds),
      decode_token(kTokenBounds),
      request(kPhaseBounds) {}

}  // namespace locus::engine
