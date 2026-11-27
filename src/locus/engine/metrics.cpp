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

// A forward call's tokens (i#1220): one decode request up to a full
// prefill chunk.
constexpr double kStepTokenBounds[] = {1,  2,   4,   8,   16,   32,
                                       64, 128, 256, 512, 1024, 2048};
// A forward call's wall time: a fast GPU decode at the bottom, a
// streamed MoE model on a cold disk (minutes a token) at the top.
constexpr double kStepSecondBounds[] = {0.005, 0.01, 0.025, 0.05, 0.1,
                                        0.25,  0.5,  1,     2.5,  5,
                                        10,    30,   60,    120,  300};

}  // namespace

EngineMetrics::EngineMetrics()
    : queue(kPhaseBounds),
      prefill(kPhaseBounds),
      ttft(kPhaseBounds),
      decode_token(kTokenBounds),
      request(kPhaseBounds),
      step_tokens{Histogram(kStepTokenBounds, 1),
                  Histogram(kStepTokenBounds, 1),
                  Histogram(kStepTokenBounds, 1)},
      step_seconds{Histogram(kStepSecondBounds),
                   Histogram(kStepSecondBounds),
                   Histogram(kStepSecondBounds)} {}

const char* EngineMetrics::phase_name(Phase p) {
    switch (p) {
        case Phase::kPrefill: return "prefill";
        case Phase::kDecode: return "decode";
        case Phase::kVerify: return "verify";
    }
    return "";
}

}  // namespace locus::engine
