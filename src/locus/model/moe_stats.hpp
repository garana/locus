#pragma once

#include <cstdint>

namespace locus::model {

/**
 * Env-gated (LOCUS_MOE_STATS=1) expert-routing telemetry for the
 * R8 streaming baseline: counts routed-expert activations per
 * layer so the true working set of a MoE run can be measured --
 * the number that decides whether prefetch/residency policies
 * can beat passive mmap streaming.
 *
 * LOCUS_MOE_TRACE=<file> also writes every pick, one "layer expert"
 * line in call order, so a residency policy can be replayed offline
 * (scripts/moe-trace-sim.py). Batched prefill records a chunk layer
 * by layer, decode token by token.
 *
 * Single-threaded by the engine contract; zero overhead when both
 * env vars are unset.
 */
struct MoeStats {
    /** @returns true when telemetry is enabled via env. */
    static bool enabled();

    /** Notes that `expert` was routed in `layer` for a token. */
    static void record(std::uint32_t layer, std::uint32_t expert,
                       std::uint32_t n_layers,
                       std::uint32_t n_expert);

    /** Prints the per-layer and aggregate summary to stderr. */
    static void report();
};

}  // namespace locus::model
