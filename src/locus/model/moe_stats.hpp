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
 * Single-threaded by the engine contract; zero overhead when the
 * env var is unset.
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
