#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

namespace locus::model {

/**
 * Step WLRU residency for routed experts (i#252): which (layer,
 * expert) pairs live on the GPU when the rest run on the CPU.
 *
 * One ranked list holds every expert seen, across all MoE layers; its
 * first `capacity` ranks are the hot zone (GPU resident). Each pick
 * moves the expert `step` ranks toward the head, shifting the experts
 * it passes down by one. When that move takes it into the zone, it is
 * promoted (uploaded), and the expert pushed from the last zone rank
 * is evicted. A new expert joins at the tail, inside the zone while
 * the list is shorter than `capacity`. scripts/moe-trace-sim.py
 * replays the same policy (wlru-step) on a recorded trace.
 *
 * Thread-safe: touch() serializes on a mutex.
 */
class ExpertCache {
  public:
    /** What one pick did to the residency. */
    struct Touch {
        /** The expert was in the zone before the pick: run it on the
         *  GPU. */
        bool hot = false;
        /** The pick moved it into the zone: upload it. */
        bool promoted = false;
        /** The pick pushed an expert out of the zone: drop it. */
        bool evicted = false;
        std::uint32_t evicted_layer = 0;
        std::uint32_t evicted_expert = 0;
    };

    /**
     * @param n_layers The model's layers (indexes keys).
     * @param n_expert The routed experts per layer.
     * @param capacity Hot zone size, in experts.
     * @param step Ranks a pick moves toward the head (at least 1).
     */
    ExpertCache(std::uint32_t n_layers, std::uint32_t n_expert,
                std::uint32_t capacity, std::uint32_t step);

    /** Records one pick of `expert` in `layer`. */
    Touch touch(std::uint32_t layer, std::uint32_t expert);

    /** @returns Whether (layer, expert) is in the zone now. */
    bool resident(std::uint32_t layer, std::uint32_t expert) const;

    std::uint32_t capacity() const { return capacity_; }
    std::uint32_t step() const { return step_; }

  private:
    std::uint32_t n_expert_;
    std::uint32_t capacity_;
    std::uint32_t step_;
    std::vector<std::uint32_t> order_;  // rank -> key
    std::vector<std::int32_t> rank_;    // key -> rank, -1 unseen
    mutable std::mutex mu_;
};

}  // namespace locus::model
