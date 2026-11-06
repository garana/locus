#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace locus::model {

/**
 * Routed-expert calls per (layer, expert, device, api), for
 * locus_moe_expert_calls_total (i#252), always on. `device` is "gpu"
 * or "cpu"; `api` is the interface that ran the expert ("cuda",
 * "vulkan", "moltenvk", or the CPU kernel set: "neon", "avx2",
 * "sse4", "scalar").
 *
 * Process-wide, like the GPU and fallback counters: a recording site
 * may hold no model (Vulkan's host-routed MoE layer). Each api gets a
 * table sized by the first model that records under it; a later
 * layer or expert past that size is not counted. Writes are relaxed
 * atomic adds after a lock-free lookup, so /metrics reads while
 * decode runs.
 *
 * Vulkan's GPU-routed MoE layers (every expert resident, routing in
 * the moe_route shader) never bring the picks to the host and are not
 * counted.
 */
class ExpertCalls {
  public:
    /** One nonzero counter. */
    struct Series {
        std::uint32_t layer;
        std::uint32_t expert;
        std::string_view device;
        std::string_view api;
        std::uint64_t calls;
    };

    /**
     * Counts one call of `expert` in `layer`.
     * @param n_layers The model's layer count (sizes a new table).
     * @param n_expert The routed experts per layer (sizes a new table).
     * @param gpu Whether the expert ran on the GPU.
     * @param api The interface that ran it; must outlive the process
     *     (a string literal or a backend name).
     */
    static void record(std::uint32_t layer, std::uint32_t expert,
                       std::uint32_t n_layers, std::uint32_t n_expert,
                       bool gpu, std::string_view api);

    /** @returns Every counter above zero, by api, layer and expert. */
    static std::vector<Series> snapshot();

    /** Drops every table (tests). Not safe while recording. */
    static void reset();
};

/**
 * @returns The api label for a model's backend: "moltenvk" or
 *     "vulkan" for Vulkan, else the backend's name ("cuda", "neon",
 *     "avx2", "sse4", "scalar").
 * @param gpu Set to whether the backend runs on the GPU.
 */
std::string_view expert_api(std::string_view backend, bool& gpu);

}  // namespace locus::model
