#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "locus/sys/histogram.hpp"

namespace locus::model {

/**
 * The phase of the forward call a thread runs (i#1219): a prompt chunk,
 * a decode, or a speculative verify. The engine sets it around each
 * forward call (RunPhaseScope); an expert run reads it for
 * locus_moe_expert_run_tokens{phase}. Prefill outside an engine.
 */
enum class RunPhase : std::uint8_t { kPrefill, kDecode, kVerify };

/** @returns This thread's phase. */
RunPhase run_phase();

/** Sets this thread's phase for its lifetime, then restores the old. */
class RunPhaseScope {
  public:
    explicit RunPhaseScope(RunPhase p);
    ~RunPhaseScope();
    RunPhaseScope(const RunPhaseScope&) = delete;
    RunPhaseScope& operator=(const RunPhaseScope&) = delete;

  private:
    RunPhase prev_;
};

/** @returns "prefill", "decode" or "verify". */
const char* run_phase_name(RunPhase p);

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

    /**
     * Counts one run of `expert` in `layer` (i#1219): one pass over its
     * weights, serving `tokens` tokens in `ns` of host-clock time, in
     * this thread's run_phase(). Only CPU and CUDA runs are timed (a
     * Vulkan or Metal expert is recorded into a command buffer, so the
     * host clock would time the recording). Other arguments as
     * record().
     */
    static void record_run(std::uint32_t layer, std::uint32_t expert,
                           std::uint32_t n_layers, std::uint32_t n_expert,
                           bool gpu, std::string_view api,
                           std::uint32_t tokens, std::uint64_t ns);

    /**
     * Adds `bytes` of `expert`'s weights that were not in RAM when it
     * was picked (i#1219, mincore before the readahead hint).
     */
    static void record_disk(std::uint32_t layer, std::uint32_t expert,
                            std::uint32_t n_layers, std::uint32_t n_expert,
                            bool gpu, std::string_view api,
                            std::uint64_t bytes);

    /** One expert's runs, their time, and its disk bytes (any above 0). */
    struct RunSeries {
        std::uint32_t layer;
        std::uint32_t expert;
        std::string_view device;
        std::string_view api;
        std::uint64_t runs;
        std::uint64_t ns;
    };
    /** @returns Every (layer, expert, device, api) with a run. */
    static std::vector<RunSeries> run_snapshot();

    /** An expert's disk bytes, summed over every api's table. */
    struct DiskSeries {
        std::uint32_t layer;
        std::uint32_t expert;
        std::uint64_t bytes;
    };
    /** @returns Every (layer, expert) with disk bytes. */
    static std::vector<DiskSeries> disk_snapshot();

    /** Tokens per run of one (device, api, phase), over all experts. */
    struct RunTokens {
        std::string_view device;
        std::string_view api;
        RunPhase phase;
        sys::Histogram::Snapshot tokens;
    };
    /** @returns Every (device, api, phase) with a run. */
    static std::vector<RunTokens> run_tokens();

    /** @returns Whether runs under `api` are timed (CPU and CUDA). */
    static bool timed(std::string_view api);

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
