#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "locus/server/engine_loop.hpp"

namespace locus::server {

/**
 * Several engines over one loaded model, serving one server (i#929):
 * each its own EngineLoop, KV pool and worker, on its own Vulkan
 * device or its own share of the CPU cores, as locus-stage's
 * executors. A request goes to the least-loaded engine and stays
 * there; its id carries the engine, so every later call reaches it.
 * With one engine it is that EngineLoop with the process's device and
 * pool, unchanged.
 */
class EngineGroup {
  public:
    /** Where one engine runs. */
    struct Slot {
        /** The Vulkan device of its KV pool and worker; -1: the
         *  process's device (and no binding). */
        int vulkan_device = -1;
        /** Its own thread pool's size (EngineLoop::Setup::threads). */
        std::size_t threads = 0;
    };

    /**
     * The slots of `n` engines: one per entry of `devices` when it is
     * not empty (it then has n entries), else n CPU shares of the cores
     * (sys::ThreadPool::share_threads).
     */
    static std::vector<Slot> plan(std::size_t n,
                                  const std::vector<int>& devices);

    /**
     * Makes one engine per slot (at least one). Each engine's KV pool is
     * made with the calling thread bound to its device, which is
     * restored after.
     */
    EngineGroup(const model::TransformerModel& m, tok::TokenId eos,
                const engine::Engine::Config& cfg,
                const std::vector<Slot>& slots);

    /** @returns How many engines serve. */
    std::size_t size() const { return loops_.size(); }

    /** EngineLoop::admit_error, against one engine's pool (all are the
     *  same size). */
    std::string admit_error(std::uint32_t n_prompt);

    /**
     * Submits to the engine that last served a prompt with the same
     * first kPrefixKeyTokens tokens (its prefix cache may hold their
     * KV, i#940), unless that engine has more than kPrefixSlack
     * requests over the least-loaded one; else, and for a shorter
     * prompt, to the least-loaded engine (the fewest requests not yet
     * released).
     * @returns The request's group id.
     */
    std::uint64_t submit(
        std::vector<tok::TokenId> prompt, std::uint32_t max_new_tokens,
        model::SamplingParams sampling = {}, std::uint64_t seed = 0,
        std::unique_ptr<model::TokenConstraint> constraint = nullptr,
        engine::LogprobsOpt logprobs = {});

    /** The EngineLoop calls, on the engine that holds `id`. @{ */
    EngineLoop::View view(std::uint64_t id);
    EngineLoop::View wait_progress(std::uint64_t id, std::size_t n_seen,
                                   std::chrono::milliseconds max_wait = {});
    EngineLoop::View wait_done(std::uint64_t id);
    EngineLoop::View wait_done_or_cancel(
        std::uint64_t id, const std::function<bool()>& alive);
    void cancel(std::uint64_t id);
    void follow(std::uint64_t id);
    bool take(std::uint64_t id, EngineLoop::View& acc);
    void unfollow(std::uint64_t id);
    /** @} */

    /** EngineLoop::set_progress_hook on every engine (one hook, run by
     *  each engine's worker). */
    void set_progress_hook(const std::function<void()>& hook);

    /** EngineLoop::release, and the request no longer counts toward its
     *  engine's load. Idempotent: a stream may release twice (on its
     *  last event, then on a failed write of it). */
    void release(std::uint64_t id);

    /** The engines' gauges, summed. */
    EngineLoop::Stats stats();
    /** Engine e's timing metrics (lock-free, as EngineLoop::metrics). */
    const engine::EngineMetrics& metrics(std::size_t e) const {
        return loops_[e]->metrics();
    }
    /** The KV cache precision (the same for every engine). */
    kv::KvType kv_type() const { return loops_.front()->kv_type(); }
    /** @returns The engine that holds group id `id` (tests, /metrics). */
    std::size_t engine_of(std::uint64_t id) const {
        return static_cast<std::size_t>(id % loops_.size());
    }

  private:
    EngineLoop& loop_of(std::uint64_t id) { return *loops_[engine_of(id)]; }
    std::uint64_t local_id(std::uint64_t id) const {
        return id / loops_.size();
    }

    /** The prompt tokens a prefix key covers: one KV block at the
     *  default geometry. */
    static constexpr std::size_t kPrefixKeyTokens = 16;
    /** How many more unreleased requests than the least-loaded engine
     *  the prefix's engine may hold and still take it. */
    static constexpr std::size_t kPrefixSlack = 1;
    /** Most prefix keys kept (then the map starts over). */
    static constexpr std::size_t kPrefixKeys = 4096;
    /** @returns A hash of the prompt's first kPrefixKeyTokens tokens. */
    static std::uint64_t prefix_key(const std::vector<tok::TokenId>& prompt);

    std::vector<std::unique_ptr<EngineLoop>> loops_;
    /** Requests submitted and not yet released: their ids, and their
     *  count per engine. Guards last_engine_ too. */
    std::mutex live_mu_;
    std::unordered_set<std::uint64_t> live_;
    std::vector<std::size_t> load_;
    /** The engine each recent prefix key went to. */
    std::unordered_map<std::uint64_t, std::size_t> last_engine_;
};

}  // namespace locus::server
