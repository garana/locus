#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "cppllm/kv/paged_cache.hpp"
#include "cppllm/model/llama.hpp"
#include "cppllm/tok/tokenizer.hpp"

namespace cppllm::engine {

/** Lifecycle of a generation request. */
enum class Status {
    kWaiting,
    kRunning,
    kDone,
    kFailed,
};

/** One generation request and its progress. */
struct Request {
    std::uint64_t id = 0;
    std::vector<tok::TokenId> prompt;
    std::uint32_t max_new_tokens = 0;
    std::vector<tok::TokenId> generated;
    Status status = Status::kWaiting;
    std::string error;

    /** Cache handle (internal to the engine). */
    kv::PagedKvCache::Seq seq;
    /** Prompt + generated tokens fed through the model so far. */
    std::uint32_t n_fed = 0;
};

/**
 * Continuous-batching engine (DESIGN.md 4.2): iteration-level
 * scheduling over the paged cache. Each step() interleaves one
 * decode token per running sequence with a bounded budget of
 * prefill tokens (chunked prefill), admits waiting requests while
 * blocks allow (prompt + decode headroom), and preempts the newest
 * running sequence by full recompute when the pool runs dry.
 *
 * Single-threaded by design: callers drive step() from one thread
 * (the M4 server owns that loop).
 */
class Engine {
  public:
    struct Config {
        /** Pool size in blocks; 0 = one full context window. */
        std::uint32_t n_blocks = 0;
        /** Max prompt tokens prefetched per step across seqs. */
        std::uint32_t prefill_budget = 64;
        /** Decode headroom required at admission, in tokens. */
        std::uint32_t decode_headroom = 16;
        /** Cap on simultaneously running sequences. */
        std::uint32_t max_running = 64;
    };

    /**
     * @param m Loaded model; must outlive the engine.
     * @param eos Token that terminates generation.
     */
    Engine(const model::LlamaModel& m, tok::TokenId eos,
           Config cfg);

    /** Same, with default Config. */
    Engine(const model::LlamaModel& m, tok::TokenId eos);

    /**
     * Enqueues a prompt for generation.
     *
     * @returns Request id for get().
     */
    std::uint64_t submit(std::vector<tok::TokenId> prompt,
                         std::uint32_t max_new_tokens);

    /**
     * Runs one scheduling iteration.
     *
     * @returns true while waiting or running work remains.
     */
    bool step();

    /** Steps until all submitted requests are done or failed. */
    void run_to_completion();

    /** @returns The request for id, or nullptr. */
    const Request* get(std::uint64_t id) const;

    /** Invoked after each newly sampled token (streaming). */
    std::function<void(const Request&, tok::TokenId)> on_token;

    std::uint32_t free_blocks() const {
        return cache_.free_blocks();
    }
    std::uint32_t total_blocks() const {
        return cache_.total_blocks();
    }

  private:
    /** Feeds up to `budget` tokens of r; samples when caught up. */
    void advance(Request& r, std::uint32_t& budget);
    /** Releases r's blocks and moves it back to the wait queue. */
    void preempt(std::uint64_t victim_id);
    void finish(Request& r, Status s, std::string error = "");
    /** @returns Blocks needed to hold n tokens. */
    std::uint32_t blocks_for(std::uint32_t n_tokens) const;

    const model::LlamaModel& model_;
    tok::TokenId eos_;
    Config cfg_;
    kv::PagedKvCache cache_;
    model::LlamaModel::Workspace ws_;
    std::vector<float> logits_;

    std::vector<std::unique_ptr<Request>> requests_;
    std::deque<std::uint64_t> waiting_;
    std::vector<std::uint64_t> running_;
};

}  // namespace cppllm::engine
