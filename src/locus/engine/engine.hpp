#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "locus/engine/prefix_cache.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/llama.hpp"
#include "locus/model/sampling.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::engine {

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

    /** Sampling parameters (default: greedy/argmax). */
    model::SamplingParams sampling;
    /** Per-request RNG (seeded at submit) for stochastic sampling. */
    std::mt19937_64 rng;
    /** Optional constrained-decoding filter (e.g. JSON grammar). */
    std::unique_ptr<model::TokenConstraint> constraint;

    /** Cache handle (internal to the engine). */
    kv::PagedKvCache::Seq seq;
    /** Prompt + generated tokens fed through the model so far. */
    std::uint32_t n_fed = 0;
    /** Logits of the last token fed (batched-decode path only,
     * where several sequences' forwards interleave in one step). */
    std::vector<float> logits;
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
        /** Ingest a prompt in one batched forward when the model
         * supports it (R10); byte-identical to per-token prefill,
         * fewer weight reads. Off falls back to per-token. */
        bool batched_prefill = true;
        /** Decode all running sequences in one batched forward per
         * step (R10 4b) -- the continuous-batching throughput win.
         * Byte-identical to the per-token scheduler. Off keeps the
         * per-sequence step(). Default-on since R10's matvec_batch
         * threading fix (07c72f5): measured >= per-token in every
         * regime on both NEON (Pi) and x86/sse4 (vx). Off keeps the
         * per-sequence step(). */
        bool batched_decode = true;
        /** Reuse KV across requests that share a prompt prefix
         * (R12: prefix caching). Byte-exact; off by default. */
        bool prefix_cache = false;
        /** Max cached prompt prefixes (LRU) when prefix_cache is on. */
        std::uint32_t prefix_cache_slots = 32;
        /** Prompt-lookup speculative decoding: draft n-gram tokens
         * from the context and verify them in one batched forward
         * (greedy only). Byte-exact; off by default; uses the
         * per-token scheduler (implies non-batched decode). */
        bool speculative = false;
        /** Trailing n-gram length matched for drafts. */
        std::uint32_t spec_ngram = 3;
        /** Max draft tokens proposed per step. */
        std::uint32_t spec_draft = 4;
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
     * @param sampling Sampling params (default greedy/argmax).
     * @param seed RNG seed; 0 draws a nondeterministic one.
     * @returns Request id for get().
     */
    std::uint64_t submit(
        std::vector<tok::TokenId> prompt,
        std::uint32_t max_new_tokens,
        model::SamplingParams sampling = {}, std::uint64_t seed = 0,
        std::unique_ptr<model::TokenConstraint> constraint = nullptr);

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

    /** @returns true when nothing is waiting or running. */
    bool idle() const {
        return waiting_.empty() && running_.empty();
    }

    /** Invoked after each newly sampled token (streaming). */
    std::function<void(const Request&, tok::TokenId)> on_token;

    std::uint32_t free_blocks() const {
        return cache_.free_blocks();
    }
    std::uint32_t total_blocks() const {
        return cache_.total_blocks();
    }

    /** Prompt positions served from the prefix cache (skipped
     * prefill), cumulative. 0 when prefix_cache is off. */
    std::uint64_t prefix_reused_tokens() const {
        return prefix_reused_tokens_;
    }

    /** Draft tokens accepted by speculative verify, cumulative. */
    std::uint64_t spec_accepted_tokens() const {
        return spec_accepted_;
    }
    /** Speculative verify forwards run, cumulative. */
    std::uint64_t spec_steps() const { return spec_steps_; }

  private:
    /** Feeds up to `budget` tokens of r; samples when caught up. */
    void advance(Request& r, std::uint32_t& budget);

    /** One iteration with cross-sequence batched decode (R10 4b):
     * prefill -> sample -> one forward_batch_decode for every
     * running sequence with a pending token. */
    bool step_batched();
    /** Feeds r's remaining PROMPT (batched or per-token) into
     * r.logits without sampling; used by step_batched's prefill
     * phase. @returns false if r finished (context overflow). */
    bool prefill_only(Request& r, std::uint32_t& budget);
    /** Samples the next token for r from r.logits, appends it, and
     * finishes r on EOS / max. @returns the sampled token. */
    void sample_from(Request& r, std::span<float> logits);
    /** Releases r's blocks and moves it back to the wait queue. */
    void preempt(std::uint64_t victim_id);
    void finish(Request& r, Status s, std::string error = "");
    /** @returns Blocks needed to hold n tokens. */
    std::uint32_t blocks_for(std::uint32_t n_tokens) const;
    /** Adopts the longest cached prompt prefix into a fresh request
     * (prefix_cache on), advancing n_fed past the shared blocks. */
    void try_adopt(Request& r);
    /** One prompt-lookup speculative-decode step: verifies drafts in
     * a single batched forward, appending the accepted prefix plus a
     * correction/bonus. @returns false if it could not run (capacity
     * / overflow) so the caller uses the per-token path. */
    bool spec_decode_step(Request& r);

    const model::LlamaModel& model_;
    tok::TokenId eos_;
    Config cfg_;
    kv::PagedKvCache cache_;
    model::LlamaModel::Workspace ws_;
    std::vector<float> logits_;
    /** n * n_vocab scratch for the batched-decode step. */
    std::vector<float> batched_logits_;

    std::vector<std::unique_ptr<Request>> requests_;
    std::deque<std::uint64_t> waiting_;
    std::vector<std::uint64_t> running_;

    std::unique_ptr<PrefixCache> prefix_cache_;
    std::uint64_t prefix_reused_tokens_ = 0;
    std::uint64_t spec_accepted_ = 0;
    std::uint64_t spec_steps_ = 0;
};

}  // namespace locus::engine
