#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"
#include "locus/pipeline/message.hpp"

namespace locus::pipeline {

/**
 * One pipeline stage (multi-server, DESIGN.md "R15+"): owns a
 * contiguous layer range [layer_begin, layer_end) of a model and its
 * own KV cache, and transforms an input message into an output message
 * via TransformerModel::forward_layers.
 *
 * Role follows the range:
 *   - first stage (layer_begin == 0): input kToken, embeds it, runs
 *     [0, end), outputs kActivation (or kLogits if also the last).
 *   - middle stage: input kActivation, runs [begin, end), outputs
 *     kActivation.
 *   - last stage (layer_end == n_layers): input kActivation, runs
 *     [begin, n_layers), outputs kLogits.
 *
 * The stage holds a full-height cache and touches only its layers'
 * rows (absolute layer indexing); sizing the cache to just the owned
 * layers is a later optimization. Each stage advances its own sequence
 * position per token: forward_layers auto-advances only on the final
 * stage, so a non-final stage bumps its own position after each token
 * (keeping every stage's KV in lockstep). CPU/CUDA only.
 */
class PipelineStage {
  public:
    /**
     * @param model The shared model (all stages load it; each executes
     *     only its layer range -- "load-full, execute-slice").
     * @param layer_begin First layer this stage runs (inclusive).
     * @param layer_end One past the last layer this stage runs.
     * @param n_blocks KV pool size in blocks (0 = model default).
     * @throws std::invalid_argument on a bad range.
     */
    PipelineStage(const model::TransformerModel& model,
                  std::uint32_t layer_begin, std::uint32_t layer_end,
                  std::uint32_t n_blocks = 0);

    /**
     * Processes one input message, returning the output message.
     * @throws std::invalid_argument if the message kind or payload
     *     size does not match this stage's role.
     */
    Message step(const Message& in);

    /**
     * Per-session variant: advances `seq` (a caller-owned KV sequence)
     * instead of the stage's built-in one, so one stage can serve many
     * concurrent sessions over its shared cache/workspace (the
     * event-loop serving model, DESIGN.md i#24). All sessions share one
     * KV pool and one forward-pass workspace; they must be stepped one
     * at a time (the single-threaded event loop does exactly that), as
     * the workspace is not reentrant.
     *
     * @param seq This session's KV sequence; its n_tokens must equal
     *     in.position (the same lockstep rule as the built-in form).
     * @throws std::invalid_argument on a role/size/lockstep mismatch,
     *     std::runtime_error if the cache is exhausted.
     */
    Message step(kv::PagedKvCache::Seq& seq, const Message& in);

    /** One entry of a batched step: a session's KV sequence and its
     * input frame. Both must outlive the step_batch() call. */
    struct BatchInput {
        kv::PagedKvCache::Seq* seq;
        const Message* in;
    };

    /** Result for one batched-step entry, matched by position to the
     * BatchInput. On success `out` is the output frame; on failure `ok`
     * is false and `err` is set -- a bad or out-of-lockstep frame, or a
     * cache exhaustion, fails ONLY its own slot and the rest of the
     * batch still runs. */
    struct BatchOutput {
        bool ok = true;
        Message out;
        std::string err;
    };

    /**
     * Batched form of the per-session step: advances N caller-owned
     * sequences in one pass through this stage's layer range, so a
     * layer's weights are read once for the whole batch instead of once
     * per session (the amortization increment 2 of the event-loop
     * serving model, DESIGN.md i#24, exists for). The sessions may sit
     * at DIFFERENT positions (ragged) -- each attends over its own KV --
     * which is the normal case once the executor coalesces independent
     * sessions.
     *
     * Byte-identical to calling step(seq, in) once per entry in order:
     * same role gating, same lockstep rule, same per-stage advance (the
     * final stage consumes the token; a non-final stage bumps each
     * session's own position). Each entry is validated independently, so
     * one malformed frame does not disturb the others.
     *
     * Requires supports_batch(); the backend floor is the model's
     * (Vulkan cannot batch). At most one entry per sequence per call --
     * the caller must not pass two frames for the same session in one
     * batch (the executor inherits this from "one step in flight per
     * session").
     *
     * @param ins One (seq, input) pair per session; order is preserved
     *     in `outs`.
     * @param outs Cleared and filled with one BatchOutput per input, by
     *     position.
     */
    void step_batch(std::span<const BatchInput> ins,
                    std::vector<BatchOutput>& outs);

    /** @returns Whether the backend can run a batched forward (and thus
     * step_batch); false means the caller must step one at a time. */
    bool supports_batch() const { return model_.supports_batch(); }

    /**
     * Runs the read -> step -> write loop over the fds until in_fd
     * reaches EOF. Takes ownership of both fds and closes them before
     * returning (so a downstream stage sees EOF and the chain drains).
     *
     * @returns true on a clean EOF shutdown, false on a read/write or
     *     protocol error.
     */
    bool run(int in_fd, int out_fd);

    /** Clears this stage's KV state so it can serve a fresh sequence:
     * releases the sequence's cache blocks and resets its position.
     * serve_stage calls this between sessions when it re-accepts after
     * a connection ends. */
    void reset();

    /** Per-session variant: releases `seq`'s blocks back to the shared
     * pool and resets it. The event-loop server calls this when a
     * session ends, so a dropped connection frees its KV at once rather
     * than holding it until the stage exits. */
    void reset(kv::PagedKvCache::Seq& seq);

    bool is_first() const { return layer_begin_ == 0; }
    bool is_last() const { return layer_end_ == n_layers_; }

  private:
    const model::TransformerModel& model_;
    std::uint32_t layer_begin_;
    std::uint32_t layer_end_;
    std::uint32_t n_layers_;
    std::uint32_t n_embd_;
    std::uint32_t n_vocab_;
    kv::PagedKvCache cache_;
    model::TransformerModel::Workspace ws_;
    kv::PagedKvCache::Seq seq_;
};

}  // namespace locus::pipeline
