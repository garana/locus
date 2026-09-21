#pragma once

#include <cstdint>

#include "locus/kv/paged_cache.hpp"
#include "locus/model/llama.hpp"
#include "locus/pipeline/message.hpp"

namespace locus::pipeline {

/**
 * One pipeline stage (multi-server, DESIGN.md "R15+"): owns a
 * contiguous layer range [layer_begin, layer_end) of a model and its
 * own KV cache, and transforms an input message into an output message
 * via LlamaModel::forward_layers.
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
    PipelineStage(const model::LlamaModel& model,
                  std::uint32_t layer_begin, std::uint32_t layer_end,
                  std::uint32_t n_blocks = 0);

    /**
     * Processes one input message, returning the output message.
     * @throws std::invalid_argument if the message kind or payload
     *     size does not match this stage's role.
     */
    Message step(const Message& in);

    /**
     * Runs the read -> step -> write loop over the fds until in_fd
     * reaches EOF. Takes ownership of both fds and closes them before
     * returning (so a downstream stage sees EOF and the chain drains).
     *
     * @returns true on a clean EOF shutdown, false on a read/write or
     *     protocol error.
     */
    bool run(int in_fd, int out_fd);

    bool is_first() const { return layer_begin_ == 0; }
    bool is_last() const { return layer_end_ == n_layers_; }

  private:
    const model::LlamaModel& model_;
    std::uint32_t layer_begin_;
    std::uint32_t layer_end_;
    std::uint32_t n_layers_;
    std::uint32_t n_embd_;
    std::uint32_t n_vocab_;
    kv::PagedKvCache cache_;
    model::LlamaModel::Workspace ws_;
    kv::PagedKvCache::Seq seq_;
};

}  // namespace locus::pipeline
