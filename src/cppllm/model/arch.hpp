#pragma once

#include <span>
#include <string>
#include <string_view>

#include "cppllm/model/llama.hpp"

namespace cppllm::model {

/**
 * Per-architecture behavior (DESIGN.md R6). The model runtime is
 * shared; what varies per architecture -- metadata keys, attention
 * tensors, the attention computation, and the KV row width -- is
 * captured here. Adding an architecture means adding one registry
 * entry plus its hooks; the loader, FFN/MoE block, engine, and
 * server are untouched.
 */
struct ArchSpec {
    /** Value of general.architecture this spec handles. */
    std::string_view name;
    std::string_view description;

    /**
     * Reads arch-specific hyperparameters (dims, yarn, gating,
     * kq_scale) after the shared keys are in.
     *
     * @throws gguf::Error on unsupported configurations.
     */
    void (*load_hparams)(const gguf::GgufFile& g,
                         const std::string& prefix, Hparams& hp);

    /** Loads one layer's attention tensors ("blk.N." prefix). */
    void (*load_attention)(const gguf::GgufFile& g,
                           const std::string& bp,
                           const Hparams& hp,
                           LlamaModel::Layer& lay);

    /**
     * Computes one token's attention for layer l at pos: reads
     * ws.xb (post-norm activations), writes ws.out, and appends
     * this position's K/V (or latent) row to the cache.
     */
    void (*attention)(const LlamaModel& m,
                      const LlamaModel::Layer& lay,
                      kv::PagedKvCache& cache,
                      kv::PagedKvCache::Seq& seq,
                      LlamaModel::Workspace& ws, std::uint32_t l,
                      std::uint32_t pos);

    /** @returns Paged-cache row width in floats. */
    std::uint32_t (*kv_dim)(const Hparams& hp);
};

/** @returns All implemented architectures. */
std::span<const ArchSpec> archs();

/** @returns The spec for `name`, or nullptr. */
const ArchSpec* find_arch(std::string_view name);

}  // namespace cppllm::model
