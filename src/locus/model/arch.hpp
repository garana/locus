#pragma once

#include <span>
#include <string>
#include <string_view>

#include "locus/model/transformer.hpp"

namespace locus::model {

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
                           TransformerModel::Layer& lay);

    /**
     * Computes one token's attention for layer l at pos: reads
     * ws.xb (post-norm activations), writes ws.out, and appends
     * this position's K/V (or latent) row to the cache.
     */
    void (*attention)(const TransformerModel& m,
                      const TransformerModel::Layer& lay,
                      kv::PagedKvCache& cache,
                      kv::PagedKvCache::Seq& seq,
                      TransformerModel::Workspace& ws, std::uint32_t l,
                      std::uint32_t pos);

    /** @returns Paged-cache row width in floats. */
    std::uint32_t (*kv_dim)(const Hparams& hp);

    /**
     * Optional whole-layer body (archs with hp.custom_layer): runs
     * layer l at pos on ws.x in place -- attention, FFN and both
     * residuals -- instead of the shared norm / attention / FFN
     * sequence. nullptr for every other arch.
     */
    void (*layer)(const TransformerModel& m,
                  const TransformerModel::Layer& lay,
                  kv::PagedKvCache& cache, kv::PagedKvCache::Seq& seq,
                  TransformerModel::Workspace& ws, std::uint32_t l,
                  std::uint32_t pos) = nullptr;

    /**
     * Optional whole-token forward (archs with hp.custom_layer whose
     * residual state spans layers: deepseek41's hyper-connections):
     * embeds `token` at seq's next position (seq.n_tokens), runs
     * every layer and writes logits; the caller then advances seq.
     * Full-model, single-stage only.
     */
    void (*forward)(const TransformerModel& m, tok::TokenId token,
                    kv::PagedKvCache& cache, kv::PagedKvCache::Seq& seq,
                    TransformerModel::Workspace& ws,
                    std::span<float> logits) = nullptr;

    /** Optional paged-cache layer count (0 / nullptr = n_layers):
     * archs that keep extra per-position rows (deepseek41). */
    std::uint32_t (*cache_layers)(const Hparams& hp) = nullptr;
};

/** @returns All implemented architectures. */
std::span<const ArchSpec> archs();

/** @returns The spec for `name`, or nullptr. */
const ArchSpec* find_arch(std::string_view name);

}  // namespace locus::model
