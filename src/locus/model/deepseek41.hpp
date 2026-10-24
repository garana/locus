#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"

namespace locus::model::deepseek41 {

/** ArchSpec hooks for "deepseek41" (DeepSeek-V4.1-Flash). */
void load_hparams(const gguf::GgufFile& g, const std::string& p,
                  Hparams& hp);
void load_layer(const gguf::GgufFile& g, const std::string& bp,
                const Hparams& hp, TransformerModel::Layer& lay);
void attention(const TransformerModel& m,
               const TransformerModel::Layer& lay, kv::PagedKvCache& cache,
               kv::PagedKvCache::Seq& seq, TransformerModel::Workspace& ws,
               std::uint32_t l, std::uint32_t pos);
std::uint32_t kv_dim(const Hparams& hp);
std::uint32_t cache_layers(const Hparams& hp);
void forward(const TransformerModel& m, tok::TokenId token,
             kv::PagedKvCache& cache, kv::PagedKvCache::Seq& seq,
             TransformerModel::Workspace& ws, std::span<float> logits);

/**
 * Engram n-gram rows for one Engram layer: toks[0] is the current
 * token's compressed id and toks[s] the one s positions back (pad
 * once blocked); writes ngram-1 orders x ngram_heads table rows.
 * Exposed for tests.
 */
void engram_rows(const Hparams::Ds41& d, std::uint32_t e,
                 std::span<const std::int64_t> toks,
                 std::span<std::uint32_t> rows);

/**
 * @returns The routed experts' weights for `picked` (indices into the
 *     router probabilities): normalized to sum 1 when
 *     hp.expert_weights_norm and more than one expert is picked, then
 *     scaled by hp.expert_weights_scale. Exposed for tests.
 */
std::vector<float> moe_weights(const Hparams& hp,
                               std::span<const float> probs,
                               std::span<const std::uint32_t> picked);

/**
 * Sinkhorn-normalizes the hc x hc mHC combine matrix in place
 * (m[dst + hc * src]): softmax over dst per src, + eps, then
 * alternating src / dst normalizations. Exposed for tests.
 */
void sinkhorn(std::span<float> m, std::uint32_t hc, std::uint32_t iters,
              float eps);

}  // namespace locus::model::deepseek41
