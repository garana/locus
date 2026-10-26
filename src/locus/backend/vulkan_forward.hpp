#pragma once

#include <span>

#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::backend {

/**
 * @returns Whether the full GPU forward implements a model with
 *     these hparams: no custom layer body (muse-spark), RMSNorm, no
 *     DSA indexer, and head width (for MLA also the latent rank)
 *     within the attention kernels' stage.
 *     The hparam half of vulkan_forward_supported.
 */
bool vulkan_hparams_supported(const model::Hparams& hp);

/**
 * @returns Whether the full GPU forward (vulkan_forward and
 *     vulkan_forward_batch) implements m: its hparams
 *     (vulkan_hparams_supported), every layer's attention shape, and
 *     its weight types. Device-independent and the same decision both
 *     forwards make from the model: false means they always decline
 *     m; true leaves only device, cache and KV-type conditions (a
 *     GPU-mapped pool, the KV types each path reads).
 */
bool vulkan_forward_supported(const model::TransformerModel& m);

/**
 * Runs one full token forward on the GPU: all matmuls (F32/Q8_0
 * shaders), rmsnorm, RoPE, SwiGLU, and paged attention reading
 * K/V straight from the GPU-mapped cache pool, recorded as one
 * command batch per token (DESIGN.md M5).
 *
 * @param row_begin First output-head row to compute.
 * @param row_end One past the last (0: every row). Logits outside the
 *     range are left unspecified; TransformerModel::forward sets them.
 * @param embd_in When not empty, the n_embd input embedding used
 *     instead of token's row (Janus image codes, i#247); token is then
 *     ignored.
 * @param normed_out When not empty, receives the n_embd normed hidden
 *     state after the final norm (what the head reads).
 * @returns true when it ran (seq advanced, logits filled); false
 *     when unsupported -- weights in types without shaders, or a
 *     cache whose pool is not GPU-mapped -- so the caller falls
 *     back to the CPU path.
 */
bool vulkan_forward(const model::TransformerModel& m, tok::TokenId token,
                    kv::PagedKvCache& cache,
                    kv::PagedKvCache::Seq& seq,
                    std::span<float> logits, std::uint32_t row_begin = 0,
                    std::uint32_t row_end = 0,
                    std::span<const float> embd_in = {},
                    std::span<float> normed_out = {});

/**
 * Runs one position for each of several sequences of one cache in a
 * single command batch (i#267: Janus's two CFG sequences). Every
 * projection reads its weights once for all the sequences (the
 * matvec_*_n kernels, up to 8 rows); rope, the KV store and attention
 * run per sequence over its own pages. Each step is what
 * vulkan_forward would do with the same inputs.
 *
 * Dense llama-family models only: no MLA, no MoE, no q / k norm or
 * qkv bias, an F32 KV cache, and every weight F32, F16, BF16 or Q8_0
 * (the types with a short-batch kernel; K-quant models are faster as
 * one vulkan_forward per sequence). The caller has checked every step
 * (TransformerModel::forward_embeddings).
 *
 * @param row_begin First head row to compute (row_end > row_begin).
 * @returns true when it ran (each seq advanced; logits[row_begin,
 *     row_end) and normed_out filled); false when unsupported, before
 *     any state changes, so the caller runs the steps one at a time.
 */
bool vulkan_forward_multi(const model::TransformerModel& m,
                          std::span<const model::EmbedStep> steps,
                          kv::PagedKvCache& cache, std::uint32_t row_begin,
                          std::uint32_t row_end);

/**
 * Runs N tokens of one sequence (a prefill chunk) through the GPU
 * in one command batch: every projection is a tiled matmul over
 * the chunk's token rows, so each weight is read once per chunk
 * instead of once per token, and attention runs one workgroup per
 * (head, token). Token-exact, not byte-identical, to N
 * vulkan_forward() calls (matmul and matvec sum in different
 * orders).
 *
 * @param tokens The chunk; capacity for tokens.size() more
 *     positions must already be ensured on seq.
 * @param logits n_vocab floats for the last token, or n * n_vocab
 *     (one row per token) when all_logits.
 * @returns true when it ran (seq advanced, logits filled); false
 *     when unsupported -- what vulkan_forward rejects, a quantized
 *     KV cache, or MoE experts that cannot route on the GPU -- so
 *     the caller runs the tokens one at a time instead.
 * @throws std::invalid_argument on a wrong logits size or missing
 *     capacity.
 */
bool vulkan_forward_batch(const model::TransformerModel& m,
                          std::span<const tok::TokenId> tokens,
                          kv::PagedKvCache& cache,
                          kv::PagedKvCache::Seq& seq,
                          std::span<float> logits, bool all_logits);

}  // namespace locus::backend
