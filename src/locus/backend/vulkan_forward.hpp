#pragma once

#include <span>

#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::backend {

/**
 * Runs one full token forward on the GPU: all matmuls (F32/Q8_0
 * shaders), rmsnorm, RoPE, SwiGLU, and paged attention reading
 * K/V straight from the GPU-mapped cache pool, recorded as one
 * command batch per token (DESIGN.md M5).
 *
 * @returns true when it ran (seq advanced, logits filled); false
 *     when unsupported -- weights in types without shaders, or a
 *     cache whose pool is not GPU-mapped -- so the caller falls
 *     back to the CPU path.
 */
bool vulkan_forward(const model::TransformerModel& m, tok::TokenId token,
                    kv::PagedKvCache& cache,
                    kv::PagedKvCache::Seq& seq,
                    std::span<float> logits);

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
