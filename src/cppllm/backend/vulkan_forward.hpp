#pragma once

#include <span>

#include "cppllm/kv/paged_cache.hpp"
#include "cppllm/model/llama.hpp"
#include "cppllm/tok/tokenizer.hpp"

namespace cppllm::backend {

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
bool vulkan_forward(const model::LlamaModel& m, tok::TokenId token,
                    kv::PagedKvCache& cache,
                    kv::PagedKvCache::Seq& seq,
                    std::span<float> logits);

}  // namespace cppllm::backend
