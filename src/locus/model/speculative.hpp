#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/tok/tokenizer.hpp"

namespace locus::model {

/**
 * Prompt-lookup draft (n-gram speculative decoding, no draft model):
 * finds the most recent earlier occurrence of the last `ngram`
 * tokens of `ctx` and proposes up to `max_draft` tokens that
 * followed it. Effective when the output echoes the input (code,
 * RAG, editing). @returns the draft tokens, or empty when there is
 * no match or ctx is too short.
 */
std::vector<tok::TokenId> ngram_draft(
    std::span<const tok::TokenId> ctx, std::uint32_t ngram,
    std::uint32_t max_draft);

}  // namespace locus::model
