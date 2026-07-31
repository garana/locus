#pragma once

#include <cstdint>
#include <random>
#include <span>

#include "locus/tok/tokenizer.hpp"

namespace locus::model {

/**
 * Token-sampling parameters. The default is greedy (temperature 0 ->
 * argmax), so callers that pass nothing keep the token-exact
 * behaviour the goldens rely on.
 */
struct SamplingParams {
    /** <= 0 selects greedy/argmax; otherwise softmax temperature. */
    float temperature = 0.0f;
    /** Keep only the top_k highest-logit tokens (0 = disabled). */
    std::uint32_t top_k = 0;
    /** Nucleus: keep the smallest set with cumulative prob >= top_p. */
    float top_p = 1.0f;
    /** Keep tokens with prob >= min_p * max_prob (0 = disabled). */
    float min_p = 0.0f;
    /** Divides positive / multiplies negative logits of repeats. */
    float repeat_penalty = 1.0f;
    /** Subtracted per prior occurrence (OpenAI frequency_penalty). */
    float frequency_penalty = 0.0f;
    /** Subtracted once if the token occurred (presence_penalty). */
    float presence_penalty = 0.0f;
    /** How many trailing history tokens the penalties consider. */
    std::uint32_t repeat_last_n = 64;
};

/**
 * Samples one token from `logits` (mutated in place: penalties and
 * temperature are applied to it). `history` is the recent token
 * stream the penalties look back over; `rng` supplies randomness
 * (seed it per request for reproducibility).
 *
 * temperature <= 0 short-circuits to argmax (rng unused), so greedy
 * decoding stays bit-exact. Filter order matches llama.cpp:
 * penalties -> top_k -> temperature/softmax -> top_p -> min_p ->
 * sample.
 */
tok::TokenId sample(std::span<float> logits, const SamplingParams& p,
                    std::span<const tok::TokenId> history,
                    std::mt19937_64& rng);

}  // namespace locus::model
