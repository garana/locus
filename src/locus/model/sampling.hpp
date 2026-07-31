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
 * A per-request token-level constraint for constrained decoding
 * (e.g. a JSON grammar). The sampler only considers tokens that
 * allows() accepts; the engine commit()s the chosen token so the
 * constraint can advance its state.
 */
class TokenConstraint {
  public:
    virtual ~TokenConstraint() = default;
    /** @returns true if `t` may be emitted next. */
    virtual bool allows(tok::TokenId t) const = 0;
    /** Advances internal state past the chosen token `t`. */
    virtual void commit(tok::TokenId t) = 0;
};

/**
 * Samples one token from `logits` (mutated in place: penalties and
 * temperature are applied to it). `history` is the recent token
 * stream the penalties look back over; `rng` supplies randomness
 * (seed it per request for reproducibility). `constraint`, when
 * non-null, restricts the choice to allowed tokens (falling back to
 * unconstrained if it would leave no candidate).
 *
 * temperature <= 0 short-circuits to argmax (rng unused, no
 * constraint) so greedy decoding stays bit-exact. Filter order
 * matches llama.cpp: penalties -> constraint -> top_k ->
 * temperature/softmax -> top_p -> min_p -> sample.
 */
tok::TokenId sample(std::span<float> logits, const SamplingParams& p,
                    std::span<const tok::TokenId> history,
                    std::mt19937_64& rng,
                    const TokenConstraint* constraint = nullptr);

}  // namespace locus::model
