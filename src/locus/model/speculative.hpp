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

/**
 * How well one request's drafts pass: decayed counts of accepted and
 * rejected draft tokens. A verify of d drafts that accepts k adds k
 * accepts, and one reject when k < d.
 */
struct DraftState {
    double accepted = 0;
    double rejected = 0;
    /** The one-token draft of a step that ran a plain decode, checked
     * against the token that decode produced (observe_guess). */
    bool has_guess = false;
    tok::TokenId guess = 0;

    /** @returns the chance that a draft token passes, given that the
     * one before it did (a prior of 1/2 with no data). */
    double accept_rate() const;
};

/** Records a verify of `drafted` tokens with `accepted` passing. */
void observe_draft(DraftState& s, std::uint32_t drafted,
                   std::uint32_t accepted);
/** Records whether a one-token draft matched the decoded token. */
void observe_guess(DraftState& s, bool hit);

/**
 * Measured time of one forward by draft length, shared by an engine's
 * requests: entry 0 is a plain decode step, entry d a verify of d
 * drafts (d + 1 rows). Each is a moving average.
 */
class DraftCost {
  public:
    /** Records one forward of `drafted` drafts that took `seconds`;
     * the first one of each length is dropped (one-time costs). */
    void observe(std::uint32_t drafted, double seconds);
    /** @returns the moving average for `drafted`, or 0 if not
     * measured yet. */
    double seconds(std::uint32_t drafted) const;

  private:
    std::vector<double> sec_;
    std::vector<std::uint32_t> seen_;
};

/**
 * Picks the draft length with the most expected tokens per second.
 * With accept rate a, a verify of d drafts yields 1 + a + ... + a^d
 * tokens (the accepted drafts plus the model's own next token). A
 * length never measured costs the longest shorter measured one plus
 * the cheapest measured cost of one more row, at most a quarter of a
 * decode step per row. That guess is low on
 * purpose, so a promising length is tried and then priced by what it
 * really costs; the times are shared by the engine's requests. On a
 * CPU each verify row costs most of a decode step and
 * short drafts win; on a GPU extra rows are nearly free and long ones
 * do.
 * @param s The request's accept counts.
 * @param cost The engine's forward times.
 * @param max_draft The most draft tokens allowed.
 * @returns 0 (run a plain decode) up to max_draft; 0 also until a
 *     plain decode has been timed.
 */
std::uint32_t choose_draft(const DraftState& s, const DraftCost& cost,
                           std::uint32_t max_draft);

}  // namespace locus::model
