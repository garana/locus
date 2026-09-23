#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/model/sampling.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::pipeline {

/** Generation policy for drive_generation. */
struct DriverParams {
    model::SamplingParams sampling;  /**< Default is greedy/argmax. */
    std::uint32_t max_tokens = 64;   /**< Cap on generated tokens. */
    std::uint64_t request_id = 1;    /**< request_id stamped on frames. */
    std::uint64_t seed = 0;          /**< RNG seed for non-greedy. */
};

/** Outcome of drive_generation. */
struct DriverResult {
    std::vector<tok::TokenId> tokens;  /**< Generated (prompt excluded). */
    bool complete = false;  /**< true if generation stopped normally (the
                             *   eos token or max_tokens); false if a
                             *   transport failure truncated it -- `tokens`
                             *   then holds only what was produced. */
};

/**
 * Drives generation across a remote pipeline stage chain from the
 * entry's side, over two already-connected sockets: `head_fd` is the
 * first stage's input (kToken frames go out here) and `logits_fd` is
 * where the last stage sends its kLogits back. For each prompt token,
 * then each sampled token, it writes one kToken frame and reads one
 * kLogits frame, samples the next token, and loops until the EOS token
 * or max_tokens.
 *
 * The stages own their KV caches and advance the position per stage, so
 * the driver keeps no cache and loads no model weights; it only needs
 * the eos id to stop (the logits width comes from the kLogits frame).
 * Greedy sampling (the default SamplingParams, temperature <= 0)
 * reproduces single-process argmax generation byte-for-byte.
 *
 * @param head_fd   Connected socket to the first stage's input.
 * @param logits_fd Connected socket the last stage sends logits on.
 * @param prompt    Prompt token ids, fed in order before generation.
 * @param eos_id    Stops generation when sampled (still included).
 * @param params    Sampling policy and limits.
 * @returns A DriverResult: the generated token ids and a `complete`
 *     flag distinguishing a normal stop (eos / max_tokens) from a
 *     transport failure that truncated generation.
 */
DriverResult drive_generation(int head_fd, int logits_fd,
                              std::span<const tok::TokenId> prompt,
                              tok::TokenId eos_id,
                              const DriverParams& params);

}  // namespace locus::pipeline
