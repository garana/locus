#pragma once

#include <cstdint>

#include "locus/model/transformer.hpp"

namespace locus::pipeline {

/**
 * Serves expert calls on one connection (expert parallelism, #781)
 * until the peer closes it: each kExpertCall (layer, expert, xb) gets a
 * kExpertOut with the same tags and model.expert_out's d, so the
 * caller gets the bits its own moe_ffn would add. Calls are answered
 * one at a time, in order.
 *
 * Anything else ends the connection: another message kind, an expert
 * outside [expert_lo, expert_hi), a layer the model has no experts in,
 * or an xb that is not n_embd floats.
 *
 * @param fd A connected socket; the caller closes it.
 * @returns The number of calls answered.
 */
std::uint64_t serve_expert_calls(const model::TransformerModel& m, int fd,
                                 std::uint32_t expert_lo,
                                 std::uint32_t expert_hi);

}  // namespace locus::pipeline
