#pragma once

#include <cstdint>
#include <span>

#include "locus/backend/cpu_ops.hpp"

namespace locus::backend::metal {

/**
 * @returns Whether the Metal backend has a matvec kernel for weights
 *     of type t (F32, Q8_0, Q4_K, Q6_K), for any shape whose columns
 *     are a whole number of t's blocks.
 */
bool matvec_supported(gguf::TensorType t);

/**
 * out = w x on the GPU (i#866): w's bytes are copied once into a
 * shared Metal buffer, cached by their host pointer (so pass only
 * weights that stay put, such as mmapped tensors), and x and out go
 * through scratch buffers. Weight types without a kernel run the
 * scalar CPU matvec.
 * @throws std::invalid_argument when x or out has the wrong size.
 */
void matvec(const Mat& w, std::span<const float> x, std::span<float> out);

/**
 * A bench hook: records `reps` copies of w's matvec in one batch (each
 * ordered after the one before, as a forward chains them), submits it
 * and waits.
 * @param out The last copy's result.
 * @returns The submit-to-done wall time per copy, in ms.
 * @throws std::invalid_argument for a type without a kernel, reps 0,
 *     or mismatched spans.
 */
double matvec_repeat(const Mat& w, std::span<const float> x,
                     std::span<float> out, std::uint32_t reps);

/** Frees every cached weight buffer (tests; the next matvec uploads
 *  again). */
void drop_weights();

}  // namespace locus::backend::metal
