#pragma once

#include <cstdint>
#include <span>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/metal/context.hpp"

namespace locus::backend::metal {

/**
 * @returns Whether the Metal backend has a matvec kernel for weights
 *     of type t (F32, Q8_0, Q4_K, Q6_K), for any shape whose columns
 *     are a whole number of t's blocks.
 */
bool matvec_supported(gguf::TensorType t);

/**
 * out = w x on the GPU (i#866): w's bytes are copied once into a
 * shared Metal buffer, cached by their host pointer, and x and out go
 * through scratch buffers. The cache checks only type and shape, so
 * pass only weights that stay put for the process (mmapped tensors):
 * another matrix of the same type and shape at a freed one's address
 * would get the old bytes. Calls from several threads run one at a
 * time. Weight types without a kernel run the scalar CPU matvec.
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

/**
 * Records out[out_off + r] (+)= dot(row r of the matrix, x[x_off ..])
 * for r < rows into ctx's open batch (or runs it alone): the forward's
 * building block. The caller holds device_mutex().
 * @param t The weight type (matvec_supported).
 * @param w The buffer holding the matrix's bytes from w_off on: a
 *     multiple of 4 bytes (16 for F32).
 * @param x_off Floats into x, a multiple of 4.
 * @param out_off Floats into out.
 * @throws std::invalid_argument for a type without a kernel, columns
 *     that are not whole blocks, or a bad x_off.
 */
void record_matvec(Context& ctx, gguf::TensorType t, std::uint32_t rows,
                   std::uint32_t cols, const Context::Buffer& w,
                   std::uint64_t w_off, const Context::Buffer& x,
                   std::uint32_t x_off, const Context::Buffer& out,
                   std::uint64_t out_off, bool accumulate);

/** @returns The bytes of a rows x cols matrix of type t (0 for a type
 *  without a kernel). */
std::size_t matvec_weight_bytes(gguf::TensorType t, std::uint32_t rows,
                                std::uint32_t cols);

}  // namespace locus::backend::metal
