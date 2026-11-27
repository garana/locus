#pragma once

#include <cstdint>
#include <span>

namespace locus::backend::metal {

/**
 * image::dense::attention on the GPU (i#1056): nq queries (q,
 * token-major, heads * hd wide) over nk keys and values (k, v, the same
 * layout), softmax(q k^T / sqrt(hd)) v into out (nq rows), every head
 * its own K / V head and no mask, through seq_attn_mma (metal::Seq's
 * attention kernel). Takes device_mutex(), so it must not be called
 * from inside a metal::Seq's scope (which holds it); runs alone and
 * waits.
 *
 * @returns false, doing nothing, without a usable Metal device or for
 *     a head width the kernel does not take (a multiple of 8, at most
 *     128): the caller runs the CPU's.
 * @throws std::invalid_argument when a span's size does not match.
 */
bool attention(std::span<const float> q, std::uint32_t nq,
               std::span<const float> k, std::span<const float> v,
               std::uint32_t nk, std::uint32_t heads, std::uint32_t hd,
               std::span<float> out);

}  // namespace locus::backend::metal
