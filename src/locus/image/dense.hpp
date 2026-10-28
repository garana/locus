#pragma once

#include <cstdint>
#include <span>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"

/** Token-major dense ops the image models share (PixArt DiT, VAE). */
namespace locus::image::dense {

/**
 * out[t][r] = w[r] . x[t] + b[r] for n tokens, token-major in and out
 * (b may be empty for no bias). Rows are split across threads; each
 * slice runs the backend's batched matvec over blocks of 64 tokens.
 */
void linear(const backend::Ops& op, const backend::Mat& w,
            std::span<const float> b, std::span<const float> x,
            std::uint32_t n, std::span<float> out);

/**
 * Multi-head attention without a mask: nq queries (q, token-major,
 * heads * hd wide) over nk keys and values (k, v, same layout),
 * softmax(q k^T / sqrt(hd)) v, into out (nq rows, token-major).
 *
 * Each head's keys are copied to a [nk][hd] matrix and its values to
 * [hd][nk], so the scores and the weighted sum run through the
 * backend's batched matvec, 64 queries at a time.
 */
void attention(const backend::Ops& op, std::span<const float> q,
               std::uint32_t nq, std::span<const float> k,
               std::span<const float> v, std::uint32_t nk,
               std::uint32_t heads, std::uint32_t hd, std::span<float> out);

}  // namespace locus::image::dense
