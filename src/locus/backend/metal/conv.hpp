#pragma once

#include <cstdint>
#include <span>

namespace locus::backend::metal {

/**
 * The 2D convolution of conv2d_mt, [in_ch][in_h][in_w] to
 * [out_ch][oh][ow], on the GPU (i#1050): an implicit GEMM over the
 * weight's [out_ch][in_ch][kh][kw] rows on the simdgroup matrices.
 * Takes device_mutex(), so it must not be called from inside a
 * metal::Seq's scope (which holds it); runs alone and waits.
 *
 * @returns false, doing nothing, for groups other than 1 or without a
 *     usable Metal device: the caller runs the CPU's.
 * @throws std::invalid_argument when a span's size does not match.
 */
bool conv2d(std::span<const float> in, std::uint32_t in_ch,
            std::uint32_t in_h, std::uint32_t in_w,
            std::span<const float> weight, std::uint32_t out_ch,
            std::uint32_t kh, std::uint32_t kw, std::span<const float> bias,
            std::uint32_t stride, std::uint32_t pad, std::uint32_t groups,
            std::span<float> out);

}  // namespace locus::backend::metal
