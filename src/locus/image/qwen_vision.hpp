#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"

namespace locus::image {

/**
 * Qwen2.5-VL's vision tower (i#530), the image half of Qwen-Image-Edit's
 * text encoder: image patches in, the embeddings the language model
 * takes in place of its image tokens out.
 *
 * Tower: a 2 x 14 x 14 patch conv (two frames, so a still image is
 * fed twice), then pre-norm blocks (RMSNorm, attention with 2D RoPE,
 * RMSNorm, SwiGLU MLP). Most blocks attend only within windows of
 * window_size pixels (112: 4 x 4 merged patches); every
 * n_wa_pattern-th block attends over the whole image. The merger
 * RMS-normalizes each patch, joins each 2 x 2 group into one token
 * (4 x 1280 wide) and maps it to the LM's width through Linear, GELU,
 * Linear.
 *
 * Weights come from llama.cpp's mmproj GGUF (clip arch, projector
 * qwen2.5vl_merger): q, k and v as separate matrices, and the patch
 * conv as two 2D kernels, one per frame. Matrices stay in the GGUF
 * mapping, so the GgufFile must outlive the tower.
 */
class QwenVision {
  public:
    /**
     * @param g The mmproj GGUF.
     * @throws gguf::Error on a missing or wrong-sized tensor or key, or
     *     another projector type.
     */
    static QwenVision load(const gguf::GgufFile& g);

    /** @returns Pixels per patch side (14). */
    std::uint32_t patch_size() const { return patch_; }
    /** @returns Patches per merged token side (2). */
    std::uint32_t merge_size() const { return merge_; }
    /** @returns The width of one output token (the LM's, 3584). */
    std::uint32_t out_dim() const { return out_; }
    /** @returns Floats per patch: 3 channels x 2 frames x 14 x 14. */
    std::uint32_t patch_dim() const { return 3 * kFrames * patch_ * patch_; }

    /**
     * The reference processor's patches for one image whose sides are
     * multiples of patch_size() * merge_size() (the processor with
     * do_resize off): x / 255, (x - image_mean) / image_std, the frame
     * repeated, and the patches in merge order (each 2 x 2 group
     * contiguous, the groups row-major).
     *
     * @param rgb h * w * 3 bytes, row-major RGB.
     * @returns (h / 14) * (w / 14) patches of patch_dim() floats, in
     *     [channel][frame][y][x] order.
     * @throws std::invalid_argument on a bad size.
     */
    std::vector<float> patches(std::span<const std::uint8_t> rgb,
                               std::uint32_t h, std::uint32_t w) const;

    /**
     * The tower and the merger on one image's patches.
     *
     * @param patches gh * gw patches as patches() lays them out.
     * @param gh Patch rows; a multiple of merge_size().
     * @param gw Patch columns; a multiple of merge_size().
     * @param op The backend ops the matrices run on.
     * @returns gh * gw / 4 tokens of out_dim() floats, token-major, in
     *     merge order (row-major over the 2 x 2 groups).
     * @throws std::invalid_argument on a bad size.
     */
    std::vector<float> encode(std::span<const float> patches,
                              std::uint32_t gh, std::uint32_t gw,
                              const backend::Ops& op) const;

    /**
     * The windows of a gh x gw patch grid, as the reference's
     * get_window_index: each window lists its merged tokens (indices
     * into the row-major merged grid), windows row-major, tokens
     * row-major within each; an edge window is smaller and an empty
     * one is dropped.
     */
    std::vector<std::vector<std::uint32_t>> windows(std::uint32_t gh,
                                                    std::uint32_t gw) const;

  private:
    static constexpr std::uint32_t kFrames = 2;

    struct Block {
        std::span<const float> ln1, ln2, q_b, k_b, v_b, out_b, gate_b, up_b,
            down_b;
        backend::Mat q, k, v, out, gate, up, down;
        bool full = false;  ///< attends over the whole image
    };

    /** 2D RoPE on one token's heads (q or k, heads x hd), for the patch
     *  at row y and column x. */
    void rope(float* t, std::uint32_t y, std::uint32_t x) const;

    std::uint32_t patch_ = 0, merge_ = 0, width_ = 0, heads_ = 0, ffn_ = 0,
                  out_ = 0, window_ = 0;
    float eps_ = 0.0f;
    float mean_[3] = {}, std_[3] = {};
    std::vector<float> inv_freq_;    ///< hd / 4 rotary frequencies
    std::vector<float> patch_w_;     ///< width x patch_dim(), F32
    backend::Mat patch_mat_;         ///< over patch_w_
    std::span<const float> post_ln_, mm0_b_, mm2_b_;
    backend::Mat mm0_, mm2_;
    std::vector<Block> blocks_;
};

}  // namespace locus::image
