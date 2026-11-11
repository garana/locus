#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/gguf/gguf.hpp"

namespace locus::image {

/**
 * Qwen-Image's VAE (diffusers AutoencoderKLQwenImage, i#533) on one
 * image: the Wan-style causal 3D VAE with a time axis of size 1.
 *
 * With a single frame every causal Conv3d sees two zero frames before
 * the image, so only its kernel's last time slice touches it, and it
 * is a Conv2d with that slice. The 3D resamplers skip their time conv
 * on the first frame, as the 2D ones do, so the encoder and decoder
 * are 2D networks:
 *
 * - residual blocks: RMS norm (x / |x| * sqrt(C) * gamma, over the
 *   channels of each pixel), SiLU, 3 x 3 conv, twice, plus a 1 x 1
 *   shortcut when the width changes;
 * - a mid block with one single-head attention over the pixels;
 * - downsample: zero pad right and bottom, 3 x 3 stride 2; upsample:
 *   nearest x 2, then a 3 x 3 conv to half the channels.
 *
 * Weights are the qwen_image_vae GGUF (diffusers names, F32, Conv3d
 * kernels as [out][in][kt * kh * kw]). The kernels are copied out at
 * load, so the GgufFile may close after it.
 */
class QwenVae {
  public:
    /**
     * @throws gguf::Error on a missing key or tensor, or a wrong size.
     */
    static QwenVae load(const gguf::GgufFile& g);

    /** @returns Latent channels (16). */
    std::uint32_t z_dim() const { return z_; }
    /** @returns Pixels per latent side: 2^(downsamples), 8 for the real
     *  model. */
    std::uint32_t scale() const;
    /** @returns Whether the GGUF holds the encoder. */
    bool has_encoder() const { return !enc_.empty(); }

    /**
     * The encoder's mean (the posterior's mode, as the edit pipeline
     * takes it), before latents_mean / latents_std.
     *
     * @param chw [3][h][w] pixels in [-1, 1]; h and w multiples of
     *     scale().
     * @returns [z_dim][h / scale()][w / scale()].
     * @throws std::invalid_argument on a bad size or no encoder.
     */
    std::vector<float> encode(std::span<const float> chw, std::uint32_t h,
                              std::uint32_t w) const;

    /**
     * The decoder, clamped to [-1, 1] as the reference.
     *
     * @param z [z_dim][h][w] latents, after latents_mean / latents_std
     *     are undone (denormalize()).
     * @returns [3][h * scale()][w * scale()].
     */
    std::vector<float> decode(std::span<const float> z, std::uint32_t h,
                              std::uint32_t w) const;

    /** z = (z - latents_mean) / latents_std, per channel, in place. */
    void normalize(std::span<float> z) const;
    /** z = z * latents_std + latents_mean, per channel, in place. */
    void denormalize(std::span<float> z) const;

    /** One conv as 2D: [out][in][k][k] weights and a bias. */
    struct Conv {
        std::vector<float> w, b;
        std::uint32_t in = 0, out = 0, k = 0;
    };
    /** A residual block (shortcut.out == 0 when there is none). */
    struct Res {
        std::vector<float> g1, g2;
        Conv c1, c2, shortcut;
    };
    /** The mid block's attention. */
    struct Attn {
        std::vector<float> g;
        Conv qkv, proj;
    };
    /** One layer of the encoder's or decoder's sequence. */
    struct Layer {
        enum class Kind { kRes, kAttn, kDown, kUp } kind;
        Res res;
        Attn attn;
        Conv conv;  ///< the resampler's conv
    };

  private:
    std::vector<float> run(const std::vector<Layer>& layers,
                           std::vector<float> x, std::uint32_t& c,
                           std::uint32_t& h, std::uint32_t& w) const;

    std::uint32_t z_ = 0, downs_ = 0;
    std::vector<float> mean_, std_;
    // encoder: conv_in, layers (down blocks, then the mid block),
    // norm_out, conv_out, quant_conv
    Conv enc_in_, enc_out_, quant_;
    std::vector<Layer> enc_;
    std::vector<float> enc_norm_;
    // decoder: post_quant_conv, conv_in, layers, norm_out, conv_out
    Conv post_quant_, dec_in_, dec_out_;
    std::vector<Layer> dec_;
    std::vector<float> dec_norm_;
};

}  // namespace locus::image
