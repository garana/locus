#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/gguf/gguf.hpp"

namespace locus::image {

/**
 * LTX-Video's VAE decoder (i#357): the Decoder of
 * Lightricks/LTX-Video's CausalVideoAutoencoder
 * (ltx_video/models/autoencoders), for the non-causal, pixel-norm,
 * timestep-conditioned decoder of the 0.9.x checkpoints. One call turns
 * a latent [C][F][H][W] (C = 128) into a video [3][(F - 1) 8 + 1][32 H]
 * [32 W], in about [-1, 1].
 *
 * - The latent is un-normalized per channel (times std-of-means, plus
 *   mean-of-means), as the pipeline's vae_per_channel_normalize does.
 * - Every conv is 3 x 3 x 3, stride 1: zero padding in space, the edge
 *   frame repeated in time (conv3d_k3).
 * - The blocks run in the reverse of the config's decoder_blocks list
 *   (the reference's up_blocks order): a "res_x" group is residual
 *   blocks (pixel norm, shift and scale from the timestep, SiLU, conv,
 *   twice, plus the input); a "compress_all" block is a conv to 8 / m
 *   times the channels, a 2 x 2 x 2 depth-to-space with the first frame
 *   dropped, plus the input shuffled and tiled the same way.
 * - Last: pixel norm, shift and scale from the timestep, SiLU, a conv
 *   to 3 x 4 x 4 channels, unpatchified to 4 x 4 pixels.
 *
 * Runs on the CPU thread pool. Weights come from an "ltxv_vae" GGUF
 * (scripts/convert-ltxv.py) and are read in place, so the GgufFile
 * must outlive the decoder.
 */
class LtxvVae {
  public:
    /**
     * @param g A GGUF whose general.architecture is "ltxv_vae".
     * @throws gguf::Error on a missing or wrong key or tensor.
     */
    static LtxvVae load(const gguf::GgufFile& g);

    /** @returns The latent channels (128). */
    std::uint32_t latent_channels() const { return latent_ch_; }
    /** @returns Video frames per latent frame step (8). */
    std::uint32_t time_scale() const { return time_scale_; }
    /** @returns Pixels per latent cell along each side (32). */
    std::uint32_t space_scale() const { return space_scale_; }

    /**
     * Decodes one latent.
     *
     * @param latent [latent_channels][f][h][w], as the sampler leaves it
     *     (before the per-channel un-normalization).
     * @param timestep The decode timestep (the pipeline's
     *     decode_timestep, for example 0.05).
     * @returns [3][(f - 1) time_scale + 1][h space_scale][w space_scale].
     * @throws std::invalid_argument on wrong sizes.
     */
    std::vector<float> decode(std::span<const float> latent, std::uint32_t f,
                              std::uint32_t h, std::uint32_t w,
                              float timestep) const;

    /**
     * A 3 x 3 x 3 conv, stride 1, as the reference's CausalConv3d with
     * zero spatial padding: in [ci][f][h][w] to out [co][f][h][w]. With
     * causal false, frame -1 reads frame 0 and frame f reads frame f - 1;
     * with causal true (the encoder), frames -2 and -1 read frame 0 and
     * nothing reads past the last. Runs across the thread pool, one item
     * per (co, frame, row).
     *
     * @param weight [co][ci][27], each kernel in (kt, ky, kx) order.
     * @param bias co floats.
     */
    static void conv3d_k3(std::span<const float> in, std::uint32_t ci,
                          std::uint32_t f, std::uint32_t h, std::uint32_t w,
                          std::span<const float> weight,
                          std::span<const float> bias, std::uint32_t co,
                          std::span<float> out, bool causal = false);

    /** @returns Whether the GGUF holds the encoder too. */
    bool has_encoder() const { return !enc_blocks_.empty(); }

    /**
     * The encoder (the reference's causal Encoder), as vae_encode with
     * vae_per_channel_normalize runs it, but taking the latent
     * distribution's mean where vae_encode samples it: patchify 4 x 4,
     * causal conv_in, the encoder blocks (res_x: pixel norm, SiLU, causal
     * conv, twice, plus the input; compress_*_res: a causal conv and a
     * space-to-depth by the block's stride, plus the input
     * space-to-depthed and averaged over channel groups; the time stride
     * first repeats frame 0), pixel norm, SiLU, causal conv_out, then
     * (mean - mean_of_means) / std_of_means.
     *
     * @param video [3][f][h][w] in [-1, 1]; f = 1 + time_scale() k, h
     *     and w multiples of space_scale().
     * @returns [latent_channels][(f - 1) / time_scale() + 1]
     *     [h / space_scale()][w / space_scale()].
     * @throws std::invalid_argument without an encoder or on bad sizes.
     */
    std::vector<float> encode(std::span<const float> video, std::uint32_t f,
                              std::uint32_t h, std::uint32_t w) const;

  private:
    struct Conv {
        std::span<const float> w, b;  ///< [co][ci][27], [co]
        std::uint32_t ci = 0, co = 0;
    };
    struct TimeMlp {  ///< Timesteps(256) -> linear_1 -> SiLU -> linear_2
        std::span<const float> w1, b1, w2, b2;
        std::uint32_t out = 0;
    };
    struct Res {
        Conv conv1, conv2;
        std::span<const float> table;  ///< [4][ch] scale_shift_table
    };
    struct Block {
        bool res = false;  ///< res_x group; otherwise compress_all
        std::uint32_t ch = 0;  ///< input channels
        std::uint32_t reduce = 1;  ///< compress_all's multiplier
        TimeMlp time;  ///< res_x: 4 x ch outputs
        std::vector<Res> layers;
        Conv conv;  ///< compress_all
    };

    std::uint32_t latent_ch_ = 0, out_ch_ = 0, patch_ = 0, time_scale_ = 1,
                  space_scale_ = 1;
    float t_scale_ = 0.0f;
    std::span<const float> mean_, std_;  ///< latent statistics
    Conv conv_in_, conv_out_;
    std::vector<Block> blocks_;
    TimeMlp last_time_;
    std::span<const float> last_table_;  ///< [2][ch]

    /** One encoder block: a res_x group, or a compress_*_res block with
     *  its (time, space) stride. */
    struct EncBlock {
        bool res = false;
        std::uint32_t ch = 0;   ///< input channels
        std::uint32_t out = 0;  ///< output channels
        std::uint32_t st = 1, ss = 1;  ///< time and space stride
        std::vector<Res> layers;  ///< res_x (their tables are empty)
        Conv conv;                ///< compress_*_res
    };
    std::uint32_t in_ch_ = 3;
    Conv enc_in_, enc_out_;
    std::vector<EncBlock> enc_blocks_;
};

}  // namespace locus::image
