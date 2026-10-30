#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/image_input.hpp"
#include "locus/image/ltxv_dit.hpp"
#include "locus/image/ltxv_vae.hpp"
#include "locus/image/t5_encoder.hpp"
#include "locus/tok/ugm_tokenizer.hpp"

namespace locus::image {

/**
 * LTX-Video text to video (i#357): the T5 encoder, the DiT, the
 * rectified-flow loop and the VAE decoder, as the reference
 * LTXVideoPipeline runs the 2B distilled checkpoint in one pass (the
 * 0.9.6 distilled config's form: the base pipeline, 8 steps from the
 * checkpoint's LinearQuadratic schedule, guidance 1, stochastic
 * sampling, no upscaler).
 *
 * - The prompt is stripped of ASCII whitespace at both ends (the
 *   reference's _text_preprocessing) and tokenized with </s>, keeping
 *   at most 256 ids; masked T5 rows are left out (the DiT gives them
 *   weight 0 anyway).
 * - Each token's coordinates are its latent (frame, row, column) times
 *   (8, 32, 32), the time divided by the frame rate (no causal
 *   first-frame shift: the 2B checkpoint does not set
 *   causal_temporal_positioning).
 * - Each step (rf.py's step) with t_next = 0 after the last: stochastic
 *   (the default, as that config sets it), x0 = x - t v, then
 *   x = (1 - t_next) x0 + t_next z with fresh noise z; or
 *   deterministic (the 0.9.8 multi-scale config's choice),
 *   x -= (t - t_next) v.
 * - Before the decode: x = x (1 - 0.025) + noise 0.025, then the
 *   decoder at timestep 0.05 (decode_noise_scale, decode_timestep).
 * - Image to video (Request::image), as inference.py conditions on a
 *   first frame at strength 1: the image is center-cropped to the
 *   clip's aspect, resized (bicubic) and blurred (3 x 3, sigma 1); the
 *   VAE encodes it to frame 0's latent, which replaces frame 0 of the
 *   start noise. Each step sets frame 0 to that latent plus
 *   image_noise * z * t^2 (fresh z), runs the DiT with frame 0's tokens
 *   at sigma 0, and steps only the other frames.
 *
 * Differences from the reference, by design: the start, step and decode
 * noise come from locus's seeded normal sampler (PixArtPipeline::
 * normal_latent), not torch's, so a seed gives a different (but
 * repeatable) clip. For image to video, the image skips the reference's
 * CRF step (an H.264 encode and decode, which needs a video codec), and
 * the encode takes the latent mean where the reference samples it.
 */
class LtxvPipeline {
  public:
    /** One video request. */
    struct Request {
        std::string prompt;
        std::uint32_t width = 512;   ///< a multiple of 32
        std::uint32_t height = 320;  ///< a multiple of 32
        std::uint32_t frames = 49;   ///< 8 k + 1
        float frame_rate = 30.0f;
        std::uint64_t seed = 0;
        bool stochastic = true;  ///< the step kind (see the class doc)
        /** Image to video: the first frame (any size; see the class doc). */
        std::optional<Rgb8> image;
        /** The noise on frame 0 each step, times t^2 (the reference's
         *  image_cond_noise_scale). */
        float image_noise = 0.15f;
    };

    /** Where the time went, in milliseconds (denoise includes T5). */
    struct Timing {
        double denoise = 0, decode = 0;
    };

    /** The largest request sides and frame count (limits()). */
    static constexpr std::uint32_t kMaxSide = 1280, kMaxFrames = 257;

    /**
     * Loads every *.gguf in `dir` and picks each part by its
     * general.architecture: "t5encoder" (with the T5 tokenizer), "ltxv"
     * and "ltxv_vae".
     *
     * @throws gguf::Error when a part is missing, twice, or the parts
     *     do not fit together, or the DiT's timestep list is not the
     *     8-step LinearQuadratic schedule.
     */
    static std::unique_ptr<LtxvPipeline> load_dir(const std::string& dir);

    /**
     * Checks a request's sizes: width and height multiples of 32 in
     * [32, kMaxSide], frames 8 k + 1 in [1, kMaxFrames], a frame rate
     * in (0, 1000], and for image to video a non-empty image and an
     * image_noise in [0, 1].
     *
     * @throws std::invalid_argument naming the bad field.
     */
    static void check(const Request& req);

    /** @returns Whether the VAE has its encoder (image to video). */
    bool can_condition() const { return vae_->has_encoder(); }

    /**
     * The first frame as load_image_to_tensor_with_resize_and_crop makes
     * it, without the CRF step: the center crop to width / height's
     * aspect, Pillow's bicubic resize, torchvision's gaussian_blur
     * (kernel 3, sigma 1, reflect padding), then [0, 255] to [-1, 1].
     *
     * @returns [3][height][width].
     * @throws std::invalid_argument on an empty image or size.
     */
    static std::vector<float> condition_image(const Rgb8& image,
                                              std::uint32_t width,
                                              std::uint32_t height);

    /**
     * Frame 0's latent from condition_image's output: the VAE encoder's
     * normalized mean.
     *
     * @returns [128][height / 32][width / 32].
     * @throws std::invalid_argument without an encoder or on a bad size.
     */
    std::vector<float> encode_image(std::span<const float> image,
                                    const Request& req) const;

    /** @returns The T5 tokenizer (for prompt_ids). */
    const tok::Tokenizer& tokenizer() const { return *tok_; }

    /**
     * Text to video.
     *
     * @param op The backend ops the DiT and T5 run on.
     * @param alive Polled before each DiT call; false cancels.
     * @returns req.frames frames, each height * width * 3 row-major RGB
     *     bytes, or empty when cancelled.
     * @throws std::invalid_argument on a bad request.
     */
    std::vector<std::vector<std::uint8_t>> generate(
        const Request& req, const backend::Ops& op,
        const std::function<bool()>& alive, Timing* timing = nullptr) const;

    /**
     * The prompt, the 8 steps and nothing else, from a given start noise
     * (for tests: the reference's).
     *
     * @param noise [128][f][h][w] with f = (frames - 1) / 8 + 1,
     *     h = height / 32, w = width / 32.
     * @param step_noise For a stochastic request, the 8 steps' noise,
     *     each shaped like noise (for tests: the reference's); empty
     *     draws it from req.seed. Ignored for a deterministic request.
     * @param first Image to video: frame 0's latent, [128][h][w]
     *     (encode_image's); empty for text to video.
     * @param first_noise With first: the 8 steps' frame-0 noise, each
     *     [128][h][w] (for tests: the reference's); empty draws it from
     *     req.seed.
     * @returns The latent after the last step, same shape, or empty when
     *     cancelled.
     */
    std::vector<float> denoise(std::span<const float> noise,
                               std::span<const float> step_noise,
                               const Request& req, const backend::Ops& op,
                               const std::function<bool()>& alive,
                               std::span<const float> first = {},
                               std::span<const float> first_noise = {}) const;

    /**
     * The decode of a denoised latent with a given decode noise.
     *
     * @returns [3][frames][height][width] in about [-1, 1].
     */
    std::vector<float> decode(std::span<const float> latent,
                              std::span<const float> decode_noise,
                              const Request& req) const;

    /**
     * The reference's postprocess per frame: [3][frames][height][width]
     * to req.frames frames of height * width * 3 RGB bytes
     * (SdVaeDecoder::to_rgb8 on each).
     */
    static std::vector<std::vector<std::uint8_t>> to_frames(
        std::span<const float> video, const Request& req);

    /**
     * The reference's linear_quadratic_schedule(n): n noise levels from
     * 1 down, half linear (to 0.025), half quadratic, in double, cast
     * to float.
     */
    static std::vector<float> schedule(std::uint32_t n);

    /** @returns The reference's token coordinates, [3][f * h * w]. */
    static std::vector<float> coords(std::uint32_t f, std::uint32_t h,
                                     std::uint32_t w, float frame_rate);

    /** The reference's _text_preprocessing: ASCII strip. */
    static std::string preprocess(std::string_view text);

    /**
     * The token ids: preprocess, encode with </s>, and keep at most
     * max_length ids, the last one </s>, as T5Tokenizer with
     * truncation does.
     */
    static std::vector<tok::TokenId> prompt_ids(const tok::Tokenizer& tok,
                                                std::string_view text,
                                                std::size_t max_length = 256);

  private:
    std::vector<std::unique_ptr<gguf::GgufFile>> files_;
    std::unique_ptr<tok::UgmTokenizer> tok_;
    std::unique_ptr<T5Encoder> t5_;
    std::unique_ptr<LtxvDit> dit_;
    std::unique_ptr<LtxvVae> vae_;
    std::vector<float> steps_;  ///< schedule(8)
};

}  // namespace locus::image
