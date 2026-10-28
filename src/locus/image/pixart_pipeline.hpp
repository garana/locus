#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/pixart_dit.hpp"
#include "locus/image/sd_vae.hpp"
#include "locus/image/t5_encoder.hpp"
#include "locus/tok/ugm_tokenizer.hpp"

namespace locus::image {

/**
 * PixArt-alpha text to image (i#146): the T5 encoder, the DiT, the
 * DPM-Solver++ loop with classifier-free guidance and the VAE decoder,
 * as diffusers' PixArtAlphaPipeline runs them
 * (docs/models/pixart-alpha.md section 1.2).
 *
 * Differences from the reference, by design:
 * - Prompt cleaning is diffusers' clean_caption=False path: strip the
 *   ASCII whitespace at both ends, lowercase ASCII letters. The full
 *   _clean_caption (URLs, HTML, CJK removal, ...) is not implemented;
 *   both give the same ids for plain English prompts.
 * - The start latent comes from locus's own seeded normal sampler, not
 *   torch's, so a seed gives a different (but repeatable) image.
 */
class PixArtPipeline {
  public:
    /** One image request. */
    struct Request {
        std::string prompt;
        std::string negative_prompt;  ///< "" as the pipeline's default
        std::uint32_t steps = 20;     ///< the pipeline's default
        float guidance = 4.5f;
        std::uint64_t seed = 0;
    };

    /** Where the time went, in milliseconds. */
    struct Timing {
        double encode = 0, denoise = 0, decode = 0;
    };

    /**
     * Loads every *.gguf in `dir` and picks each part by its
     * general.architecture: "t5encoder" (with the T5 tokenizer),
     * "pixart" and "autoencoder_kl".
     *
     * @throws gguf::Error when a part is missing, twice, or the parts
     *     do not fit together (T5 width, latent channels).
     */
    static std::unique_ptr<PixArtPipeline> load_dir(const std::string& dir);

    /** @returns The image side in pixels (512). */
    std::uint32_t image_size() const {
        return dit_->latent_size() * vae_->upscale();
    }
    /** @returns The GGUF's default guidance scale (4.5). */
    float default_guidance() const { return guidance_; }

    /**
     * Text to an RGB image.
     *
     * @param req The request; steps 1 to 1000.
     * @param op The backend ops (a CPU backend).
     * @param alive Polled before each DiT call; false cancels.
     * @param timing Filled when not null.
     * @returns image_size()^2 * 3 bytes, row-major RGB, or empty when
     *     cancelled.
     * @throws std::invalid_argument on a bad request.
     */
    std::vector<std::uint8_t> generate(const Request& req,
                                       const backend::Ops& op,
                                       const std::function<bool()>& alive,
                                       Timing* timing = nullptr) const;

    /**
     * The denoising loop and the decode from a given start latent (for
     * tests: the reference's latent).
     *
     * @returns [3][S][S] in about [-1, 1], or empty when cancelled.
     */
    std::vector<float> generate_from(std::span<const float> latent,
                                     const Request& req,
                                     const backend::Ops& op,
                                     const std::function<bool()>& alive,
                                     Timing* timing = nullptr) const;

    /** diffusers' clean_caption=False: ASCII strip and lowercase. */
    static std::string preprocess(std::string_view text);

    /**
     * The pipeline's token ids: preprocess, encode with </s>, and keep
     * at most max_length (120) ids, the last one </s>, as T5Tokenizer
     * with truncation does.
     */
    static std::vector<tok::TokenId> prompt_ids(const tok::Tokenizer& tok,
                                                std::string_view text,
                                                std::size_t max_length = 120);

    /**
     * n standard normal floats from `seed` (mt19937_64 and the
     * Box-Muller transform): locus's start latent.
     */
    static std::vector<float> normal_latent(std::uint64_t seed,
                                            std::size_t n);

  private:
    std::vector<std::unique_ptr<gguf::GgufFile>> files_;
    std::unique_ptr<tok::UgmTokenizer> tok_;
    std::unique_ptr<T5Encoder> t5_;
    std::unique_ptr<PixArtDit> dit_;
    std::unique_ptr<SdVaeDecoder> vae_;
    const gguf::GgufFile* dit_file_ = nullptr;
    float guidance_ = 4.5f;
};

}  // namespace locus::image
