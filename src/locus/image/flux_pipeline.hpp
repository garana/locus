#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/clip_text.hpp"
#include "locus/image/flux_dit.hpp"
#include "locus/image/flux_sampler.hpp"
#include "locus/image/sd_vae.hpp"
#include "locus/image/t5_encoder.hpp"
#include "locus/tok/clip_tokenizer.hpp"
#include "locus/tok/ugm_tokenizer.hpp"

namespace locus::image {

/**
 * FLUX.1 text to image (i#372), as diffusers' FluxPipeline runs it:
 * CLIP-L's pooled prompt, T5-XXL's states of the prompt padded to
 * max_sequence_length (no attention mask: the pipeline runs T5 on the
 * padded ids), the flow-matching Euler loop with the guidance scale
 * embedded (dev, Kontext; schnell has no guidance embedding), and the
 * VAE decoder.
 *
 * Differences from the reference, by design: the start latent comes
 * from locus's seeded normal sampler (PixArtPipeline::normal_latent),
 * not torch's, so a seed gives a different (but repeatable) image.
 */
class FluxPipeline {
  public:
    /** One image request. */
    struct Request {
        std::string prompt;
        std::uint32_t width = 1024, height = 1024;  ///< multiples of 16
        std::uint32_t steps = 0;   ///< 0: the model's default
        float guidance = 3.5f;     ///< ignored without a guidance embedding
        std::uint32_t max_sequence_length = 512;  ///< T5 tokens
        std::uint64_t seed = 0;
    };

    /** Where the time went, in milliseconds. */
    struct Timing {
        double encode = 0, denoise = 0, decode = 0;
    };

    /**
     * Loads every *.gguf in `dir` and picks each part by its
     * general.architecture: "t5encoder" (with the T5 tokenizer),
     * "clip_text" (with the CLIP tokenizer), "flux" and
     * "autoencoder_kl".
     *
     * @throws gguf::Error when a part is missing, twice, or the parts
     *     do not fit together (T5 and CLIP widths, latent channels).
     */
    static std::unique_ptr<FluxPipeline> load_dir(const std::string& dir);

    /**
     * Checks a request's size and fields; generate runs it before it
     * draws the noise, so a bad size allocates nothing.
     *
     * @throws std::invalid_argument naming the bad field.
     */
    static void validate(const Request& req);

    /** @returns The default step count: 4 without a guidance embedding
     *  (schnell), else 28. */
    std::uint32_t default_steps() const {
        return dit_->guidance_embeds() ? 28 : 4;
    }
    /** @returns Whether the model takes a guidance scale. */
    bool guidance_embeds() const { return dit_->guidance_embeds(); }

    /**
     * Text to an RGB image.
     *
     * @param req The request: sides 16 to 1024, multiples of 16; steps
     *     0 to 1000; max_sequence_length 1 to 512.
     * @param op The backend ops (a CPU backend).
     * @param alive Polled before each transformer call; false cancels.
     * @param timing Filled when not null.
     * @returns width * height * 3 bytes, row-major RGB, or empty when
     *     cancelled.
     * @throws std::invalid_argument on a bad request.
     */
    std::vector<std::uint8_t> generate(const Request& req,
                                       const backend::Ops& op,
                                       const std::function<bool()>& alive,
                                       Timing* timing = nullptr) const;

    /**
     * The loop and the decode from a given start latent (for tests: the
     * reference's latent).
     *
     * @param latent [16][height / 8][width / 8].
     * @returns [3][height][width] in about [-1, 1], or empty when
     *     cancelled.
     */
    std::vector<float> generate_from(std::span<const float> latent,
                                     const Request& req,
                                     const backend::Ops& op,
                                     const std::function<bool()>& alive,
                                     Timing* timing = nullptr) const;

    /**
     * The pipeline's T5 ids: the prompt with </s>, truncated to
     * max_length (the last kept id </s>) and padded with <pad> to
     * max_length, as T5TokenizerFast with padding="max_length" gives.
     */
    std::vector<tok::TokenId> t5_ids(const std::string& prompt,
                                     std::uint32_t max_length) const;

  private:
    std::vector<std::unique_ptr<gguf::GgufFile>> files_;
    std::unique_ptr<tok::UgmTokenizer> t5_tok_;
    std::unique_ptr<tok::ClipTokenizer> clip_tok_;
    std::unique_ptr<T5Encoder> t5_;
    std::unique_ptr<ClipTextEncoder> clip_;
    std::unique_ptr<FluxDit> dit_;
    std::unique_ptr<FluxSampler> sampler_;
    std::unique_ptr<SdVaeDecoder> vae_;
};

}  // namespace locus::image
