#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/image_input.hpp"
#include "locus/image/qwen_image_dit.hpp"
#include "locus/image/qwen_image_sampler.hpp"
#include "locus/image/qwen_vae.hpp"
#include "locus/image/qwen_vision.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::image {

/**
 * Qwen-Image-Edit (i#616), as diffusers' QwenImageEditPipeline runs it:
 * the reference image sized for the VAE and for the vision tower
 * (qwen_edit_reference), the prompt and the negative prompt through
 * Qwen2.5-VL with the edit template (qwen_edit_text_embeds), the
 * reference's latents (qwen_edit_reference_latents), the denoising loop
 * with true CFG (qwen_edit_denoise) and the VAE decoder.
 *
 * Differences from the reference, by design: the start noise comes
 * from locus's seeded normal sampler (PixArtPipeline::normal_latent),
 * not torch's, so a seed gives a different (but repeatable) image; and
 * the vision tower's image is resized with Pillow's BICUBIC, where the
 * pipeline's fast processor uses torchvision's (i#601).
 */
class QwenEditPipeline {
  public:
    /** One edit request. */
    struct Request {
        std::string prompt;
        /** With true_cfg_scale over 1, the pipeline's true CFG. */
        std::optional<std::string> negative_prompt;
        Rgb8 reference;
        /** The output size; 0 for the VAE image's (about area pixels at
         *  the reference's aspect). Floored to multiples of 2 x the VAE
         *  scale. */
        std::uint32_t width = 0, height = 0;
        std::uint32_t steps = 50;
        float true_cfg_scale = 4.0f;
        std::uint64_t seed = 0;
        /** The reference's pixel count for the VAE (the pipeline's
         *  1024 x 1024; smaller in tests). */
        std::uint64_t area = 1024 * 1024;
        /** true (diffusers): the reference is encoded at about `area`
         *  pixels whatever the output size. false (i#670): at the
         *  output's area when that is smaller, so a small edit runs
         *  fewer reference tokens through the MMDiT and the vision
         *  tower. As FLUX Kontext's auto_resize. */
        bool auto_resize = true;
    };

    /** Where the time went, in milliseconds. */
    struct Timing {
        double encode = 0, denoise = 0, decode = 0;
    };

    /**
     * Loads every *.gguf in `dir` and picks each part by its
     * general.architecture: "qwen2vl" (the language model and its
     * tokenizer), "clip" (the Qwen2.5-VL vision tower, llama.cpp's
     * mmproj), "qwen_image" and "qwen_image_vae".
     *
     * @throws gguf::Error when a part is missing, twice, or the parts
     *     do not fit together.
     */
    static std::unique_ptr<QwenEditPipeline> load_dir(const std::string& dir);

    /**
     * Builds the pipeline from GGUFs the caller keeps alive (tests).
     *
     * @param tok The GGUF whose tokenizer to use (the language model's,
     *     in load_dir).
     * @throws gguf::Error when the parts do not fit together.
     */
    static std::unique_ptr<QwenEditPipeline> from_parts(
        const gguf::GgufFile& lm, const gguf::GgufFile& vision,
        const gguf::GgufFile& dit, const gguf::GgufFile& vae,
        const gguf::GgufFile& tok);

    /**
     * The edit, as RGB bytes.
     *
     * @param op The backend ops for the vision tower, the MMDiT and the
     *     VAE (the language model runs on the CPU).
     * @param alive Polled before each MMDiT step; false cancels.
     * @param out_w, out_h Receive the output size.
     * @returns out_w * out_h * 3 bytes, row-major RGB (the pipeline's
     *     PIL output: x * 255 rounded), or empty when cancelled.
     * @throws std::invalid_argument on a bad request.
     */
    std::vector<std::uint8_t> edit(const Request& req,
                                   const backend::Ops& op,
                                   const std::function<bool()>& alive,
                                   std::uint32_t& out_w,
                                   std::uint32_t& out_h,
                                   Timing* timing = nullptr) const;

    /**
     * The edit from given start noise (for tests: the reference's).
     *
     * @param noise The packed start latents of the output size.
     * @returns [height][width][3] floats in [0, 1] (output_type "np"),
     *     or empty when cancelled.
     */
    std::vector<float> edit_from(std::span<const float> noise,
                                 const Request& req, const backend::Ops& op,
                                 const std::function<bool()>& alive,
                                 Timing* timing = nullptr) const;

    /**
     * The output size of a request: width and height, or the VAE
     * image's size, floored to multiples of 2 x the VAE scale.
     *
     * @throws std::invalid_argument on an empty reference or a side
     *     that floors to 0.
     */
    std::pair<std::uint32_t, std::uint32_t> output_size(
        const Request& req) const;

    /** @returns The packed latent channels per token (64). */
    std::uint32_t channels() const { return dit_->channels(); }
    /** @returns The VAE's spatial scale (8). */
    std::uint32_t vae_scale() const { return vae_->scale(); }

  private:
    QwenEditPipeline() = default;

    /** The prompt's conditioning rows with the reference's vision
     *  tower rows (grid_h x grid_w merged tokens, encoded once for
     *  every prompt of the edit). */
    std::vector<float> text_rows(const std::string& prompt,
                                 std::span<const float> image_rows,
                                 std::uint32_t grid_h, std::uint32_t grid_w,
                                 std::uint32_t& n) const;

    std::vector<std::unique_ptr<gguf::GgufFile>> files_;
    std::unique_ptr<tok::Tokenizer> tok_;
    std::unique_ptr<model::TransformerModel> lm_;
    std::unique_ptr<QwenVision> vision_;
    std::unique_ptr<QwenImageDit> dit_;
    std::unique_ptr<QwenImageSampler> sampler_;
    std::unique_ptr<QwenVae> vae_;
};

}  // namespace locus::image
