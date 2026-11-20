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
#include "locus/image/qwen_image_dit.hpp"
#include "locus/image/qwen_image_sampler.hpp"
#include "locus/image/qwen_vae.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::image {

/** diffusers' QwenImagePipeline prompt_template_encode (0.35.1), split
 *  around the user's prompt. */
inline constexpr std::string_view kImageTemplateHead =
    "<|im_start|>system\nDescribe the image by detailing the color, "
    "shape, size, texture, quantity, text, spatial relationships of the "
    "objects and background:<|im_end|>\n<|im_start|>user\n";
/** The template after the prompt. */
inline constexpr std::string_view kImageTemplateTail =
    "<|im_end|>\n<|im_start|>assistant\n";
/** prompt_template_encode_start_idx: the system part's tokens. */
inline constexpr std::uint32_t kImageDropTokens = 34;
/** tokenizer_max_length: the prompt's token cap before the drop. */
inline constexpr std::uint32_t kImageTokenizerMax = 1024;

/**
 * The text-to-image template with `prompt`, tokenized with special
 * tokens and cut to kImageTokenizerMax + kImageDropTokens tokens, as
 * the pipeline's tokenizer call truncates it.
 */
std::vector<tok::TokenId> qwen_image_prompt(const tok::Tokenizer& tok,
                                            std::string_view prompt);

/**
 * The conditioning rows of one text-to-image prompt: the ids through
 * the qwen2vl language model (text only, so every M-RoPE axis is the
 * token's position), its last hidden states after the final norm, the
 * first `drop` positions dropped and at most `max_rows` kept
 * (max_sequence_length).
 *
 * @param lm The qwen2vl language model (CPU backend).
 * @returns min(ids.size() - drop, max_rows) rows of lm's width.
 * @throws std::invalid_argument when ids has no more than `drop` tokens.
 */
std::vector<float> qwen_image_text_embeds(const model::TransformerModel& lm,
                                          std::span<const tok::TokenId> ids,
                                          std::uint32_t drop,
                                          std::uint32_t max_rows);

/**
 * Qwen-Image text to image (i#1091), as diffusers' QwenImagePipeline
 * runs it: the prompt and the negative prompt through Qwen2.5-VL with
 * the text-to-image template (qwen_image_text_embeds), the denoising
 * loop with true CFG on the noise tokens alone (qwen_edit_denoise with
 * no reference) and the VAE decoder.
 *
 * As QwenEditPipeline, the start noise comes from locus's seeded normal
 * sampler, so a seed gives a different (but repeatable) image than
 * diffusers.
 */
class QwenImagePipeline {
  public:
    /** One text-to-image request. */
    struct Request {
        std::string prompt;
        /** With true_cfg_scale over 1, the pipeline's true CFG. */
        std::optional<std::string> negative_prompt;
        /** Pixels; floored to multiples of 2 x the VAE scale. The
         *  pipeline's default is 1024 x 1024. */
        std::uint32_t width = 1024, height = 1024;
        std::uint32_t steps = 50;
        float true_cfg_scale = 4.0f;
        std::uint64_t seed = 0;
        /** Text rows kept after the template's system part (1 to
         *  1024; the pipeline's default 512). */
        std::uint32_t max_sequence_length = 512;
    };

    /** Where the time went, in milliseconds. */
    struct Timing {
        double encode = 0, denoise = 0, decode = 0;
    };

    /**
     * Loads every *.gguf in `dir` and picks each part by its
     * general.architecture: "qwen2vl" (the language model and its
     * tokenizer), "qwen_image" and "qwen_image_vae". A "clip" GGUF (the
     * vision tower) is not a part: a directory with one is a
     * Qwen-Image-Edit set (QwenEditPipeline).
     *
     * @throws gguf::Error when a part is missing, twice, or the parts
     *     do not fit together.
     */
    static std::unique_ptr<QwenImagePipeline> load_dir(
        const std::string& dir);

    /**
     * Builds the pipeline from GGUFs the caller keeps alive (tests).
     *
     * @param tok The GGUF whose tokenizer to use (the language model's,
     *     in load_dir).
     * @throws gguf::Error when the parts do not fit together.
     */
    static std::unique_ptr<QwenImagePipeline> from_parts(
        const gguf::GgufFile& lm, const gguf::GgufFile& dit,
        const gguf::GgufFile& vae, const gguf::GgufFile& tok);

    /**
     * The image, as RGB bytes.
     *
     * @param op The backend ops for the MMDiT and the VAE (the language
     *     model runs on the CPU).
     * @param alive Polled before each MMDiT step; false cancels.
     * @param out_w, out_h Receive the output size.
     * @returns out_w * out_h * 3 bytes, row-major RGB (the pipeline's
     *     PIL output: x * 255 rounded), or empty when cancelled.
     * @throws std::invalid_argument on a bad request.
     */
    std::vector<std::uint8_t> generate(const Request& req,
                                       const backend::Ops& op,
                                       const std::function<bool()>& alive,
                                       std::uint32_t& out_w,
                                       std::uint32_t& out_h,
                                       Timing* timing = nullptr) const;

    /**
     * The image from given start noise (for tests).
     *
     * @param noise The packed start latents of the output size.
     * @returns [height][width][3] floats in [0, 1] (output_type "np"),
     *     or empty when cancelled.
     */
    std::vector<float> generate_from(std::span<const float> noise,
                                     const Request& req,
                                     const backend::Ops& op,
                                     const std::function<bool()>& alive,
                                     Timing* timing = nullptr) const;

    /**
     * The output size of a request, floored to multiples of 2 x the
     * VAE scale.
     *
     * @throws std::invalid_argument on a side that floors to 0.
     */
    std::pair<std::uint32_t, std::uint32_t> output_size(
        const Request& req) const;

    /** @returns The packed latent channels per token (64). */
    std::uint32_t channels() const { return dit_->channels(); }
    /** @returns The VAE's spatial scale (8). */
    std::uint32_t vae_scale() const { return vae_->scale(); }
    /** @returns The GGUFs load_dir used, by name; others it skipped. */
    const std::vector<std::string>& file_paths() const { return paths_; }

  private:
    QwenImagePipeline() = default;

    std::vector<float> text_rows(const std::string& prompt,
                                 std::uint32_t max_rows,
                                 std::uint32_t& n) const;

    std::vector<std::unique_ptr<gguf::GgufFile>> files_;
    std::vector<std::string> paths_;
    std::unique_ptr<tok::Tokenizer> tok_;
    std::unique_ptr<model::TransformerModel> lm_;
    std::unique_ptr<QwenImageDit> dit_;
    std::unique_ptr<QwenImageSampler> sampler_;
    std::unique_ptr<QwenVae> vae_;
};

}  // namespace locus::image
