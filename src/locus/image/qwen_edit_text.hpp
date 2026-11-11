#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/image/qwen_vision.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::image {

/**
 * Qwen-Image-Edit's text conditioning (i#574), as diffusers'
 * QwenImageEditPipeline._get_qwen_prompt_embeds: the edit template with
 * the user's instruction and the reference image, through Qwen2.5-VL
 * (the vision tower and the qwen2vl language model with M-RoPE), its
 * last hidden states after the final norm, the template's system part
 * (the first kEditDropTokens positions) dropped.
 */

/** diffusers' prompt_template_encode up to the image (0.35.1). */
inline constexpr std::string_view kEditTemplateHead =
    "<|im_start|>system\nDescribe the key features of the input image "
    "(color, shape, size, texture, objects, background), then explain how "
    "the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while "
    "maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|>";
/** The template between the image and the instruction. */
inline constexpr std::string_view kEditTemplateMid = "<|vision_end|>";
/** The template after the instruction. */
inline constexpr std::string_view kEditTemplateTail =
    "<|im_end|>\n<|im_start|>assistant\n";
/** prompt_template_encode_start_idx: the system part's tokens. */
inline constexpr std::uint32_t kEditDropTokens = 64;

/** One edit prompt's token ids and where its image span sits. */
struct EditPrompt {
    std::vector<tok::TokenId> ids;
    /** The first image token (the one after <|vision_start|>). */
    std::uint32_t image_at = 0;
    /** Image tokens: one per merged 2 x 2 patch group. */
    std::uint32_t n_image = 0;
};

/**
 * The template with `instruction`, its <|image_pad|> repeated n_image
 * times as the Qwen2.5-VL processor expands it, tokenized with special
 * tokens.
 *
 * @throws std::invalid_argument when the tokenizer has no single
 *     <|image_pad|> token, or n_image is 0.
 */
EditPrompt qwen_edit_prompt(const tok::Tokenizer& tok,
                            std::string_view instruction,
                            std::uint32_t n_image);

/**
 * The conditioning rows of one edit prompt.
 *
 * @param lm The qwen2vl language model (M-RoPE; CPU backend).
 * @param vision The vision tower; its out_dim() is lm's width.
 * @param ids The prompt (qwen_edit_prompt, or any ids with one image
 *     span); its image-span tokens are replaced by the tower's output.
 * @param image_at The image span's first position.
 * @param patches gh * gw patches of the reference image, as
 *     QwenVision::patches() lays them out.
 * @param gh Patch rows (a multiple of merge_size()).
 * @param gw Patch columns (a multiple of merge_size()).
 * @param drop Leading positions dropped (kEditDropTokens).
 * @param op The backend ops the vision tower runs on.
 * @returns (ids.size() - drop) rows of lm's width, token-major.
 * @throws std::invalid_argument on a size that does not match.
 */
std::vector<float> qwen_edit_text_embeds(const model::TransformerModel& lm,
                                         const QwenVision& vision,
                                         std::span<const tok::TokenId> ids,
                                         std::uint32_t image_at,
                                         std::span<const float> patches,
                                         std::uint32_t gh, std::uint32_t gw,
                                         std::uint32_t drop,
                                         const backend::Ops& op);

}  // namespace locus::image
