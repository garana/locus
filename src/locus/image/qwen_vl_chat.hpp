#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "locus/image/image_input.hpp"
#include "locus/image/janus_chat.hpp"
#include "locus/image/qwen_vision.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::image {

/**
 * Chat with images on Qwen2.5-VL (i#1092): its chat template, its
 * processor's image sizing and patches, and the M-RoPE positions of a
 * prompt with any number of images, as transformers' Qwen2.5-VL
 * (processor and get_rope_index) builds them.
 */

/** The template's system prompt when the chat sets none. */
inline constexpr std::string_view kQwenVlSystemPrompt =
    "You are a helpful assistant.";

/** The processor's pixel bounds (preprocessor_config.json). */
inline constexpr std::uint64_t kQwenVlMinPixels = 3136;
inline constexpr std::uint64_t kQwenVlMaxPixels = 12845056;

/**
 * The chat template's text (chat_template.jinja with
 * add_generation_prompt): the system turn ("<|im_start|>system\n" +
 * system + "<|im_end|>\n", kQwenVlSystemPrompt when `system` is
 * unset), then each turn as "<|im_start|>" + role + "\n" + text +
 * "<|im_end|>\n", then "<|im_start|>assistant\n". Each
 * kImagePlaceholder in a turn becomes
 * "<|vision_start|><|image_pad|><|vision_end|>"; qwen_vl_chat_ids
 * expands the pad to the image's tokens.
 *
 * @param turns The conversation; the last turn must be the user's.
 * @throws std::invalid_argument on an unknown role, no turns, or a
 *     last turn that is not the user's.
 */
std::string qwen_vl_chat_text(std::span<const ChatTurn> turns,
                              const std::optional<std::string>& system);

/** One image as the processor prepares it. */
struct QwenVlImage {
    /** gh * gw patches (QwenVision::patches). */
    std::vector<float> patches;
    /** Patch rows and columns (multiples of the merge size). */
    std::uint32_t gh = 0, gw = 0;
};

/**
 * The processor's image: smart_resize to sides that are multiples of
 * patch x merge (28), the area within [min_pixels, max_pixels], with
 * Pillow's BICUBIC (the fast processor uses torchvision's, i#601),
 * then QwenVision::patches.
 *
 * @throws std::invalid_argument on an empty image.
 */
QwenVlImage qwen_vl_image(const QwenVision& v, const Rgb8& img,
                          std::uint64_t max_pixels = kQwenVlMaxPixels);

/** A tokenized chat prompt and where its images sit. */
struct QwenVlPrompt {
    std::vector<tok::TokenId> ids;
    /** The first <|image_pad|> of each image, in order. */
    std::vector<std::uint32_t> image_at;
    /** One (t, h, w) rope position per id. */
    std::vector<model::MropePos> pos;
    /** The position of the first generated token (every axis). */
    std::uint32_t next = 0;
};

/**
 * Tokenizes qwen_vl_chat_text's output and expands the k-th
 * <|image_pad|> to grids[k] merged tokens (gh * gw / merge^2), then
 * the rope positions as get_rope_index sets them: text at t = h = w,
 * advancing by one; an image's merged tokens at (s, s + row, s + col)
 * with s the position after the text before it, and the text after it
 * from s + max(rows, cols).
 *
 * @param grids Each image's merged rows and columns, in order.
 * @throws std::invalid_argument when the tokenizer has no single
 *     <|image_pad|> token, or the pads and the grids differ in count.
 */
QwenVlPrompt qwen_vl_chat_ids(
    const tok::Tokenizer& tok, std::string_view text,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> grids);

/**
 * The prompt's prefill: each image through the vision tower, its rows
 * in place of its pads, and the ids through the qwen2vl language model
 * with p.pos, in chunks of lm.prefill_chunk() positions.
 *
 * @param images The prompt's images, in order (qwen_vl_image).
 * @param seq A sequence with p.ids.size() positions ensured.
 * @param logits Out: n_vocab floats, the last position's.
 * @param op The backend ops the vision tower runs on (the language
 *     model runs on its own backend, which must take M-RoPE: a CPU
 *     one).
 * @param alive Polled between chunks; false stops early.
 * @returns false when alive() stopped it.
 * @throws std::invalid_argument when the images and p do not match.
 */
bool qwen_vl_prefill(const model::TransformerModel& lm,
                     const QwenVision& vision, const QwenVlPrompt& p,
                     std::span<const QwenVlImage> images,
                     kv::PagedKvCache& cache, kv::PagedKvCache::Seq& seq,
                     model::TransformerModel::Workspace& ws,
                     std::span<float> logits, const backend::Ops& op,
                     const std::function<bool()>& alive = {});

/**
 * One decode step: `token` at rope position `pos` on every axis.
 *
 * @param logits Out: n_vocab floats.
 */
void qwen_vl_step(const model::TransformerModel& lm, tok::TokenId token,
                  std::uint32_t pos, kv::PagedKvCache& cache,
                  kv::PagedKvCache::Seq& seq,
                  model::TransformerModel::Workspace& ws,
                  std::span<float> logits);

}  // namespace locus::image
