#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "locus/image/janus_vision.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::image {

/** The Janus reference processor's default system prompt. */
inline constexpr std::string_view kJanusSystemPrompt =
    "You are a helpful language and vision assistant. You are able to "
    "understand the visual content that the user provides, and assist "
    "the user with a variety of tasks using natural language.";

/** The text that stands for one image inside a turn. */
inline constexpr std::string_view kImagePlaceholder = "<image_placeholder>";

/** One chat turn; an image is kImagePlaceholder inside the text. */
struct ChatTurn {
    std::string role;  ///< "user" or "assistant"
    std::string text;
};

/**
 * The Janus reference's prompt text for a chat
 * (apply_sft_template_for_multi_turn_prompts, sft_format "deepseek"):
 * the system prompt and "\n\n" when it is not empty, then each user
 * turn as "<|User|>: <text>\n\n" and each assistant turn as
 * "<|Assistant|>: <text>" + eos, every text Python-stripped, then
 * "<|Assistant|>:" for the reply. The role labels are Janus-Pro's own
 * special tokens (its README's examples).
 *
 * @param turns The conversation; the last turn must be the user's.
 * @param system The system prompt (kJanusSystemPrompt by default in
 *     the reference); empty for none.
 * @param eos The end-of-sentence token's text.
 * @throws std::invalid_argument on an unknown role, no turns, or a
 *     last turn that is not the user's.
 */
std::string janus_chat_text(std::span<const ChatTurn> turns,
                            std::string_view system, std::string_view eos);

/**
 * Tokenizes the prompt text (with BOS) and expands each
 * <image_placeholder> id to <begin_of_image>, tokens() placeholders and
 * <end_of_image>, as the reference's add_image_token does.
 *
 * @param n_images Set to the number of images in the prompt.
 * @returns The ids; image k fills the k-th run of placeholders.
 */
std::vector<tok::TokenId> janus_chat_ids(const tok::Tokenizer& tok,
                                         std::string_view text,
                                         const JanusVision& v,
                                         std::size_t& n_images);

}  // namespace locus::image
