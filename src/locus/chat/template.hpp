#pragma once

#include <span>
#include <string>
#include <string_view>

#include "locus/gguf/gguf.hpp"

namespace locus::chat {

/** One chat turn. */
struct Message {
    std::string role;  // "system", "user", "assistant"
    std::string content;
};

/**
 * Chat prompt formatter (DESIGN.md R2). The GGUF
 * `tokenizer.chat_template` Jinja source is not executed;
 * instead it is fingerprinted to one of the well-known families
 * and rendered by a hardcoded formatter, like llama.cpp's classic
 * path. Unknown templates fall back to plain "role: content"
 * concatenation.
 *
 * BOS is never emitted here -- the tokenizer adds it via
 * encode(add_bos). Role markers that are control tokens are
 * matched by the BPE special-token pass during encoding.
 */
class ChatTemplate {
  public:
    enum class Family {
        kPlain,     // "role: content" lines
        kLlama2,    // [INST] ... [/INST]
        kLlama3,    // <|start_header_id|>...<|eot_id|>
        kChatML,    // <|im_start|>...<|im_end|> (Qwen and kin)
        kZephyr,    // <|user|> ... </s> (TinyLlama-chat)
        kDeepSeek,  // <U+FF5C>User/Assistant markers
    };

    ChatTemplate() = default;
    explicit ChatTemplate(Family family) : family_(family) {}

    /** Fingerprints `tokenizer.chat_template` (kPlain if absent). */
    static ChatTemplate from_gguf(const gguf::GgufFile& g);

    /**
     * Renders messages into a model-ready prompt.
     *
     * @param add_assistant_prompt Append the assistant header so
     *     the model continues as the assistant (the normal case).
     */
    std::string apply(std::span<const Message> messages,
                      bool add_assistant_prompt = true) const;

    Family family() const { return family_; }

    /** @returns A short family name for logs. */
    std::string_view name() const;

  private:
    Family family_ = Family::kPlain;
};

}  // namespace locus::chat
