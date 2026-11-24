#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"

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
 * matched by the BPE special-token pass during encoding; render()
 * marks the message text, which encode_literal() keeps as text.
 */
class ChatTemplate {
  public:
    enum class Family {
        kPlain,     // "role: content" lines
        // [INST] ... [/INST], llama.cpp's four variants (i#1152): a
        // template without <<SYS>> (system text before the first user
        // turn); with <<SYS>>; that plus <s> before each later [INST];
        // and <<SYS>> with every message stripped.
        kLlama2,
        kLlama2Sys,
        kLlama2SysBos,
        kLlama2SysStrip,
        kLlama3,    // <|start_header_id|>...<|eot_id|>
        kChatML,    // <|im_start|>...<|im_end|> (Qwen and kin)
        kZephyr,    // <|user|> ... </s> (TinyLlama-chat)
        kDeepSeek,  // <<U+FF5C>User<U+FF5C>> markers (DeepSeek-V3)
        kDeepSeek2, // "User: " / "Assistant: " (DeepSeek-V2)
        kDeepSeekCoder,  // "### Instruction:" / <|EOT|> (Coder V1)
    };

    ChatTemplate() = default;
    explicit ChatTemplate(Family family) : family_(family) {}

    /** Fingerprints `tokenizer.chat_template` (kPlain if absent). */
    static ChatTemplate from_gguf(const gguf::GgufFile& g);

    /**
     * The template a name picks, for a GGUF that carries none
     * (`--chat-template NAME`, i#1152): the names name() returns, as
     * llama.cpp's built-in template names are ("llama2-sys-strip").
     *
     * @returns The template, or nullopt for an unknown name.
     */
    static std::optional<ChatTemplate> from_name(std::string_view name);

    /** @returns Every name from_name() takes, comma separated. */
    static std::string_view names();

    /**
     * Renders messages into a model-ready prompt.
     *
     * @param add_assistant_prompt Append the assistant header so
     *     the model continues as the assistant (the normal case).
     */
    std::string apply(std::span<const Message> messages,
                      bool add_assistant_prompt = true) const;

    /** A rendered prompt and where its message text lies. */
    struct Prompt {
        std::string text;
        /** Byte ranges of `text` copied from message content. */
        std::vector<tok::TextRange> content;
    };

    /**
     * As apply(), and records each message's content range, for
     * tok::Tokenizer::encode_literal().
     */
    Prompt render(std::span<const Message> messages,
                  bool add_assistant_prompt = true) const;

    Family family() const { return family_; }

    /** @returns A short family name for logs. */
    std::string_view name() const;

  private:
    Family family_ = Family::kPlain;
};

}  // namespace locus::chat
