#include "locus/chat/template.hpp"

namespace locus::chat {

namespace {

// DeepSeek control tokens; U+FF5C is the fullwidth bar, U+2581 the
// SentencePiece space.
const std::string kDsUser = "<\xef\xbd\x9cUser\xef\xbd\x9c>";
const std::string kDsAssistant = "<\xef\xbd\x9c"
                                 "Assistant\xef\xbd\x9c>";
const std::string kDsEos = "<\xef\xbd\x9c"
                           "end\xe2\x96\x81of\xe2\x96\x81sentence"
                           "\xef\xbd\x9c>";

/** s without leading and trailing whitespace, as llama.cpp's trim()
 *  (isspace: space, \t, \n, \v, \f, \r). */
std::string_view trim(std::string_view s) {
    constexpr std::string_view ws = " \t\n\v\f\r";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string_view::npos) {
        return {};
    }
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

// Every family, in the order names() lists them.
constexpr ChatTemplate::Family kFamilies[] = {
    ChatTemplate::Family::kPlain,         ChatTemplate::Family::kLlama2,
    ChatTemplate::Family::kLlama2Sys,     ChatTemplate::Family::kLlama2SysBos,
    ChatTemplate::Family::kLlama2SysStrip, ChatTemplate::Family::kLlama3,
    ChatTemplate::Family::kChatML,        ChatTemplate::Family::kZephyr,
    ChatTemplate::Family::kDeepSeek,      ChatTemplate::Family::kDeepSeek2,
    ChatTemplate::Family::kDeepSeekCoder};

}  // namespace

std::optional<ChatTemplate> ChatTemplate::from_name(std::string_view name) {
    for (const Family f : kFamilies) {
        if (ChatTemplate(f).name() == name) {
            return ChatTemplate(f);
        }
    }
    return std::nullopt;
}

std::string_view ChatTemplate::names() {
    return "plain, llama2, llama2-sys, llama2-sys-bos, llama2-sys-strip, "
           "llama3, chatml, zephyr, deepseek, deepseek2, deepseek-coder";
}

ChatTemplate ChatTemplate::from_gguf(const gguf::GgufFile& g) {
    const auto t = g.get_string("tokenizer.chat_template");
    if (!t) {
        return ChatTemplate(Family::kPlain);
    }
    const std::string_view s = *t;
    auto has = [&](std::string_view needle) {
        return s.find(needle) != std::string_view::npos;
    };
    if (has("<|start_header_id|>")) {
        return ChatTemplate(Family::kLlama3);
    }
    if (has("<|im_start|>")) {
        return ChatTemplate(Family::kChatML);
    }
    if (has("[INST]")) {
        // llama.cpp's Llama 2 variant rules, in its order (i#1152).
        if (has("content.strip()")) {
            return ChatTemplate(Family::kLlama2SysStrip);
        }
        if (has("bos_token + '[INST]")) {
            return ChatTemplate(Family::kLlama2SysBos);
        }
        if (has("<<SYS>>")) {
            return ChatTemplate(Family::kLlama2Sys);
        }
        return ChatTemplate(Family::kLlama2);
    }
    if (has("<|user|>")) {
        return ChatTemplate(Family::kZephyr);
    }
    // DeepSeek-V3 uses fullwidth-bar markers like <U+FF5C>User<U+FF5C>.
    if (has("\xef\xbd\x9cUser\xef\xbd\x9c")) {
        return ChatTemplate(Family::kDeepSeek);
    }
    // DeepSeek-V2 (and Coder-V2): plain "User: " / "Assistant: "
    // turns; the same fingerprint llama.cpp uses.
    if (has("'Assistant: ' + message['content'] + eos_token")) {
        return ChatTemplate(Family::kDeepSeek2);
    }
    // DeepSeek-Coder V1 instruct (deepseek-coder-33b-instruct), with
    // llama.cpp's fingerprint.
    if (has("### Instruction:") && has("<|EOT|>")) {
        return ChatTemplate(Family::kDeepSeekCoder);
    }
    return ChatTemplate(Family::kPlain);
}

std::string ChatTemplate::apply(std::span<const Message> messages,
                                bool add_assistant_prompt) const {
    return render(messages, add_assistant_prompt).text;
}

ChatTemplate::Prompt ChatTemplate::render(
    std::span<const Message> messages, bool add_assistant_prompt) const {
    Prompt p;
    std::string& out = p.text;
    // Appends message text and records its range.
    const auto put = [&](std::string_view s) {
        p.content.push_back({out.size(), out.size() + s.size()});
        out += s;
    };
    switch (family_) {
        case Family::kLlama3: {
            for (const Message& m : messages) {
                out += "<|start_header_id|>" + m.role +
                       "<|end_header_id|>\n\n";
                put(m.content);
                out += "<|eot_id|>";
            }
            if (add_assistant_prompt) {
                out += "<|start_header_id|>assistant"
                       "<|end_header_id|>\n\n";
            }
            break;
        }
        case Family::kChatML: {
            for (const Message& m : messages) {
                out += "<|im_start|>" + m.role + "\n";
                put(m.content);
                out += "<|im_end|>\n";
            }
            if (add_assistant_prompt) {
                out += "<|im_start|>assistant\n";
            }
            break;
        }
        case Family::kZephyr: {
            for (const Message& m : messages) {
                out += "<|" + m.role + "|>\n";
                put(m.content);
                out += "</s>\n";
            }
            if (add_assistant_prompt) {
                out += "<|assistant|>\n";
            }
            break;
        }
        case Family::kLlama2:
        case Family::kLlama2Sys:
        case Family::kLlama2SysBos:
        case Family::kLlama2SysStrip: {
            // llama.cpp's LLM_CHAT_TEMPLATE_LLAMA_2* rendering, string
            // for string (i#1152): an assistant turn ends with </s>,
            // and the next user turn opens a new [INST] (<s>[INST] in
            // the BOS variant). The model answers after [/INST], so
            // there is no assistant header to add.
            const bool sys = family_ != Family::kLlama2;
            const bool bos = family_ == Family::kLlama2SysBos;
            const bool strip = family_ == Family::kLlama2SysStrip;
            bool inside = true;  // the opening [INST] below
            out += "[INST] ";
            for (const Message& m : messages) {
                const std::string_view c =
                    strip ? trim(m.content) : std::string_view(m.content);
                if (!inside) {
                    inside = true;
                    out += bos ? "<s>[INST] " : "[INST] ";
                }
                if (m.role == "system") {
                    if (sys) {
                        out += "<<SYS>>\n";
                        put(c);
                        out += "\n<</SYS>>\n\n";
                    } else {
                        put(c);
                        out += "\n";
                    }
                } else if (m.role == "user") {
                    put(c);
                    out += " [/INST]";
                } else {
                    put(c);
                    out += "</s>";
                    inside = false;
                }
            }
            break;
        }
        case Family::kDeepSeek: {
            for (const Message& m : messages) {
                if (m.role == "user") {
                    out += kDsUser;
                    put(m.content);
                } else if (m.role == "assistant") {
                    out += kDsAssistant;
                    put(m.content);
                    out += kDsEos;
                } else {
                    put(m.content);
                    out += "\n\n";
                }
            }
            if (add_assistant_prompt) {
                out += kDsAssistant;
            }
            break;
        }
        case Family::kDeepSeek2: {
            for (const Message& m : messages) {
                if (m.role == "user") {
                    out += "User: ";
                    put(m.content);
                    out += "\n\n";
                } else if (m.role == "assistant") {
                    out += "Assistant: ";
                    put(m.content);
                    out += kDsEos;
                } else {
                    put(m.content);
                    out += "\n\n";
                }
            }
            if (add_assistant_prompt) {
                out += "Assistant:";
            }
            break;
        }
        case Family::kDeepSeekCoder: {
            for (const Message& m : messages) {
                if (m.role == "user") {
                    out += "### Instruction:\n";
                    put(m.content);
                    out += "\n";
                } else if (m.role == "assistant") {
                    out += "### Response:\n";
                    put(m.content);
                    out += "\n<|EOT|>\n";
                } else {
                    put(m.content);
                }
            }
            if (add_assistant_prompt) {
                out += "### Response:\n";
            }
            break;
        }
        case Family::kPlain: {
            for (const Message& m : messages) {
                if (!out.empty()) {
                    out += "\n";
                }
                put(m.content);
            }
            break;
        }
    }
    return p;
}

std::string_view ChatTemplate::name() const {
    switch (family_) {
        case Family::kLlama2: return "llama2";
        case Family::kLlama2Sys: return "llama2-sys";
        case Family::kLlama2SysBos: return "llama2-sys-bos";
        case Family::kLlama2SysStrip: return "llama2-sys-strip";
        case Family::kLlama3: return "llama3";
        case Family::kChatML: return "chatml";
        case Family::kZephyr: return "zephyr";
        case Family::kDeepSeek: return "deepseek";
        case Family::kDeepSeek2: return "deepseek2";
        case Family::kDeepSeekCoder: return "deepseek-coder";
        case Family::kPlain: return "plain";
    }
    return "plain";
}

}  // namespace locus::chat
