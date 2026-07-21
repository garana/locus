#include "locus/chat/template.hpp"

namespace locus::chat {

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
        return ChatTemplate(Family::kLlama2);
    }
    if (has("<|user|>")) {
        return ChatTemplate(Family::kZephyr);
    }
    // DeepSeek uses fullwidth-bar markers like <U+FF5C>User<U+FF5C>.
    if (has("\xef\xbd\x9cUser\xef\xbd\x9c")) {
        return ChatTemplate(Family::kDeepSeek);
    }
    return ChatTemplate(Family::kPlain);
}

std::string ChatTemplate::apply(std::span<const Message> messages,
                                bool add_assistant_prompt) const {
    std::string out;
    switch (family_) {
        case Family::kLlama3: {
            for (const Message& m : messages) {
                out += "<|start_header_id|>" + m.role +
                       "<|end_header_id|>\n\n" + m.content +
                       "<|eot_id|>";
            }
            if (add_assistant_prompt) {
                out += "<|start_header_id|>assistant"
                       "<|end_header_id|>\n\n";
            }
            break;
        }
        case Family::kChatML: {
            for (const Message& m : messages) {
                out += "<|im_start|>" + m.role + "\n" +
                       m.content + "<|im_end|>\n";
            }
            if (add_assistant_prompt) {
                out += "<|im_start|>assistant\n";
            }
            break;
        }
        case Family::kZephyr: {
            for (const Message& m : messages) {
                out += "<|" + m.role + "|>\n" + m.content +
                       "</s>\n";
            }
            if (add_assistant_prompt) {
                out += "<|assistant|>\n";
            }
            break;
        }
        case Family::kLlama2: {
            // system folds into the first user turn.
            std::string system;
            for (const Message& m : messages) {
                if (m.role == "system") {
                    system = m.content;
                } else if (m.role == "user") {
                    out += "[INST] ";
                    if (!system.empty()) {
                        out += "<<SYS>>\n" + system +
                               "\n<</SYS>>\n\n";
                        system.clear();
                    }
                    out += m.content + " [/INST]";
                } else {
                    out += " " + m.content + " ";
                }
            }
            break;
        }
        case Family::kDeepSeek: {
            for (const Message& m : messages) {
                if (m.role == "user") {
                    out += "\xef\xbd\x9cUser\xef\xbd\x9c" +
                           m.content;
                } else if (m.role == "assistant") {
                    out += "\xef\xbd\x9c"
                           "Assistant\xef\xbd\x9c" +
                           m.content;
                } else {
                    out += m.content;
                }
            }
            if (add_assistant_prompt) {
                out += "\xef\xbd\x9c"
                       "Assistant\xef\xbd\x9c";
            }
            break;
        }
        case Family::kPlain: {
            for (const Message& m : messages) {
                if (!out.empty()) {
                    out += "\n";
                }
                out += m.content;
            }
            break;
        }
    }
    return out;
}

std::string_view ChatTemplate::name() const {
    switch (family_) {
        case Family::kLlama2: return "llama2";
        case Family::kLlama3: return "llama3";
        case Family::kChatML: return "chatml";
        case Family::kZephyr: return "zephyr";
        case Family::kDeepSeek: return "deepseek";
        case Family::kPlain: return "plain";
    }
    return "plain";
}

}  // namespace locus::chat
