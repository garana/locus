#include "locus/image/janus_chat.hpp"

#include <stdexcept>

#include "locus/image/janus.hpp"

namespace locus::image {

std::string janus_chat_text(std::span<const ChatTurn> turns,
                            std::string_view system, std::string_view eos) {
    if (turns.empty()) {
        throw std::invalid_argument("janus chat: no turns");
    }
    if (turns.back().role != "user") {
        throw std::invalid_argument("janus chat: the last turn must be the "
                                    "user's");
    }
    std::string out;
    if (!system.empty()) {
        out += system;
        out += "\n\n";
    }
    // Janus-Pro's role labels (its own special tokens, as in the
    // README's Janus-Pro examples; the plain "User" / "Assistant" of
    // the repo's inference.py are Janus-1.3B's).
    for (const ChatTurn& t : turns) {
        if (t.role == "user") {
            out += "<|User|>: ";
            out += python_strip(t.text);
            out += "\n\n";
        } else if (t.role == "assistant") {
            out += "<|Assistant|>: ";
            out += python_strip(t.text);
            out += eos;
        } else {
            throw std::invalid_argument("janus chat: unsupported role " +
                                        t.role);
        }
    }
    out += "<|Assistant|>:";
    return out;
}

std::vector<tok::TokenId> janus_chat_ids(const tok::Tokenizer& tok,
                                         std::string_view text,
                                         const JanusVision& v,
                                         std::size_t& n_images) {
    const auto ids = tok.encode(text, true);
    std::vector<tok::TokenId> out;
    out.reserve(ids.size());
    n_images = 0;
    for (const tok::TokenId id : ids) {
        if (id != v.placeholder()) {
            out.push_back(id);
            continue;
        }
        out.push_back(v.begin_of_image());
        out.insert(out.end(), v.tokens(), v.placeholder());
        out.push_back(v.end_of_image());
        ++n_images;
    }
    return out;
}

}  // namespace locus::image
