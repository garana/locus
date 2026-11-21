#include "locus/image/janus_chat.hpp"

#include <stdexcept>

#include "locus/image/janus.hpp"

namespace locus::image {

std::string janus_chat_text(std::span<const ChatTurn> turns,
                            std::string_view system, std::string_view eos,
                            std::vector<tok::TextRange>* content) {
    if (turns.empty()) {
        throw std::invalid_argument("janus chat: no turns");
    }
    if (turns.back().role != "user") {
        throw std::invalid_argument("janus chat: the last turn must be the "
                                    "user's");
    }
    std::string out;
    // Appends the client's text and records its range.
    const auto put = [&](std::string_view s) {
        if (content) content->push_back({out.size(), out.size() + s.size()});
        out += s;
    };
    if (!system.empty()) {
        put(system);
        out += "\n\n";
    }
    // Janus-Pro's role labels (its own special tokens, as in the
    // README's Janus-Pro examples; the plain "User" / "Assistant" of
    // the repo's inference.py are Janus-1.3B's).
    // A turn's text, with its image placeholders (the server's, never
    // the client's) left out of the client's ranges.
    const auto put_turn = [&](std::string_view text) {
        for (;;) {
            const auto at = text.find(kImagePlaceholder);
            put(text.substr(0, at));
            if (at == std::string_view::npos) break;
            out += kImagePlaceholder;
            text.remove_prefix(at + kImagePlaceholder.size());
        }
    };
    for (const ChatTurn& t : turns) {
        if (t.role == "user") {
            out += "<|User|>: ";
            put_turn(python_strip(t.text));
            out += "\n\n";
        } else if (t.role == "assistant") {
            out += "<|Assistant|>: ";
            put_turn(python_strip(t.text));
            out += eos;
        } else {
            throw std::invalid_argument("janus chat: unsupported role " +
                                        t.role);
        }
    }
    out += "<|Assistant|>:";
    return out;
}

std::vector<tok::TokenId> janus_chat_ids(
    const tok::Tokenizer& tok, std::string_view text, const JanusVision& v,
    std::size_t& n_images, std::span<const tok::TextRange> content) {
    const auto ids = tok.encode_literal(text, true, content);
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
