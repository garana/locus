#include "locus/tok/wpm_tokenizer.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "locus/tok/unicode.hpp"
#include "locus/tok/unicode_flags.hpp"

namespace locus::tok {

namespace {

/** The word-start marker of the converted vocab, U+2581. */
constexpr std::string_view kSpace = "\xe2\x96\x81";
constexpr std::uint32_t kReplacement = 0xFFFD;

/**
 * Decodes the codepoint at s[i] and advances i, as llama.cpp's
 * `unicode_cpt_from_utf8` (and its caller) does: a bad lead byte, a
 * bad or missing continuation byte or an overlong form yields U+FFFD
 * and consumes one byte.
 */
std::uint32_t next_cp_or_fffd(std::string_view s, std::size_t& i) {
    const auto b = [&](std::size_t k) {
        return static_cast<std::uint32_t>(static_cast<unsigned char>(s[k]));
    };
    const auto cont = [&](std::size_t k) {
        return k < s.size() && (b(k) & 0xc0) == 0x80;
    };
    const std::uint32_t c = b(i);
    if ((c & 0x80) == 0) {
        i += 1;
        return c;
    }
    if ((c & 0x40) != 0) {
        if ((c & 0x20) == 0) {
            if (cont(i + 1)) {
                const std::uint32_t r = ((c & 0x1f) << 6) | (b(i + 1) & 0x3f);
                if (r >= 0x80) {
                    i += 2;
                    return r;
                }
            }
        } else if ((c & 0x10) == 0) {
            if (cont(i + 1) && cont(i + 2)) {
                const std::uint32_t r = ((c & 0x0f) << 12) |
                                        ((b(i + 1) & 0x3f) << 6) |
                                        (b(i + 2) & 0x3f);
                if (r >= 0x800) {
                    i += 3;
                    return r;
                }
            }
        } else if ((c & 0x08) == 0) {
            if (cont(i + 1) && cont(i + 2) && cont(i + 3)) {
                const std::uint32_t r =
                    ((c & 0x07) << 18) | ((b(i + 1) & 0x3f) << 12) |
                    ((b(i + 2) & 0x3f) << 6) | (b(i + 3) & 0x3f);
                if (r >= 0x10000 && r <= 0x10ffff) {
                    i += 4;
                    return r;
                }
            }
        }
    }
    i += 1;
    return kReplacement;
}

/** The CJK ideograph ranges of BERT's `_is_chinese_char`, with the
 *  0x2B920 start HF's Rust code uses (llama.cpp's choice). */
bool is_cjk(std::uint32_t c) {
    return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
           (c >= 0x20000 && c <= 0x2A6DF) ||
           (c >= 0x2A700 && c <= 0x2B73F) ||
           (c >= 0x2B740 && c <= 0x2B81F) ||
           (c >= 0x2B920 && c <= 0x2CEAF) ||
           (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
}

}  // namespace

WpmTokenizer::WpmTokenizer(Config config) : cfg_(std::move(config)) {
    const auto n = cfg_.tokens.size();
    if (cfg_.types.size() != n) {
        throw std::invalid_argument("wpm: tokens and types differ in size");
    }
    const auto in_range = [n](TokenId id) {
        return id >= 0 && static_cast<std::size_t>(id) < n;
    };
    if (!in_range(cfg_.cls) || !in_range(cfg_.sep) || !in_range(cfg_.unk) ||
        !in_range(cfg_.pad)) {
        throw std::invalid_argument("wpm: special token id out of range");
    }
    token_to_id_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto& t = cfg_.tokens[i];
        if (t.empty()) t = "[EMPTY_" + std::to_string(i) + "]";
        // A duplicate piece maps to its last id, as in llama.cpp.
        token_to_id_[t] = static_cast<TokenId>(i);
        max_token_len_ = std::max(max_token_len_, t.size());
    }
}

WpmTokenizer WpmTokenizer::from_gguf(const gguf::GgufFile& g) {
    if (g.get_string("tokenizer.ggml.model") != "bert") {
        throw gguf::Error("tokenizer model is not WordPiece (\"bert\")");
    }
    const auto* tokens = g.get_array("tokenizer.ggml.tokens");
    if (tokens == nullptr || tokens->empty()) {
        throw gguf::Error("tokenizer tokens missing");
    }
    Config cfg;
    cfg.tokens.reserve(tokens->size());
    for (const auto& v : *tokens) {
        if (v.type != gguf::ValueType::kString) {
            throw gguf::Error("token entry is not a string");
        }
        cfg.tokens.push_back(v.s);
    }
    if (const auto* types = g.get_array("tokenizer.ggml.token_type")) {
        for (const auto& v : *types) {
            cfg.types.push_back(static_cast<std::int32_t>(v.i));
        }
    }
    if (cfg.types.size() != cfg.tokens.size()) {
        cfg.types.assign(cfg.tokens.size(),
                         static_cast<std::int32_t>(TokenType::kNormal));
    }
    auto id_of = [&](std::string_view key, TokenId fallback) {
        auto v = g.get_uint(key);
        return v ? static_cast<TokenId>(*v) : fallback;
    };
    // llama.cpp's BERT defaults; "seperator" is the GGUF key's spelling.
    cfg.cls = id_of("tokenizer.ggml.bos_token_id", 101);
    cfg.sep = id_of("tokenizer.ggml.seperator_token_id", 102);
    cfg.unk = id_of("tokenizer.ggml.unknown_token_id", 100);
    cfg.pad = id_of("tokenizer.ggml.padding_token_id", 0);
    try {
        return WpmTokenizer(std::move(cfg));
    } catch (const std::invalid_argument& e) {
        throw gguf::Error(e.what());
    }
}

std::vector<std::string> WpmTokenizer::words(std::string_view text) const {
    std::vector<std::string> out(1);
    for (std::size_t i = 0; i < text.size();) {
        std::uint32_t c = next_cp_or_fffd(text, i);
        if (cfg_.strip_accents) c = cpt_nfd(c);
        const auto f = cpt_flags(c);
        if ((f & cpt::kWhitespace) != 0) {
            if (!out.back().empty()) out.emplace_back();
            continue;
        }
        if (c == 0 || c == kReplacement || (f & cpt::kControl) != 0) continue;
        if (cfg_.strip_accents && (f & cpt::kAccentMark) != 0) continue;
        std::string s;
        append_utf8(s, cfg_.lowercase ? cpt_lower(c) : c);
        if ((f & cpt::kPunctuation) != 0 ||
            (c < 0x7F && (f & cpt::kSymbol) != 0) || is_cjk(c)) {
            if (!out.back().empty()) out.emplace_back();
            out.back() = std::move(s);
            out.emplace_back();
        } else {
            out.back() += s;
        }
    }
    if (out.back().empty()) out.pop_back();
    return out;
}

std::vector<TokenId> WpmTokenizer::encode(std::string_view text,
                                          bool add_special) const {
    std::vector<TokenId> out;
    if (add_special) out.push_back(cfg_.cls);
    std::string w;
    for (const auto& word : words(text)) {
        w.assign(kSpace);
        w += word;
        const std::string_view wv = w;
        const std::size_t start = out.size();
        for (std::size_t i = 0; i < wv.size();) {
            bool found = false;
            // Longest first, up to the longest piece in the vocab.
            for (std::size_t j = std::min(wv.size(), i + max_token_len_ + 1);
                 j > i; --j) {
                const auto it = token_to_id_.find(wv.substr(i, j - i));
                if (it != token_to_id_.end()) {
                    out.push_back(it->second);
                    i = j;
                    found = true;
                    break;
                }
            }
            if (!found) {
                out.resize(start);
                break;
            }
        }
        if (out.size() == start) out.push_back(cfg_.unk);
    }
    if (add_special) out.push_back(cfg_.sep);
    return out;
}

std::string WpmTokenizer::decode(std::span<const TokenId> ids) const {
    std::string out;
    for (const TokenId id : ids) {
        const auto p = piece(id);
        const auto t = static_cast<TokenType>(
            cfg_.types[static_cast<std::size_t>(id)]);
        if (t == TokenType::kControl || t == TokenType::kUnused) continue;
        for (std::size_t k = 0; k < p.size();) {
            if (p.substr(k, kSpace.size()) == kSpace) {
                out += ' ';
                k += kSpace.size();
            } else {
                out += p[k++];
            }
        }
    }
    if (!out.empty() && out.front() == ' ') out.erase(0, 1);
    return out;
}

std::string_view WpmTokenizer::piece(TokenId id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= cfg_.tokens.size()) {
        throw std::out_of_range("wpm: token id out of range");
    }
    return cfg_.tokens[static_cast<std::size_t>(id)];
}

}  // namespace locus::tok
