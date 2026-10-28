#include "locus/tok/ugm_tokenizer.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace locus::tok {

namespace {

/** SentencePiece's escaped space, U+2581 LOWER ONE EIGHTH BLOCK. */
constexpr std::string_view kSpace = "\xe2\x96\x81";

/** Score penalty below the lowest piece for an unknown character
 *  (SentencePiece's kUnkPenalty). */
constexpr float kUnkPenalty = 10.0f;

/** At most this many charsmap matches count, as SentencePiece's
 *  kMaxTrieResultsSize. */
constexpr std::size_t kMaxTrieResults = 32;

/** @returns The byte length of the UTF-8 character led by `c`, from
 *      its high nibble (SentencePiece's OneCharLen). */
std::size_t one_char_len(unsigned char c) {
    static constexpr std::uint8_t kLen[16] = {1, 1, 1, 1, 1, 1, 1, 1,
                                              1, 1, 1, 1, 2, 2, 3, 4};
    return kLen[c >> 4];
}

bool is_trail(unsigned char c) { return (c & 0xc0) == 0x80; }

bool valid_cp(std::uint32_t c) {
    return c < 0xd800 || (c >= 0xe000 && c <= 0x10ffff);
}

/**
 * SentencePiece's IsValidDecodeUTF8: one well-formed character at the
 * start of `s` (no overlong form, no surrogate).
 *
 * @returns Its byte length, or 0 when malformed.
 */
std::size_t valid_utf8_len(std::string_view s) {
    const auto b = [&](std::size_t i) {
        return static_cast<unsigned char>(s[i]);
    };
    const std::size_t n = s.size();
    if (b(0) < 0x80) return 1;
    if (n >= 2 && (b(0) & 0xe0) == 0xc0) {
        const std::uint32_t cp = ((b(0) & 0x1fu) << 6) | (b(1) & 0x3fu);
        if (is_trail(b(1)) && cp >= 0x80 && valid_cp(cp)) return 2;
    } else if (n >= 3 && (b(0) & 0xf0) == 0xe0) {
        const std::uint32_t cp = ((b(0) & 0x0fu) << 12) |
                                 ((b(1) & 0x3fu) << 6) | (b(2) & 0x3fu);
        if (is_trail(b(1)) && is_trail(b(2)) && cp >= 0x800 &&
            valid_cp(cp)) {
            return 3;
        }
    } else if (n >= 4 && (b(0) & 0xf8) == 0xf0) {
        const std::uint32_t cp = ((b(0) & 0x07u) << 18) |
                                 ((b(1) & 0x3fu) << 12) |
                                 ((b(2) & 0x3fu) << 6) | (b(3) & 0x3fu);
        if (is_trail(b(1)) && is_trail(b(2)) && is_trail(b(3)) &&
            cp >= 0x10000 && valid_cp(cp)) {
            return 4;
        }
    }
    return 0;
}

/** Python's str.isspace for the ASCII range, which is what HF strips
 *  around lstrip/rstrip added tokens in practice. */
bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

std::string_view trim_right(std::string_view s) {
    while (!s.empty() && is_ws(s.back())) s.remove_suffix(1);
    return s;
}

std::string_view trim_left(std::string_view s) {
    while (!s.empty() && is_ws(s.front())) s.remove_prefix(1);
    return s;
}

}  // namespace

UgmTokenizer::UgmTokenizer(Config config) : cfg_(std::move(config)) {
    const std::size_t n = cfg_.tokens.size();
    if (n == 0) {
        throw std::invalid_argument("empty vocabulary");
    }
    if (cfg_.scores.size() != n || cfg_.types.size() != n) {
        throw std::invalid_argument("vocab array size mismatch");
    }
    for (TokenId id : {cfg_.eos, cfg_.unk, cfg_.pad}) {
        if (id < 0 || static_cast<std::size_t>(id) >= n) {
            throw std::invalid_argument("special token id out of range");
        }
    }

    // Charsmap: a uint32 LE byte size of the darts-clone trie, the
    // trie units, then the NUL-terminated replacement strings.
    const auto& cm = cfg_.charsmap;
    if (!cm.empty()) {
        if (cm.size() < 4) {
            throw std::invalid_argument("charsmap too short");
        }
        const std::uint32_t trie_bytes =
            std::uint32_t{cm[0]} | (std::uint32_t{cm[1]} << 8) |
            (std::uint32_t{cm[2]} << 16) | (std::uint32_t{cm[3]} << 24);
        if (trie_bytes % 4 != 0 || trie_bytes == 0 ||
            trie_bytes > cm.size() - 4) {
            throw std::invalid_argument("charsmap trie size invalid");
        }
        trie_.resize(trie_bytes / 4);
        for (std::size_t i = 0; i < trie_.size(); ++i) {
            const std::uint8_t* p = cm.data() + 4 + 4 * i;
            trie_[i] = std::uint32_t{p[0]} | (std::uint32_t{p[1]} << 8) |
                       (std::uint32_t{p[2]} << 16) |
                       (std::uint32_t{p[3]} << 24);
        }
        replacements_ = std::string_view(
            reinterpret_cast<const char*>(cm.data()) + 4 + trie_bytes,
            cm.size() - 4 - trie_bytes);
    }

    min_score_ = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < n; ++i) {
        const auto type = static_cast<TokenType>(cfg_.types[i]);
        const auto& tok = cfg_.tokens[i];
        if (type == TokenType::kNormal) {
            pieces_.emplace(tok, static_cast<TokenId>(i));
            max_piece_len_ = std::max(max_piece_len_, tok.size());
            min_score_ = std::min(min_score_, cfg_.scores[i]);
        } else if (type == TokenType::kUserDefined && !tok.empty()) {
            user_defined_.emplace_back(tok, static_cast<TokenId>(i));
        }
    }
    if (pieces_.empty()) {
        min_score_ = 0;
    }
    // Longest first, so a token that is a prefix of another loses.
    std::stable_sort(user_defined_.begin(), user_defined_.end(),
                     [](const auto& a, const auto& b) {
                         return a.first.size() > b.first.size();
                     });
}

UgmTokenizer UgmTokenizer::from_gguf(const gguf::GgufFile& g) {
    if (g.get_string("tokenizer.ggml.model") != "t5") {
        throw gguf::Error("tokenizer model is not unigram (\"t5\")");
    }
    const auto* tokens = g.get_array("tokenizer.ggml.tokens");
    const auto* scores = g.get_array("tokenizer.ggml.scores");
    const auto* types = g.get_array("tokenizer.ggml.token_type");
    if (tokens == nullptr || tokens->empty() || scores == nullptr ||
        types == nullptr) {
        throw gguf::Error("tokenizer tokens, scores or types missing");
    }
    Config cfg;
    cfg.tokens.reserve(tokens->size());
    for (const auto& v : *tokens) {
        if (v.type != gguf::ValueType::kString) {
            throw gguf::Error("token entry is not a string");
        }
        cfg.tokens.push_back(v.s);
    }
    for (const auto& v : *scores) {
        cfg.scores.push_back(static_cast<float>(v.f));
    }
    for (const auto& v : *types) {
        cfg.types.push_back(static_cast<std::int32_t>(v.i));
    }
    if (const auto* cm = g.get_array("tokenizer.ggml.precompiled_charsmap")) {
        cfg.charsmap.reserve(cm->size());
        for (const auto& v : *cm) {
            cfg.charsmap.push_back(static_cast<std::uint8_t>(v.u));
        }
    }
    auto id_of = [&](std::string_view key, TokenId fallback) {
        auto v = g.get_uint(key);
        return v ? static_cast<TokenId>(*v) : fallback;
    };
    cfg.eos = id_of("tokenizer.ggml.eos_token_id", 1);
    cfg.unk = id_of("tokenizer.ggml.unknown_token_id", 2);
    cfg.pad = id_of("tokenizer.ggml.padding_token_id", 0);
    cfg.add_space_prefix =
        g.get_bool("tokenizer.ggml.add_space_prefix").value_or(true);
    cfg.remove_extra_whitespaces =
        g.get_bool("tokenizer.ggml.remove_extra_whitespaces")
            .value_or(true);
    cfg.add_eos = g.get_bool("tokenizer.ggml.add_eos_token").value_or(true);
    try {
        return UgmTokenizer(std::move(cfg));
    } catch (const std::invalid_argument& e) {
        throw gguf::Error(e.what());
    }
}

std::pair<std::string_view, std::size_t> UgmTokenizer::charsmap_prefix(
    std::string_view s) const {
    // darts-clone commonPrefixSearch: walk the double array one byte
    // at a time; every unit with a leaf ends a rule. Unit fields:
    // label = u & (1<<31 | 0xff), has_leaf = bit 8, value = low 31
    // bits, offset = (u >> 10) << ((u & 1<<9) >> 6).
    if (trie_.empty() || s.empty()) return {{}, 0};
    const auto offset = [](std::uint32_t u) -> std::size_t {
        return static_cast<std::size_t>(u >> 10) << ((u & (1u << 9)) >> 6);
    };
    std::size_t best_len = 0;
    std::uint32_t best_value = 0;
    std::size_t found = 0;
    std::size_t pos = offset(trie_[0]);
    for (std::size_t i = 0; i < s.size(); ++i) {
        const auto c = static_cast<unsigned char>(s[i]);
        pos ^= c;
        if (pos >= trie_.size()) break;
        const std::uint32_t unit = trie_[pos];
        if ((unit & ((1u << 31) | 0xffu)) != c) break;
        pos ^= offset(unit);
        if ((unit >> 8) & 1u) {
            if (pos >= trie_.size()) break;
            if (found < kMaxTrieResults) {
                best_len = i + 1;
                best_value = trie_[pos] & 0x7fffffffu;
            }
            ++found;
        }
    }
    if (best_len == 0 || best_value >= replacements_.size()) {
        return {{}, 0};
    }
    std::string_view rep = replacements_.substr(best_value);
    rep = rep.substr(0, rep.find('\0'));
    return {rep, best_len};
}

std::string UgmTokenizer::normalize(std::string_view text) const {
    // SentencePiece Normalizer::Normalize with escape_whitespaces.
    auto next = [&](std::string_view in)
        -> std::pair<std::string_view, std::size_t> {
        auto [rep, used] = charsmap_prefix(in);
        if (used != 0) return {rep, used};
        if (const std::size_t n = valid_utf8_len(in); n != 0) {
            return {in.substr(0, n), n};
        }
        return {"\xef\xbf\xbd", 1};  // U+FFFD for one malformed byte
    };
    std::string out;
    if (cfg_.remove_extra_whitespaces) {
        while (!text.empty()) {
            const auto [rep, used] = next(text);
            if (rep != " ") break;
            text.remove_prefix(used);
        }
    }
    if (text.empty()) return out;
    out.reserve(text.size() * 3 / 2 + kSpace.size());
    if (cfg_.add_space_prefix) out += kSpace;
    bool prev_space = cfg_.remove_extra_whitespaces;
    while (!text.empty()) {
        auto [rep, used] = next(text);
        if (prev_space) {
            while (!rep.empty() && rep.front() == ' ') rep.remove_prefix(1);
        }
        if (!rep.empty()) {
            for (char c : rep) {
                if (c == ' ') {
                    out += kSpace;
                } else {
                    out += c;
                }
            }
            prev_space = rep.back() == ' ';
        }
        text.remove_prefix(used);
        if (!cfg_.remove_extra_whitespaces) prev_space = false;
    }
    if (cfg_.remove_extra_whitespaces) {
        while (out.size() >= kSpace.size() &&
               std::string_view(out).substr(out.size() - kSpace.size()) ==
                   kSpace) {
            out.resize(out.size() - kSpace.size());
        }
    }
    return out;
}

void UgmTokenizer::encode_chunk(std::string_view text,
                                std::vector<TokenId>& out) const {
    const std::string norm = normalize(text);
    const std::size_t size = norm.size();
    if (size == 0) return;

    // Model::EncodeOptimized: best[i] is the best path over norm[0, i).
    struct Node {
        TokenId id = -1;
        float score = 0;
        std::ptrdiff_t start = -1;
    };
    std::vector<Node> best(size + 1);
    const float unk_score = min_score_ - kUnkPenalty;
    const std::string_view sv(norm);
    std::size_t start = 0;
    while (start < size) {
        const float here = best[start].score;
        const std::size_t mblen = std::min(
            one_char_len(static_cast<unsigned char>(norm[start])),
            size - start);
        bool has_single = false;
        const std::size_t max_end =
            std::min(size, start + max_piece_len_);
        for (std::size_t end = start + 1; end <= max_end; ++end) {
            const auto it = pieces_.find(sv.substr(start, end - start));
            if (it == pieces_.end()) continue;
            Node& t = best[end];
            const float cand =
                cfg_.scores[static_cast<std::size_t>(it->second)] + here;
            if (t.start == -1 || cand > t.score) {
                t = {it->second, cand, static_cast<std::ptrdiff_t>(start)};
            }
            if (end - start == mblen) has_single = true;
        }
        if (!has_single) {
            Node& t = best[start + mblen];
            const float cand = unk_score + here;
            if (t.start == -1 || cand > t.score) {
                t = {cfg_.unk, cand, static_cast<std::ptrdiff_t>(start)};
            }
        }
        start += mblen;
    }

    std::vector<TokenId> ids;
    for (std::size_t end = size; end > 0;) {
        const Node& node = best[end];
        ids.push_back(node.id);
        end = static_cast<std::size_t>(node.start);
    }
    // A run of unknown pieces becomes one unknown token.
    for (auto it = ids.rbegin(); it != ids.rend(); ++it) {
        if (*it == cfg_.unk && !out.empty() && out.back() == cfg_.unk) {
            continue;
        }
        out.push_back(*it);
    }
}

std::vector<TokenId> UgmTokenizer::encode(std::string_view text,
                                          bool add_special) const {
    std::vector<TokenId> out;
    // Split at user-defined tokens; strip whitespace on both sides of
    // each, as HF's lstrip/rstrip added tokens do.
    std::string_view rest = text;
    while (!rest.empty()) {
        std::size_t at = std::string_view::npos;
        const std::pair<std::string, TokenId>* hit = nullptr;
        for (const auto& ud : user_defined_) {
            const std::size_t p = rest.find(ud.first);
            if (p < at) {
                at = p;
                hit = &ud;
            }
        }
        if (hit == nullptr) {
            encode_chunk(rest, out);
            break;
        }
        encode_chunk(trim_right(rest.substr(0, at)), out);
        out.push_back(hit->second);
        rest = trim_left(rest.substr(at + hit->first.size()));
    }
    if (add_special && cfg_.add_eos) {
        out.push_back(cfg_.eos);
    }
    return out;
}

std::string UgmTokenizer::decode(std::span<const TokenId> ids) const {
    std::string out;
    for (TokenId id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= cfg_.tokens.size()) {
            throw std::out_of_range("token id out of range");
        }
        const auto type =
            static_cast<TokenType>(cfg_.types[static_cast<std::size_t>(id)]);
        if (type == TokenType::kControl || type == TokenType::kUnused) {
            continue;
        }
        const std::string_view p = cfg_.tokens[static_cast<std::size_t>(id)];
        for (std::size_t i = 0; i < p.size();) {
            if (p.substr(i, kSpace.size()) == kSpace) {
                out += ' ';
                i += kSpace.size();
            } else {
                out += p[i++];
            }
        }
    }
    if (cfg_.add_space_prefix && !out.empty() && out.front() == ' ') {
        out.erase(0, 1);
    }
    return out;
}

std::string_view UgmTokenizer::piece(TokenId id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= cfg_.tokens.size()) {
        throw std::out_of_range("token id out of range");
    }
    return cfg_.tokens[static_cast<std::size_t>(id)];
}

}  // namespace locus::tok
