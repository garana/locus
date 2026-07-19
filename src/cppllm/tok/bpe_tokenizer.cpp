#include "cppllm/tok/bpe_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace cppllm::tok {

namespace {

/**
 * GPT-2 byte<->codepoint alphabet: printable Latin-1 bytes map to
 * themselves, the rest to 256+n in ascending byte order.
 */
struct ByteAlphabet {
    std::array<std::uint32_t, 256> to_cp;
    std::unordered_map<std::uint32_t, std::uint8_t> to_byte;

    ByteAlphabet() {
        auto printable = [](int b) {
            return (b >= '!' && b <= '~') ||
                   (b >= 0xa1 && b <= 0xac) ||
                   (b >= 0xae && b <= 0xff);
        };
        std::uint32_t next = 256;
        for (int b = 0; b < 256; ++b) {
            to_cp[static_cast<std::size_t>(b)] =
                printable(b) ? static_cast<std::uint32_t>(b)
                             : next++;
            to_byte[to_cp[static_cast<std::size_t>(b)]] =
                static_cast<std::uint8_t>(b);
        }
    }
};

const ByteAlphabet& alphabet() {
    static const ByteAlphabet a;
    return a;
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xc0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    } else {
        out += static_cast<char>(0xe0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    }
}

/** Decodes one UTF-8 codepoint at s[i]; advances i. */
std::uint32_t next_cp(std::string_view s, std::size_t& i) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    std::size_t len = 1;
    std::uint32_t cp = b0;
    if ((b0 & 0xe0) == 0xc0) {
        len = 2;
        cp = b0 & 0x1f;
    } else if ((b0 & 0xf0) == 0xe0) {
        len = 3;
        cp = b0 & 0x0f;
    } else if ((b0 & 0xf8) == 0xf0) {
        len = 4;
        cp = b0 & 0x07;
    }
    if (i + len > s.size()) {
        len = 1;
        cp = b0;
    } else {
        for (std::size_t k = 1; k < len; ++k) {
            cp = (cp << 6) |
                 (static_cast<unsigned char>(s[i + k]) & 0x3f);
        }
    }
    i += len;
    return cp;
}

bool is_ws(std::uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == 0x0b || cp == 0x85 || cp == 0xa0 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202f || cp == 0x205f ||
           cp == 0x3000;
}

bool is_digit(std::uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

/**
 * Letter test: exact for ASCII; non-ASCII defaults to "letter"
 * unless it is a known space (see header caveat).
 */
bool is_letter(std::uint32_t cp) {
    if (cp < 0x80) {
        return (cp >= 'a' && cp <= 'z') ||
               (cp >= 'A' && cp <= 'Z');
    }
    return !is_ws(cp);
}

/** One pretokenized chunk: [begin, end) byte range. */
struct Chunk {
    std::size_t begin, end;
};

/**
 * Hand-rolled equivalent of the llama3/gpt2 split regexes over
 * decoded codepoints. `cps`/`offs` hold the codepoints of the
 * text and their byte offsets (offs has one extra end entry).
 */
std::vector<Chunk> pretokenize(std::string_view text,
                               const std::string& pre) {
    std::vector<std::uint32_t> cps;
    std::vector<std::size_t> offs;
    for (std::size_t i = 0; i < text.size();) {
        offs.push_back(i);
        cps.push_back(next_cp(text, i));
    }
    offs.push_back(text.size());
    const std::size_t n = cps.size();
    // llama3 caps digit runs at 3 and allows one leading
    // non-letter before a letter run; gpt-2 style does not.
    const bool llama3 = pre == "llama-bpe" || pre == "llama3" ||
                        pre == "deepseek-v3" ||
                        pre == "deepseek-r1-qwen";

    std::vector<Chunk> out;
    std::size_t i = 0;
    auto is_contraction = [&](std::size_t at) -> std::size_t {
        if (cps[at] != '\'' || at + 1 >= n) {
            return 0;
        }
        auto low = [&](std::size_t k) {
            std::uint32_t c = cps[k];
            return c >= 'A' && c <= 'Z' ? c + 32 : c;
        };
        const std::uint32_t c1 = low(at + 1);
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
            return 2;
        }
        if (at + 2 < n) {
            const std::uint32_t c2 = low(at + 2);
            if ((c1 == 'r' && c2 == 'e') ||
                (c1 == 'v' && c2 == 'e') ||
                (c1 == 'l' && c2 == 'l')) {
                return 3;
            }
        }
        return 0;
    };

    while (i < n) {
        const std::size_t start = i;
        if (std::size_t len = is_contraction(i)) {
            i += len;
        } else if (is_letter(cps[i]) ||
                   // llama3: `[^\r\n\p{L}\p{N}]?\p{L}+` -- any
                   // single non-CRLF non-letter non-digit char
                   // (including a space or tab) prefixes a run.
                   (llama3 && cps[i] != '\r' && cps[i] != '\n' &&
                    !is_digit(cps[i]) && i + 1 < n &&
                    is_letter(cps[i + 1])) ||
                   (!llama3 && cps[i] == ' ' && i + 1 < n &&
                    is_letter(cps[i + 1]))) {
            ++i;  // the optional prefix char
            while (i < n && is_letter(cps[i])) {
                ++i;
            }
        } else if (is_digit(cps[i]) ||
                   (!llama3 && cps[i] == ' ' && i + 1 < n &&
                    is_digit(cps[i + 1]))) {
            if (cps[i] == ' ') {
                ++i;
            }
            std::size_t run = 0;
            while (i < n && is_digit(cps[i]) &&
                   (!llama3 || run < 3)) {
                ++i;
                ++run;
            }
        } else if (!is_ws(cps[i]) ||
                   (cps[i] == ' ' && i + 1 < n &&
                    !is_ws(cps[i + 1]) &&
                    !is_letter(cps[i + 1]) &&
                    !is_digit(cps[i + 1]))) {
            // ` ?[punct]+` then (llama3) trailing newlines.
            if (cps[i] == ' ') {
                ++i;
            }
            while (i < n && !is_ws(cps[i]) &&
                   !is_letter(cps[i]) && !is_digit(cps[i])) {
                ++i;
            }
            if (llama3) {
                while (i < n &&
                       (cps[i] == '\r' || cps[i] == '\n')) {
                    ++i;
                }
            }
        } else {
            // Whitespace: `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`.
            std::size_t j = i;
            while (j < n && is_ws(cps[j])) {
                ++j;
            }
            std::size_t last_nl = i;
            bool has_nl = false;
            for (std::size_t k = i; k < j; ++k) {
                if (cps[k] == '\r' || cps[k] == '\n') {
                    last_nl = k;
                    has_nl = true;
                }
            }
            if (has_nl && last_nl + 1 == j) {
                i = j;  // run ends in newlines: one chunk
            } else if (j < n && j - i > 1) {
                i = j - 1;  // keep one space for the next word
            } else {
                i = j;
            }
        }
        if (i == start) {
            ++i;  // safety: never stall
        }
        out.push_back({offs[start], offs[i]});
    }
    return out;
}

}  // namespace

BpeTokenizer::BpeTokenizer(Config config) : cfg_(std::move(config)) {
    const std::size_t n = cfg_.tokens.size();
    if (n == 0) {
        throw std::invalid_argument("empty vocabulary");
    }
    if (cfg_.types.empty()) {
        cfg_.types.resize(
            n, static_cast<std::int32_t>(TokenType::kNormal));
    }
    if (cfg_.types.size() != n) {
        throw std::invalid_argument("vocab array size mismatch");
    }
    for (TokenId id : {cfg_.bos, cfg_.eos}) {
        if (id < 0 || static_cast<std::size_t>(id) >= n) {
            throw std::invalid_argument("special id out of range");
        }
    }
    token_to_id_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        token_to_id_.emplace(cfg_.tokens[i],
                             static_cast<TokenId>(i));
        const auto t = static_cast<TokenType>(cfg_.types[i]);
        if (t == TokenType::kControl ||
            t == TokenType::kUserDefined) {
            specials_.emplace_back(cfg_.tokens[i],
                                   static_cast<TokenId>(i));
        }
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const auto& a, const auto& b) {
                  return a.first.size() > b.first.size();
              });
    merge_rank_.reserve(cfg_.merges.size());
    for (std::size_t r = 0; r < cfg_.merges.size(); ++r) {
        const std::string& m = cfg_.merges[r];
        const std::size_t sp = m.find(' ');
        if (sp == std::string::npos) {
            throw std::invalid_argument("malformed merge rule");
        }
        std::string key = m.substr(0, sp) + '\x01' +
                          m.substr(sp + 1);
        merge_rank_.emplace(std::move(key),
                            static_cast<std::uint32_t>(r));
    }
}

BpeTokenizer BpeTokenizer::from_gguf(const gguf::GgufFile& g) {
    if (g.get_string("tokenizer.ggml.model") != "gpt2") {
        throw gguf::Error("tokenizer model is not BPE (\"gpt2\")");
    }
    const auto* tokens = g.get_array("tokenizer.ggml.tokens");
    const auto* merges = g.get_array("tokenizer.ggml.merges");
    if (tokens == nullptr || tokens->empty()) {
        throw gguf::Error("tokenizer.ggml.tokens missing");
    }
    if (merges == nullptr) {
        throw gguf::Error("tokenizer.ggml.merges missing");
    }

    Config cfg;
    cfg.tokens.reserve(tokens->size());
    for (const auto& v : *tokens) {
        cfg.tokens.push_back(v.s);
    }
    if (const auto* types =
            g.get_array("tokenizer.ggml.token_type")) {
        for (const auto& v : *types) {
            cfg.types.push_back(static_cast<std::int32_t>(v.i));
        }
    }
    cfg.merges.reserve(merges->size());
    for (const auto& v : *merges) {
        cfg.merges.push_back(v.s);
    }
    if (auto pre = g.get_string("tokenizer.ggml.pre")) {
        cfg.pre = std::string(*pre);
    }
    auto id_of = [&](std::string_view key, TokenId fallback) {
        auto v = g.get_uint(key);
        return v ? static_cast<TokenId>(*v) : fallback;
    };
    cfg.bos = id_of("tokenizer.ggml.bos_token_id", 0);
    cfg.eos = id_of("tokenizer.ggml.eos_token_id", 0);
    cfg.add_bos =
        g.get_bool("tokenizer.ggml.add_bos_token").value_or(true);
    return BpeTokenizer(std::move(cfg));
}

void BpeTokenizer::bpe_word(std::string_view word,
                            std::vector<TokenId>& out) const {
    // Bytes -> alphabet codepoints, one symbol per byte.
    std::vector<std::string> syms;
    syms.reserve(word.size());
    for (char c : word) {
        std::string s;
        append_utf8(s, alphabet().to_cp[static_cast<std::size_t>(
                           static_cast<unsigned char>(c))]);
        syms.push_back(std::move(s));
    }
    // Lowest-rank-first pair merging.
    while (syms.size() > 1) {
        std::uint32_t best_rank =
            std::numeric_limits<std::uint32_t>::max();
        std::size_t best = syms.size();
        for (std::size_t i = 0; i + 1 < syms.size(); ++i) {
            auto it = merge_rank_.find(syms[i] + '\x01' +
                                       syms[i + 1]);
            if (it != merge_rank_.end() &&
                it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best == syms.size()) {
            break;
        }
        syms[best] += syms[best + 1];
        syms.erase(syms.begin() +
                   static_cast<std::ptrdiff_t>(best + 1));
    }
    for (const std::string& s : syms) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        }
        // Single alphabet chars are always in a well-formed
        // byte-level vocab; silently skip pathological misses.
    }
}

std::vector<TokenId> BpeTokenizer::encode(std::string_view text,
                                          bool add_bos) const {
    std::vector<TokenId> out;
    if (add_bos && cfg_.add_bos) {
        out.push_back(cfg_.bos);
    }
    // Split on special tokens (longest first), BPE in between.
    std::size_t at = 0;
    while (at < text.size()) {
        std::size_t sp_pos = std::string_view::npos;
        std::size_t sp_idx = 0;
        for (std::size_t k = 0; k < specials_.size(); ++k) {
            const std::size_t p =
                text.find(specials_[k].first, at);
            if (p < sp_pos) {
                sp_pos = p;
                sp_idx = k;
            }
        }
        const std::size_t seg_end =
            sp_pos == std::string_view::npos ? text.size()
                                             : sp_pos;
        const std::string_view seg =
            text.substr(at, seg_end - at);
        for (const auto& [b, e] : pretokenize(seg, cfg_.pre)) {
            bpe_word(seg.substr(b, e - b), out);
        }
        if (sp_pos == std::string_view::npos) {
            break;
        }
        out.push_back(specials_[sp_idx].second);
        at = sp_pos + specials_[sp_idx].first.size();
    }
    return out;
}

std::string BpeTokenizer::decode(
    std::span<const TokenId> ids) const {
    std::string out;
    for (TokenId id : ids) {
        const std::string_view p = piece(id);
        switch (type_of(id)) {
            case TokenType::kControl:
            case TokenType::kUnused:
                break;
            default: {
                for (std::size_t i = 0; i < p.size();) {
                    const std::uint32_t cp = next_cp(p, i);
                    auto it = alphabet().to_byte.find(cp);
                    if (it != alphabet().to_byte.end()) {
                        out += static_cast<char>(it->second);
                    }
                }
                break;
            }
        }
    }
    return out;
}

std::string_view BpeTokenizer::piece(TokenId id) const {
    if (id < 0 ||
        static_cast<std::size_t>(id) >= cfg_.tokens.size()) {
        throw std::out_of_range("token id out of range");
    }
    return cfg_.tokens[static_cast<std::size_t>(id)];
}

}  // namespace cppllm::tok
