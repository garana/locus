#include "locus/tok/bpe_tokenizer.hpp"

#include "locus/tok/unicode.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace locus::tok {

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

/** An inclusive codepoint range of a character class. */
struct CpRange {
    std::uint32_t lo, hi;
};

template <std::size_t N>
bool in_class(const std::array<CpRange, N>& cls, std::uint32_t cp) {
    for (const auto& r : cls) {
        if (cp >= r.lo && cp <= r.hi) return true;
    }
    return false;
}

// The deepseek-llm pre-tokenizer's classes, copied from the Janus-Pro
// tokenizer.json regexes (i#226).
constexpr std::array<CpRange, 87> kDsLetters = {{
    {0x41, 0x5A}, {0x61, 0x7A}, {0xB5, 0xB5}, {0xC0, 0xD6},
    {0xD8, 0xF6}, {0xF8, 0x1BA}, {0x1BC, 0x1BF}, {0x1C4, 0x293},
    {0x295, 0x2AF}, {0x370, 0x373}, {0x376, 0x376}, {0x377, 0x377},
    {0x37B, 0x37D}, {0x37F, 0x37F}, {0x386, 0x386}, {0x388, 0x38A},
    {0x38C, 0x38C}, {0x38E, 0x3A1}, {0x3A3, 0x3F5}, {0x3F7, 0x481},
    {0x48A, 0x52F}, {0x531, 0x556}, {0x10A0, 0x10C5}, {0x13A0, 0x13F5},
    {0x13F8, 0x13FD}, {0x1C90, 0x1CBA}, {0x1CBD, 0x1CBF},
    {0x1D00, 0x1D2B}, {0x1D6B, 0x1D77}, {0x1D79, 0x1D9A},
    {0x1E00, 0x1F15}, {0x1F18, 0x1F1D}, {0x1F20, 0x1F45},
    {0x1F48, 0x1F4D}, {0x1F50, 0x1F57}, {0x1F59, 0x1F59},
    {0x1F5B, 0x1F5B}, {0x1F5D, 0x1F5D}, {0x1F5F, 0x1F7D},
    {0x1F80, 0x1FB4}, {0x1FB6, 0x1FBC}, {0x1FBE, 0x1FBE},
    {0x1FC2, 0x1FC4}, {0x1FC6, 0x1FCC}, {0x1FD0, 0x1FD3},
    {0x1FD6, 0x1FDB}, {0x1FE0, 0x1FEC}, {0x1FF2, 0x1FF4},
    {0x1FF6, 0x1FFC}, {0x2102, 0x2102}, {0x2107, 0x2107},
    {0x210A, 0x2113}, {0x2115, 0x2115}, {0x2119, 0x211D},
    {0x2124, 0x2124}, {0x2126, 0x2126}, {0x2128, 0x2128},
    {0x212A, 0x212D}, {0x212F, 0x2134}, {0x2139, 0x2139},
    {0x213C, 0x213F}, {0x2145, 0x2149}, {0x214E, 0x214E},
    {0x2183, 0x2183}, {0x2184, 0x2184}, {0x2C00, 0x2C7B},
    {0x2C7E, 0x2CE4}, {0x2CEB, 0x2CEE}, {0x2CF2, 0x2CF2},
    {0x2CF3, 0x2CF3}, {0xA640, 0xA66D}, {0xA680, 0xA69B},
    {0xA722, 0xA76F}, {0xA771, 0xA787}, {0xA78B, 0xA78E},
    {0xAB70, 0xABBF}, {0xFB00, 0xFB06}, {0xFB13, 0xFB17},
    {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A}, {0x10400, 0x1044F},
    {0x104B0, 0x104D3}, {0x104D8, 0x104FB}, {0x10C80, 0x10CB2},
    {0x10CC0, 0x10CF2}, {0x118A0, 0x118DF}, {0x1E900, 0x1E943},
}};
constexpr std::array<CpRange, 6> kDsPunct = {{
    {0x21, 0x2F}, {0x3A, 0x7E}, {0xFF01, 0xFF0F}, {0xFF1A, 0xFF5E},
    {0x2018, 0x201F}, {0x3000, 0x3002},
}};
constexpr std::array<CpRange, 3> kDsCjk = {{
    {0x4E00, 0x9FA5}, {0x800, 0x4E00}, {0xAC00, 0xD7FF},
}};

/**
 * The deepseek-llm pre-tokenizer (Janus-Pro / DeepSeek-LLM), as its
 * tokenizer.json defines it: a sequence of splits, each applied to
 * every piece the previous one produced, matches isolated as their own
 * pieces and the gaps between them kept:
 *   [\r\n]  |  \s?[letters]+  |  \s?[punct]+  |  \s+$  |  [cjk]+
 * then every numeric character on its own. ByteLevel follows with no
 * further split. `cps` / `offs` are the text's codepoints and byte
 * offsets (offs has one extra end entry).
 */
std::vector<Chunk> pretokenize_deepseek_llm(
    const std::vector<std::uint32_t>& cps,
    const std::vector<std::size_t>& offs) {
    struct Piece {
        std::size_t b, e;  // codepoint indices
    };
    std::vector<Piece> pieces;
    if (!cps.empty()) pieces.push_back({0, cps.size()});
    // One split stage: match(b, e, i) returns the end of a match that
    // starts at i (inside [b, e)), or i for no match there.
    auto stage = [&](auto match) {
        std::vector<Piece> next;
        for (const Piece& p : pieces) {
            std::size_t gap = p.b;
            for (std::size_t i = p.b; i < p.e;) {
                const std::size_t j = match(p.b, p.e, i);
                if (j > i) {
                    if (gap < i) next.push_back({gap, i});
                    next.push_back({i, j});
                    gap = i = j;
                } else {
                    ++i;
                }
            }
            if (gap < p.e) next.push_back({gap, p.e});
        }
        pieces.swap(next);
    };
    // `\s?[cls]+`: a whitespace char joins only when a class run follows
    // it; otherwise the run starts at i itself.
    auto ws_run = [&](const auto& cls) {
        return [&cps, &cls](std::size_t, std::size_t e, std::size_t i) {
            std::size_t k = i;
            if (is_white(cps[i]) && i + 1 < e && in_class(cls, cps[i + 1])) {
                k = i + 1;
            } else if (!in_class(cls, cps[i])) {
                return i;
            }
            while (k < e && in_class(cls, cps[k])) ++k;
            return k;
        };
    };
    stage([&](std::size_t, std::size_t, std::size_t i) {
        return cps[i] == '\r' || cps[i] == '\n' ? i + 1 : i;
    });
    stage(ws_run(kDsLetters));
    stage(ws_run(kDsPunct));
    stage([&](std::size_t, std::size_t e, std::size_t i) {  // \s+$
        for (std::size_t k = i; k < e; ++k) {
            if (!is_white(cps[k])) return i;
        }
        return e;
    });
    stage([&](std::size_t, std::size_t e, std::size_t i) {  // [cjk]+
        std::size_t k = i;
        while (k < e && in_class(kDsCjk, cps[k])) ++k;
        return k;
    });
    stage([&](std::size_t, std::size_t, std::size_t i) {  // digits
        return is_numeric(cps[i]) ? i + 1 : i;
    });
    std::vector<Chunk> out;
    out.reserve(pieces.size());
    for (const Piece& p : pieces) out.push_back({offs[p.b], offs[p.e]});
    return out;
}

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
    if (pre == "deepseek-llm") {
        return pretokenize_deepseek_llm(cps, offs);
    }
    const std::size_t n = cps.size();
    // llama3 caps digit runs at 3 and allows one leading
    // non-letter before a letter run; gpt-2 style does not. qwen2
    // (Qwen2/Qwen3) is llama3's split with single digits (`\p{N}`
    // instead of `\p{N}{1,3}`).
    const bool qwen2 = pre == "qwen2";
    // gpt-4o (o200k; Muse Spark) also splits case runs, as llama.cpp
    // does, with letters that are not a-z as "upper" and letters that
    // are not A-Z as "lower" (so non-ASCII is both):
    //   [^\r\n\p{L}\p{N}]?U*L+C? | [^\r\n\p{L}\p{N}]?U+L*C?
    // C is a case-insensitive contraction suffix (there is no
    // standalone one), and punctuation runs also take trailing '/'.
    const bool gpt4o = pre == "gpt-4o";
    const bool llama3 = pre == "llama-bpe" || pre == "llama3" ||
                        pre == "deepseek-v3" || pre == "joyai-llm" ||
                        pre == "deepseek-r1-qwen" || qwen2 || gpt4o;
    const std::size_t digit_cap = qwen2 ? 1 : 3;
    auto is_upper = [&](std::size_t k) {
        return is_letter(cps[k]) && !(cps[k] >= 'a' && cps[k] <= 'z');
    };
    auto is_lower = [&](std::size_t k) {
        return is_letter(cps[k]) && !(cps[k] >= 'A' && cps[k] <= 'Z');
    };

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
        if (gpt4o && (is_letter(cps[i]) ||
                      (cps[i] != '\r' && cps[i] != '\n' &&
                       !is_digit(cps[i]) && i + 1 < n &&
                       is_letter(cps[i + 1])))) {
            if (!is_letter(cps[i])) {
                ++i;  // the optional prefix char
            }
            // U*L+: the longest upper run, backed off to the last
            // position that can start the lower run; else U+L*.
            std::size_t j = i;
            while (j < n && is_upper(j)) {
                ++j;
            }
            std::size_t k = j;
            while (k > i && !(k < n && is_lower(k))) {
                --k;
            }
            if (k < n && is_lower(k)) {
                while (k < n && is_lower(k)) {
                    ++k;
                }
                i = k;
            } else {
                i = j;  // U+ with no lower letter after it
            }
            if (i < n) {
                i += is_contraction(i);
            }
        } else if (std::size_t len = gpt4o ? 0 : is_contraction(i)) {
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
                   (!llama3 || run < digit_cap)) {
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
                       (cps[i] == '\r' || cps[i] == '\n' ||
                        (gpt4o && cps[i] == '/'))) {
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
        // Skip EMPTY special pieces: an empty needle makes the
        // encode split loop (text.find("", at) == at) match at the
        // current position without advancing -> infinite loop +
        // unbounded output. A hostile GGUF can declare an empty
        // kControl/kUserDefined token, so guard here.
        if ((t == TokenType::kControl ||
             t == TokenType::kUserDefined) &&
            !cfg_.tokens[i].empty()) {
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
        if (v.type != gguf::ValueType::kString) {
            throw gguf::Error("token entry is not a string");
        }
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
        if (v.type != gguf::ValueType::kString) {
            throw gguf::Error("merge entry is not a string");
        }
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
    // When the key is absent, llama.cpp defaults add_bos by
    // pretokenizer family: the llama3 family prepends BOS, the
    // rest of the BPE world (deepseek, gpt2, ...) does not.
    const bool bos_default =
        cfg.pre == "llama-bpe" || cfg.pre == "llama3";
    cfg.add_bos = g.get_bool("tokenizer.ggml.add_bos_token")
                      .value_or(bos_default);
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
        // Advance by at least one byte so a zero-width special can
        // never stall the loop (defence-in-depth; empty specials are
        // already filtered at construction).
        const std::size_t adv = specials_[sp_idx].first.size();
        at = sp_pos + (adv > 0 ? adv : 1);
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

}  // namespace locus::tok
