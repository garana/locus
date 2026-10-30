#include "locus/tok/clip_tokenizer.hpp"

#include <array>
#include <cctype>
#include <cstdlib>
#include <limits>

#include "locus/tok/unicode.hpp"

namespace locus::tok {

namespace {

constexpr std::string_view kStart = "<|startoftext|>";
constexpr std::string_view kEnd = "<|endoftext|>";

/**
 * GPT-2 byte -> codepoint alphabet (CLIP's bytes_to_unicode): the
 * printable Latin-1 bytes map to themselves, the rest to 256 + n in
 * ascending byte order.
 */
const std::array<std::uint32_t, 256>& byte_cps() {
    static const auto a = [] {
        std::array<std::uint32_t, 256> t{};
        std::uint32_t next = 256;
        for (int b = 0; b < 256; ++b) {
            const bool printable = (b >= '!' && b <= '~') ||
                                   (b >= 0xa1 && b <= 0xac) ||
                                   (b >= 0xae && b <= 0xff);
            t[static_cast<std::size_t>(b)] =
                printable ? static_cast<std::uint32_t>(b) : next++;
        }
        return t;
    }();
    return a;
}

/** Python's `\s` in str patterns: White_Space plus U+001C..U+001F. */
bool is_space(std::uint32_t cp) {
    return is_white(cp) || (cp >= 0x1c && cp <= 0x1f);
}

/**
 * Python's str.lower() for ASCII, Latin-1, Latin Extended-A, Greek and
 * Cyrillic; other codepoints are unchanged. U+0130 lowers to "i" plus
 * U+0307, so it returns two codepoints through `second`.
 */
std::uint32_t lower(std::uint32_t cp, std::uint32_t& second) {
    second = 0;
    if (cp < 0x80) {
        return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
    }
    if (cp >= 0xc0 && cp <= 0xde && cp != 0xd7) return cp + 0x20;
    if (cp == 0x130) {
        second = 0x307;
        return 'i';
    }
    if ((cp >= 0x100 && cp <= 0x137) || (cp >= 0x14a && cp <= 0x177)) {
        return cp % 2 == 0 ? cp + 1 : cp;
    }
    if ((cp >= 0x139 && cp <= 0x148) || (cp >= 0x179 && cp <= 0x17e)) {
        return cp % 2 == 1 ? cp + 1 : cp;
    }
    if (cp == 0x178) return 0xff;
    if (cp >= 0x391 && cp <= 0x3a9 && cp != 0x3a2) return cp + 0x20;
    if (cp == 0x386) return 0x3ac;
    if (cp >= 0x388 && cp <= 0x38a) return cp + 0x25;
    if (cp == 0x38c) return 0x3cc;
    if (cp == 0x38e || cp == 0x38f) return cp + 0x3f;
    if (cp >= 0x410 && cp <= 0x42f) return cp + 0x20;
    if (cp >= 0x400 && cp <= 0x40f) return cp + 0x50;
    return cp;
}

/**
 * `\p{L}` approximately: exact for ASCII and Latin-1; above that, a
 * letter unless a space, a number, a combining mark or in a
 * punctuation or symbol block.
 */
bool is_letter(std::uint32_t cp) {
    if (cp < 0x80) {
        return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    }
    if (cp < 0x100) {
        return cp == 0xaa || cp == 0xb5 || cp == 0xba ||
               (cp >= 0xc0 && cp != 0xd7 && cp != 0xf7);
    }
    if (is_white(cp) || is_numeric(cp)) return false;
    struct Range {
        std::uint32_t lo, hi;
    };
    // Combining marks, general punctuation, currency, letterlike and
    // number forms, arrows to miscellaneous symbols and arrows,
    // supplemental punctuation, CJK symbols, variation selectors, CJK
    // compatibility forms, fullwidth ASCII punctuation, emoji and
    // pictographs.
    static constexpr Range kNotLetters[] = {
        {0x300, 0x36f},     {0x2000, 0x206f},   {0x20a0, 0x20ff},
        {0x2100, 0x218f},   {0x2190, 0x2bff},   {0x2e00, 0x2e7f},
        {0x3000, 0x303f},   {0xfe00, 0xfe0f},   {0xfe30, 0xfe4f},
        {0xff01, 0xff0f},   {0xff1a, 0xff20},   {0xff3b, 0xff40},
        {0xff5b, 0xff65},   {0x1f000, 0x1faff},
    };
    for (const auto& r : kNotLetters) {
        if (cp >= r.lo && cp <= r.hi) return false;
    }
    return true;
}

/**
 * The deterministic part of ftfy.fix_text (the reference cleans each
 * prompt with it), in its order: HTML entities (only when the text has
 * no "<"), Latin ligatures, fullwidth forms, curly quotes, terminal
 * escapes, control characters. Not done: mojibake repair
 * (fix_encoding), C1 and surrogate fixes, NFC; line breaks need no
 * fix since every one becomes a space next.
 */
std::vector<std::uint32_t> ftfy_subset(std::string_view text) {
    std::vector<std::uint32_t> in;
    for (std::size_t i = 0; i < text.size();) in.push_back(next_cp(text, i));
    const bool html = text.find('<') == std::string_view::npos;
    std::vector<std::uint32_t> out;
    out.reserve(in.size());
    const auto push = [&](std::string_view ascii) {
        for (const char c : ascii) out.push_back(static_cast<unsigned char>(c));
    };
    for (std::size_t i = 0; i < in.size(); ++i) {
        std::uint32_t cp = in[i];
        if (html && cp == '&') {
            // &name; or &#NN; or &#xHH;, at most 24 characters inside.
            std::size_t e = i + 1;
            while (e < in.size() && e - i <= 25 && in[e] != ';' &&
                   in[e] < 0x80 &&
                   (std::isalnum(static_cast<int>(in[e])) || in[e] == '#')) {
                ++e;
            }
            if (e < in.size() && in[e] == ';' && e > i + 1) {
                std::string name;
                for (std::size_t k = i + 1; k < e; ++k) {
                    name += static_cast<char>(in[k]);
                }
                std::uint32_t v = 0;
                if (name == "amp") v = '&';
                else if (name == "lt") v = '<';
                else if (name == "gt") v = '>';
                else if (name == "quot") v = '"';
                else if (name == "apos") v = '\'';
                else if (name == "nbsp") v = 0xa0;
                else if (name.size() > 1 && name[0] == '#') {
                    const bool hex = name[1] == 'x' || name[1] == 'X';
                    const std::string digits = name.substr(hex ? 2 : 1);
                    if (!digits.empty() && digits.size() <= 7) {
                        char* end = nullptr;
                        const unsigned long n =
                            std::strtoul(digits.c_str(), &end, hex ? 16 : 10);
                        if (*end == '\0' && n > 0 && n <= 0x10ffff) {
                            v = static_cast<std::uint32_t>(n);
                        }
                    }
                }
                if (v != 0) {
                    in[e] = v;  // re-read as the decoded character
                    i = e - 1;
                    continue;
                }
            }
        }
        // Latin ligatures.
        switch (cp) {
            case 0x132: push("IJ"); continue;
            case 0x133: push("ij"); continue;
            case 0x149: out.push_back(0x2bc); out.push_back('n'); continue;
            case 0x1f1: push("DZ"); continue;
            case 0x1f2: push("Dz"); continue;
            case 0x1f3: push("dz"); continue;
            case 0x1c4: push("D"); out.push_back(0x17d); continue;
            case 0x1c5: push("D"); out.push_back(0x17e); continue;
            case 0x1c6: push("d"); out.push_back(0x17e); continue;
            case 0x1c7: push("LJ"); continue;
            case 0x1c8: push("Lj"); continue;
            case 0x1c9: push("lj"); continue;
            case 0x1ca: push("NJ"); continue;
            case 0x1cb: push("Nj"); continue;
            case 0x1cc: push("nj"); continue;
            case 0xfb00: push("ff"); continue;
            case 0xfb01: push("fi"); continue;
            case 0xfb02: push("fl"); continue;
            case 0xfb03: push("ffi"); continue;
            case 0xfb04: push("ffl"); continue;
            case 0xfb05: out.push_back(0x17f); push("t"); continue;
            case 0xfb06: push("st"); continue;
            default: break;
        }
        // Fullwidth forms (the ASCII block and the fullwidth signs; the
        // halfwidth Katakana and Hangul are left as they are).
        if (cp == 0x3000) cp = ' ';
        else if (cp >= 0xff01 && cp <= 0xff5e) cp -= 0xfee0;
        else if (cp == 0xffe3) {
            // NFKC of the fullwidth macron: a space and a combining one.
            out.push_back(' ');
            cp = 0x304;
        } else if (cp >= 0xffe0 && cp <= 0xffe6) {
            static constexpr std::uint32_t kSigns[] = {0xa2, 0xa3, 0xac, 0,
                                                       0xa6, 0xa5, 0x20a9};
            cp = kSigns[cp - 0xffe0];
        }
        // Curly quotes.
        if (cp == 0x2bc || (cp >= 0x2018 && cp <= 0x201b)) cp = '\'';
        else if (cp >= 0x201c && cp <= 0x201f) cp = '"';
        out.push_back(cp);
    }
    // Terminal escapes: ESC [ (digits | ;)* letter.
    std::vector<std::uint32_t> clean;
    clean.reserve(out.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i] == 0x1b && i + 1 < out.size() && out[i + 1] == '[') {
            std::size_t e = i + 2;
            while (e < out.size() &&
                   ((out[e] >= '0' && out[e] <= '9') || out[e] == ';')) {
                ++e;
            }
            if (e < out.size() && out[e] < 0x80 &&
                std::isalpha(static_cast<int>(out[e]))) {
                i = e;
                continue;
            }
        }
        const std::uint32_t c = out[i];
        const bool control = c <= 0x08 || c == 0x0b ||
                             (c >= 0x0e && c <= 0x1f) || c == 0x7f ||
                             (c >= 0x206a && c <= 0x206f) || c == 0xfeff ||
                             (c >= 0xfff9 && c <= 0xfffc);
        if (!control) clean.push_back(c);
    }
    return clean;
}

}  // namespace

ClipTokenizer ClipTokenizer::from_gguf(const gguf::GgufFile& g) {
    if (g.get_string("tokenizer.ggml.model") != "clip") {
        throw gguf::Error("tokenizer model is not \"clip\"");
    }
    const auto* tokens = g.get_array("tokenizer.ggml.tokens");
    const auto* merges = g.get_array("tokenizer.ggml.merges");
    if (tokens == nullptr || tokens->empty() || merges == nullptr) {
        throw gguf::Error("clip: tokenizer tokens or merges missing");
    }
    if (tokens->size() >
        static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
        throw gguf::Error("clip: too many tokens");
    }
    ClipTokenizer t;
    t.vocab_.reserve(tokens->size());
    for (std::size_t i = 0; i < tokens->size(); ++i) {
        const auto& v = (*tokens)[i];
        if (v.type != gguf::ValueType::kString) {
            throw gguf::Error("clip: token entry is not a string");
        }
        t.vocab_.emplace(v.s, static_cast<TokenId>(i));
    }
    t.ranks_.reserve(merges->size());
    for (std::size_t i = 0; i < merges->size(); ++i) {
        const auto& v = (*merges)[i];
        if (v.type != gguf::ValueType::kString) {
            throw gguf::Error("clip: merge entry is not a string");
        }
        // The first rank wins, like CLIPTokenizer's dict built in order.
        t.ranks_.emplace(v.s, static_cast<std::uint32_t>(i));
    }
    const auto id = [&](std::string_view key) {
        const auto v = g.get_uint(key);
        if (!v || *v >= tokens->size()) {
            throw gguf::Error("clip: missing or bad " + std::string(key));
        }
        return static_cast<TokenId>(*v);
    };
    t.bos_ = id("tokenizer.ggml.bos_token_id");
    t.eos_ = id("tokenizer.ggml.eos_token_id");
    t.pad_ = id("tokenizer.ggml.padding_token_id");
    t.unk_ = id("tokenizer.ggml.unknown_token_id");
    return t;
}

void ClipTokenizer::bpe(const std::string& word,
                        std::vector<TokenId>& out) const {
    // Symbols are the word's codepoints; the last carries "</w>".
    std::vector<std::string> sym;
    for (std::size_t i = 0; i < word.size();) {
        const std::size_t b = i;
        next_cp(word, i);
        sym.emplace_back(word.substr(b, i - b));
    }
    if (sym.empty()) return;
    sym.back() += "</w>";
    std::string key;
    while (sym.size() > 1) {
        std::uint32_t best = std::numeric_limits<std::uint32_t>::max();
        std::size_t at = 0;
        for (std::size_t k = 0; k + 1 < sym.size(); ++k) {
            key.assign(sym[k]).append(" ").append(sym[k + 1]);
            const auto it = ranks_.find(key);
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                at = k;
            }
        }
        if (best == std::numeric_limits<std::uint32_t>::max()) break;
        // Merge every non-overlapping occurrence of that pair, left to
        // right, as the reference does.
        const std::string first = sym[at], second = sym[at + 1];
        std::vector<std::string> next;
        next.reserve(sym.size());
        for (std::size_t k = 0; k < sym.size();) {
            if (k + 1 < sym.size() && sym[k] == first &&
                sym[k + 1] == second) {
                next.push_back(first + second);
                k += 2;
            } else {
                next.push_back(std::move(sym[k]));
                ++k;
            }
        }
        sym = std::move(next);
    }
    for (const auto& s : sym) {
        const auto it = vocab_.find(s);
        out.push_back(it != vocab_.end() ? it->second : unk_);
    }
}

std::vector<TokenId> ClipTokenizer::tokenize(std::string_view text) const {
    // The special tokens split the text first, wherever they are, as the
    // reference's added-token split does; each piece is cleaned alone.
    std::vector<TokenId> out;
    while (!text.empty()) {
        std::size_t at = std::string_view::npos;
        std::string_view special;
        for (const std::string_view s : {kStart, kEnd}) {
            const std::size_t p = text.find(s);
            if (p < at) {
                at = p;
                special = s;
            }
        }
        tokenize_piece(text.substr(0, at), out);
        if (at == std::string_view::npos) break;
        const auto it = vocab_.find(std::string(special));
        out.push_back(it != vocab_.end() ? it->second : unk_);
        text.remove_prefix(at + special.size());
    }
    return out;
}

void ClipTokenizer::tokenize_piece(std::string_view text,
                                   std::vector<TokenId>& out) const {
    // Clean: whitespace runs to one space, ends trimmed; then lower.
    const std::vector<std::uint32_t> fixed = ftfy_subset(text);
    std::vector<std::uint32_t> cps;
    bool pending_space = false;
    for (std::size_t i = 0; i < fixed.size(); ++i) {
        const std::uint32_t cp = fixed[i];
        if (is_space(cp)) {
            pending_space = !cps.empty();
            continue;
        }
        if (pending_space) {
            cps.push_back(' ');
            pending_space = false;
        }
        // Python lowers a capital sigma that ends a word to the final
        // form: after a letter and not before one.
        if (cp == 0x3a3) {
            const bool after = i > 0 && is_letter(fixed[i - 1]);
            const bool before = i + 1 < fixed.size() && is_letter(fixed[i + 1]);
            cps.push_back(after && !before ? 0x3c2 : 0x3c3);
            continue;
        }
        std::uint32_t second = 0;
        cps.push_back(lower(cp, second));
        if (second != 0) cps.push_back(second);
    }

    const auto& alpha = byte_cps();
    const auto emit = [&](std::size_t b, std::size_t e) {
        std::string utf8, mapped;
        for (std::size_t k = b; k < e; ++k) append_utf8(utf8, cps[k]);
        for (const unsigned char c : utf8) append_utf8(mapped, alpha[c]);
        bpe(mapped, out);
    };
    const std::size_t n = cps.size();
    for (std::size_t i = 0; i < n;) {
        if (cps[i] == ' ') {
            ++i;
            continue;
        }
        // The regex's alternatives, in its order (the special tokens are
        // already split out).
        if (cps[i] == '\'' && i + 1 < n) {
            const std::uint32_t c1 = cps[i + 1];
            const std::uint32_t c2 = i + 2 < n ? cps[i + 2] : 0;
            std::size_t len = 0;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                len = 2;
            } else if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
                       (c1 == 'l' && c2 == 'l')) {
                len = 3;
            }
            if (len != 0) {
                emit(i, i + len);
                i += len;
                continue;
            }
        }
        std::size_t e = i + 1;
        if (is_letter(cps[i])) {
            while (e < n && is_letter(cps[e])) ++e;
        } else if (!is_numeric(cps[i])) {
            while (e < n && !is_space(cps[e]) && !is_letter(cps[e]) &&
                   !is_numeric(cps[e])) {
                ++e;
            }
        }
        emit(i, e);
        i = e;
    }
}

std::vector<TokenId> ClipTokenizer::encode(std::string_view text,
                                           std::uint32_t length) const {
    std::vector<TokenId> ids = tokenize(text);
    const std::size_t room = length >= 2 ? length - 2 : 0;
    if (ids.size() > room) ids.resize(room);
    ids.insert(ids.begin(), bos_);
    ids.push_back(eos_);
    ids.resize(std::max<std::size_t>(ids.size(), length), pad_);
    return ids;
}

}  // namespace locus::tok
