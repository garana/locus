#include "locus/tok/tokenizer.hpp"

#include <charconv>
#include <cstdio>
#include <limits>
#include <queue>
#include <stdexcept>

#include "locus/tok/bpe_tokenizer.hpp"
#include "locus/tok/ugm_tokenizer.hpp"
#include "locus/tok/wpm_tokenizer.hpp"

namespace locus::tok {

std::unique_ptr<Tokenizer> tokenizer_from_gguf(
    const gguf::GgufFile& g) {
    const auto model = g.get_string("tokenizer.ggml.model");
    if (model == "llama") {
        return std::make_unique<SpmTokenizer>(
            SpmTokenizer::from_gguf(g));
    }
    if (model == "gpt2") {
        return std::make_unique<BpeTokenizer>(
            BpeTokenizer::from_gguf(g));
    }
    if (model == "t5") {
        return std::make_unique<UgmTokenizer>(
            UgmTokenizer::from_gguf(g));
    }
    if (model == "bert") {
        return std::make_unique<WpmTokenizer>(
            WpmTokenizer::from_gguf(g));
    }
    throw gguf::Error("unsupported tokenizer.ggml.model");
}

namespace {

/** SPM "meta" space marker, U+2581 LOWER ONE EIGHTH BLOCK. */
constexpr std::string_view kSpace = "\xe2\x96\x81";

/** @returns Length in bytes of the UTF-8 sequence starting at c. */
std::size_t utf8_len(unsigned char c) {
    if ((c & 0x80) == 0x00) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;  // invalid lead byte: treat as a lone byte
}

std::string normalize(std::string_view text, bool add_space_prefix) {
    std::string out;
    out.reserve(text.size() + kSpace.size());
    if (add_space_prefix && !text.empty()) {
        out += kSpace;
    }
    for (char c : text) {
        if (c == ' ') {
            out += kSpace;
        } else {
            out += c;
        }
    }
    return out;
}

}  // namespace

SpmTokenizer::SpmTokenizer(Config config) : cfg_(std::move(config)) {
    const std::size_t n = cfg_.tokens.size();
    if (n == 0) {
        throw std::invalid_argument("empty vocabulary");
    }
    if (cfg_.scores.empty()) {
        cfg_.scores.resize(n, 0.0f);
    }
    if (cfg_.types.empty()) {
        cfg_.types.resize(
            n, static_cast<std::int32_t>(TokenType::kNormal));
    }
    if (cfg_.scores.size() != n || cfg_.types.size() != n) {
        throw std::invalid_argument("vocab array size mismatch");
    }
    for (TokenId id : {cfg_.bos, cfg_.eos, cfg_.unk}) {
        if (id < 0 || static_cast<std::size_t>(id) >= n) {
            throw std::invalid_argument("special token id out of range");
        }
    }
    token_to_id_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        token_to_id_.emplace(cfg_.tokens[i],
                             static_cast<TokenId>(i));
    }
}

SpmTokenizer SpmTokenizer::from_gguf(const gguf::GgufFile& g) {
    auto model = g.get_string("tokenizer.ggml.model");
    if (model != "llama") {
        throw gguf::Error("tokenizer model is not SPM (\"llama\")");
    }
    const auto* tokens = g.get_array("tokenizer.ggml.tokens");
    if (tokens == nullptr || tokens->empty()) {
        throw gguf::Error("tokenizer.ggml.tokens missing or empty");
    }

    Config cfg;
    cfg.tokens.reserve(tokens->size());
    for (const auto& v : *tokens) {
        if (v.type != gguf::ValueType::kString) {
            throw gguf::Error("token entry is not a string");
        }
        cfg.tokens.push_back(v.s);
    }
    if (const auto* scores = g.get_array("tokenizer.ggml.scores")) {
        for (const auto& v : *scores) {
            cfg.scores.push_back(static_cast<float>(v.f));
        }
    }
    if (const auto* types = g.get_array("tokenizer.ggml.token_type")) {
        for (const auto& v : *types) {
            cfg.types.push_back(static_cast<std::int32_t>(v.i));
        }
    }

    auto id_of = [&](std::string_view key, TokenId fallback) {
        auto v = g.get_uint(key);
        return v ? static_cast<TokenId>(*v) : fallback;
    };
    cfg.bos = id_of("tokenizer.ggml.bos_token_id", 1);
    cfg.eos = id_of("tokenizer.ggml.eos_token_id", 2);
    cfg.unk = id_of("tokenizer.ggml.unknown_token_id", 0);
    cfg.add_space_prefix =
        g.get_bool("tokenizer.ggml.add_space_prefix").value_or(true);
    return SpmTokenizer(std::move(cfg));
}

std::vector<TokenId> SpmTokenizer::encode(std::string_view text,
                                          bool add_bos) const {
    std::vector<TokenId> out;
    if (add_bos) {
        out.push_back(cfg_.bos);
    }
    const std::string norm = normalize(text, cfg_.add_space_prefix);
    if (norm.empty()) {
        return out;
    }

    // Split into UTF-8 codepoint symbols (offset, length pairs), linked
    // so a merge is O(1). A merged-away symbol keeps len 0.
    struct Sym {
        std::size_t off;
        std::size_t len;
        std::ptrdiff_t prev;
        std::ptrdiff_t next;
    };
    std::vector<Sym> syms;
    for (std::size_t i = 0; i < norm.size();) {
        std::size_t len = utf8_len(static_cast<unsigned char>(norm[i]));
        if (len > norm.size() - i) {
            len = norm.size() - i;  // truncated trailing sequence
        }
        const auto k = static_cast<std::ptrdiff_t>(syms.size());
        syms.push_back({i, len, k - 1, k + 1});
        i += len;
    }
    syms.back().next = -1;

    // Greedy SPM: repeatedly merge the adjacent pair whose joined
    // piece has the highest score (leftmost wins ties), as in
    // llama.cpp. A heap of candidate pairs makes it O(n log n) (i#625
    // F5: the rescan after every merge was O(n^2) on one long word). A
    // pair is stale once either symbol changed; their lengths only
    // grow, so a sum that no longer matches finds it.
    struct Pair {
        float score;
        std::size_t off;  // the left symbol's: leftmost first on a tie
        std::ptrdiff_t left;
        std::size_t len;
    };
    const auto lower = [](const Pair& a, const Pair& b) {
        return a.score < b.score || (a.score == b.score && a.off > b.off);
    };
    std::priority_queue<Pair, std::vector<Pair>, decltype(lower)> heap(
        lower);
    const auto push = [&](std::ptrdiff_t l) {
        if (l < 0 || syms[static_cast<std::size_t>(l)].next < 0) {
            return;
        }
        const Sym& a = syms[static_cast<std::size_t>(l)];
        const Sym& b = syms[static_cast<std::size_t>(a.next)];
        const auto it = token_to_id_.find(
            std::string_view(norm.data() + a.off, a.len + b.len));
        if (it == token_to_id_.end()) {
            return;
        }
        const float s = cfg_.scores[static_cast<std::size_t>(it->second)];
        // The rescan this replaces never picked a score at or below
        // lowest() (or a NaN): skip them, which also keeps the heap's
        // order strict.
        if (s > std::numeric_limits<float>::lowest()) {
            heap.push({s, a.off, l, a.len + b.len});
        }
    };
    for (std::size_t i = 0; i + 1 < syms.size(); ++i) {
        push(static_cast<std::ptrdiff_t>(i));
    }
    while (!heap.empty()) {
        const Pair p = heap.top();
        heap.pop();
        Sym& a = syms[static_cast<std::size_t>(p.left)];
        if (a.len == 0 || a.next < 0) {
            continue;
        }
        Sym& b = syms[static_cast<std::size_t>(a.next)];
        if (a.len + b.len != p.len) {
            continue;  // stale
        }
        a.len += b.len;
        b.len = 0;
        a.next = b.next;
        if (b.next >= 0) {
            syms[static_cast<std::size_t>(b.next)].prev = p.left;
        }
        push(a.prev);
        push(p.left);
    }

    for (std::ptrdiff_t k = 0; k >= 0;
         k = syms[static_cast<std::size_t>(k)].next) {
        const Sym& s = syms[static_cast<std::size_t>(k)];
        std::string_view piece(norm.data() + s.off, s.len);
        if (auto it = token_to_id_.find(piece);
            it != token_to_id_.end()) {
            out.push_back(it->second);
            continue;
        }
        // Byte fallback: emit <0xNN> tokens; unk if absent.
        for (std::size_t k = 0; k < s.len; ++k) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>",
                          static_cast<unsigned char>(piece[k]));
            auto it = token_to_id_.find(std::string_view(buf, 6));
            out.push_back(it == token_to_id_.end() ? cfg_.unk
                                                   : it->second);
        }
    }
    return out;
}

std::string SpmTokenizer::decode(std::span<const TokenId> ids) const {
    std::string out;
    for (TokenId id : ids) {
        std::string_view p = piece(id);  // bounds-checks id
        switch (type_of(id)) {
            case TokenType::kControl:
            case TokenType::kUnused:
                break;  // renders as nothing
            case TokenType::kByte: {
                // Piece format is "<0xNN>". The piece is raw vocab
                // from an untrusted GGUF, so parse without throwing:
                // a malformed byte piece (non-hex NN) is emitted
                // verbatim rather than aborting decode (which runs
                // in the SSE lambdas outside any try/catch).
                unsigned byte = 0;
                if (p.size() == 6 && p.rfind("<0x", 0) == 0 &&
                    p[5] == '>' &&
                    std::from_chars(p.data() + 3, p.data() + 5,
                                    byte, 16)
                            .ec == std::errc()) {
                    out += static_cast<char>(byte);
                } else {
                    out += p;
                }
                break;
            }
            default: {
                for (std::size_t i = 0; i < p.size();) {
                    if (p.substr(i, kSpace.size()) == kSpace) {
                        out += ' ';
                        i += kSpace.size();
                    } else {
                        out += p[i++];
                    }
                }
                break;
            }
        }
    }
    return out;
}

std::string_view SpmTokenizer::piece(TokenId id) const {
    if (id < 0 ||
        static_cast<std::size_t>(id) >= cfg_.tokens.size()) {
        throw std::out_of_range("token id out of range");
    }
    return cfg_.tokens[static_cast<std::size_t>(id)];
}

}  // namespace locus::tok
