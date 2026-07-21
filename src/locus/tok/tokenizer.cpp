#include "locus/tok/tokenizer.hpp"

#include <cstdio>
#include <limits>
#include <stdexcept>

#include "locus/tok/bpe_tokenizer.hpp"

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

    // Split into UTF-8 codepoint symbols (offset, length pairs).
    struct Sym {
        std::size_t off;
        std::size_t len;
    };
    std::vector<Sym> syms;
    for (std::size_t i = 0; i < norm.size();) {
        std::size_t len = utf8_len(static_cast<unsigned char>(norm[i]));
        if (len > norm.size() - i) {
            len = norm.size() - i;  // truncated trailing sequence
        }
        syms.push_back({i, len});
        i += len;
    }

    // Greedy SPM: repeatedly merge the adjacent pair whose joined
    // piece has the highest score (leftmost wins ties), as in
    // llama.cpp. O(n^2) is fine at prompt sizes; optimize later if
    // profiling says so.
    auto lookup = [&](const Sym& a, const Sym& b) {
        auto it = token_to_id_.find(std::string_view(
            norm.data() + a.off, a.len + b.len));
        return it == token_to_id_.end() ? -1 : it->second;
    };
    while (syms.size() > 1) {
        float best_score = std::numeric_limits<float>::lowest();
        std::size_t best = syms.size();
        for (std::size_t i = 0; i + 1 < syms.size(); ++i) {
            TokenId id = lookup(syms[i], syms[i + 1]);
            if (id >= 0 &&
                cfg_.scores[static_cast<std::size_t>(id)] > best_score) {
                best_score = cfg_.scores[static_cast<std::size_t>(id)];
                best = i;
            }
        }
        if (best == syms.size()) {
            break;
        }
        syms[best].len += syms[best + 1].len;
        syms.erase(syms.begin() +
                   static_cast<std::ptrdiff_t>(best + 1));
    }

    for (const Sym& s : syms) {
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
                // Piece format is "<0xNN>".
                if (p.size() == 6 && p.substr(0, 3) == "<0x") {
                    out += static_cast<char>(
                        std::stoi(std::string(p.substr(3, 2)),
                                  nullptr, 16));
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
