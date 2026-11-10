#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "locus/tok/tokenizer.hpp"

namespace locus::tok {

/**
 * WordPiece tokenizer of BERT-family models (`tokenizer.ggml.model` ==
 * "bert"), as llama.cpp's WPM tokenizer encodes; the GGUF converter
 * stores a word-start piece as "▁piece" and a continuation ("##piece")
 * as "piece".
 *
 * 1. Normalize: an invalid UTF-8 byte becomes U+FFFD; with
 *    strip_accents, each codepoint maps to the first codepoint of its
 *    canonical decomposition (cpt_nfd) and accent marks are dropped;
 *    U+0000, U+FFFD and control characters are dropped; with
 *    lowercase, each codepoint is lowercased.
 * 2. Split into words at whitespace; a punctuation character, an ASCII
 *    symbol or a CJK ideograph is a word of its own.
 * 3. Each word, with "▁" prepended, is split by greedy longest match
 *    from the left; if some position has no piece, the whole word is
 *    the unknown token.
 * 4. With add_special, CLS comes first and SEP last.
 *
 * Text that spells a special token ("[SEP]") is not parsed as one:
 * user text cannot inject control tokens. llama.cpp parses it by
 * default (llama-tokenize --no-parse-special matches this class).
 */
class WpmTokenizer final : public Tokenizer {
  public:
    /** Vocabulary and special-token configuration. */
    struct Config {
        std::vector<std::string> tokens;
        /** TokenType per token; same length as tokens. */
        std::vector<std::int32_t> types;
        TokenId cls = 101;
        TokenId sep = 102;
        TokenId unk = 100;
        TokenId pad = 0;
        bool lowercase = true;
        bool strip_accents = true;
    };

    /** @throws std::invalid_argument when types and tokens differ in
     *      size or a special id is out of range. */
    explicit WpmTokenizer(Config config);

    /** @throws gguf::Error when `tokenizer.ggml.model` is not "bert"
     *      or the tokens are missing. */
    static WpmTokenizer from_gguf(const gguf::GgufFile& g);

    /**
     * @param text UTF-8 input.
     * @param add_special Add CLS first and SEP last.
     */
    std::vector<TokenId> encode(std::string_view text,
                                bool add_special) const override;

    /** @returns The pieces joined, "▁" as a space, without the leading
     *      space; control tokens render as nothing.
     *  @throws std::out_of_range on an id outside the vocabulary. */
    std::string decode(std::span<const TokenId> ids) const override;

    std::string_view piece(TokenId id) const override;
    std::size_t vocab_size() const override { return cfg_.tokens.size(); }
    /** The CLS token, which llama.cpp calls BOS. */
    TokenId bos_id() const override { return cfg_.cls; }
    /** BERT has no EOS: -1. */
    TokenId eos_id() const override { return -1; }
    TokenId sep_id() const { return cfg_.sep; }
    TokenId pad_id() const { return cfg_.pad; }

    /** @returns The words of steps 1 and 2 (for tests). */
    std::vector<std::string> words(std::string_view text) const;

  private:
    /** Lets token_to_id_ be searched by string_view without a copy. */
    struct ViewHash {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const {
            return std::hash<std::string_view>{}(s);
        }
    };

    Config cfg_;
    std::unordered_map<std::string, TokenId, ViewHash, std::equal_to<>>
        token_to_id_;
    std::size_t max_token_len_ = 0;
};

}  // namespace locus::tok
