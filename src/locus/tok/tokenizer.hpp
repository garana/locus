#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "locus/gguf/gguf.hpp"

namespace locus::tok {

/** Token id type used across the runtime. */
using TokenId = std::int32_t;

/** Token roles, matching GGUF `tokenizer.ggml.token_type` values. */
enum class TokenType : std::int32_t {
    kUndefined = 0,
    kNormal = 1,
    kUnknown = 2,
    kControl = 3,
    kUserDefined = 4,
    kUnused = 5,
    kByte = 6,
};

/** A byte range [begin, end) of a text to encode. */
struct TextRange {
    std::size_t begin = 0;
    std::size_t end = 0;
};

/** Interface every tokenizer family implements. */
class Tokenizer {
  public:
    virtual ~Tokenizer() = default;

    /** @returns Token ids; prepends BOS when add_bos. */
    virtual std::vector<TokenId> encode(std::string_view text,
                                        bool add_bos) const = 0;

    /**
     * As encode(), but a control token's text that overlaps one of
     * `literal`'s ranges is encoded as plain text, so a chat message
     * cannot open a role (i#1128). User-defined tokens still match,
     * unless `user_defined` is set. Tokenizers without a special-token
     * pass encode as encode().
     *
     * @param literal Ranges of `text` that came from a client.
     * @param user_defined Also keep user-defined tokens' text as text
     *     (Orpheus' speech markers and audio codes are user-defined).
     * @returns Token ids; prepends BOS when add_bos.
     */
    virtual std::vector<TokenId> encode_literal(
        std::string_view text, bool add_bos,
        std::span<const TextRange> literal,
        bool user_defined = false) const {
        (void)literal;
        (void)user_defined;
        return encode(text, add_bos);
    }

    /** @returns UTF-8 text; control tokens render as nothing. */
    virtual std::string decode(
        std::span<const TokenId> ids) const = 0;

    /** @returns The raw vocab string for a token id. */
    virtual std::string_view piece(TokenId id) const = 0;

    virtual std::size_t vocab_size() const = 0;
    virtual TokenId bos_id() const = 0;
    virtual TokenId eos_id() const = 0;
};

/**
 * Builds the right tokenizer for a model file, dispatching on
 * `tokenizer.ggml.model` ("llama" -> SPM, "gpt2" -> BPE, "t5" ->
 * unigram, "bert" -> WordPiece).
 *
 * @throws gguf::Error on unknown tokenizer families.
 */
std::unique_ptr<Tokenizer> tokenizer_from_gguf(
    const gguf::GgufFile& g);

/**
 * SentencePiece-style (SPM) tokenizer for Llama-family models.
 *
 * Encoding uses greedy highest-score bigram merging over UTF-8
 * symbols, matching llama.cpp's SPM behavior: spaces become U+2581
 * (the "meta" underscore), unknown symbols fall back to `<0xNN>`
 * byte tokens, and remaining misses map to the unk token.
 */
class SpmTokenizer final : public Tokenizer {
  public:
    /** Vocabulary and special-token configuration. */
    struct Config {
        std::vector<std::string> tokens;
        /** Merge priority per token; same length as tokens. */
        std::vector<float> scores;
        /** TokenType per token; same length as tokens. */
        std::vector<std::int32_t> types;
        TokenId bos = 1;
        TokenId eos = 2;
        TokenId unk = 0;
        /** Prefix input with a space before encoding. */
        bool add_space_prefix = true;
    };

    /**
     * @param config Vocabulary; tokens/scores/types must have equal
     *     sizes and the special ids must be in range.
     * @throws std::invalid_argument on inconsistent config.
     */
    explicit SpmTokenizer(Config config);

    /**
     * Builds a tokenizer from GGUF metadata.
     *
     * @param g A parsed model whose `tokenizer.ggml.model` is
     *     "llama" (SPM).
     * @throws gguf::Error if required metadata is missing/invalid.
     */
    static SpmTokenizer from_gguf(const gguf::GgufFile& g);

    /**
     * As llama.cpp with parse_special: the text of a control, unknown
     * or user-defined token encodes as that token (i#1150), and each
     * run of text between them takes the SPM space prefix when it
     * follows one (or starts the input).
     *
     * @param text UTF-8 input.
     * @param add_bos Prepend the BOS token.
     * @returns Token ids; empty input yields just BOS (if asked).
     */
    std::vector<TokenId> encode(std::string_view text,
                                bool add_bos) const override;

    /** As encode(), but a control or unknown token's text that overlaps
     *  a `literal` range (a chat message) encodes as text; so does a
     *  user-defined one's when `user_defined` is set. */
    std::vector<TokenId> encode_literal(
        std::string_view text, bool add_bos,
        std::span<const TextRange> literal,
        bool user_defined = false) const override;

    /**
     * @param ids Any sequence of in-range token ids.
     * @returns UTF-8 text; control tokens render as nothing, byte
     *     tokens as their byte. May carry the SPM leading space.
     * @throws std::out_of_range on an id outside the vocabulary.
     */
    std::string decode(std::span<const TokenId> ids) const override;

    /** @returns The piece (raw vocab string) for a token id. */
    std::string_view piece(TokenId id) const override;

    std::size_t vocab_size() const override {
        return cfg_.tokens.size();
    }
    TokenId bos_id() const override { return cfg_.bos; }
    TokenId eos_id() const override { return cfg_.eos; }
    TokenId unk_id() const { return cfg_.unk; }

  private:
    TokenType type_of(TokenId id) const {
        return static_cast<TokenType>(cfg_.types[
            static_cast<std::size_t>(id)]);
    }

    /** SPM-encodes plain text (no special tokens), appending to out;
     *  `space_prefix` adds the leading space. */
    void encode_text(std::string_view text, bool space_prefix,
                     std::vector<TokenId>& out) const;

    Config cfg_;
    std::unordered_map<std::string_view, TokenId> token_to_id_;
    /** Control, unknown and user-defined tokens with text, longest
     *  first: the special-token pass matches these. */
    std::vector<std::pair<std::string, TokenId>> specials_;
};

}  // namespace locus::tok
