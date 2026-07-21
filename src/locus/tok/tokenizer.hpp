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

/** Interface every tokenizer family implements. */
class Tokenizer {
  public:
    virtual ~Tokenizer() = default;

    /** @returns Token ids; prepends BOS when add_bos. */
    virtual std::vector<TokenId> encode(std::string_view text,
                                        bool add_bos) const = 0;

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
 * `tokenizer.ggml.model` ("llama" -> SPM, "gpt2" -> BPE).
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
     * @param text UTF-8 input.
     * @param add_bos Prepend the BOS token.
     * @returns Token ids; empty input yields just BOS (if asked).
     */
    std::vector<TokenId> encode(std::string_view text,
                                bool add_bos) const override;

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

    Config cfg_;
    std::unordered_map<std::string_view, TokenId> token_to_id_;
};

}  // namespace locus::tok
