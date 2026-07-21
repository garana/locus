#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::tok {

/**
 * Byte-level BPE tokenizer (GPT-2 family), as used by Llama-3,
 * DeepSeek, Qwen, and most non-SPM GGUF models
 * (`tokenizer.ggml.model` == "gpt2").
 *
 * Pipeline: special-token split -> pretokenize (a hand-rolled
 * scanner equivalent to the model's split regex) -> GPT-2
 * byte-to-unicode mapping -> lowest-rank-first pair merging over
 * `tokenizer.ggml.merges` -> vocab lookup.
 *
 * Unicode caveat: ASCII classification is exact; non-ASCII
 * codepoints outside the common space/punctuation ranges are
 * treated as letters. Exactness is golden-tested against
 * llama.cpp for ASCII-dominant text (see tests).
 */
class BpeTokenizer final : public Tokenizer {
  public:
    /** Vocabulary, merge rules, and special-token config. */
    struct Config {
        std::vector<std::string> tokens;
        /** TokenType per token; same length as tokens. */
        std::vector<std::int32_t> types;
        /** Merge rules "left right", rank = index. */
        std::vector<std::string> merges;
        /** Pretokenizer id from tokenizer.ggml.pre. */
        std::string pre = "gpt-2";
        TokenId bos = 0;
        TokenId eos = 0;
        bool add_bos = true;
    };

    /** @throws std::invalid_argument on inconsistent config. */
    explicit BpeTokenizer(Config config);

    /** @throws gguf::Error if required metadata is missing. */
    static BpeTokenizer from_gguf(const gguf::GgufFile& g);

    std::vector<TokenId> encode(std::string_view text,
                                bool add_bos) const override;
    std::string decode(std::span<const TokenId> ids) const override;
    std::string_view piece(TokenId id) const override;

    std::size_t vocab_size() const override {
        return cfg_.tokens.size();
    }
    TokenId bos_id() const override { return cfg_.bos; }
    TokenId eos_id() const override { return cfg_.eos; }

  private:
    void bpe_word(std::string_view word,
                  std::vector<TokenId>& out) const;

    TokenType type_of(TokenId id) const {
        return static_cast<TokenType>(
            cfg_.types[static_cast<std::size_t>(id)]);
    }

    Config cfg_;
    std::unordered_map<std::string_view, TokenId> token_to_id_;
    /** "left\x01right" -> rank. */
    std::unordered_map<std::string, std::uint32_t> merge_rank_;
    /** Special (control/user-defined) pieces, longest first. */
    std::vector<std::pair<std::string_view, TokenId>> specials_;
};

}  // namespace locus::tok
