#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "locus/tok/tokenizer.hpp"

namespace locus::tok {

/**
 * SentencePiece unigram tokenizer (`tokenizer.ggml.model` == "t5"),
 * as transformers' T5Tokenizer (legacy) encodes:
 *
 * 1. The text is split at user-defined tokens (T5's `<extra_id_N>`),
 *    which become their own ids; the whitespace before one is dropped
 *    (they are lstrip tokens).
 * 2. Each other chunk is normalized as SentencePiece's
 *    `Normalizer::Normalize` does: the precompiled charsmap (NFKC for
 *    T5) maps the longest matching prefix to its replacement, an
 *    invalid UTF-8 byte becomes U+FFFD; leading and repeated spaces
 *    are dropped and trailing ones removed (remove_extra_whitespaces);
 *    spaces become U+2581; a U+2581 is prepended (add_space_prefix).
 * 3. The normalized chunk is split by the unigram Viterbi search
 *    (`Model::EncodeOptimized`): the best sum of piece scores over
 *    the normal pieces, one unknown token per character no piece
 *    covers (scored min_score - 10), and runs of unknowns merged.
 *    Recent sentencepiece also rebases the running score once it
 *    passes 1e5 in magnitude; this class does not, so float rounding
 *    could pick a different near-tie only after thousands of pieces.
 * 4. With add_special, EOS is appended (T5 has no BOS).
 */
class UgmTokenizer final : public Tokenizer {
  public:
    /** Vocabulary and normalizer configuration. */
    struct Config {
        std::vector<std::string> tokens;
        std::vector<float> scores;
        std::vector<std::int32_t> types;
        /** The SentencePiece precompiled charsmap (may be empty). */
        std::vector<std::uint8_t> charsmap;
        TokenId eos = 1;
        TokenId unk = 2;
        TokenId pad = 0;
        bool add_space_prefix = true;
        bool remove_extra_whitespaces = true;
        bool add_eos = true;
    };

    /** @throws std::invalid_argument on an inconsistent config or a
     *      malformed charsmap. */
    explicit UgmTokenizer(Config config);

    /** @throws gguf::Error when `tokenizer.ggml.model` is not "t5" or
     *      the tokenizer metadata is missing. */
    static UgmTokenizer from_gguf(const gguf::GgufFile& g);

    /**
     * @param text UTF-8 input.
     * @param add_special Append EOS (when the model asks for it). T5
     *     has no BOS, so the flag means "add the special tokens".
     */
    std::vector<TokenId> encode(std::string_view text,
                                bool add_special) const override;

    /** @returns The pieces joined, U+2581 as a space, without the
     *      leading space; control and unused tokens render as nothing.
     *  @throws std::out_of_range on an id outside the vocabulary. */
    std::string decode(std::span<const TokenId> ids) const override;

    std::string_view piece(TokenId id) const override;
    std::size_t vocab_size() const override { return cfg_.tokens.size(); }
    /** T5 has no BOS: -1. */
    TokenId bos_id() const override { return -1; }
    TokenId eos_id() const override { return cfg_.eos; }
    TokenId pad_id() const { return cfg_.pad; }

    /** @returns `text` normalized as step 2 describes (for tests). */
    std::string normalize(std::string_view text) const;

  private:
    /** Steps 2 and 3 for one chunk. */
    void encode_chunk(std::string_view text, std::vector<TokenId>& out) const;
    /** The charsmap's longest rule for a prefix of `s`:
     *  {replacement, bytes consumed}, consumed 0 when none. */
    std::pair<std::string_view, std::size_t> charsmap_prefix(
        std::string_view s) const;

    /** Lets pieces_ be searched by string_view without a copy. */
    struct ViewHash {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const {
            return std::hash<std::string_view>{}(s);
        }
    };

    Config cfg_;
    std::vector<std::uint32_t> trie_;  ///< darts-clone double array
    std::string_view replacements_;    ///< NUL-separated strings
    /** Normal pieces only: the ones the Viterbi search may pick. */
    std::unordered_map<std::string, TokenId, ViewHash, std::equal_to<>>
        pieces_;
    std::vector<std::pair<std::string, TokenId>> user_defined_;
    /** True for a byte that starts some user-defined token. */
    std::array<bool, 256> ud_first_{};
    std::size_t max_piece_len_ = 0;
    float min_score_ = 0;
};

}  // namespace locus::tok
