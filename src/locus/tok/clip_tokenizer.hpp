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
 * The CLIP BPE tokenizer (i#372), as transformers' CLIPTokenizer with
 * padding="max_length" and truncation: FLUX's pooled-prompt tokenizer.
 *
 * The text goes through the deterministic part of ftfy.fix_text (HTML
 * entities, Latin ligatures, fullwidth forms, curly quotes, terminal
 * escapes, control characters), is cleaned (whitespace runs become one
 * space, the ends are trimmed), lowercased, and split like
 * CLIPTokenizer's regex: the two
 * special tokens, the contractions 's 't 're 've 'm 'll 'd, letter
 * runs, single digits, and runs of anything else that is not a space.
 * Each piece maps its UTF-8 bytes to the GPT-2 byte alphabet, gets
 * "</w>" on its last symbol, and is merged by merge rank.
 *
 * Differences from the reference, all for non-ASCII text: ftfy's
 * mojibake repair and its NFC normalization are not done (valid,
 * composed text is unchanged by both), and the Unicode classes are
 * approximate: lowercasing covers Latin-1, Latin Extended-A, Greek and
 * Cyrillic, and a non-ASCII codepoint is a letter unless it is a
 * space, a number, a combining mark or in a punctuation or symbol
 * block.
 */
class ClipTokenizer {
  public:
    /**
     * @param g A GGUF with tokenizer.ggml.model "clip", its tokens,
     *     merges and the bos / eos / pad ids.
     * @throws gguf::Error on missing or malformed tokenizer keys.
     */
    static ClipTokenizer from_gguf(const gguf::GgufFile& g);

    /**
     * @param text The prompt.
     * @returns The BPE ids of the text, without bos / eos.
     */
    std::vector<TokenId> tokenize(std::string_view text) const;

    /**
     * @param text The prompt.
     * @param length The padded length (77, the model's context).
     * @returns bos, the first length - 2 ids of tokenize(text), eos,
     *     then pad ids up to length.
     */
    std::vector<TokenId> encode(std::string_view text,
                                std::uint32_t length) const;

    TokenId bos() const { return bos_; }
    TokenId eos() const { return eos_; }
    TokenId pad() const { return pad_; }

  private:
    void bpe(const std::string& word, std::vector<TokenId>& out) const;

    std::unordered_map<std::string, TokenId> vocab_;
    /** "left right" -> merge rank. */
    std::unordered_map<std::string, std::uint32_t> ranks_;
    TokenId bos_ = 0, eos_ = 0, pad_ = 0, unk_ = 0;
};

}  // namespace locus::tok
