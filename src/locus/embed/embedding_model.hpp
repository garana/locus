#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/embed/bert.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::embed {

/**
 * A dedicated embedding model (locus-server --embedding-model): a BERT
 * GGUF with its tokenizer, owning the mapping its encoder points into.
 * Task prefixes ("query: ", "search_query: ") are the client's job.
 */
class EmbeddingModel {
  public:
    /** The tokens one encoder pass may hold: inputs are grouped into
     *  passes of at most this many tokens (one input may exceed it on
     *  its own), which bounds the activations (about 70 MB for
     *  bge-small, 180 MB for bge-large). */
    static constexpr std::size_t kPassTokens = 4096;

    /**
     * @param path A GGUF of arch "bert" with a WordPiece tokenizer.
     * @param name The model id requests use; empty = the file name,
     *     as locus-server names the chat model.
     * @throws gguf::Error when the file is not such a model.
     */
    static std::unique_ptr<EmbeddingModel> load(const std::string& path,
                                                std::string name = {});

    const std::string& name() const { return name_; }
    std::uint32_t width() const { return enc_.width(); }
    /** @returns The most tokens one input may have, CLS and SEP included. */
    std::uint32_t context() const { return enc_.context(); }

    /**
     * Embeds each input: tokenized with CLS and SEP, encoded in passes
     * of up to kPassTokens tokens, pooled and L2-normalized.
     *
     * @param inputs UTF-8 texts.
     * @param op The backend ops the matrices run on.
     * @param tokens Set to the total token count, when not null.
     * @returns One width()-float vector per input.
     * @throws std::invalid_argument when an input has more than
     *     context() tokens; the message names the input's index.
     */
    std::vector<std::vector<float>> embed(
        const std::vector<std::string>& inputs, const backend::Ops& op,
        std::size_t* tokens = nullptr) const;

  private:
    EmbeddingModel(std::unique_ptr<gguf::GgufFile> g,
                   std::unique_ptr<tok::Tokenizer> tok, BertEncoder enc,
                   std::string name);

    std::unique_ptr<gguf::GgufFile> gguf_;  ///< enc_ points into it
    std::unique_ptr<tok::Tokenizer> tok_;
    BertEncoder enc_;
    std::string name_;
};

}  // namespace locus::embed
