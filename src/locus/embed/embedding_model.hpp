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
 * An encoder model: a BERT or XLM-R GGUF with its tokenizer, owning the
 * mapping its encoder points into. With pooling rank it is a reranker
 * (locus-server --rerank-model, rank()), else an embedding model
 * (--embedding-model, embed()). Task prefixes ("query: ",
 * "search_query: ") are the client's job.
 */
class EmbeddingModel {
  public:
    /** The tokens one encoder pass may hold: inputs are grouped into
     *  passes of at most this many tokens (one input may exceed it on
     *  its own), which bounds the activations (about 70 MB for
     *  bge-small, 180 MB for bge-large). */
    static constexpr std::size_t kPassTokens = 4096;

    /** A pass runs on the GPU ops, when embed() / rank() get them, from
     *  kGpuMinTokens tokens and kGpuMacs of work (tokens x
     *  BertEncoder::token_macs()); below either the GPU's fixed cost
     *  per call wins. On an M2 Pro (MoltenVK), CPU against Vulkan:
     *  bge-m3 45 tokens 190 / 206 ms, 120 tokens 588 / 339; nomic-embed
     *  v1.5 120 tokens 217 / 108; bge-small 240 tokens 112 / 135, 480
     *  tokens 216 / 205. */
    static constexpr std::size_t kGpuMinTokens = 100;
    static constexpr std::uint64_t kGpuMacs = 10'000'000'000ull;

    /**
     * @param path A GGUF of arch "bert" with a WordPiece tokenizer
     *     (BERT) or a SentencePiece unigram one (XLM-R).
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
     * Embeds each input: tokenized with the model's special tokens
     * (CLS and SEP, or <s> and </s>), encoded in passes
     * of up to kPassTokens tokens, pooled and L2-normalized.
     *
     * @param inputs UTF-8 texts.
     * @param op The backend ops the matrices run on.
     * @param tokens Set to the total token count, when not null.
     * @param gpu GPU ops (backend::vulkan_image_ops()) for the passes of
     *     at least kGpuMinTokens and kGpuMacs, or null for op
     *     everywhere.
     * @returns One width()-float vector per input.
     * @throws std::invalid_argument when an input has more than
     *     context() tokens (the message names the input's index), or
     *     the model is a reranker.
     */
    std::vector<std::vector<float>> embed(
        const std::vector<std::string>& inputs, const backend::Ops& op,
        std::size_t* tokens = nullptr,
        const backend::Ops* gpu = nullptr) const;

    /** @returns Whether the model is a reranker (pooling rank). */
    bool reranker() const { return enc_.pooling() == Pooling::kRank; }

    /**
     * Scores each document against the query, as llama-server's
     * /v1/rerank does: each pair is [BOS] query [EOS] [SEP] document
     * [EOS], each special token when the GGUF's add_bos / add_eos /
     * add_sep says so (XLM-R: <s> q </s></s> d </s>), encoded in
     * passes of up to kPassTokens tokens; the score is the classifier's
     * first raw logit. A pair longer than context() keeps the start of
     * its document.
     *
     * @param tokens Set to the total token count, when not null.
     * @param gpu As for embed().
     * @returns One score per document, in their order.
     * @throws std::invalid_argument when the model is not a reranker,
     *     or the query alone leaves no room for a document.
     */
    std::vector<float> rank(const std::string& query,
                            const std::vector<std::string>& documents,
                            const backend::Ops& op,
                            std::size_t* tokens = nullptr,
                            const backend::Ops* gpu = nullptr) const;

  private:
    /** The special tokens of a pair; -1 = none. */
    struct PairFormat {
        tok::TokenId bos = -1, eos = -1, sep = -1;
    };

    EmbeddingModel(std::unique_ptr<gguf::GgufFile> g,
                   std::unique_ptr<tok::Tokenizer> tok, BertEncoder enc,
                   std::string name, PairFormat pair);

    std::unique_ptr<gguf::GgufFile> gguf_;  ///< enc_ points into it
    std::unique_ptr<tok::Tokenizer> tok_;
    BertEncoder enc_;
    std::string name_;
    PairFormat pair_;
};

}  // namespace locus::embed
