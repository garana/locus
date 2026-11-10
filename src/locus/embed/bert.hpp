#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::embed {

/** How token states become one vector, as GGUF's `pooling_type`. */
enum class Pooling : std::uint32_t {
    kNone = 0,
    kMean = 1,  ///< The average over the tokens.
    kCls = 2,   ///< The first token's state (CLS).
    kLast = 3,  ///< The last token's state.
    kRank = 4,  ///< A reranker's classifier head (not an embedding).
};

/**
 * The BERT encoder of embedding models (i#502), GGUF arch "bert", as
 * llama.cpp runs it: BERT (bge-small, e5-small-v2) and XLM-R (bge-m3),
 * whose converter already drops the position rows XLM-R skips, so
 * row t is position t for both:
 *
 * x = LayerNorm(token + position + token_type[0]); then per layer,
 * post-norm: x = LayerNorm(x + O * attention(x)), then
 * x = LayerNorm(x + down * gelu(up * x)). Every matrix has a bias.
 * Attention is bidirectional (no mask within a sequence), scaled by
 * 1/sqrt(head_dim). GELU is the exact erf form BERT was trained
 * with; llama.cpp uses the tanh form, which moves bge-small's vectors
 * by a cosine of about 1e-6.
 *
 * Matrices stay in the GGUF mapping, so the GgufFile must outlive the
 * encoder.
 */
class BertEncoder {
  public:
    /**
     * @param g A GGUF whose general.architecture is "bert".
     * @throws gguf::Error on a missing key, a missing or wrong-sized
     *     tensor, or a pooling type other than mean, CLS, last or rank
     *     (rank needs cls.output, and takes cls when present).
     */
    static BertEncoder load(const gguf::GgufFile& g);

    /** @returns The output width (384 for bge-small). */
    std::uint32_t width() const { return width_; }
    /** @returns The most tokens one sequence may have (512). */
    std::uint32_t context() const { return context_; }
    Pooling pooling() const { return pooling_; }

    /**
     * Runs the encoder over several sequences in one pass: the
     * matrices see every token of the batch, attention stays within
     * each sequence.
     *
     * @param batch Token ids per sequence, as the tokenizer gives them
     *     (CLS or <s> first, SEP or </s> last).
     * @param op The backend ops the matrices run on.
     * @returns Per sequence, the token states: size() x width() floats,
     *     token-major.
     * @throws std::invalid_argument on an empty sequence, one longer
     *     than context(), or an id outside the vocabulary.
     */
    std::vector<std::vector<float>> encode(
        std::span<const std::vector<tok::TokenId>> batch,
        const backend::Ops& op) const;

    /**
     * A reranker's scores (pooling() kRank): encode(), then the
     * classifier on each sequence's first (CLS) state: tanh(cls x)
     * when the model has cls, then cls.output.
     *
     * @returns labels() raw logits per sequence (bge-reranker-v2-m3
     *     has one: the relevance score).
     * @throws std::invalid_argument when the model is not a reranker,
     *     or as encode() does.
     */
    std::vector<std::vector<float>> rank(
        std::span<const std::vector<tok::TokenId>> batch,
        const backend::Ops& op) const;

    /** @returns The classifier's output count (0 unless kRank). */
    std::uint32_t labels() const { return labels_; }

    /**
     * encode(), then pooling() per sequence and an L2 normalization.
     *
     * @returns One width()-float unit vector per sequence (a zero
     *     vector stays zero).
     */
    std::vector<std::vector<float>> embed(
        std::span<const std::vector<tok::TokenId>> batch,
        const backend::Ops& op) const;

    /**
     * Pools one sequence's token states.
     *
     * @param states n x width floats, token-major.
     * @param n Token count, at least 1.
     * @param width Row width.
     * @param pooling kMean, kCls or kLast.
     * @returns width floats, L2-normalized.
     */
    static std::vector<float> pool(std::span<const float> states,
                                   std::uint32_t n, std::uint32_t width,
                                   Pooling pooling);

  private:
    struct Linear {
        backend::Mat w;
        std::vector<float> b;
    };
    struct Norm {
        std::vector<float> w, b;
    };
    struct Layer {
        Linear q, k, v, o, up, down;
        Norm attn_norm, out_norm;
    };

    std::uint32_t width_ = 0, heads_ = 0, ffn_ = 0, vocab_ = 0,
                  context_ = 0;
    float eps_ = 0.0f;
    Pooling pooling_ = Pooling::kCls;
    backend::Mat tok_embd_, pos_embd_;
    std::vector<float> type0_;
    Norm embd_norm_;
    std::vector<Layer> layers_;
    /** The reranker's classifier; cls_ may be absent (data null). */
    Linear cls_, cls_out_;
    std::uint32_t labels_ = 0;
};

}  // namespace locus::embed
