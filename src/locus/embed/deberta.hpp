#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::embed {

/**
 * The DeBERTa v3 encoder with its sequence-classifier head (i#1158),
 * GGUF arch "deberta-v2" (scripts/text/convert-deberta.py), as
 * transformers' DebertaV2ForSequenceClassification computes it:
 *
 * x = LayerNorm(token): no absolute positions, no token types. Then
 * per layer, post-norm as BERT: x = LayerNorm(x + O * attention(x)),
 * then x = LayerNorm(x + down * gelu(up * x)), every matrix with a
 * bias and GELU in its exact erf form.
 *
 * Attention is disentangled. With r = LayerNorm(rel_embd), the
 * 2 * position_buckets relative embeddings, and the layer's own q and
 * k projections on them (share_att_key), pk = Wk r + bk and
 * pq = Wq r + bq, the score of query i on key j in each head is
 *
 *     (q_i . k_j + q_i . pk[c] + k_j . pq[c]) / sqrt(3 * head_dim)
 *
 * with c = clamp(bucket(i - j) + position_buckets, 0,
 * 2 * position_buckets - 1): content to content, content to position
 * (c2p) and position to content (p2c). bucket() is transformers'
 * make_log_bucket_position: the distance itself within half the
 * buckets, logarithmic beyond. The positions are relative only, so a
 * sequence may be longer than context(), as transformers allows.
 *
 * classify() runs the head: the first ([CLS]) state through cls and
 * GELU (the pooler), then cls.output, one logit per label.
 *
 * Matrices stay in the GGUF mapping, so the GgufFile must outlive the
 * encoder.
 */
class DebertaEncoder {
  public:
    /** The most tokens one sequence may have: the attention costs
     *  tokens squared per head. */
    static constexpr std::uint32_t kMaxTokens = 8192;

    /**
     * @param g A GGUF whose general.architecture is "deberta-v2".
     * @throws gguf::Error on a missing key, a missing or wrong-sized
     *     tensor, or a pooling type other than 4 (a classifier head).
     */
    static DebertaEncoder load(const gguf::GgufFile& g);

    /** @returns The hidden width (768 for DeBERTa v3 base). */
    std::uint32_t width() const { return width_; }
    /** @returns The length the model was trained at (512): longer
     *  inputs run, but callers that window should use this. */
    std::uint32_t context() const { return context_; }
    /** @returns The classifier's label names, in logit order. */
    const std::vector<std::string>& labels() const { return labels_; }
    /** @returns The multiply-adds of one token through the layer and
     *  head matrices (the token embedding is a lookup, not counted). */
    std::uint64_t token_macs() const { return token_macs_; }

    /**
     * Runs the encoder over several sequences in one pass: the
     * matrices see every token of the batch, attention stays within
     * each sequence.
     *
     * @param batch Token ids per sequence, as the tokenizer gives them
     *     ([CLS] first, [SEP] last).
     * @param op The backend ops the matrices run on.
     * @returns Per sequence, the token states: size() x width() floats,
     *     token-major.
     * @throws std::invalid_argument on an empty sequence, one longer
     *     than kMaxTokens, or an id outside the vocabulary.
     */
    std::vector<std::vector<float>> encode(
        std::span<const std::vector<tok::TokenId>> batch,
        const backend::Ops& op) const;

    /**
     * encode(), then the classifier head on each sequence's first
     * state.
     *
     * @returns labels().size() raw logits per sequence.
     * @throws std::invalid_argument as encode() does.
     */
    std::vector<std::vector<float>> classify(
        std::span<const std::vector<tok::TokenId>> batch,
        const backend::Ops& op) const;

    /**
     * transformers' make_log_bucket_position, in float32 as it runs.
     *
     * @param rel A relative position, query minus key.
     * @param buckets position_buckets (256 for DeBERTa v3).
     * @param max_position max_relative_positions (512).
     * @returns The bucket, in about [-buckets, buckets].
     */
    static std::int32_t bucket(std::int32_t rel, std::uint32_t buckets,
                               std::uint32_t max_position);

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

    /** One layer's attention over one sequence of m tokens: q, k and
     *  v are m x width (token-major) at the sequence's rows, out the
     *  same. pk and pq hold relative rows [lo, lo + nr). */
    void attend(std::span<const float> q, std::span<const float> k,
                std::span<const float> v, std::uint32_t m,
                std::span<const float> pk, std::span<const float> pq,
                std::uint32_t lo, std::uint32_t nr,
                std::span<float> out) const;

    std::uint32_t width_ = 0, heads_ = 0, ffn_ = 0, vocab_ = 0,
                  context_ = 0, buckets_ = 0, max_rel_ = 0;
    float eps_ = 0.0f;
    backend::Mat tok_embd_;
    Norm embd_norm_;
    /** LayerNorm(rel_embd), 2 * buckets_ rows of width_, at load. */
    std::vector<float> rel_;
    std::vector<Layer> layers_;
    Linear cls_, cls_out_;
    std::vector<std::string> labels_;
    std::uint64_t token_macs_ = 0;
};

}  // namespace locus::embed
