#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::image {

/**
 * The T5 v1.1 text encoder (i#308), as transformers' T5EncoderModel:
 * PixArt-alpha's prompt encoder (T5 v1.1 XXL: 24 blocks, 4096 wide,
 * 64 heads of 64).
 *
 * Each block: x += O * attention(rms(x)), then
 * x += down * (gelu_tanh(gate * rms(x)) * (up * rms(x))); a final rms
 * norm. RMS norms have no bias and no mean. Attention has no 1/sqrt(d)
 * scale; instead a learned bias per (bucket, head) from block 0 is
 * added to every block's scores, the bucket being a function of the
 * key position minus the query position (relative_bucket).
 *
 * The encoder runs on the valid tokens only. With a padded, masked
 * input the valid rows are the same, since a masked key gets weight 0
 * and the bias depends only on the distance.
 *
 * Weights come from a "t5encoder" GGUF (scripts/convert-t5-encoder.py,
 * llama.cpp's tensor names). Matrices stay in the GGUF mapping, so the
 * GgufFile must outlive the encoder.
 */
class T5Encoder {
  public:
    /**
     * @param g A GGUF whose general.architecture is "t5encoder".
     * @throws gguf::Error on a missing key or a missing or wrong-sized
     *     tensor.
     */
    static T5Encoder load(const gguf::GgufFile& g);

    /** @returns The output width (d_model, 4096 for XXL). */
    std::uint32_t width() const { return width_; }
    /** @returns The rows of the token embedding (32128). */
    std::uint32_t vocab() const { return vocab_; }

    /**
     * @param ids Token ids, ending with </s> as T5Tokenizer gives them.
     * @param op The backend ops the matrices run on.
     * @returns ids.size() x width() floats, token-major.
     * @throws std::invalid_argument on an empty input or an id outside
     *     the vocabulary.
     */
    std::vector<float> encode(std::span<const tok::TokenId> ids,
                              const backend::Ops& op) const;

    /**
     * transformers' _relative_position_bucket, bidirectional.
     *
     * @param d Key position minus query position.
     * @param buckets Bucket count (32): half for d > 0, half for the
     *     rest.
     * @param max_distance Distance where the log-spaced buckets end
     *     (128); farther distances share the last bucket.
     * @returns The bucket, in [0, buckets).
     */
    static std::uint32_t relative_bucket(std::int32_t d,
                                         std::uint32_t buckets,
                                         std::uint32_t max_distance);

  private:
    struct Block {
        std::vector<float> attn_norm, ffn_norm;
        backend::Mat q, k, v, o, gate, up, down;
    };

    std::uint32_t width_ = 0, heads_ = 0, head_dim_ = 0, ffn_ = 0,
                  vocab_ = 0, buckets_ = 0, max_distance_ = 0;
    float eps_ = 0.0f;
    backend::Mat embd_;
    /** [bucket][head], from block 0. */
    std::vector<float> rel_bias_;
    std::vector<float> out_norm_;
    std::vector<Block> blocks_;
};

}  // namespace locus::image
