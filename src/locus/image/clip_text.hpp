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
 * The CLIP text encoder (i#372), as transformers' CLIPTextModel: FLUX's
 * pooled-prompt encoder (CLIP-L: 12 layers, 768 wide, 12 heads of 64).
 *
 * Token plus position embeddings, then per layer
 * x += out(attention(ln1(x))) with a causal mask, and
 * x += fc2(quick_gelu(fc1(ln2(x)))), where quick_gelu(v) is
 * v * sigmoid(1.702 v); a final layer norm. The pooled output is the
 * final row at the end token: with the legacy config eos_token_id 2
 * that is the first position of the largest id (transformers'
 * argmax), else the first position of the end token.
 *
 * Weights come from a "clip_text" GGUF (scripts/convert-flux.py), all
 * F32. Matrices stay in the GGUF mapping, so the GgufFile must outlive
 * the encoder.
 */
class ClipTextEncoder {
  public:
    /** The encoder's output for one prompt. */
    struct Output {
        std::vector<float> hidden;  ///< ids.size() x width, token-major
        std::vector<float> pooled;  ///< width floats
    };

    /**
     * @param g A GGUF whose general.architecture is "clip_text".
     * @throws gguf::Error on a missing key or a missing or wrong-sized
     *     tensor.
     */
    static ClipTextEncoder load(const gguf::GgufFile& g);

    std::uint32_t width() const { return width_; }
    /** @returns The position embedding rows (77). */
    std::uint32_t context() const { return context_; }

    /**
     * @param ids Token ids as ClipTokenizer::encode gives them, at most
     *     context() of them.
     * @param op The backend ops the matrices run on.
     * @throws std::invalid_argument on an empty or too long input, or an
     *     id outside the vocabulary.
     */
    Output encode(std::span<const tok::TokenId> ids,
                  const backend::Ops& op) const;

  private:
    struct Linear {
        backend::Mat w;
        std::vector<float> b;
    };
    struct Layer {
        std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
        Linear q, k, v, out, fc1, fc2;
    };

    std::uint32_t width_ = 0, heads_ = 0, ffn_ = 0, context_ = 0,
                  vocab_ = 0, legacy_eos_ = 0;
    tok::TokenId eos_ = 0;
    float eps_ = 0.0f;
    /** [vocab][width] and [context][width], in the GGUF mapping. */
    const float* token_embd_ = nullptr;
    const float* pos_embd_ = nullptr;
    std::vector<float> final_w_, final_b_;
    std::vector<Layer> layers_;
};

}  // namespace locus::image
