#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "cppllm/backend/cpu_ops.hpp"
#include "cppllm/gguf/gguf.hpp"
#include "cppllm/tok/tokenizer.hpp"

namespace cppllm::model {

/** Llama-family hyperparameters read from GGUF metadata. */
struct Hparams {
    std::uint32_t n_embd = 0;
    std::uint32_t n_layers = 0;
    std::uint32_t n_heads = 0;
    std::uint32_t n_kv_heads = 0;
    std::uint32_t n_ff = 0;
    std::uint32_t n_vocab = 0;
    std::uint32_t head_dim = 0;
    /** Max context the model was trained for. */
    std::uint32_t n_ctx = 0;
    float rms_eps = 1e-5f;
    float rope_freq_base = 10000.0f;
};

/**
 * Single-sequence Llama-family runtime over the scalar CPU backend
 * (DESIGN.md milestone M2). Weights stay in the GGUF mapping; the
 * GgufFile must outlive the model. M3 replaces State's contiguous
 * KV vectors with the paged cache.
 */
class LlamaModel {
  public:
    /**
     * Wires up weights and validates shapes against hyperparams.
     *
     * @param g A parsed model with general.architecture "llama".
     * @throws gguf::Error on missing tensors or shape mismatches.
     */
    static LlamaModel load(const gguf::GgufFile& g);

    const Hparams& hparams() const { return hp_; }

    /** Per-sequence KV cache and scratch buffers. */
    struct State {
        /** Per layer: n_ctx * n_kv_heads * head_dim floats. */
        std::vector<std::vector<float>> k, v;
        /** Tokens already in the cache. */
        std::uint32_t n = 0;
        /** Scratch (sized by make_state). */
        std::vector<float> x, xb, xb2, q, att, gate, up, out;
    };

    /** @returns A fresh State sized for this model. */
    State make_state() const;

    /**
     * Runs one token through the model, appending to the cache.
     *
     * @param token In-vocab token id.
     * @param st State whose st.n equals this token's position.
     * @param logits Out; n_vocab floats.
     * @throws std::invalid_argument on position/context misuse.
     */
    void forward(tok::TokenId token, State& st,
                 std::span<float> logits) const;

  private:
    struct Layer {
        std::span<const float> attn_norm, ffn_norm;
        backend::Mat wq, wk, wv, wo, w_gate, w_up, w_down;
    };

    Hparams hp_;
    backend::Mat embd_;
    std::span<const float> out_norm_;
    backend::Mat out_w_;
    std::vector<Layer> layers_;
};

/** @returns The index of the largest logit (greedy sampling). */
tok::TokenId argmax(std::span<const float> logits);

}  // namespace cppllm::model
