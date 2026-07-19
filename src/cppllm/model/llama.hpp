#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "cppllm/backend/cpu_ops.hpp"
#include "cppllm/backend/registry.hpp"
#include "cppllm/gguf/gguf.hpp"
#include "cppllm/kv/paged_cache.hpp"
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
 * Llama-family runtime over the scalar CPU backend, reading and
 * writing KV state through the paged cache so any number of
 * sequences can interleave (DESIGN.md M2/M3). Weights stay in the
 * GGUF mapping; the GgufFile must outlive the model.
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

    /**
     * Routes heavy ops through the given backend (default: the
     * best available at load time; see backend::backends()).
     */
    void use_backend(const backend::Backend& b) { backend_ = &b; }

    /** @returns The backend currently doing the math. */
    const backend::Backend& active_backend() const {
        return *backend_;
    }

    /** Reusable per-thread scratch buffers (no sequence state). */
    struct Workspace {
        std::vector<float> x, xb, xb2, q, att, gate, up, out;
    };

    /** @returns A workspace sized for this model. */
    Workspace make_workspace() const;

    /**
     * @param n_blocks Pool size in blocks; 0 sizes the pool for
     *     one full context window.
     * @returns A paged cache with this model's geometry.
     */
    kv::PagedKvCache make_cache(std::uint32_t n_blocks = 0) const;

    /**
     * Runs one token at position seq.n_tokens, appending K/V to
     * seq and advancing it. Routed to the GPU when the active
     * backend provides a full forward (vulkan); CPU ops
     * otherwise.
     *
     * @param token In-vocab token id.
     * @param cache Cache the sequence lives in; capacity for one
     *     more position must already be ensured.
     * @param logits Out; n_vocab floats.
     * @throws std::invalid_argument on vocab/context misuse.
     */
    void forward(tok::TokenId token, kv::PagedKvCache& cache,
                 kv::PagedKvCache::Seq& seq, Workspace& ws,
                 std::span<float> logits) const;

    /** Per-layer weights (read by GPU backends; else internal). */
    struct Layer {
        std::span<const float> attn_norm, ffn_norm;
        backend::Mat wq, wk, wv, wo, w_gate, w_up, w_down;
    };

    const std::vector<Layer>& layers() const { return layers_; }
    const backend::Mat& embedding() const { return embd_; }
    const backend::Mat& output_weight() const { return out_w_; }
    std::span<const float> output_norm() const {
        return out_norm_;
    }

    /**
     * @returns llama3 rope-scaling divisors (head_dim/2 floats,
     *     from rope_freqs.weight), or empty when unscaled.
     */
    std::span<const float> rope_factors() const {
        return rope_factors_;
    }

  private:
    Hparams hp_;
    const backend::Backend* backend_ = nullptr;
    std::span<const float> rope_factors_;
    backend::Mat embd_;
    std::span<const float> out_norm_;
    backend::Mat out_w_;
    std::vector<Layer> layers_;
};

/** @returns The index of the largest logit (greedy sampling). */
tok::TokenId argmax(std::span<const float> logits);

}  // namespace cppllm::model
