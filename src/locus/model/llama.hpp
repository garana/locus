#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::model {

/** Model architecture families the runtime implements. */
enum class Arch {
    kLlama,
    kDeepseek2,
};

/** MoE gating functions (deepseek2.expert_gating_func values). */
enum class GatingFunc : std::uint32_t {
    kSoftmax = 1,
    kSigmoid = 2,
};

/** Llama-family hyperparameters read from GGUF metadata. */
struct Hparams {
    Arch arch = Arch::kLlama;
    std::uint32_t n_embd = 0;
    std::uint32_t n_layers = 0;
    std::uint32_t n_heads = 0;
    std::uint32_t n_kv_heads = 0;
    std::uint32_t n_ff = 0;
    std::uint32_t n_vocab = 0;
    std::uint32_t head_dim = 0;
    /** MoE: expert count (0 = dense FFN). */
    std::uint32_t n_expert = 0;
    /** MoE: experts activated per token. */
    std::uint32_t n_expert_used = 0;
    /** MoE: always-on shared experts (deepseek2). */
    std::uint32_t n_expert_shared = 0;
    /** MoE: first layers that use the dense FFN (deepseek2). */
    std::uint32_t n_dense_lead = 0;
    /** MoE: per-expert FFN width (deepseek2; else n_ff). */
    std::uint32_t n_ff_exp = 0;
    /** MoE: routed-expert weight multiplier. */
    float expert_weights_scale = 1.0f;
    /** MoE: renormalize the top-k weights. */
    bool expert_weights_norm = false;
    GatingFunc gating = GatingFunc::kSoftmax;
    /** MoE: routing groups (V3/K2 group-limited routing). */
    std::uint32_t n_group = 1;
    std::uint32_t n_group_used = 1;
    /** MLA: query compression rank (0 = direct wq). */
    std::uint32_t q_lora_rank = 0;
    /** MLA (deepseek2): latent and per-head dims. */
    std::uint32_t kv_lora_rank = 0;
    std::uint32_t qk_nope_dim = 0;
    std::uint32_t qk_rope_dim = 0;
    std::uint32_t v_head_dim = 0;
    /** Attention score scale (arch-dependent; yarn mscale^2). */
    float kq_scale = 0.0f;
    /** YARN rope correction (freq_scale 1 = off). */
    backend::Yarn yarn;
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
        /** MoE: router probabilities and expert accumulator. */
        std::vector<float> router, moe_acc;
        /** MLA: kv_a projection, absorbed q, weighted latent,
         * compressed query. */
        std::vector<float> kv_a, q_abs, latent, q_a;
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

    /**
     * A 3-D expert tensor: n_expert equally-sized matrices,
     * contiguous in the mapped file.
     */
    struct ExpertMat {
        backend::Mat base;
        std::uint64_t expert_bytes = 0;

        /** @returns Expert e's matrix view. */
        backend::Mat expert(std::uint32_t e) const {
            backend::Mat m = base;
            m.data = base.data + e * expert_bytes;
            return m;
        }
    };

    /** Per-layer weights (read by GPU backends; else internal). */
    struct Layer {
        std::span<const float> attn_norm, ffn_norm;
        backend::Mat wq, wk, wv, wo;
        /** Dense FFN (unused when the layer is MoE). */
        backend::Mat w_gate, w_up, w_down;
        /** MoE router and expert tensors. */
        backend::Mat gate_inp;
        ExpertMat gate_exps, up_exps, down_exps;
        /** MoE shared experts (deepseek2; fused matrices). */
        backend::Mat gate_shexp, up_shexp, down_shexp;
        /** MLA (deepseek2): latent projections and norms. The
         * decompression weights come fused (wkv_b) in older
         * conversions or split (wk_b/wv_b, k_b pre-transposed)
         * in MLA-cache-era GGUFs. */
        backend::Mat wkv_a, wkv_b, wk_b, wv_b;
        std::span<const float> kv_a_norm;
        /** MLA: compressed-query path (q_lora_rank > 0). */
        backend::Mat wq_a, wq_b;
        std::span<const float> q_a_norm;
        /** MoE: selection bias (V3/K2 e_score_correction). */
        std::span<const float> exp_probs_b;

        /** @returns true when this layer routes experts. */
        bool is_moe() const { return gate_inp.rows > 0; }
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
    void moe_ffn(const Layer& lay, Workspace& ws,
                 std::uint32_t layer) const;

    Hparams hp_;
    const struct ArchSpec* spec_ = nullptr;
    const backend::Backend* backend_ = nullptr;
    std::span<const float> rope_factors_;
    backend::Mat embd_;
    std::span<const float> out_norm_;
    backend::Mat out_w_;
    std::vector<Layer> layers_;
};

/** @returns The index of the largest logit (greedy sampling). */
tok::TokenId argmax(std::span<const float> logits);

/**
 * MoE expert selection shared by the CPU and GPU forwards:
 * applies the gating function to raw router logits, adds the
 * V3/K2 selection bias, applies group-limited routing, picks the
 * top-k, and returns (expert, mixing weight) pairs with
 * normalization and expert_weights_scale applied.
 */
std::vector<std::pair<std::uint32_t, float>> moe_select(
    const Hparams& hp, const LlamaModel::Layer& lay,
    std::span<float> router_logits);

}  // namespace locus::model
