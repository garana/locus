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
    /** DSA lightning indexer (glm-dsa; 0 heads = no indexer). */
    std::uint32_t idx_heads = 0;
    std::uint32_t idx_dim = 0;
    std::uint32_t idx_top_k = 0;
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
        /** DSA indexer: per-head queries, head weights, scores
         * over cached positions, selected positions. */
        std::vector<float> idx_q, idx_w, idx_scores;
        std::vector<std::uint32_t> idx_sel;
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
     * Computes a sentence embedding for `tokens`: runs them through
     * the layer stack (updating seq's KV) and returns the final
     * normed hidden state of the LAST token (last-token pooling),
     * n_embd floats, unnormalized. The caller L2-normalizes if it
     * wants unit vectors.
     *
     * CPU/CUDA backends only -- the Vulkan full-forward does not
     * surface the hidden state.
     *
     * @param out n_embd floats.
     * @throws std::invalid_argument on an empty batch, wrong `out`
     *     size, or a backend without a CPU-driver forward.
     */
    void embed(std::span<const tok::TokenId> tokens,
               kv::PagedKvCache& cache, kv::PagedKvCache::Seq& seq,
               Workspace& ws, std::span<float> out) const;

    /**
     * Runs N tokens of one sequence in a single pass (R10 batched
     * forward): each weight is applied to all N tokens before the
     * next weight, so a streamed weight is read once per layer,
     * not once per token. Byte-identical to N sequential forward()
     * calls (weight-stationary, not a GEMM). Only the last token's
     * logits are produced (prefill semantics). Currently the dense
     * llama path only; MLA/MoE archs fall back is the caller's job.
     *
     * @param tokens In-vocab token ids; capacity for tokens.size()
     *     more positions must already be ensured on seq.
     * @param logits Out; n_vocab floats for the LAST token, or
     *     n * n_vocab (one per position, in order) when all_logits.
     * @param all_logits Emit every position's logits (speculative
     *     verify) instead of only the last (prefill).
     * @throws std::invalid_argument on misuse or unsupported arch.
     */
    void forward_batch(std::span<const tok::TokenId> tokens,
                       kv::PagedKvCache& cache,
                       kv::PagedKvCache::Seq& seq, Workspace& ws,
                       std::span<float> logits,
                       bool all_logits = false) const;

    /** @returns true when the batched forwards support this model
     * (every CPU/CUDA backend + arch; Vulkan opts out). */
    bool supports_batch() const;

    /**
     * Decodes one token from each of N DIFFERENT sequences in a
     * single pass (R10 step 4b, continuous-batch decode): token i
     * belongs to seqs[i] and attends to its own KV cache, while
     * the weight-bearing ops are batched across all N so each
     * weight is read once per layer instead of once per sequence.
     * Byte-identical to N separate forward() calls (the sequences
     * are independent). Produces logits for every token
     * (n * n_vocab, token-major) and advances each seq by one.
     *
     * @param tokens One in-vocab token id per sequence.
     * @param seqs   Sequence per token; each must have capacity
     *     for one more position ensured.
     * @param logits Out; n * n_vocab floats, token i at [i*n_vocab].
     */
    void forward_batch_decode(
        std::span<const tok::TokenId> tokens,
        kv::PagedKvCache& cache,
        std::span<kv::PagedKvCache::Seq* const> seqs,
        Workspace& ws, std::span<float> logits) const;

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
        /** DSA indexer (glm-dsa): head weights, shared key,
         * per-head queries from the q latent, key layernorm. */
        backend::Mat idx_proj, idx_k, idx_q_b;
        std::span<const float> idx_k_norm, idx_k_norm_b;

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

    /**
     * Batched MoE (R10 step 3b): routes n tokens (ffn-normed
     * inputs in xbf, n*n_embd), then applies the routed experts
     * WEIGHT-STATIONARY -- each expert's gate/up/down is read once
     * across the tokens that picked it -- accumulating into the
     * residual x (n*n_embd). Byte-identical to n moe_ffn() calls:
     * each token's mixture is summed in moe_select order (routed
     * then shared) before adding to x.
     */
    void moe_ffn_batch(const Layer& lay, std::uint32_t layer,
                       const std::vector<float>& xbf,
                       std::vector<float>& x, std::uint32_t n,
                       Workspace& ws) const;

    Hparams hp_;
    const struct ArchSpec* spec_ = nullptr;
    const backend::Backend* backend_ = nullptr;
    /** Weights live in a read-only file mapping; gates the
     * LOCUS_WEIGHT_WINDOW DONTNEED policy (see gguf.hpp). */
    bool file_backed_ = false;
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

/**
 * Row-split matvec across the R9 thread pool: slices w into row
 * ranges dispatched through the SAME backend kernel into
 * disjoint spans of out, so logits are bit-identical for every
 * thread count. LOCUS_THREADS caps the fan-out (default:
 * hardware cores; 1 runs inline). Small matrices stay inline.
 */
void matvec_mt(const backend::Ops& op, const backend::Mat& w,
               std::span<const float> x, std::span<float> out);

/**
 * Weight-stationary batched matvec (R10): applies w to each of
 * the n input vectors (x_batch is n * w.cols contiguous) into
 * out_batch (n * w.rows contiguous). The batch loop is inside so
 * w is touched once and stays hot in cache / page cache across
 * the n tokens. Byte-identical to n matvec() calls -- the win is
 * fewer weight reads, not different arithmetic.
 *
 * With LOCUS_BATCH_DEQUANT set, dispatches instead to a dequant-
 * amortized kernel that dequantizes each weight row to f32 once and
 * dots it against all n columns (cuts dequant work n-fold). That
 * path is token-exact but not byte-identical to the fused matvec,
 * so it is opt-in and covered by its own approximate-equality test.
 */
void matvec_batch(const backend::Ops& op, const backend::Mat& w,
                  std::span<const float> x_batch,
                  std::span<float> out_batch, std::uint32_t n);

/**
 * Dequant-amortized batched matvec (R10 step 5): dequantizes each
 * weight row to f32 once via op.dequant_row, then dots it against
 * all n token columns in plain scalar f32. Cuts dequant work from
 * once-per-(row,token) to once-per-row. Token-exact and
 * deterministic, but not byte-identical to the fused matvec, so
 * matvec_batch only routes here when LOCUS_BATCH_DEQUANT is set.
 * Exposed for its dedicated approximate-equality test.
 */
void matvec_batch_deq(const backend::Ops& op, const backend::Mat& w,
                      std::span<const float> x_batch,
                      std::span<float> out_batch, std::uint32_t n);

/**
 * DSA lightning-indexer score for one cached position:
 * sum over heads h of w[h] * relu(dot(q[h*d .. h*d+d), k)).
 * Global scale factors are omitted -- they do not change the
 * top-k order the score exists to produce.
 */
float dsa_index_score(std::span<const float> q,
                      std::span<const float> w,
                      std::span<const float> k);

}  // namespace locus::model
