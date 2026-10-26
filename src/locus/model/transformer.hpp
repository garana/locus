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

/** Pre-attention / pre-FFN / final normalization kind. */
enum class NormType {
    kRmsNorm,    /**< default: llama/deepseek2/glm/qwen */
    kLayerNorm,  /**< weight-only LayerNorm (dbrx) */
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
    /** MoE: shared-expert FFN width when it differs from the routed
     * width (qwen2moe); 0 falls back to n_ff_exp * n_expert_shared. */
    std::uint32_t n_ff_shexp = 0;
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
    /** GQA rope pairs (x[i], x[i + head_dim/2]) (NEOX: Qwen2 /
     * Qwen3) instead of interleaved (x[2i], x[2i+1]). */
    bool rope_neox = false;
    /** YARN rope correction (freq_scale 1 = off). */
    backend::Yarn yarn;
    /** Quantized-activation (R16) mode at load: on for every arch
     * (llama.cpp's algorithm, and the fast CPU int8 dots);
     * LOCUS_F32_ACTIVATIONS=1 or use_q8k_activations(false) keeps
     * the f32-activation dot for exactness checks. */
    bool q8k_default = true;
    /** Max context the model was trained for. */
    std::uint32_t n_ctx = 0;
    float rms_eps = 1e-5f;
    float rope_freq_base = 10000.0f;
    /** Normalization kind for attn/ffn/output norms (default RMS). */
    NormType norm_type = NormType::kRmsNorm;
    /** The arch runs its own layer body (ArchSpec::layer:
     * muse-spark). Only the CPU single-token forward implements it:
     * no batch paths, no GPU full forward. */
    bool custom_layer = false;
    /** Weightless RMS norm of the token embedding (muse-spark). */
    bool embd_norm = false;
    /** Logits multiplier, then tanh softcap at +/- this (0 = off). */
    float logit_scale = 1.0f;
    float final_logit_softcap = 0.0f;
    /** Epsilon of the weightless post-branch norms (muse-spark). */
    float post_norm_eps = 1e-6f;
    /** Sliding-window attention: the window (0 = none) and which
     * layers use it (muse-spark: those rope at rope_freq_base_swa;
     * the others attend globally without rope). */
    std::uint32_t swa_window = 0;
    std::vector<bool> swa_layers;
    float rope_freq_base_swa = 10000.0f;
    /** Latent MoE: the width experts read and write (muse-spark
     * moe_latent_size; 0 = n_embd). */
    std::uint32_t moe_latent = 0;

    /** deepseek41 (DeepSeek-V4.1): see deepseek41.cpp. */
    struct Ds41 {
        /** Per layer: compression ratio (0 = window only), the layer
         * whose compressed rows it reads, and the layer whose top-k
         * it reuses (-1 when the ratio is 0). */
        std::vector<std::uint32_t> ratio;
        std::vector<std::int32_t> kv_src, topk_src;
        std::uint32_t window = 0;
        float compress_rope_base = 0.0f;
        std::uint32_t rope_dims = 0;
        std::uint32_t idx_heads = 0, idx_dim = 0, idx_top_k = 0;
        std::int32_t cand_src = -1;
        std::uint32_t cand_blocks = 0, cand_block_size = 0;
        std::uint32_t out_groups = 0, out_rank = 0;
        std::uint32_t hc = 0, sinkhorn_iters = 0;
        float hc_eps = 0.0f;
        std::vector<float> clamp_exp, clamp_shexp;
        /** Engram: layers, n-gram size, heads per n-gram, row width,
         * per-layer multipliers / bucket offsets / bucket counts, pad
         * id and the token -> compressed-id map. */
        std::vector<std::uint32_t> engram_layers;
        std::uint32_t ngram = 0, ngram_heads = 0, engram_dim = 0;
        std::vector<std::uint64_t> engram_mult, engram_off, engram_size;
        std::uint32_t engram_pad = 0;
        std::vector<std::int32_t> token_map;
        /** Cache layer of kv-source layer l's compressed rows. */
        std::vector<std::int32_t> comp_slot;
    };
    Ds41 ds41;
};

/**
 * Output-head rows a forward computes: logits [begin, end) only, the
 * rest -inf. The default (end 0) is every row. A caller that can only
 * sample a known token range (speech: the audio band) skips the rest of
 * the head, which for a large vocabulary is a big share of the weights
 * read per token.
 */
struct LogitRows {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;  ///< one past the last row; 0 = all rows
};

/**
 * Llama-family runtime over the scalar CPU backend, reading and
 * writing KV state through the paged cache so any number of
 * sequences can interleave (DESIGN.md M2/M3). Weights stay in the
 * GGUF mapping; the GgufFile must outlive the model.
 */
class TransformerModel {
  public:
    /**
     * Wires up weights and validates shapes against hyperparams.
     *
     * For slice-only loading (multi-server, DESIGN.md R15+) a pipeline
     * stage loads only the layers it owns: pass [layer_begin, layer_end)
     * and only those transformer blocks are wired up (their pages are
     * the only per-layer weights that ever fault in). layer_end == 0
     * (the default) means the whole model, so the single-process path is
     * unchanged. Layers stay indexed by absolute number; a skipped
     * layer's slot is a default-constructed (empty) Layer, so reading
     * one is silent rather than loud -- forward_layers' range check is
     * what turns an out-of-range run into a clear error. make_cache
     * sizes the KV cache to the loaded range.
     *
     * @param g A parsed model with general.architecture "llama".
     * @param layer_begin First layer to load (inclusive; default 0).
     * @param layer_end One past the last layer to load; 0 (default) or
     *     past the end means all layers.
     * @throws gguf::Error on missing tensors or shape mismatches; Error
     *     on an empty/inverted layer range.
     */
    static TransformerModel load(const gguf::GgufFile& g,
                           std::uint32_t layer_begin = 0,
                           std::uint32_t layer_end = 0);

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

    /**
     * Opt-in ggml-parity mode (DESIGN.md R16): when enabled, QK_K
     * weight matmuls integer-dot a Q8_K-quantized activation, and
     * block-32 weights (Q8_0 / Q4_0 / IQ4_NL) a Q8_0-quantized one
     * (matching llama.cpp's algorithm) via Ops::matvec_q8k. On by
     * default (Hparams::q8k_default) since the CPU int8 dots made it
     * 1.7x faster than the f32-activation dot on NEON (the scalar,
     * NEON and CUDA Q8_K dots replay one accumulation order). Off
     * (or LOCUS_F32_ACTIVATIONS=1) keeps the f32 dot, the reference
     * for exactness checks.
     */
    void use_q8k_activations(bool on) { q8k_activations_ = on; }

    /** @returns Whether Q8_K-activation mode is enabled. */
    bool q8k_activations() const { return q8k_activations_; }

    /**
     * @returns The active backend's ops as the forward uses them:
     *     Ops::matvec_q8k cleared unless Q8_K-activation mode is on.
     *     Arch hooks take these, never active_backend().ops directly,
     *     so use_q8k_activations holds for every arch.
     */
    backend::Ops ops() const;

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

    /** @returns Paged-cache layers: the loaded slice, or the arch's
     * own count (ArchSpec::cache_layers). */
    std::uint32_t cache_layer_count() const;

    /**
     * @param n_blocks Pool size in blocks; 0 sizes the pool for
     *     one full context window.
     * @param kv_type KV storage precision (DESIGN.md R14). kF32 is
     *     the default; kQ8/kQ4 shrink the resident cache and require
     *     a CPU backend (GPU external KV storage is F32-only).
     * @returns A paged cache with this model's geometry.
     */
    kv::PagedKvCache make_cache(
        std::uint32_t n_blocks = 0,
        kv::KvType kv_type = kv::KvType::kF32) const;

    /**
     * Bytes the paged KV pool for this model would occupy at `n_blocks`
     * (0 = the make_cache default), WITHOUT allocating it. Lets a caller
     * budget memory before constructing caches -- e.g. the stage server
     * refusing a --workers x --kv-blocks request that would exceed
     * available RAM. Mirrors make_cache's geometry exactly, so the number
     * matches what make_cache(n_blocks, kv_type) then allocates.
     *
     * Measures the HOST pool. When a backend provides its own KV
     * allocator (Vulkan's alloc_kv), make_cache takes the pool from the
     * device instead, and this host figure would not reflect it -- moot
     * today, since pipeline stages run CPU/CUDA and never take that path.
     * @returns the pool size in bytes.
     */
    std::size_t kv_pool_bytes(
        std::uint32_t n_blocks = 0,
        kv::KvType kv_type = kv::KvType::kF32) const;

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
     * @param rows Head rows to compute; logits outside them are -inf
     *     on every backend (default: all rows).
     * @throws std::invalid_argument on vocab/context misuse or a bad
     *     row range.
     */
    void forward(tok::TokenId token, kv::PagedKvCache& cache,
                 kv::PagedKvCache::Seq& seq, Workspace& ws,
                 std::span<float> logits, LogitRows rows = {}) const;

    /**
     * Runs the transformer layers [layer_begin, layer_end) for one
     * token at position seq.n_tokens. This is the building block for
     * pipeline-parallel execution: each pipeline stage owns a
     * contiguous layer range and calls this for its slice.
     *
     * Input: when layer_begin == 0 the first stage embeds `token`;
     * otherwise `hidden_in` (n_embd floats) is the residual stream
     * handed over from the previous stage and `token` is ignored.
     *
     * Output: when layer_end == n_layers the last stage applies the
     * final norm + output projection and `out` receives the logits
     * (n_vocab floats); otherwise `out` receives the residual stream
     * (n_embd floats) to hand to the next stage.
     *
     * seq.n_tokens advances only when layer_end == n_layers (the
     * token is not fully consumed until the last layer runs), so the
     * whole stack run as [0,k) then [k,n_layers) against one shared
     * cache/seq is byte-identical to a single forward(). The KV cache
     * is indexed by absolute layer number, so `cache` must cover the
     * layers this call runs. CPU/CUDA backends only (the Vulkan
     * full-forward does not surface the hidden state); forward()
     * keeps the Vulkan path.
     *
     * @param token In-vocab token id (used only when layer_begin==0).
     * @param hidden_in Residual stream from the previous stage, n_embd
     *     floats, when layer_begin > 0. At layer_begin == 0 it is empty
     *     (embed `token`) or an n_embd input embedding used instead of
     *     the token's row (Janus image tokens; not for arches with their
     *     own forward).
     * @param layer_begin First layer to run (inclusive).
     * @param layer_end One past the last layer to run.
     * @param out n_vocab floats when layer_end==n_layers, else n_embd.
     * @param rows Head rows the last stage computes (see forward());
     *     here only a hint: rows outside it are left unspecified.
     * @throws std::invalid_argument on a bad range, vocab/context
     *     misuse, or wrong hidden_in / out size.
     */
    void forward_layers(tok::TokenId token,
                        std::span<const float> hidden_in,
                        std::uint32_t layer_begin,
                        std::uint32_t layer_end, kv::PagedKvCache& cache,
                        kv::PagedKvCache::Seq& seq, Workspace& ws,
                        std::span<float> out, LogitRows rows = {}) const;

    /**
     * One position through the whole model, like forward(), with two
     * additions for models that feed embeddings and read the hidden
     * state (Janus image codes, i#247): an optional input embedding in
     * place of `token`'s row, and the final normed hidden state (what
     * the head reads) copied out. Runs the fused Vulkan forward when
     * the backend is Vulkan and supports the model, else
     * forward_layers, so the CPU result is unchanged.
     *
     * @param token The token (ignored when embd_in is set).
     * @param embd_in Empty, or n_embd floats used instead of the
     *     token's embedding row.
     * @param logits n_vocab floats; rows outside `rows` are
     *     unspecified.
     * @param rows Head rows to compute (see forward()).
     * @param normed_out n_embd floats that receive the normed hidden
     *     state.
     * @throws std::invalid_argument on a wrong embd_in / normed_out
     *     size, a token out of vocab (when embd_in is empty), a bad
     *     row range, or an input embedding for an arch with its own
     *     forward.
     */
    void forward_embedding(tok::TokenId token,
                           std::span<const float> embd_in,
                           kv::PagedKvCache& cache,
                           kv::PagedKvCache::Seq& seq, Workspace& ws,
                           std::span<float> logits, LogitRows rows,
                           std::span<float> normed_out) const;

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
     * On the Vulkan backend the chunk runs as GPU matmuls
     * (vulkan_forward_batch): token-exact, not byte-identical.
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

    /** @returns true when forward_batch can ingest a prompt chunk:
     * every supports_batch() backend, plus Vulkan, which runs the
     * chunk through its GPU batched forward (one token at a time
     * when that path does not support the model). */
    bool supports_batch_prefill() const;

    /** @returns The prompt tokens per batched forward this backend
     * prefers: 512 on Vulkan (a GPU matmul needs many token rows to
     * fill the device), 64 elsewhere. */
    std::uint32_t prefill_chunk() const;

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
     * Batched, layer-range counterpart of forward_layers (event-loop
     * executor, DESIGN.md i#24): runs N DIFFERENT sequences through
     * layers [layer_begin, layer_end) in one pass -- each token or
     * activation attends to its own seq, while the weight-bearing ops are
     * batched across all N so each weight is read once per layer. The
     * batched analog of a pipeline stage's step, byte-identical to N
     * separate forward_layers calls.
     *
     * Role mirrors forward_layers. First stage (layer_begin == 0) embeds
     * tokens[i]; a later stage loads hidden_in[i*n_embd ...]. Last stage
     * (layer_end == n_layers) emits logits[i*n_vocab ...] and advances
     * each seq by one; a non-final stage emits the residual stream into
     * out[i*n_embd ...] and leaves the seqs (the final stage consumes the
     * token). Advancing matches forward_layers, so the caller for a
     * non-final stage bumps its seqs itself, exactly as PipelineStage
     * does for the single path.
     *
     * @param tokens One in-vocab id per sequence (first stage); empty
     *     when !first.
     * @param hidden_in n*n_embd residual-stream floats (later stage);
     *     empty when first.
     * @param seqs One sequence per batch slot; each needs capacity for
     *     one more position ensured.
     * @param out n*n_vocab (last) or n*n_embd (otherwise), token-major.
     * @throws std::invalid_argument on misuse, a bad or outside-slice
     *     layer range, or an unsupported backend.
     */
    void forward_batch_layers(
        std::span<const tok::TokenId> tokens,
        std::span<const float> hidden_in, std::uint32_t layer_begin,
        std::uint32_t layer_end, kv::PagedKvCache& cache,
        std::span<kv::PagedKvCache::Seq* const> seqs, Workspace& ws,
        std::span<float> out) const;

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
        /** Optional attention q/k/v biases (qwen2moe); empty spans
         * on arches without them (llama/deepseek2/glm) -> no-op. */
        std::span<const float> wq_bias, wk_bias, wv_bias;
        /** Optional per-head q/k RMS norm weights, head_dim wide
         * (qwen3moe); empty on arches without them. */
        std::span<const float> attn_q_norm, attn_k_norm;

        /** @returns Whether any q/k/v projection bias is present. */
        bool has_qkv_bias() const {
            return !wq_bias.empty() || !wk_bias.empty() ||
                   !wv_bias.empty();
        }

        /** @returns Whether both per-head q and k norms are present
         * (qwen3moe); one without the other is not a known arch. */
        bool has_qk_norm() const {
            return !attn_q_norm.empty() && !attn_k_norm.empty();
        }

        /**
         * @returns Whether q, k and v are separate projections of the
         *     GQA shapes (n_heads * head_dim and n_kv_heads * head_dim
         *     rows). False for dbrx's fused qkv (held in wq, wk and wv
         *     empty) and for MLA layers (no wk / wv).
         */
        bool has_separate_qkv(const Hparams& hp) const {
            const std::uint64_t kv =
                std::uint64_t{hp.n_kv_heads} * hp.head_dim;
            return wq.rows == std::uint64_t{hp.n_heads} * hp.head_dim &&
                   wk.rows == kv && wv.rows == kv;
        }

        /** Dense FFN (unused when the layer is MoE). */
        backend::Mat w_gate, w_up, w_down;
        /** MoE router and expert tensors. */
        backend::Mat gate_inp;
        ExpertMat gate_exps, up_exps, down_exps;
        /** MoE shared experts (deepseek2; fused matrices). */
        backend::Mat gate_shexp, up_shexp, down_shexp;
        /** MoE gated shared expert (qwen2moe): [n_embd->1] logit;
         * empty when the shared expert is ungated (deepseek2). */
        backend::Mat gate_inp_shexp;
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

        /** muse-spark: attention output gate [n_embd -> q_dim]
         * (sigmoid), norm-preserving residual scales (stream alpha,
         * branch beta), latent-MoE projections and norms, and the
         * post-FFN norm weight. Empty on other arches. */
        backend::Mat attn_gate;
        std::span<const float> attn_resid_alpha, attn_resid_beta;
        std::span<const float> ffn_resid_alpha, ffn_resid_beta;
        backend::Mat ffn_latent_down, ffn_latent_up;
        std::span<const float> ffn_pre_exp_norm, ffn_post_exp_norm;
        std::span<const float> post_ffw_norm;

        /** deepseek41: attention sinks, grouped output LoRA, mHC
         * mixes (fn / base / scale per sublayer), the KV compressor
         * (kv-source layers) and the Engram embedding (its layers;
         * engram_table is the shared per_layer_token_embd). The q
         * LoRA, kv projection and indexer reuse wq_a / q_a_norm /
         * wq_b, wkv_a / kv_a_norm and idx_*. */
        std::span<const float> attn_sinks;
        backend::Mat attn_out_a, attn_out_b;
        backend::Mat hc_attn_fn, hc_ffn_fn;
        std::span<const float> hc_attn_base, hc_attn_scale;
        std::span<const float> hc_ffn_base, hc_ffn_scale;
        backend::Mat comp_kv, comp_gate;
        std::span<const float> comp_norm;
        backend::Mat engram_kv, engram_gate, engram_table;

        /** @returns true when this layer routes experts. */
        bool is_moe() const { return gate_inp.rows > 0; }
    };

    const std::vector<Layer>& layers() const { return layers_; }
    /** First absolute layer index this model loaded (0 for a full
     * load; a stage's slice start under slice-only loading). */
    std::uint32_t layer_begin() const { return layer_begin_; }
    /** One past the last absolute layer index this model loaded
     * (n_layers for a full load). */
    std::uint32_t layer_end() const { return layer_end_; }
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
    /** Applies the model's norm (RMS or LayerNorm per hp_.norm_type)
     * to x scaled by w, into out; uses hp_.rms_eps. */
    /** Applies hp_.logit_scale and the final logit softcap in place
     * (single-token path: only custom_layer arches set them). */
    void scale_logits(std::span<float> logits) const;

    void apply_norm(std::span<const float> x,
                    std::span<const float> w, float eps,
                    std::span<float> out) const;

    /** Amortized LM-head projection shared by the batched-forward paths
     * (i#52): normalizes n token hidden states and projects them through
     * out_w_ with one matvec_batch instead of n per-token passes. See
     * the definition for the layout and byte-exactness contract. */
    void project_head_batch(const backend::Ops& op,
                            std::span<const float> x,
                            std::span<float> xbf,
                            std::span<float> logits,
                            std::uint32_t n) const;

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
    /** R16 Q8_K-activation opt-in (see use_q8k_activations); default
     * off keeps the f32 path and stable cross-backend goldens. */
    bool q8k_activations_ = false;
    /** Weights live in a read-only file mapping; gates the
     * LOCUS_WEIGHT_WINDOW DONTNEED policy (see gguf.hpp). */
    bool file_backed_ = false;
    /** R8 madvise readahead runs per token (see weights_stream). */
    bool advise_weights_ = false;
    std::span<const float> rope_factors_;
    backend::Mat embd_;
    std::span<const float> out_norm_;
    backend::Mat out_w_;
    std::vector<Layer> layers_;
    /** Loaded layer range [layer_begin_, layer_end_) under slice-only
     * loading; [0, n_layers) for a normal full load. layers_ is still
     * sized to n_layers and indexed absolutely; entries outside this
     * range are default-constructed and never accessed. */
    std::uint32_t layer_begin_ = 0;
    std::uint32_t layer_end_ = 0;
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
    const Hparams& hp, const TransformerModel::Layer& lay,
    std::span<float> router_logits);

/**
 * Whether a model's weights are expected to stream from storage while
 * it runs, which is when the per-token R8 madvise readahead pays: more
 * than half of RAM, or RAM unknown. A model that fits stays resident
 * after its first pass, and madvise on resident pages is not free (on
 * macOS a 3B model's per-layer hints cost ~4.5 ms per token).
 *
 * @param weight_bytes Total tensor bytes of the model.
 * @param ram_bytes Physical RAM (sys::total_ram_bytes(); 0 = unknown).
 */
bool weights_stream(std::uint64_t weight_bytes, std::uint64_t ram_bytes);

/**
 * @returns How many pool threads model-level parallel loops may use:
 *     the performance-core count on heterogeneous CPUs, else the pool
 *     width; an explicit LOCUS_THREADS overrides it, up to the pool
 *     (1 runs inline).
 */
std::size_t mt_fanout();

/**
 * Row-split matvec across the R9 thread pool: slices w into row
 * ranges dispatched through the SAME backend kernel into
 * disjoint spans of out, so logits are bit-identical for every
 * thread count. LOCUS_THREADS caps the fan-out (default:
 * hardware cores; 1 runs inline). Small matrices stay inline.
 */
void matvec_mt(const backend::Ops& op, const backend::Mat& w,
               std::span<const float> x, std::span<float> out);

/** One matvec of a matvec_mt_group: out = w * x. */
struct MatvecJob {
    const backend::Mat& w;
    std::span<const float> x;
    std::span<float> out;
};

/**
 * Several independent matvecs (q / k / v, gate / up) in ONE thread
 * pool dispatch: their rows are sliced together, each job getting
 * slices in proportion to its rows, so the pool wakes once instead of
 * once per matrix. Each row is computed by the same kernel as
 * matvec_mt, so outputs are bit-identical to issuing the jobs one by
 * one.
 */
void matvec_mt_group(const backend::Ops& op,
                     std::span<const MatvecJob> jobs);

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
