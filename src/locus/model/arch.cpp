#include "locus/model/arch.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "locus/model/gguf_load.hpp"

namespace locus::model {

namespace {

using namespace load_util;
using backend::Mat;
using backend::Ops;

// ----------------------------------------------------------------
// llama (dense or Mixtral-style MoE, GQA attention)
// ----------------------------------------------------------------

void llama_hparams(const gguf::GgufFile&, const std::string&,
                   Hparams& hp) {
    hp.arch = Arch::kLlama;
    if (hp.n_heads == 0 || hp.n_embd % hp.n_heads != 0) {
        throw gguf::Error("invalid head configuration");
    }
    hp.head_dim = hp.n_embd / hp.n_heads;
    if (hp.n_kv_heads == 0 || hp.n_heads % hp.n_kv_heads != 0) {
        throw gguf::Error("invalid kv head configuration");
    }
    hp.n_ff_exp = hp.n_ff;
    hp.expert_weights_norm = true;  // Mixtral renormalizes
    hp.kq_scale =
        1.0f / std::sqrt(static_cast<float>(hp.head_dim));
}

void llama_attention_tensors(const gguf::GgufFile& g,
                             const std::string& bp,
                             const Hparams& hp,
                             LlamaModel::Layer& lay) {
    const std::uint32_t kv_dim = hp.n_kv_heads * hp.head_dim;
    lay.wq = need_mat(g, bp + "attn_q.weight", hp.n_embd,
                      hp.n_embd);
    lay.wk = need_mat(g, bp + "attn_k.weight", hp.n_embd, kv_dim);
    lay.wv = need_mat(g, bp + "attn_v.weight", hp.n_embd, kv_dim);
    lay.wo = need_mat(g, bp + "attn_output.weight", hp.n_embd,
                      hp.n_embd);
}

void llama_attention(const LlamaModel& m,
                     const LlamaModel::Layer& lay,
                     kv::PagedKvCache& cache,
                     kv::PagedKvCache::Seq& seq,
                     LlamaModel::Workspace& ws, std::uint32_t l,
                     std::uint32_t pos) {
    using namespace locus::backend;
    const Hparams& hp = m.hparams();
    const Ops& op = m.active_backend().ops;
    const std::uint32_t hd = hp.head_dim;
    const std::size_t kv_dim =
        static_cast<std::size_t>(hp.n_kv_heads) * hd;
    const std::uint32_t group = hp.n_heads / hp.n_kv_heads;

    op.matvec(lay.wq, ws.xb, ws.q);
    rope_norm(ws.q, hp.n_heads, hd, pos, hp.rope_freq_base,
              m.rope_factors());

    // Quantized KV (DESIGN.md R14): compute this position's K/V into
    // scratch, store it quantized, and dequant the layer's past rows
    // into a transient matrix so the attention math below is
    // arithmetically identical to the F32 zero-copy path.
    std::vector<float> kmat, vmat;
    const float* kbase = nullptr;
    const float* vbase = nullptr;
    std::size_t k_stride = kv_dim;
    if (cache.quantized()) {
        std::vector<float> kbuf(kv_dim), vbuf(kv_dim);
        op.matvec(lay.wk, ws.xb, kbuf);
        op.matvec(lay.wv, ws.xb, vbuf);
        rope_norm(kbuf, hp.n_kv_heads, hd, pos, hp.rope_freq_base,
                  m.rope_factors());
        cache.store_row(seq, l, pos, false, kbuf);
        cache.store_row(seq, l, pos, true, vbuf);
        const std::size_t n = pos + 1;
        kmat.resize(n * kv_dim);
        vmat.resize(n * kv_dim);
        for (std::uint32_t t = 0; t <= pos; ++t) {
            cache.load_row(seq, l, t, false,
                           {kmat.data() + t * kv_dim, kv_dim});
            cache.load_row(seq, l, t, true,
                           {vmat.data() + t * kv_dim, kv_dim});
        }
        kbase = kmat.data();
        vbase = vmat.data();
    } else {
        float* krow = cache.k(seq, l, pos);
        float* vrow = cache.v(seq, l, pos);
        op.matvec(lay.wk, ws.xb, {krow, kv_dim});
        op.matvec(lay.wv, ws.xb, {vrow, kv_dim});
        rope_norm({krow, kv_dim}, hp.n_kv_heads, hd, pos,
                  hp.rope_freq_base, m.rope_factors());
    }
    // In F32 mode rows live in the cache with a fixed per-position
    // stride; a lambda hides the two layouts from the head loops.
    auto krow_at = [&](std::uint32_t t) -> const float* {
        return kbase ? kbase + static_cast<std::size_t>(t) * k_stride
                     : cache.k(seq, l, t);
    };
    auto vrow_at = [&](std::uint32_t t) -> const float* {
        return vbase ? vbase + static_cast<std::size_t>(t) * k_stride
                     : cache.v(seq, l, t);
    };

    for (std::uint32_t h = 0; h < hp.n_heads; ++h) {
        const float* qh =
            ws.q.data() + static_cast<std::size_t>(h) * hd;
        const std::size_t kvh =
            static_cast<std::size_t>(h / group) * hd;
        std::span<float> att(ws.att.data(), pos + 1);
        for (std::uint32_t t = 0; t <= pos; ++t) {
            const float* kt = krow_at(t) + kvh;
            float s = 0.0f;
            for (std::uint32_t i = 0; i < hd; ++i) {
                s += qh[i] * kt[i];
            }
            att[t] = s * hp.kq_scale;
        }
        softmax_inplace(att);
        float* oh =
            ws.out.data() + static_cast<std::size_t>(h) * hd;
        for (std::uint32_t i = 0; i < hd; ++i) {
            oh[i] = 0.0f;
        }
        for (std::uint32_t t = 0; t <= pos; ++t) {
            const float* vt = vrow_at(t) + kvh;
            const float a = att[t];
            for (std::uint32_t i = 0; i < hd; ++i) {
                oh[i] += a * vt[i];
            }
        }
    }
}

std::uint32_t llama_kv_dim(const Hparams& hp) {
    return hp.n_kv_heads * hp.head_dim;
}

// ----------------------------------------------------------------
// deepseek2 (MLA latent attention, DeepSeek MoE)
// ----------------------------------------------------------------

void deepseek2_hparams(const gguf::GgufFile& g,
                       const std::string& p, Hparams& hp) {
    hp.arch = Arch::kDeepseek2;
    hp.q_lora_rank = static_cast<std::uint32_t>(
        g.get_uint(p + "attention.q_lora_rank").value_or(0));
    hp.kv_lora_rank = need_uint(g, p + "attention.kv_lora_rank");
    hp.qk_rope_dim = need_uint(g, p + "rope.dimension_count");
    // Newer conversions store per-head dims in *_mla and repurpose
    // key/value_length for the absorbed sizes; prefer the former.
    const std::uint32_t qk = static_cast<std::uint32_t>(
        g.get_uint(p + "attention.key_length_mla")
            .value_or(need_uint(g, p + "attention.key_length")));
    hp.qk_nope_dim = qk - hp.qk_rope_dim;
    hp.v_head_dim = static_cast<std::uint32_t>(
        g.get_uint(p + "attention.value_length_mla")
            .value_or(
                need_uint(g, p + "attention.value_length")));
    hp.head_dim = qk;
    hp.n_expert_shared = static_cast<std::uint32_t>(
        g.get_uint(p + "expert_shared_count").value_or(0));
    hp.n_dense_lead = static_cast<std::uint32_t>(
        g.get_uint(p + "leading_dense_block_count").value_or(0));
    hp.n_ff_exp = static_cast<std::uint32_t>(
        g.get_uint(p + "expert_feed_forward_length")
            .value_or(hp.n_ff));
    hp.expert_weights_scale =
        get_float(g, p + "expert_weights_scale", 1.0f);
    hp.expert_weights_norm =
        g.get_bool(p + "expert_weights_norm").value_or(false);
    hp.gating = static_cast<GatingFunc>(
        g.get_uint(p + "expert_gating_func")
            .value_or(static_cast<std::uint64_t>(
                GatingFunc::kSoftmax)));
    hp.n_group = static_cast<std::uint32_t>(
        g.get_uint(p + "expert_group_count").value_or(1));
    hp.n_group_used = static_cast<std::uint32_t>(
        g.get_uint(p + "expert_group_used_count").value_or(1));
    if (hp.n_group == 0 || hp.n_group_used == 0 ||
        hp.n_group_used > hp.n_group ||
        (hp.n_expert > 0 && hp.n_expert % hp.n_group != 0)) {
        throw gguf::Error("invalid expert group configuration");
    }

    // YARN: frequencies are corrected; the cos/sin mscale cancels
    // for DeepSeek (mscale == mscale_all_dim) and the attention
    // scale carries mscale^2 instead.
    const float factor =
        get_float(g, p + "rope.scaling.factor", 1.0f);
    const float log_mul = get_float(
        g, p + "rope.scaling.yarn_log_multiplier", 0.0f);
    if (g.get_string(p + "rope.scaling.type") == "yarn" &&
        factor > 0.0f) {
        hp.yarn.freq_scale = 1.0f / factor;
        hp.yarn.n_ctx_orig = static_cast<std::uint32_t>(
            g.get_uint(p + "rope.scaling.original_context_length")
                .value_or(hp.n_ctx));
        const float mscale = 1.0f + log_mul * std::log(factor);
        hp.kq_scale =
            mscale * mscale / std::sqrt(static_cast<float>(qk));
    } else {
        hp.kq_scale = 1.0f / std::sqrt(static_cast<float>(qk));
    }
}

void deepseek2_attention_tensors(const gguf::GgufFile& g,
                                 const std::string& bp,
                                 const Hparams& hp,
                                 LlamaModel::Layer& lay) {
    const std::uint32_t qk = hp.qk_nope_dim + hp.qk_rope_dim;
    if (hp.q_lora_rank > 0) {
        lay.wq_a = need_mat(g, bp + "attn_q_a.weight", hp.n_embd,
                            hp.q_lora_rank);
        lay.q_a_norm = need_vec(g, bp + "attn_q_a_norm.weight",
                                hp.q_lora_rank);
        lay.wq_b = need_mat(g, bp + "attn_q_b.weight",
                            hp.q_lora_rank, hp.n_heads * qk);
    } else {
        lay.wq = need_mat(g, bp + "attn_q.weight", hp.n_embd,
                          hp.n_heads * qk);
    }
    lay.wkv_a =
        need_mat(g, bp + "attn_kv_a_mqa.weight", hp.n_embd,
                 hp.kv_lora_rank + hp.qk_rope_dim);
    lay.kv_a_norm = need_vec(g, bp + "attn_kv_a_norm.weight",
                             hp.kv_lora_rank);
    if (g.find_tensor(bp + "attn_kv_b.weight") != nullptr) {
        lay.wkv_b = need_mat(
            g, bp + "attn_kv_b.weight", hp.kv_lora_rank,
            hp.n_heads * (hp.qk_nope_dim + hp.v_head_dim));
    } else {
        // MLA-cache-era split: 3-D per-head tensors, with k_b
        // stored transposed ([rank x nope] per head) so
        // absorption is a plain matvec; v_b matches the fused
        // W_uv slices. Heads are outermost, so flattening ne[1]
        // and ne[2] into rows preserves the memory layout.
        lay.wk_b = need_mat3(g, bp + "attn_k_b.weight",
                             hp.qk_nope_dim, hp.kv_lora_rank,
                             hp.n_heads)
                       .base;
        lay.wk_b.rows = hp.n_heads * hp.kv_lora_rank;
        lay.wv_b = need_mat3(g, bp + "attn_v_b.weight",
                             hp.kv_lora_rank, hp.v_head_dim,
                             hp.n_heads)
                       .base;
        lay.wv_b.rows = hp.n_heads * hp.v_head_dim;
    }
    lay.wo = need_mat(g, bp + "attn_output.weight",
                      hp.n_heads * hp.v_head_dim, hp.n_embd);
}

/**
 * MLA attention over the cached latents. When sel is non-null
 * only the n_att positions it lists (ascending) are attended --
 * the DSA path; a null sel attends all n_att = pos + 1 cached
 * positions, which is the dense case and bit-identical to the
 * pre-DSA implementation.
 */
void mla_attention(const LlamaModel& m,
                   const LlamaModel::Layer& lay,
                   kv::PagedKvCache& cache,
                   kv::PagedKvCache::Seq& seq,
                   LlamaModel::Workspace& ws, std::uint32_t l,
                   std::uint32_t pos, const std::uint32_t* sel,
                   std::uint32_t n_att) {
    using namespace locus::backend;
    const Hparams& hp = m.hparams();
    const Ops& op = m.active_backend().ops;
    const std::uint32_t rank = hp.kv_lora_rank;
    const std::uint32_t nope = hp.qk_nope_dim;
    const std::uint32_t rope = hp.qk_rope_dim;
    const std::uint32_t vd = hp.v_head_dim;
    const std::uint32_t qk = nope + rope;

    // Query: direct, or through the compressed q_lora path.
    if (hp.q_lora_rank > 0) {
        op.matvec(lay.wq_a, ws.xb, ws.q_a);
        rmsnorm(ws.q_a, lay.q_a_norm, hp.rms_eps, ws.q_a);
        matvec_mt(op, lay.wq_b, ws.q_a, ws.q);
    } else {
        matvec_mt(op, lay.wq, ws.xb, ws.q);
    }
    // Latent row: rms-normed c_kv followed by roped shared k_pe.
    op.matvec(lay.wkv_a, ws.xb, ws.kv_a);
    float* row = cache.k(seq, l, pos);
    rmsnorm({ws.kv_a.data(), rank}, lay.kv_a_norm, hp.rms_eps,
            {row, rank});
    std::copy(ws.kv_a.begin() + rank, ws.kv_a.end(), row + rank);
    rope_norm_yarn({row + rank, rope}, 1, rope, pos,
                   hp.rope_freq_base, hp.yarn);

    for (std::uint32_t h = 0; h < hp.n_heads; ++h) {
        float* qh =
            ws.q.data() + static_cast<std::size_t>(h) * qk;
        rope_norm_yarn({qh + nope, rope}, 1, rope, pos,
                       hp.rope_freq_base, hp.yarn);

        // Weight absorption: q_abs = W_uk_h^T q_nope, so scores
        // are dot products against the cached latents directly.
        // Split-form k_b is already transposed per head.
        if (lay.wk_b.rows > 0) {
            const Mat k_b = mat_rows(lay.wk_b, h * rank, rank);
            op.matvec(k_b, {qh, nope}, ws.q_abs);
        } else {
            const Mat w_uk =
                mat_rows(lay.wkv_b, h * (nope + vd), nope);
            matvec_t(w_uk, {qh, nope}, ws.q_abs);
        }

        std::span<float> att(ws.att.data(), n_att);
        for (std::uint32_t j = 0; j < n_att; ++j) {
            const std::uint32_t t = sel != nullptr ? sel[j] : j;
            const float* ckv = cache.k(seq, l, t);
            float s = 0.0f;
            for (std::uint32_t i = 0; i < rank; ++i) {
                s += ws.q_abs[i] * ckv[i];
            }
            for (std::uint32_t i = 0; i < rope; ++i) {
                s += qh[nope + i] * ckv[rank + i];
            }
            att[j] = s * hp.kq_scale;
        }
        softmax_inplace(att);

        // o_h = W_uv_h (sum_t a_t c_kv(t)).
        std::fill(ws.latent.begin(), ws.latent.end(), 0.0f);
        for (std::uint32_t j = 0; j < n_att; ++j) {
            const std::uint32_t t = sel != nullptr ? sel[j] : j;
            const float* ckv = cache.k(seq, l, t);
            const float a = att[j];
            for (std::uint32_t i = 0; i < rank; ++i) {
                ws.latent[i] += a * ckv[i];
            }
        }
        const Mat w_uv =
            lay.wv_b.rows > 0
                ? mat_rows(lay.wv_b, h * vd, vd)
                : mat_rows(lay.wkv_b, h * (nope + vd) + nope,
                           vd);
        op.matvec(w_uv, ws.latent,
                  {ws.out.data() +
                       static_cast<std::size_t>(h) * vd,
                   vd});
    }
}

void deepseek2_attention(const LlamaModel& m,
                         const LlamaModel::Layer& lay,
                         kv::PagedKvCache& cache,
                         kv::PagedKvCache::Seq& seq,
                         LlamaModel::Workspace& ws,
                         std::uint32_t l, std::uint32_t pos) {
    mla_attention(m, lay, cache, seq, ws, l, pos, nullptr,
                  pos + 1);
}

std::uint32_t deepseek2_kv_dim(const Hparams& hp) {
    // One latent row per position; the paged cache's V half goes
    // unused (simplicity > 2x space; still ~9x smaller than
    // caching materialized K/V heads).
    return hp.kv_lora_rank + hp.qk_rope_dim;
}

/**
 * GLM-5.2 (DSA): MLA with q_lora + split k_b/v_b and V3-style
 * sigmoid gating -- computationally the deepseek2 stack -- plus
 * the DSA lightning indexer, which picks the top_k cached
 * positions the MLA attention sees. For contexts <= top_k the
 * selection keeps everything and DSA equals dense MLA, which is
 * what the llama.cpp golden anchors.
 */
void glm_dsa_hparams(const gguf::GgufFile& g,
                     const std::string& p, Hparams& hp) {
    deepseek2_hparams(g, p, hp);
    // The trailing MTP block(s) are not part of the causal
    // stack: llama.cpp likewise loads and ignores them.
    const std::uint32_t nextn = static_cast<std::uint32_t>(
        g.get_uint(p + "nextn_predict_layers").value_or(0));
    if (nextn >= hp.n_layers) {
        throw gguf::Error("invalid nextn_predict_layers");
    }
    hp.n_layers -= nextn;
    hp.idx_heads = static_cast<std::uint32_t>(
        g.get_uint(p + "attention.indexer.head_count")
            .value_or(0));
    hp.idx_dim = static_cast<std::uint32_t>(
        g.get_uint(p + "attention.indexer.key_length")
            .value_or(0));
    hp.idx_top_k = static_cast<std::uint32_t>(
        g.get_uint(p + "attention.indexer.top_k")
            .value_or(2048));
    if (hp.idx_heads == 0 || hp.idx_dim == 0 ||
        hp.q_lora_rank == 0) {
        // No indexer metadata: dense-equivalent DSA only, so
        // the usable context stays capped at top_k.
        hp.n_ctx = std::min(hp.n_ctx, hp.idx_top_k);
        hp.idx_heads = 0;
        hp.idx_dim = 0;
        hp.idx_top_k = 0;
    }
}

void glm_dsa_attention_tensors(const gguf::GgufFile& g,
                               const std::string& bp,
                               const Hparams& hp,
                               LlamaModel::Layer& lay) {
    deepseek2_attention_tensors(g, bp, hp, lay);
    if (hp.idx_heads == 0) {
        return;
    }
    lay.idx_proj = need_mat(g, bp + "indexer.proj.weight",
                            hp.n_embd, hp.idx_heads);
    lay.idx_k = need_mat(g, bp + "indexer.attn_k.weight",
                         hp.n_embd, hp.idx_dim);
    lay.idx_q_b =
        need_mat(g, bp + "indexer.attn_q_b.weight",
                 hp.q_lora_rank, hp.idx_heads * hp.idx_dim);
    lay.idx_k_norm = need_vec(g, bp + "indexer.k_norm.weight",
                              hp.idx_dim);
    lay.idx_k_norm_b = need_vec(g, bp + "indexer.k_norm.bias",
                                hp.idx_dim);
}

/** LayerNorm with bias, in place (the indexer k norm). */
void layernorm_inplace(std::span<float> x,
                       std::span<const float> w,
                       std::span<const float> b, float eps) {
    const std::size_t n = x.size();
    float mean = 0.0f;
    for (float v : x) {
        mean += v;
    }
    mean /= static_cast<float>(n);
    float var = 0.0f;
    for (float v : x) {
        var += (v - mean) * (v - mean);
    }
    var /= static_cast<float>(n);
    const float inv = 1.0f / std::sqrt(var + eps);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

/**
 * Half-split (non-interleaved) rope on the first d dims of x:
 * pair (i, i + d/2). The DSA reference is explicit that the
 * indexer's rope is NOT interleaved, unlike the main MLA rope.
 */
void rope_half(float* x, std::uint32_t d, std::uint32_t pos,
               float base) {
    const std::uint32_t half = d / 2;
    for (std::uint32_t i = 0; i < half; ++i) {
        const float freq = std::pow(
            base, -2.0f * static_cast<float>(i) /
                      static_cast<float>(d));
        const float a = static_cast<float>(pos) * freq;
        const float c = std::cos(a);
        const float s = std::sin(a);
        const float x0 = x[i];
        const float x1 = x[i + half];
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

void glm_dsa_attention(const LlamaModel& m,
                       const LlamaModel::Layer& lay,
                       kv::PagedKvCache& cache,
                       kv::PagedKvCache::Seq& seq,
                       LlamaModel::Workspace& ws, std::uint32_t l,
                       std::uint32_t pos) {
    using namespace locus::backend;
    const Hparams& hp = m.hparams();
    const Ops& op = m.active_backend().ops;
    const std::uint32_t n_pos = pos + 1;
    const std::uint32_t* sel = nullptr;
    std::uint32_t n_att = n_pos;

    if (hp.idx_heads > 0) {
        const std::uint32_t off =
            hp.kv_lora_rank + hp.qk_rope_dim;
        const std::uint32_t d = hp.idx_dim;
        // Compute and cache this position's indexer key:
        // layernorm(W_k x), roped half-split on the rope dims.
        float* krow = cache.k(seq, l, pos) + off;
        op.matvec(lay.idx_k, ws.xb, {krow, d});
        layernorm_inplace({krow, d}, lay.idx_k_norm,
                          lay.idx_k_norm_b, hp.rms_eps);
        rope_half(krow, hp.qk_rope_dim, pos, hp.rope_freq_base);

        if (n_pos > hp.idx_top_k) {
            // Indexer queries share the MLA q latent path.
            op.matvec(lay.wq_a, ws.xb, ws.q_a);
            rmsnorm(ws.q_a, lay.q_a_norm, hp.rms_eps, ws.q_a);
            matvec_mt(op, lay.idx_q_b, ws.q_a, ws.idx_q);
            for (std::uint32_t h = 0; h < hp.idx_heads; ++h) {
                rope_half(ws.idx_q.data() +
                              static_cast<std::size_t>(h) * d,
                          hp.qk_rope_dim, pos,
                          hp.rope_freq_base);
            }
            op.matvec(lay.idx_proj, ws.xb, ws.idx_w);
            for (std::uint32_t t = 0; t < n_pos; ++t) {
                ws.idx_scores[t] = dsa_index_score(
                    ws.idx_q, ws.idx_w,
                    {cache.k(seq, l, t) + off, d});
            }
            ws.idx_sel.resize(n_pos);
            for (std::uint32_t t = 0; t < n_pos; ++t) {
                ws.idx_sel[t] = t;
            }
            std::nth_element(
                ws.idx_sel.begin(),
                ws.idx_sel.begin() + hp.idx_top_k,
                ws.idx_sel.end(),
                [&](std::uint32_t a, std::uint32_t b) {
                    return ws.idx_scores[a] > ws.idx_scores[b];
                });
            ws.idx_sel.resize(hp.idx_top_k);
            std::sort(ws.idx_sel.begin(), ws.idx_sel.end());
            sel = ws.idx_sel.data();
            n_att = hp.idx_top_k;
        }
    }
    mla_attention(m, lay, cache, seq, ws, l, pos, sel, n_att);
}

std::uint32_t glm_dsa_kv_dim(const Hparams& hp) {
    // MLA latent row plus the cached indexer key.
    return hp.kv_lora_rank + hp.qk_rope_dim + hp.idx_dim;
}

// ---- qwen2moe (Qwen1.5/2 MoE): GQA + q/k/v bias + gated shared
// expert. Reuses the llama GQA attention math (a clone with bias
// adds, so llama_attention stays untouched) and the shared MoE
// forward (gated shared expert via Layer.gate_inp_shexp, wired in
// model/llama.cpp). ----

void qwen2moe_hparams(const gguf::GgufFile& g, const std::string&,
                      Hparams& hp) {
    hp.arch = Arch::kLlama;  // GQA attention + standard MoE forward
    if (hp.n_heads == 0 || hp.n_embd % hp.n_heads != 0) {
        throw gguf::Error("invalid head configuration");
    }
    hp.head_dim = hp.n_embd / hp.n_heads;
    if (hp.n_kv_heads == 0 || hp.n_heads % hp.n_kv_heads != 0) {
        throw gguf::Error("invalid kv head configuration");
    }
    // Routed / shared expert widths are absent from this GGUF's
    // metadata (only feed_forward_length); take them from the expert
    // tensor shapes (ne[1]) so the loader's shape checks pass.
    const auto* ge = g.find_tensor("blk.0.ffn_gate_exps.weight");
    const auto* gs = g.find_tensor("blk.0.ffn_gate_shexp.weight");
    if (ge == nullptr || gs == nullptr) {
        throw gguf::Error("qwen2moe: missing expert tensors");
    }
    hp.n_ff_exp = static_cast<std::uint32_t>(ge->ne[1]);    // routed
    hp.n_ff_shexp = static_cast<std::uint32_t>(gs->ne[1]);  // shared
    hp.n_expert_shared = 1;  // one gated shared expert
    hp.expert_weights_norm = false;  // Qwen1.5-MoE: raw softmax probs
                                     // (norm_topk_prob=false)
    hp.kq_scale =
        1.0f / std::sqrt(static_cast<float>(hp.head_dim));
}

void qwen2moe_attention_tensors(const gguf::GgufFile& g,
                                const std::string& bp,
                                const Hparams& hp,
                                LlamaModel::Layer& lay) {
    const std::uint32_t kv_dim = hp.n_kv_heads * hp.head_dim;
    lay.wq = need_mat(g, bp + "attn_q.weight", hp.n_embd, hp.n_embd);
    lay.wk = need_mat(g, bp + "attn_k.weight", hp.n_embd, kv_dim);
    lay.wv = need_mat(g, bp + "attn_v.weight", hp.n_embd, kv_dim);
    lay.wo = need_mat(g, bp + "attn_output.weight", hp.n_embd,
                      hp.n_embd);
    // Qwen2 carries q/k/v projection bias (no output bias).
    lay.wq_bias = need_vec(g, bp + "attn_q.bias", hp.n_embd);
    lay.wk_bias = need_vec(g, bp + "attn_k.bias", kv_dim);
    lay.wv_bias = need_vec(g, bp + "attn_v.bias", kv_dim);
}

// Clone of llama_attention with q/k/v projection bias added right
// after each projection (before RoPE), matching Qwen2. llama_attention
// itself is left untouched.
void qwen2moe_attention(const LlamaModel& m,
                        const LlamaModel::Layer& lay,
                        kv::PagedKvCache& cache,
                        kv::PagedKvCache::Seq& seq,
                        LlamaModel::Workspace& ws, std::uint32_t l,
                        std::uint32_t pos) {
    using namespace locus::backend;
    const Hparams& hp = m.hparams();
    const Ops& op = m.active_backend().ops;
    const std::uint32_t hd = hp.head_dim;
    const std::size_t kv_dim =
        static_cast<std::size_t>(hp.n_kv_heads) * hd;
    const std::uint32_t group = hp.n_heads / hp.n_kv_heads;
    auto add_bias = [](std::span<float> d,
                       std::span<const float> b) {
        if (!b.empty()) {
            for (std::size_t i = 0; i < d.size(); ++i) {
                d[i] += b[i];
            }
        }
    };

    op.matvec(lay.wq, ws.xb, ws.q);
    add_bias({ws.q.data(), static_cast<std::size_t>(hp.n_embd)},
             lay.wq_bias);
    rope_neox_yarn(ws.q, hp.n_heads, hd, pos, hp.rope_freq_base,
                   Yarn{});  // Qwen2 uses NEOX rope, not interleaved

    std::vector<float> kmat, vmat;
    const float* kbase = nullptr;
    const float* vbase = nullptr;
    std::size_t k_stride = kv_dim;
    if (cache.quantized()) {
        std::vector<float> kbuf(kv_dim), vbuf(kv_dim);
        op.matvec(lay.wk, ws.xb, kbuf);
        op.matvec(lay.wv, ws.xb, vbuf);
        add_bias(kbuf, lay.wk_bias);
        add_bias(vbuf, lay.wv_bias);
        rope_neox_yarn(kbuf, hp.n_kv_heads, hd, pos,
                       hp.rope_freq_base, Yarn{});
        cache.store_row(seq, l, pos, false, kbuf);
        cache.store_row(seq, l, pos, true, vbuf);
        const std::size_t n = pos + 1;
        kmat.resize(n * kv_dim);
        vmat.resize(n * kv_dim);
        for (std::uint32_t t = 0; t <= pos; ++t) {
            cache.load_row(seq, l, t, false,
                           {kmat.data() + t * kv_dim, kv_dim});
            cache.load_row(seq, l, t, true,
                           {vmat.data() + t * kv_dim, kv_dim});
        }
        kbase = kmat.data();
        vbase = vmat.data();
    } else {
        float* krow = cache.k(seq, l, pos);
        float* vrow = cache.v(seq, l, pos);
        op.matvec(lay.wk, ws.xb, {krow, kv_dim});
        op.matvec(lay.wv, ws.xb, {vrow, kv_dim});
        add_bias({krow, kv_dim}, lay.wk_bias);
        add_bias({vrow, kv_dim}, lay.wv_bias);
        rope_neox_yarn({krow, kv_dim}, hp.n_kv_heads, hd, pos,
                       hp.rope_freq_base, Yarn{});
    }
    auto krow_at = [&](std::uint32_t t) -> const float* {
        return kbase ? kbase + static_cast<std::size_t>(t) * k_stride
                     : cache.k(seq, l, t);
    };
    auto vrow_at = [&](std::uint32_t t) -> const float* {
        return vbase ? vbase + static_cast<std::size_t>(t) * k_stride
                     : cache.v(seq, l, t);
    };

    for (std::uint32_t h = 0; h < hp.n_heads; ++h) {
        const float* qh =
            ws.q.data() + static_cast<std::size_t>(h) * hd;
        const std::size_t kvh =
            static_cast<std::size_t>(h / group) * hd;
        std::span<float> att(ws.att.data(), pos + 1);
        for (std::uint32_t t = 0; t <= pos; ++t) {
            const float* kt = krow_at(t) + kvh;
            float s = 0.0f;
            for (std::uint32_t i = 0; i < hd; ++i) {
                s += qh[i] * kt[i];
            }
            att[t] = s * hp.kq_scale;
        }
        softmax_inplace(att);
        float* oh =
            ws.out.data() + static_cast<std::size_t>(h) * hd;
        for (std::uint32_t i = 0; i < hd; ++i) {
            oh[i] = 0.0f;
        }
        for (std::uint32_t t = 0; t <= pos; ++t) {
            const float* vt = vrow_at(t) + kvh;
            const float a = att[t];
            for (std::uint32_t i = 0; i < hd; ++i) {
                oh[i] += a * vt[i];
            }
        }
    }
}

// ---- dbrx (Databricks DBRX): GQA with a FUSED qkv projection +
// clip_qkv clamp, LayerNorm (weight-only, via the norm-type seam),
// and fine-grained MoE (16 experts / 4 active, no shared expert).
// Attention is a new fn (fused split + clamp + NEOX rope); the norm
// dispatch + gated-shexp-less MoE are the shared forward. ----

void dbrx_hparams(const gguf::GgufFile& g, const std::string& p,
                  Hparams& hp) {
    hp.arch = Arch::kLlama;  // GQA attention + standard MoE forward
    if (hp.n_heads == 0 || hp.n_embd % hp.n_heads != 0) {
        throw gguf::Error("invalid head configuration");
    }
    hp.head_dim = hp.n_embd / hp.n_heads;
    if (hp.n_kv_heads == 0 || hp.n_heads % hp.n_kv_heads != 0) {
        throw gguf::Error("invalid kv head configuration");
    }
    hp.n_ff_exp = hp.n_ff;         // expert width = feed_forward_length
    hp.n_expert_shared = 0;        // no shared expert
    hp.expert_weights_norm = true;  // DBRX renormalizes the top-k
    hp.norm_type = NormType::kLayerNorm;  // DBRX uses LayerNorm
    hp.rms_eps =
        get_float(g, p + "attention.layer_norm_epsilon", 1e-5f);
    hp.kq_scale =
        1.0f / std::sqrt(static_cast<float>(hp.head_dim));
}

void dbrx_attention_tensors(const gguf::GgufFile& g,
                            const std::string& bp,
                            const Hparams& hp,
                            LlamaModel::Layer& lay) {
    const std::uint32_t kv_dim = hp.n_kv_heads * hp.head_dim;
    // Fused Wqkv [n_embd -> n_embd + 2*kv_dim]; held in wq, split at
    // compute time (wk/wv stay empty). No projection bias.
    lay.wq = need_mat(g, bp + "attn_qkv.weight", hp.n_embd,
                      hp.n_embd + 2 * kv_dim);
    lay.wo = need_mat(g, bp + "attn_output.weight", hp.n_embd,
                      hp.n_embd);
    // DBRX names the pre-MoE norm attn_output_norm (vs ffn_norm);
    // attn_norm + output_norm load generically by common name.
    lay.ffn_norm =
        need_vec(g, bp + "attn_output_norm.weight", hp.n_embd);
}

// Fused-QKV GQA attention with clip_qkv clamp and NEOX rope. One
// matvec produces [q|k|v], clamped to +/-8 (DBRX clamp_kqv), then
// split; the rest mirrors llama_attention. llama_attention untouched.
void dbrx_attention(const LlamaModel& m,
                    const LlamaModel::Layer& lay,
                    kv::PagedKvCache& cache,
                    kv::PagedKvCache::Seq& seq,
                    LlamaModel::Workspace& ws, std::uint32_t l,
                    std::uint32_t pos) {
    using namespace locus::backend;
    const Hparams& hp = m.hparams();
    const Ops& op = m.active_backend().ops;
    const std::uint32_t hd = hp.head_dim;
    const std::size_t kv_dim =
        static_cast<std::size_t>(hp.n_kv_heads) * hd;
    const std::size_t q_dim =
        static_cast<std::size_t>(hp.n_heads) * hd;
    const std::uint32_t group = hp.n_heads / hp.n_kv_heads;
    constexpr float kClamp = 8.0f;  // DBRX clamp_kqv (gguf-confirmed)
    const float rb = hp.rope_freq_base;

    std::vector<float> qkv(q_dim + 2 * kv_dim);
    op.matvec(lay.wq, ws.xb, qkv);  // fused q|k|v
    for (float& v : qkv) {
        v = std::clamp(v, -kClamp, kClamp);
    }
    std::copy(qkv.begin(), qkv.begin() + q_dim, ws.q.begin());
    rope_neox_yarn(ws.q, hp.n_heads, hd, pos, rb, Yarn{});
    const float* ksrc = qkv.data() + q_dim;
    const float* vsrc = qkv.data() + q_dim + kv_dim;

    std::vector<float> kmat, vmat;
    const float* kbase = nullptr;
    const float* vbase = nullptr;
    const std::size_t k_stride = kv_dim;
    if (cache.quantized()) {
        std::vector<float> kbuf(ksrc, ksrc + kv_dim);
        std::vector<float> vbuf(vsrc, vsrc + kv_dim);
        rope_neox_yarn(kbuf, hp.n_kv_heads, hd, pos, rb, Yarn{});
        cache.store_row(seq, l, pos, false, kbuf);
        cache.store_row(seq, l, pos, true, vbuf);
        const std::size_t n = pos + 1;
        kmat.resize(n * kv_dim);
        vmat.resize(n * kv_dim);
        for (std::uint32_t t = 0; t <= pos; ++t) {
            cache.load_row(seq, l, t, false,
                           {kmat.data() + t * kv_dim, kv_dim});
            cache.load_row(seq, l, t, true,
                           {vmat.data() + t * kv_dim, kv_dim});
        }
        kbase = kmat.data();
        vbase = vmat.data();
    } else {
        float* krow = cache.k(seq, l, pos);
        float* vrow = cache.v(seq, l, pos);
        std::copy(ksrc, ksrc + kv_dim, krow);
        std::copy(vsrc, vsrc + kv_dim, vrow);
        rope_neox_yarn({krow, kv_dim}, hp.n_kv_heads, hd, pos, rb,
                       Yarn{});
    }
    auto krow_at = [&](std::uint32_t t) -> const float* {
        return kbase ? kbase + static_cast<std::size_t>(t) * k_stride
                     : cache.k(seq, l, t);
    };
    auto vrow_at = [&](std::uint32_t t) -> const float* {
        return vbase ? vbase + static_cast<std::size_t>(t) * k_stride
                     : cache.v(seq, l, t);
    };

    for (std::uint32_t h = 0; h < hp.n_heads; ++h) {
        const float* qh =
            ws.q.data() + static_cast<std::size_t>(h) * hd;
        const std::size_t kvh =
            static_cast<std::size_t>(h / group) * hd;
        std::span<float> att(ws.att.data(), pos + 1);
        for (std::uint32_t t = 0; t <= pos; ++t) {
            const float* kt = krow_at(t) + kvh;
            float s = 0.0f;
            for (std::uint32_t i = 0; i < hd; ++i) {
                s += qh[i] * kt[i];
            }
            att[t] = s * hp.kq_scale;
        }
        softmax_inplace(att);
        float* oh =
            ws.out.data() + static_cast<std::size_t>(h) * hd;
        for (std::uint32_t i = 0; i < hd; ++i) {
            oh[i] = 0.0f;
        }
        for (std::uint32_t t = 0; t <= pos; ++t) {
            const float* vt = vrow_at(t) + kvh;
            const float a = att[t];
            for (std::uint32_t i = 0; i < hd; ++i) {
                oh[i] += a * vt[i];
            }
        }
    }
}

const ArchSpec kArchs[] = {
    {"llama",
     "Llama family (dense or Mixtral-style MoE, GQA)",
     &llama_hparams, &llama_attention_tensors, &llama_attention,
     &llama_kv_dim},
    {"deepseek2",
     "DeepSeek-V2 family (MLA latent attention, DeepSeek MoE)",
     &deepseek2_hparams, &deepseek2_attention_tensors,
     &deepseek2_attention, &deepseek2_kv_dim},
    {"glm-dsa",
     "GLM-5.2 family (MLA + DeepSeek MoE + DSA lightning "
     "indexer beyond top_k tokens)",
     &glm_dsa_hparams, &glm_dsa_attention_tensors,
     &glm_dsa_attention, &glm_dsa_kv_dim},
    {"qwen2moe",
     "Qwen1.5/2 MoE (GQA + q/k/v bias, routed + gated shared "
     "expert)",
     &qwen2moe_hparams, &qwen2moe_attention_tensors,
     &qwen2moe_attention, &llama_kv_dim},
    {"dbrx",
     "DBRX (GQA, fused QKV + clip_qkv, LayerNorm, fine-grained "
     "MoE 16x/4)",
     &dbrx_hparams, &dbrx_attention_tensors, &dbrx_attention,
     &llama_kv_dim},
};

}  // namespace

std::span<const ArchSpec> archs() { return kArchs; }

const ArchSpec* find_arch(std::string_view name) {
    for (const ArchSpec& a : kArchs) {
        if (a.name == name) {
            return &a;
        }
    }
    return nullptr;
}

}  // namespace locus::model
