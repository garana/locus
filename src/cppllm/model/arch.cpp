#include "cppllm/model/arch.hpp"

#include <cmath>

#include "cppllm/model/gguf_load.hpp"

namespace cppllm::model {

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
    using namespace cppllm::backend;
    const Hparams& hp = m.hparams();
    const Ops& op = m.active_backend().ops;
    const std::uint32_t hd = hp.head_dim;
    const std::size_t kv_dim =
        static_cast<std::size_t>(hp.n_kv_heads) * hd;
    const std::uint32_t group = hp.n_heads / hp.n_kv_heads;

    float* krow = cache.k(seq, l, pos);
    float* vrow = cache.v(seq, l, pos);
    op.matvec(lay.wq, ws.xb, ws.q);
    op.matvec(lay.wk, ws.xb, {krow, kv_dim});
    op.matvec(lay.wv, ws.xb, {vrow, kv_dim});
    rope_norm(ws.q, hp.n_heads, hd, pos, hp.rope_freq_base,
              m.rope_factors());
    rope_norm({krow, kv_dim}, hp.n_kv_heads, hd, pos,
              hp.rope_freq_base, m.rope_factors());

    for (std::uint32_t h = 0; h < hp.n_heads; ++h) {
        const float* qh =
            ws.q.data() + static_cast<std::size_t>(h) * hd;
        const std::size_t kvh =
            static_cast<std::size_t>(h / group) * hd;
        std::span<float> att(ws.att.data(), pos + 1);
        for (std::uint32_t t = 0; t <= pos; ++t) {
            const float* kt = cache.k(seq, l, t) + kvh;
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
            const float* vt = cache.v(seq, l, t) + kvh;
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
    if (g.get_uint(p + "attention.q_lora_rank").value_or(0) !=
        0) {
        throw gguf::Error("q_lora_rank models not yet supported");
    }
    hp.kv_lora_rank = need_uint(g, p + "attention.kv_lora_rank");
    hp.qk_rope_dim = need_uint(g, p + "rope.dimension_count");
    const std::uint32_t qk =
        need_uint(g, p + "attention.key_length");
    hp.qk_nope_dim = qk - hp.qk_rope_dim;
    hp.v_head_dim = need_uint(g, p + "attention.value_length");
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
    lay.wq = need_mat(g, bp + "attn_q.weight", hp.n_embd,
                      hp.n_heads * qk);
    lay.wkv_a =
        need_mat(g, bp + "attn_kv_a_mqa.weight", hp.n_embd,
                 hp.kv_lora_rank + hp.qk_rope_dim);
    lay.kv_a_norm = need_vec(g, bp + "attn_kv_a_norm.weight",
                             hp.kv_lora_rank);
    lay.wkv_b =
        need_mat(g, bp + "attn_kv_b.weight", hp.kv_lora_rank,
                 hp.n_heads * (hp.qk_nope_dim + hp.v_head_dim));
    lay.wo = need_mat(g, bp + "attn_output.weight",
                      hp.n_heads * hp.v_head_dim, hp.n_embd);
}

void deepseek2_attention(const LlamaModel& m,
                         const LlamaModel::Layer& lay,
                         kv::PagedKvCache& cache,
                         kv::PagedKvCache::Seq& seq,
                         LlamaModel::Workspace& ws,
                         std::uint32_t l, std::uint32_t pos) {
    using namespace cppllm::backend;
    const Hparams& hp = m.hparams();
    const Ops& op = m.active_backend().ops;
    const std::uint32_t rank = hp.kv_lora_rank;
    const std::uint32_t nope = hp.qk_nope_dim;
    const std::uint32_t rope = hp.qk_rope_dim;
    const std::uint32_t vd = hp.v_head_dim;
    const std::uint32_t qk = nope + rope;

    // Latent row: rms-normed c_kv followed by roped shared k_pe.
    op.matvec(lay.wq, ws.xb, ws.q);
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
        const Mat w_uk =
            mat_rows(lay.wkv_b, h * (nope + vd), nope);
        matvec_t(w_uk, {qh, nope}, ws.q_abs);

        std::span<float> att(ws.att.data(), pos + 1);
        for (std::uint32_t t = 0; t <= pos; ++t) {
            const float* ckv = cache.k(seq, l, t);
            float s = 0.0f;
            for (std::uint32_t i = 0; i < rank; ++i) {
                s += ws.q_abs[i] * ckv[i];
            }
            for (std::uint32_t i = 0; i < rope; ++i) {
                s += qh[nope + i] * ckv[rank + i];
            }
            att[t] = s * hp.kq_scale;
        }
        softmax_inplace(att);

        // o_h = W_uv_h (sum_t a_t c_kv(t)).
        std::fill(ws.latent.begin(), ws.latent.end(), 0.0f);
        for (std::uint32_t t = 0; t <= pos; ++t) {
            const float* ckv = cache.k(seq, l, t);
            const float a = att[t];
            for (std::uint32_t i = 0; i < rank; ++i) {
                ws.latent[i] += a * ckv[i];
            }
        }
        const Mat w_uv =
            mat_rows(lay.wkv_b, h * (nope + vd) + nope, vd);
        op.matvec(w_uv, ws.latent,
                  {ws.out.data() +
                       static_cast<std::size_t>(h) * vd,
                   vd});
    }
}

std::uint32_t deepseek2_kv_dim(const Hparams& hp) {
    // One latent row per position; the paged cache's V half goes
    // unused (simplicity > 2x space; still ~9x smaller than
    // caching materialized K/V heads).
    return hp.kv_lora_rank + hp.qk_rope_dim;
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

}  // namespace cppllm::model
