#include "cppllm/model/llama.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "cppllm/backend/vulkan_forward.hpp"

namespace cppllm::model {

namespace {

using backend::Mat;
using gguf::Error;
using gguf::GgufFile;
using gguf::TensorInfo;

std::uint32_t need_uint(const GgufFile& g, const std::string& key) {
    auto v = g.get_uint(key);
    if (!v) {
        throw Error("missing metadata: " + key);
    }
    return static_cast<std::uint32_t>(*v);
}

float get_float(const GgufFile& g, const std::string& key,
                float fallback) {
    const gguf::Value* v = g.find(key);
    if (v == nullptr) {
        return fallback;
    }
    if (v->type != gguf::ValueType::kFloat32 &&
        v->type != gguf::ValueType::kFloat64) {
        throw Error(key + " is not a float");
    }
    return static_cast<float>(v->f);
}

/** Fetches a tensor as a Mat, enforcing its 2-D shape. */
Mat need_mat(const GgufFile& g, const std::string& name,
             std::uint32_t cols, std::uint32_t rows) {
    const TensorInfo* t = g.find_tensor(name);
    if (t == nullptr) {
        throw Error("missing tensor: " + name);
    }
    if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != 1 ||
        t->ne[3] != 1) {
        throw Error("unexpected shape for tensor: " + name);
    }
    return Mat{t->type, g.tensor_data(*t).data(), rows, cols};
}

/** Fetches a 3-D expert tensor, enforcing its shape. */
LlamaModel::ExpertMat need_mat3(const GgufFile& g,
                                const std::string& name,
                                std::uint32_t cols,
                                std::uint32_t rows,
                                std::uint32_t n_expert) {
    const TensorInfo* t = g.find_tensor(name);
    if (t == nullptr) {
        throw Error("missing tensor: " + name);
    }
    if (t->ne[0] != cols || t->ne[1] != rows ||
        t->ne[2] != n_expert || t->ne[3] != 1) {
        throw Error("unexpected shape for tensor: " + name);
    }
    LlamaModel::ExpertMat em;
    em.base = Mat{t->type, g.tensor_data(*t).data(), rows, cols};
    em.expert_bytes = t->nbytes / n_expert;
    return em;
}

/** Fetches a 1-D F32 tensor (norm weights). */
std::span<const float> need_vec(const GgufFile& g,
                                const std::string& name,
                                std::uint32_t len) {
    const TensorInfo* t = g.find_tensor(name);
    if (t == nullptr) {
        throw Error("missing tensor: " + name);
    }
    if (t->type != gguf::TensorType::kF32 || t->ne[0] != len ||
        t->nelements() != len) {
        throw Error("norm tensor must be F32 1-D: " + name);
    }
    return {reinterpret_cast<const float*>(g.tensor_data(*t).data()),
            len};
}

}  // namespace

LlamaModel LlamaModel::load(const GgufFile& g) {
    const auto arch_name = g.get_string("general.architecture");
    LlamaModel m;
    Hparams& hp = m.hp_;
    std::string p;
    if (arch_name == "llama") {
        hp.arch = Arch::kLlama;
        p = "llama.";
    } else if (arch_name == "deepseek2") {
        hp.arch = Arch::kDeepseek2;
        p = "deepseek2.";
    } else {
        throw Error("unsupported general.architecture");
    }

    hp.n_embd = need_uint(g, p + "embedding_length");
    hp.n_layers = need_uint(g, p + "block_count");
    hp.n_heads = need_uint(g, p + "attention.head_count");
    hp.n_kv_heads = static_cast<std::uint32_t>(
        g.get_uint(p + "attention.head_count_kv")
            .value_or(hp.n_heads));
    hp.n_ff = need_uint(g, p + "feed_forward_length");
    hp.n_ctx = need_uint(g, p + "context_length");
    hp.n_expert = static_cast<std::uint32_t>(
        g.get_uint(p + "expert_count").value_or(0));
    hp.n_expert_used = static_cast<std::uint32_t>(
        g.get_uint(p + "expert_used_count").value_or(0));
    if (hp.n_expert > 0 &&
        (hp.n_expert_used == 0 ||
         hp.n_expert_used > hp.n_expert)) {
        throw Error("invalid expert_used_count");
    }
    hp.rms_eps = get_float(
        g, p + "attention.layer_norm_rms_epsilon", 1e-5f);
    hp.rope_freq_base =
        get_float(g, p + "rope.freq_base", 10000.0f);

    if (hp.arch == Arch::kDeepseek2) {
        if (g.get_uint(p + "attention.q_lora_rank")
                .value_or(0) != 0) {
            throw Error(
                "q_lora_rank models not yet supported");
        }
        hp.kv_lora_rank =
            need_uint(g, p + "attention.kv_lora_rank");
        hp.qk_rope_dim =
            need_uint(g, p + "rope.dimension_count");
        const std::uint32_t qk =
            need_uint(g, p + "attention.key_length");
        hp.qk_nope_dim = qk - hp.qk_rope_dim;
        hp.v_head_dim =
            need_uint(g, p + "attention.value_length");
        hp.head_dim = qk;
        hp.n_expert_shared = static_cast<std::uint32_t>(
            g.get_uint(p + "expert_shared_count").value_or(0));
        hp.n_dense_lead = static_cast<std::uint32_t>(
            g.get_uint(p + "leading_dense_block_count")
                .value_or(0));
        hp.n_ff_exp = static_cast<std::uint32_t>(
            g.get_uint(p + "expert_feed_forward_length")
                .value_or(hp.n_ff));
        hp.expert_weights_scale = get_float(
            g, p + "expert_weights_scale", 1.0f);
        hp.expert_weights_norm =
            g.get_bool(p + "expert_weights_norm")
                .value_or(false);
        hp.gating = static_cast<GatingFunc>(
            g.get_uint(p + "expert_gating_func")
                .value_or(static_cast<std::uint64_t>(
                    GatingFunc::kSoftmax)));

        // YARN: frequencies are corrected; the cos/sin mscale
        // cancels for DeepSeek, and the attention scale carries
        // mscale^2 instead (llama.cpp deepseek2 semantics).
        const float factor =
            get_float(g, p + "rope.scaling.factor", 1.0f);
        const float log_mul = get_float(
            g, p + "rope.scaling.yarn_log_multiplier", 0.0f);
        if (g.get_string(p + "rope.scaling.type") == "yarn" &&
            factor > 0.0f) {
            hp.yarn.freq_scale = 1.0f / factor;
            hp.yarn.n_ctx_orig = static_cast<std::uint32_t>(
                g.get_uint(
                     p + "rope.scaling.original_context_length")
                    .value_or(hp.n_ctx));
            const float mscale =
                1.0f + log_mul * std::log(factor);
            hp.kq_scale =
                mscale * mscale /
                std::sqrt(static_cast<float>(qk));
        } else {
            hp.kq_scale =
                1.0f / std::sqrt(static_cast<float>(qk));
        }
    } else {
        if (hp.n_heads == 0 || hp.n_embd % hp.n_heads != 0) {
            throw Error("invalid head configuration");
        }
        hp.head_dim = hp.n_embd / hp.n_heads;
        if (hp.n_kv_heads == 0 ||
            hp.n_heads % hp.n_kv_heads != 0) {
            throw Error("invalid kv head configuration");
        }
        hp.n_ff_exp = hp.n_ff;
        hp.expert_weights_norm = true;  // Mixtral renormalizes
        hp.kq_scale =
            1.0f / std::sqrt(static_cast<float>(hp.head_dim));
    }

    const TensorInfo* embd = g.find_tensor("token_embd.weight");
    if (embd == nullptr) {
        throw Error("missing tensor: token_embd.weight");
    }
    hp.n_vocab = static_cast<std::uint32_t>(embd->ne[1]);
    m.embd_ = need_mat(g, "token_embd.weight", hp.n_embd,
                       hp.n_vocab);

    const std::uint32_t kv_dim = hp.n_kv_heads * hp.head_dim;
    m.layers_.reserve(hp.n_layers);
    for (std::uint32_t l = 0; l < hp.n_layers; ++l) {
        const std::string bp = "blk." + std::to_string(l) + ".";
        Layer lay;
        lay.attn_norm = need_vec(g, bp + "attn_norm.weight",
                                 hp.n_embd);
        lay.ffn_norm =
            need_vec(g, bp + "ffn_norm.weight", hp.n_embd);
        if (hp.arch == Arch::kDeepseek2) {
            const std::uint32_t qk =
                hp.qk_nope_dim + hp.qk_rope_dim;
            lay.wq = need_mat(g, bp + "attn_q.weight", hp.n_embd,
                              hp.n_heads * qk);
            lay.wkv_a = need_mat(
                g, bp + "attn_kv_a_mqa.weight", hp.n_embd,
                hp.kv_lora_rank + hp.qk_rope_dim);
            lay.kv_a_norm = need_vec(
                g, bp + "attn_kv_a_norm.weight",
                hp.kv_lora_rank);
            lay.wkv_b = need_mat(
                g, bp + "attn_kv_b.weight", hp.kv_lora_rank,
                hp.n_heads * (hp.qk_nope_dim + hp.v_head_dim));
            lay.wo = need_mat(g, bp + "attn_output.weight",
                              hp.n_heads * hp.v_head_dim,
                              hp.n_embd);
        } else {
            lay.wq = need_mat(g, bp + "attn_q.weight", hp.n_embd,
                              hp.n_embd);
            lay.wk = need_mat(g, bp + "attn_k.weight", hp.n_embd,
                              kv_dim);
            lay.wv = need_mat(g, bp + "attn_v.weight", hp.n_embd,
                              kv_dim);
            lay.wo = need_mat(g, bp + "attn_output.weight",
                              hp.n_embd, hp.n_embd);
        }
        const bool moe_layer =
            hp.n_expert > 0 && l >= hp.n_dense_lead;
        if (moe_layer) {
            lay.gate_inp = need_mat(g, bp + "ffn_gate_inp.weight",
                                    hp.n_embd, hp.n_expert);
            lay.gate_exps =
                need_mat3(g, bp + "ffn_gate_exps.weight",
                          hp.n_embd, hp.n_ff_exp, hp.n_expert);
            lay.up_exps =
                need_mat3(g, bp + "ffn_up_exps.weight",
                          hp.n_embd, hp.n_ff_exp, hp.n_expert);
            lay.down_exps =
                need_mat3(g, bp + "ffn_down_exps.weight",
                          hp.n_ff_exp, hp.n_embd, hp.n_expert);
            if (hp.n_expert_shared > 0) {
                const std::uint32_t sh =
                    hp.n_ff_exp * hp.n_expert_shared;
                lay.gate_shexp =
                    need_mat(g, bp + "ffn_gate_shexp.weight",
                             hp.n_embd, sh);
                lay.up_shexp =
                    need_mat(g, bp + "ffn_up_shexp.weight",
                             hp.n_embd, sh);
                lay.down_shexp =
                    need_mat(g, bp + "ffn_down_shexp.weight",
                             sh, hp.n_embd);
            }
        } else {
            lay.w_gate = need_mat(g, bp + "ffn_gate.weight",
                                  hp.n_embd, hp.n_ff);
            lay.w_up = need_mat(g, bp + "ffn_up.weight",
                                hp.n_embd, hp.n_ff);
            lay.w_down = need_mat(g, bp + "ffn_down.weight",
                                  hp.n_ff, hp.n_embd);
        }
        m.layers_.push_back(lay);
    }

    m.backend_ = &backend::best_backend();
    if (g.find_tensor("rope_freqs.weight") != nullptr) {
        m.rope_factors_ =
            need_vec(g, "rope_freqs.weight", hp.head_dim / 2);
    }
    m.out_norm_ = need_vec(g, "output_norm.weight", hp.n_embd);
    // Tied embeddings when output.weight is absent.
    m.out_w_ = g.find_tensor("output.weight") != nullptr
                   ? need_mat(g, "output.weight", hp.n_embd,
                              hp.n_vocab)
                   : m.embd_;
    return m;
}

LlamaModel::Workspace LlamaModel::make_workspace() const {
    Workspace ws;
    const bool mla = hp_.arch == Arch::kDeepseek2;
    ws.x.resize(hp_.n_embd);
    ws.xb.resize(hp_.n_embd);
    ws.xb2.resize(hp_.n_embd);
    ws.q.resize(mla ? hp_.n_heads * hp_.head_dim : hp_.n_embd);
    ws.att.resize(hp_.n_ctx);
    const std::uint32_t ff = std::max(
        {hp_.n_ff, hp_.n_ff_exp,
         hp_.n_ff_exp * hp_.n_expert_shared});
    ws.gate.resize(ff);
    ws.up.resize(ff);
    ws.out.resize(mla ? hp_.n_heads * hp_.v_head_dim
                      : hp_.n_embd);
    ws.router.resize(hp_.n_expert);
    ws.moe_acc.resize(hp_.n_expert > 0 ? hp_.n_embd : 0);
    if (mla) {
        ws.kv_a.resize(hp_.kv_lora_rank + hp_.qk_rope_dim);
        ws.q_abs.resize(hp_.kv_lora_rank);
        ws.latent.resize(hp_.kv_lora_rank);
    }
    return ws;
}

kv::PagedKvCache LlamaModel::make_cache(
    std::uint32_t n_blocks) const {
    kv::PagedKvCache::Geometry geom;
    geom.n_layers = hp_.n_layers;
    // MLA caches one latent row (c_kv + roped k_pe) per position;
    // the V half of each block goes unused (simplicity > 2x
    // space; still ~9x smaller than caching full K/V heads).
    geom.kv_dim = hp_.arch == Arch::kDeepseek2
                      ? hp_.kv_lora_rank + hp_.qk_rope_dim
                      : hp_.n_kv_heads * hp_.head_dim;
    geom.block_tokens = 16;
    geom.n_blocks =
        n_blocks != 0
            ? n_blocks
            : (hp_.n_ctx + geom.block_tokens - 1) /
                  geom.block_tokens;
    if (backend_->ops.alloc_kv != nullptr) {
        float* storage = backend_->ops.alloc_kv(
            kv::PagedKvCache::pool_floats(geom));
        if (storage != nullptr) {
            return kv::PagedKvCache(geom, storage);
        }
    }
    return kv::PagedKvCache(geom);
}

void LlamaModel::forward(tok::TokenId token,
                         kv::PagedKvCache& cache,
                         kv::PagedKvCache::Seq& seq, Workspace& ws,
                         std::span<float> logits) const {
    using namespace cppllm::backend;

    if (token < 0 ||
        static_cast<std::uint32_t>(token) >= hp_.n_vocab) {
        throw std::invalid_argument("token id out of vocab");
    }
    if (backend_->name == "vulkan" &&
        vulkan_forward(*this, token, cache, seq, logits)) {
        return;
    }
    if (seq.n_tokens >= hp_.n_ctx) {
        throw std::invalid_argument("context window exhausted");
    }
    if (seq.n_tokens >= cache.capacity(seq)) {
        throw std::invalid_argument("seq capacity not ensured");
    }
    const std::uint32_t pos = seq.n_tokens;
    const backend::Ops& op = backend_->ops;

    op.dequant_row(embd_, static_cast<std::uint32_t>(token),
                   ws.x);

    for (std::uint32_t l = 0; l < hp_.n_layers; ++l) {
        const Layer& lay = layers_[l];

        rmsnorm(ws.x, lay.attn_norm, hp_.rms_eps, ws.xb);
        if (hp_.arch == Arch::kDeepseek2) {
            attention_mla(lay, cache, seq, ws, l, pos);
        } else {
            attention_dense(lay, cache, seq, ws, l, pos);
        }
        op.matvec(lay.wo, ws.out, ws.xb2);
        for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
            ws.x[i] += ws.xb2[i];
        }

        rmsnorm(ws.x, lay.ffn_norm, hp_.rms_eps, ws.xb);
        if (!lay.is_moe()) {
            op.matvec(lay.w_gate, ws.xb, ws.gate);
            op.matvec(lay.w_up, ws.xb, ws.up);
            silu_mul({ws.gate.data(), hp_.n_ff},
                     {ws.up.data(), hp_.n_ff},
                     {ws.gate.data(), hp_.n_ff});
            op.matvec(lay.w_down,
                      {ws.gate.data(), hp_.n_ff}, ws.xb2);
            for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
                ws.x[i] += ws.xb2[i];
            }
        } else {
            moe_ffn(lay, ws);
            for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
                ws.x[i] += ws.moe_acc[i];
            }
        }
    }

    rmsnorm(ws.x, out_norm_, hp_.rms_eps, ws.xb);
    op.matvec(out_w_, ws.xb, logits);
    seq.n_tokens = pos + 1;
}

void LlamaModel::attention_dense(const Layer& lay,
                                 kv::PagedKvCache& cache,
                                 kv::PagedKvCache::Seq& seq,
                                 Workspace& ws, std::uint32_t l,
                                 std::uint32_t pos) const {
    using namespace cppllm::backend;
    const Ops& op = backend_->ops;
    const std::uint32_t hd = hp_.head_dim;
    const std::size_t kv_dim =
        static_cast<std::size_t>(hp_.n_kv_heads) * hd;
    const std::uint32_t group = hp_.n_heads / hp_.n_kv_heads;

    float* krow = cache.k(seq, l, pos);
    float* vrow = cache.v(seq, l, pos);
    op.matvec(lay.wq, ws.xb, ws.q);
    op.matvec(lay.wk, ws.xb, {krow, kv_dim});
    op.matvec(lay.wv, ws.xb, {vrow, kv_dim});
    rope_norm(ws.q, hp_.n_heads, hd, pos, hp_.rope_freq_base,
              rope_factors_);
    rope_norm({krow, kv_dim}, hp_.n_kv_heads, hd, pos,
              hp_.rope_freq_base, rope_factors_);

    for (std::uint32_t h = 0; h < hp_.n_heads; ++h) {
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
            att[t] = s * hp_.kq_scale;
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

void LlamaModel::attention_mla(const Layer& lay,
                               kv::PagedKvCache& cache,
                               kv::PagedKvCache::Seq& seq,
                               Workspace& ws, std::uint32_t l,
                               std::uint32_t pos) const {
    using namespace cppllm::backend;
    const Ops& op = backend_->ops;
    const std::uint32_t rank = hp_.kv_lora_rank;
    const std::uint32_t nope = hp_.qk_nope_dim;
    const std::uint32_t rope = hp_.qk_rope_dim;
    const std::uint32_t vd = hp_.v_head_dim;
    const std::uint32_t qk = nope + rope;

    // Latent K/V row for this position: rms-normed c_kv followed
    // by the roped shared k_pe (DESIGN.md R5 latent cache).
    op.matvec(lay.wq, ws.xb, ws.q);
    op.matvec(lay.wkv_a, ws.xb, ws.kv_a);
    float* row = cache.k(seq, l, pos);
    rmsnorm({ws.kv_a.data(), rank}, lay.kv_a_norm, hp_.rms_eps,
            {row, rank});
    std::copy(ws.kv_a.begin() + rank, ws.kv_a.end(), row + rank);
    rope_norm_yarn({row + rank, rope}, 1, rope, pos,
                   hp_.rope_freq_base, hp_.yarn);

    for (std::uint32_t h = 0; h < hp_.n_heads; ++h) {
        float* qh =
            ws.q.data() + static_cast<std::size_t>(h) * qk;
        rope_norm_yarn({qh + nope, rope}, 1, rope, pos,
                       hp_.rope_freq_base, hp_.yarn);

        // Weight absorption: q_abs = W_uk_h^T q_nope, so scores
        // are dot products against the cached latents directly.
        const Mat w_uk = mat_rows(
            lay.wkv_b, h * (nope + vd), nope);
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
            att[t] = s * hp_.kq_scale;
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
        const Mat w_uv = mat_rows(
            lay.wkv_b, h * (nope + vd) + nope, vd);
        op.matvec(w_uv, ws.latent,
                  {ws.out.data() +
                       static_cast<std::size_t>(h) * vd,
                   vd});
    }
}

void LlamaModel::moe_ffn(const Layer& lay, Workspace& ws) const {
    using namespace cppllm::backend;
    const Ops& op = backend_->ops;
    const std::uint32_t n_ff = hp_.n_ff_exp;

    op.matvec(lay.gate_inp, ws.xb, ws.router);
    if (hp_.gating == GatingFunc::kSoftmax) {
        softmax_inplace(ws.router);
    } else {
        for (float& v : ws.router) {
            v = 1.0f / (1.0f + std::exp(-v));
        }
    }
    std::vector<std::uint32_t> picked;
    for (std::uint32_t k = 0; k < hp_.n_expert_used; ++k) {
        std::uint32_t best = hp_.n_expert;
        for (std::uint32_t e = 0; e < hp_.n_expert; ++e) {
            const bool taken =
                std::find(picked.begin(), picked.end(), e) !=
                picked.end();
            if (!taken && (best == hp_.n_expert ||
                           ws.router[e] > ws.router[best])) {
                best = e;
            }
        }
        picked.push_back(best);
    }
    float wsum = 1.0f;
    if (hp_.expert_weights_norm) {
        wsum = 0.0f;
        for (auto e : picked) {
            wsum += ws.router[e];
        }
    }
    std::fill(ws.moe_acc.begin(), ws.moe_acc.end(), 0.0f);
    auto swiglu_into_acc = [&](const Mat& wg, const Mat& wu,
                               const Mat& wd, std::uint32_t ff,
                               float wgt) {
        op.matvec(wg, ws.xb, {ws.gate.data(), ff});
        op.matvec(wu, ws.xb, {ws.up.data(), ff});
        silu_mul({ws.gate.data(), ff}, {ws.up.data(), ff},
                 {ws.gate.data(), ff});
        op.matvec(wd, {ws.gate.data(), ff}, ws.xb2);
        for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
            ws.moe_acc[i] += wgt * ws.xb2[i];
        }
    };
    for (auto e : picked) {
        const float wgt = ws.router[e] / wsum *
                          hp_.expert_weights_scale;
        swiglu_into_acc(lay.gate_exps.expert(e),
                        lay.up_exps.expert(e),
                        lay.down_exps.expert(e), n_ff, wgt);
    }
    if (hp_.n_expert_shared > 0) {
        swiglu_into_acc(lay.gate_shexp, lay.up_shexp,
                        lay.down_shexp,
                        n_ff * hp_.n_expert_shared, 1.0f);
    }
}

tok::TokenId argmax(std::span<const float> logits) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return static_cast<tok::TokenId>(best);
}

}  // namespace cppllm::model
