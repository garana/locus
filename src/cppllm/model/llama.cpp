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
    if (g.get_string("general.architecture") != "llama") {
        throw Error("general.architecture is not llama");
    }

    LlamaModel m;
    Hparams& hp = m.hp_;
    hp.n_embd = need_uint(g, "llama.embedding_length");
    hp.n_layers = need_uint(g, "llama.block_count");
    hp.n_heads = need_uint(g, "llama.attention.head_count");
    hp.n_kv_heads = static_cast<std::uint32_t>(
        g.get_uint("llama.attention.head_count_kv")
            .value_or(hp.n_heads));
    hp.n_ff = need_uint(g, "llama.feed_forward_length");
    hp.n_ctx = need_uint(g, "llama.context_length");
    hp.n_expert = static_cast<std::uint32_t>(
        g.get_uint("llama.expert_count").value_or(0));
    hp.n_expert_used = static_cast<std::uint32_t>(
        g.get_uint("llama.expert_used_count").value_or(0));
    if (hp.n_expert > 0 &&
        (hp.n_expert_used == 0 ||
         hp.n_expert_used > hp.n_expert)) {
        throw Error("invalid expert_used_count");
    }
    hp.rms_eps = get_float(
        g, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    hp.rope_freq_base =
        get_float(g, "llama.rope.freq_base", 10000.0f);

    if (hp.n_heads == 0 || hp.n_embd % hp.n_heads != 0) {
        throw Error("invalid head configuration");
    }
    hp.head_dim = hp.n_embd / hp.n_heads;
    if (hp.n_kv_heads == 0 || hp.n_heads % hp.n_kv_heads != 0) {
        throw Error("invalid kv head configuration");
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
        const std::string p = "blk." + std::to_string(l) + ".";
        Layer lay;
        lay.attn_norm = need_vec(g, p + "attn_norm.weight",
                                 hp.n_embd);
        lay.ffn_norm = need_vec(g, p + "ffn_norm.weight", hp.n_embd);
        lay.wq = need_mat(g, p + "attn_q.weight", hp.n_embd,
                          hp.n_embd);
        lay.wk = need_mat(g, p + "attn_k.weight", hp.n_embd, kv_dim);
        lay.wv = need_mat(g, p + "attn_v.weight", hp.n_embd, kv_dim);
        lay.wo = need_mat(g, p + "attn_output.weight", hp.n_embd,
                          hp.n_embd);
        if (hp.n_expert > 0) {
            lay.gate_inp = need_mat(g, p + "ffn_gate_inp.weight",
                                    hp.n_embd, hp.n_expert);
            lay.gate_exps =
                need_mat3(g, p + "ffn_gate_exps.weight",
                          hp.n_embd, hp.n_ff, hp.n_expert);
            lay.up_exps =
                need_mat3(g, p + "ffn_up_exps.weight", hp.n_embd,
                          hp.n_ff, hp.n_expert);
            lay.down_exps =
                need_mat3(g, p + "ffn_down_exps.weight", hp.n_ff,
                          hp.n_embd, hp.n_expert);
        } else {
            lay.w_gate = need_mat(g, p + "ffn_gate.weight",
                                  hp.n_embd, hp.n_ff);
            lay.w_up = need_mat(g, p + "ffn_up.weight",
                                hp.n_embd, hp.n_ff);
            lay.w_down = need_mat(g, p + "ffn_down.weight",
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
    ws.x.resize(hp_.n_embd);
    ws.xb.resize(hp_.n_embd);
    ws.xb2.resize(hp_.n_embd);
    ws.q.resize(hp_.n_embd);
    ws.att.resize(hp_.n_ctx);
    ws.gate.resize(hp_.n_ff);
    ws.up.resize(hp_.n_ff);
    ws.out.resize(hp_.n_embd);
    ws.router.resize(hp_.n_expert);
    ws.moe_acc.resize(hp_.n_expert > 0 ? hp_.n_embd : 0);
    return ws;
}

kv::PagedKvCache LlamaModel::make_cache(
    std::uint32_t n_blocks) const {
    kv::PagedKvCache::Geometry geom;
    geom.n_layers = hp_.n_layers;
    geom.kv_dim = hp_.n_kv_heads * hp_.head_dim;
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
    const std::uint32_t hd = hp_.head_dim;
    const std::size_t kv_dim =
        static_cast<std::size_t>(hp_.n_kv_heads) * hd;
    const std::uint32_t group = hp_.n_heads / hp_.n_kv_heads;
    const backend::Ops& op = backend_->ops;

    op.dequant_row(embd_, static_cast<std::uint32_t>(token),
                   ws.x);

    for (std::uint32_t l = 0; l < hp_.n_layers; ++l) {
        const Layer& lay = layers_[l];

        // Attention block.
        rmsnorm(ws.x, lay.attn_norm, hp_.rms_eps, ws.xb);
        float* krow = cache.k(seq, l, pos);
        float* vrow = cache.v(seq, l, pos);
        op.matvec(lay.wq, ws.xb, ws.q);
        op.matvec(lay.wk, ws.xb, {krow, kv_dim});
        op.matvec(lay.wv, ws.xb, {vrow, kv_dim});
        rope_norm(ws.q, hp_.n_heads, hd, pos, hp_.rope_freq_base,
                  rope_factors_);
        rope_norm({krow, kv_dim}, hp_.n_kv_heads, hd, pos,
                  hp_.rope_freq_base, rope_factors_);

        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        for (std::uint32_t h = 0; h < hp_.n_heads; ++h) {
            const float* qh = ws.q.data() +
                              static_cast<std::size_t>(h) * hd;
            const std::size_t kvh =
                static_cast<std::size_t>(h / group) * hd;
            std::span<float> att(ws.att.data(), pos + 1);
            for (std::uint32_t t = 0; t <= pos; ++t) {
                const float* kt = cache.k(seq, l, t) + kvh;
                float s = 0.0f;
                for (std::uint32_t i = 0; i < hd; ++i) {
                    s += qh[i] * kt[i];
                }
                att[t] = s * scale;
            }
            softmax_inplace(att);
            float* oh = ws.out.data() +
                        static_cast<std::size_t>(h) * hd;
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
        op.matvec(lay.wo, ws.out, ws.xb2);
        for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
            ws.x[i] += ws.xb2[i];
        }

        // Feed-forward block (SwiGLU; dense or MoE).
        rmsnorm(ws.x, lay.ffn_norm, hp_.rms_eps, ws.xb);
        if (hp_.n_expert == 0) {
            op.matvec(lay.w_gate, ws.xb, ws.gate);
            op.matvec(lay.w_up, ws.xb, ws.up);
            silu_mul(ws.gate, ws.up, ws.gate);
            op.matvec(lay.w_down, ws.gate, ws.xb2);
            for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
                ws.x[i] += ws.xb2[i];
            }
        } else {
            // Router: softmax over all experts, take the top-k,
            // renormalize their weights (Mixtral-style gating).
            op.matvec(lay.gate_inp, ws.xb, ws.router);
            softmax_inplace(ws.router);
            std::vector<std::uint32_t> picked;
            for (std::uint32_t k = 0; k < hp_.n_expert_used;
                 ++k) {
                std::uint32_t best = hp_.n_expert;
                for (std::uint32_t e = 0; e < hp_.n_expert;
                     ++e) {
                    const bool taken =
                        std::find(picked.begin(), picked.end(),
                                  e) != picked.end();
                    if (!taken &&
                        (best == hp_.n_expert ||
                         ws.router[e] > ws.router[best])) {
                        best = e;
                    }
                }
                picked.push_back(best);
            }
            float wsum = 0.0f;
            for (auto e : picked) {
                wsum += ws.router[e];
            }
            std::fill(ws.moe_acc.begin(), ws.moe_acc.end(),
                      0.0f);
            for (auto e : picked) {
                const float wgt = ws.router[e] / wsum;
                op.matvec(lay.gate_exps.expert(e), ws.xb,
                          ws.gate);
                op.matvec(lay.up_exps.expert(e), ws.xb, ws.up);
                silu_mul(ws.gate, ws.up, ws.gate);
                op.matvec(lay.down_exps.expert(e), ws.gate,
                          ws.xb2);
                for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
                    ws.moe_acc[i] += wgt * ws.xb2[i];
                }
            }
            for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
                ws.x[i] += ws.moe_acc[i];
            }
        }
    }

    rmsnorm(ws.x, out_norm_, hp_.rms_eps, ws.xb);
    op.matvec(out_w_, ws.xb, logits);
    seq.n_tokens = pos + 1;
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
