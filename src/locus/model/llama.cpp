#include "locus/model/llama.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "locus/backend/vulkan_forward.hpp"
#include "locus/model/arch.hpp"
#include "locus/model/moe_stats.hpp"
#include "locus/model/gguf_load.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::model {

using namespace load_util;
using backend::Mat;
using gguf::Error;
using gguf::GgufFile;
using gguf::TensorInfo;

namespace {

/**
 * R8 readahead (expert + layer) is on by default: it measured
 * 33% faster cold streaming on GLM-5.2 with bit-identical
 * output, and madvise on resident pages is a no-op for warm
 * models. LOCUS_NO_READAHEAD=1 opts out (A/B benchmarks).
 */
bool readahead_enabled() {
    return std::getenv("LOCUS_NO_READAHEAD") == nullptr;
}

/** Hints a whole matrix for readahead; empty mats are skipped. */
void advise_mat(const backend::Mat& m) {
    if (m.data != nullptr && m.rows > 0) {
        sys::advise_willneed(
            m.data, m.rows * backend::mat_row_bytes(m));
    }
}

/**
 * Applies f to every statically-known weight of a layer --
 * attention (incl. MLA projections and the DSA indexer), dense
 * FFN, router and shared experts. Routed experts are excluded:
 * their identity is only known after routing (see moe_ffn).
 * Shared by the R8 layer readahead and the R9 static pinning.
 */
template <class F>
void for_each_static_mat(const LlamaModel::Layer& lay, F&& f) {
    f(lay.wq);
    f(lay.wk);
    f(lay.wv);
    f(lay.wo);
    f(lay.w_gate);
    f(lay.w_up);
    f(lay.w_down);
    f(lay.gate_inp);
    f(lay.gate_shexp);
    f(lay.up_shexp);
    f(lay.down_shexp);
    f(lay.wkv_a);
    f(lay.wkv_b);
    f(lay.wk_b);
    f(lay.wv_b);
    f(lay.wq_a);
    f(lay.wq_b);
    f(lay.idx_proj);
    f(lay.idx_k);
    f(lay.idx_q_b);
}

void advise_layer_statics(const LlamaModel::Layer& lay) {
    for_each_static_mat(
        lay, [](const backend::Mat& m) { advise_mat(m); });
}

}  // namespace

void matvec_mt(const backend::Ops& op, const Mat& w,
               std::span<const float> x, std::span<float> out) {
    // Below this many rows per slice the dispatch overhead wins.
    constexpr std::uint32_t kMinRowsPerSlice = 64;
    auto& pool = sys::ThreadPool::instance();
    std::size_t t = pool.parallelism();
    if (const char* env = std::getenv("LOCUS_THREADS")) {
        const long v = std::atol(env);
        if (v >= 1) {
            t = std::min<std::size_t>(
                t, static_cast<std::size_t>(v));
        }
    }
    t = std::min<std::size_t>(t, w.rows / kMinRowsPerSlice);
    if (t <= 1) {
        op.matvec(w, x, out);
        return;
    }
    pool.parallel_for(t, [&](std::size_t i) {
        const std::uint32_t r0 = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(w.rows) * i / t);
        const std::uint32_t r1 = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(w.rows) * (i + 1) / t);
        op.matvec(backend::mat_rows(w, r0, r1 - r0), x,
                  out.subspan(r0, r1 - r0));
    });
}

float dsa_index_score(std::span<const float> q,
                      std::span<const float> w,
                      std::span<const float> k) {
    const std::size_t d = k.size();
    float score = 0.0f;
    for (std::size_t h = 0; h < w.size(); ++h) {
        const float* qh = q.data() + h * d;
        float dot = 0.0f;
        for (std::size_t i = 0; i < d; ++i) {
            dot += qh[i] * k[i];
        }
        if (dot > 0.0f) {  // ReLU
            score += w[h] * dot;
        }
    }
    return score;
}

LlamaModel LlamaModel::load(const GgufFile& g) {
    const auto arch_name = g.get_string("general.architecture");
    const ArchSpec* spec =
        find_arch(arch_name.value_or("<missing>"));
    if (spec == nullptr) {
        std::string supported;
        for (const ArchSpec& a : archs()) {
            supported += supported.empty() ? "" : ", ";
            supported += a.name;
        }
        throw Error("unsupported general.architecture \"" +
                    std::string(arch_name.value_or("")) +
                    "\" (supported: " + supported + ")");
    }

    LlamaModel m;
    m.spec_ = spec;
    Hparams& hp = m.hp_;
    const std::string p = std::string(spec->name) + ".";

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
    spec->load_hparams(g, p, hp);

    const TensorInfo* embd = g.find_tensor("token_embd.weight");
    if (embd == nullptr) {
        throw Error("missing tensor: token_embd.weight");
    }
    hp.n_vocab = static_cast<std::uint32_t>(embd->ne[1]);
    m.embd_ = need_mat(g, "token_embd.weight", hp.n_embd,
                       hp.n_vocab);

    m.layers_.reserve(hp.n_layers);
    for (std::uint32_t l = 0; l < hp.n_layers; ++l) {
        const std::string bp = "blk." + std::to_string(l) + ".";
        Layer lay;
        lay.attn_norm = need_vec(g, bp + "attn_norm.weight",
                                 hp.n_embd);
        lay.ffn_norm =
            need_vec(g, bp + "ffn_norm.weight", hp.n_embd);
        spec->load_attention(g, bp, hp, lay);
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
            if (g.find_tensor(bp + "exp_probs_b.bias") !=
                nullptr) {
                lay.exp_probs_b = need_vec(
                    g, bp + "exp_probs_b.bias", hp.n_expert);
            }
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

    // R9 static pinning (opt-in): wire every non-expert weight
    // into RAM so per-token streaming is only the routed
    // experts. Policy decision on a small-RAM host, hence the
    // env gate; failures degrade to plain demand paging.
    if (std::getenv("LOCUS_PIN_STATIC") != nullptr) {
        std::size_t locked = 0, failed = 0;
        auto pin_bytes = [&](const void* p, std::size_t n) {
            if (p == nullptr || n == 0) {
                return;
            }
            (sys::lock_resident(p, n) ? locked : failed) += n;
        };
        auto pin = [&](const Mat& w) {
            if (w.data != nullptr && w.rows > 0) {
                pin_bytes(w.data,
                          w.rows * backend::mat_row_bytes(w));
            }
        };
        for (const Layer& lay : m.layers_) {
            for_each_static_mat(lay, pin);
            pin_bytes(lay.attn_norm.data(),
                      lay.attn_norm.size_bytes());
            pin_bytes(lay.ffn_norm.data(),
                      lay.ffn_norm.size_bytes());
            pin_bytes(lay.kv_a_norm.data(),
                      lay.kv_a_norm.size_bytes());
            pin_bytes(lay.q_a_norm.data(),
                      lay.q_a_norm.size_bytes());
            pin_bytes(lay.exp_probs_b.data(),
                      lay.exp_probs_b.size_bytes());
            pin_bytes(lay.idx_k_norm.data(),
                      lay.idx_k_norm.size_bytes());
            pin_bytes(lay.idx_k_norm_b.data(),
                      lay.idx_k_norm_b.size_bytes());
        }
        pin(m.embd_);
        if (m.out_w_.data != m.embd_.data) {
            pin(m.out_w_);
        }
        pin_bytes(m.out_norm_.data(), m.out_norm_.size_bytes());
        pin_bytes(m.rope_factors_.data(),
                  m.rope_factors_.size_bytes());
        std::fprintf(stderr,
                     "locus: pinned %zu MB static weights"
                     " (%zu MB failed)\n",
                     locked >> 20, failed >> 20);
    }
    return m;
}

LlamaModel::Workspace LlamaModel::make_workspace() const {
    Workspace ws;
    const bool mla = hp_.kv_lora_rank > 0;
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
        ws.q_a.resize(hp_.q_lora_rank);
    }
    if (hp_.idx_heads > 0) {
        ws.idx_q.resize(static_cast<std::size_t>(hp_.idx_heads) *
                        hp_.idx_dim);
        ws.idx_w.resize(hp_.idx_heads);
        ws.idx_scores.resize(hp_.n_ctx);
        ws.idx_sel.reserve(hp_.idx_top_k);
    }
    return ws;
}

kv::PagedKvCache LlamaModel::make_cache(
    std::uint32_t n_blocks) const {
    kv::PagedKvCache::Geometry geom;
    geom.n_layers = hp_.n_layers;
    geom.kv_dim = spec_->kv_dim(hp_);
    geom.block_tokens = 16;
    // Default pool covers min(n_ctx, 4096) tokens: long-context
    // models (128k+) would otherwise demand tens of GB up front.
    // Callers wanting more pass n_blocks explicitly.
    const std::uint32_t cap_tokens =
        std::min(hp_.n_ctx, 4096u);
    geom.n_blocks =
        n_blocks != 0
            ? n_blocks
            : (cap_tokens + geom.block_tokens - 1) /
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
    using namespace locus::backend;

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

    // R8 layer readahead: while layer l computes, ask the kernel
    // to page in layer l+1's static weights (and the output head
    // after the last layer). Routed experts are covered
    // separately at selection time (moe_ffn).
    const bool layer_ra = readahead_enabled();

    for (std::uint32_t l = 0; l < hp_.n_layers; ++l) {
        const Layer& lay = layers_[l];
        if (layer_ra) {
            if (l + 1 < hp_.n_layers) {
                advise_layer_statics(layers_[l + 1]);
            } else {
                advise_mat(out_w_);
            }
        }

        rmsnorm(ws.x, lay.attn_norm, hp_.rms_eps, ws.xb);
        spec_->attention(*this, lay, cache, seq, ws, l, pos);
        matvec_mt(op, lay.wo, ws.out, ws.xb2);
        for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
            ws.x[i] += ws.xb2[i];
        }

        rmsnorm(ws.x, lay.ffn_norm, hp_.rms_eps, ws.xb);
        if (!lay.is_moe()) {
            matvec_mt(op, lay.w_gate, ws.xb, ws.gate);
            matvec_mt(op, lay.w_up, ws.xb, ws.up);
            silu_mul({ws.gate.data(), hp_.n_ff},
                     {ws.up.data(), hp_.n_ff},
                     {ws.gate.data(), hp_.n_ff});
            matvec_mt(op, lay.w_down, {ws.gate.data(), hp_.n_ff},
                      ws.xb2);
            for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
                ws.x[i] += ws.xb2[i];
            }
        } else {
            moe_ffn(lay, ws, l);
            for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
                ws.x[i] += ws.moe_acc[i];
            }
        }
    }

    rmsnorm(ws.x, out_norm_, hp_.rms_eps, ws.xb);
    matvec_mt(op, out_w_, ws.xb, logits);
    seq.n_tokens = pos + 1;
}

std::vector<std::pair<std::uint32_t, float>> moe_select(
    const Hparams& hp, const LlamaModel::Layer& lay,
    std::span<float> router) {
    if (hp.gating == GatingFunc::kSoftmax) {
        backend::softmax_inplace(router);
    } else {
        for (float& v : router) {
            v = 1.0f / (1.0f + std::exp(-v));
        }
    }
    // Selection scores add the V3/K2 correction bias (weights do
    // not); group-limited routing masks all but the best groups,
    // each scored by the sum of its top-2 selection scores.
    std::vector<float> sel(router.begin(), router.end());
    if (!lay.exp_probs_b.empty()) {
        for (std::uint32_t e = 0; e < hp.n_expert; ++e) {
            sel[e] += lay.exp_probs_b[e];
        }
    }
    if (hp.n_group > 1) {
        const std::uint32_t per = hp.n_expert / hp.n_group;
        std::vector<float> gscore(hp.n_group);
        for (std::uint32_t gi = 0; gi < hp.n_group; ++gi) {
            float top1 = -1e30f, top2 = -1e30f;
            for (std::uint32_t e = gi * per;
                 e < (gi + 1) * per; ++e) {
                if (sel[e] > top1) {
                    top2 = top1;
                    top1 = sel[e];
                } else if (sel[e] > top2) {
                    top2 = sel[e];
                }
            }
            gscore[gi] = top1 + (per > 1 ? top2 : 0.0f);
        }
        std::vector<bool> keep(hp.n_group, false);
        for (std::uint32_t k = 0; k < hp.n_group_used; ++k) {
            std::uint32_t best = hp.n_group;
            for (std::uint32_t gi = 0; gi < hp.n_group; ++gi) {
                if (!keep[gi] && (best == hp.n_group ||
                                  gscore[gi] > gscore[best])) {
                    best = gi;
                }
            }
            keep[best] = true;
        }
        for (std::uint32_t e = 0; e < hp.n_expert; ++e) {
            if (!keep[e / per]) {
                sel[e] = -1e30f;
            }
        }
    }
    std::vector<std::uint32_t> picked;
    for (std::uint32_t k = 0; k < hp.n_expert_used; ++k) {
        std::uint32_t best = hp.n_expert;
        for (std::uint32_t e = 0; e < hp.n_expert; ++e) {
            const bool taken =
                std::find(picked.begin(), picked.end(), e) !=
                picked.end();
            if (!taken && (best == hp.n_expert ||
                           sel[e] > sel[best])) {
                best = e;
            }
        }
        picked.push_back(best);
    }
    float wsum = 1.0f;
    if (hp.expert_weights_norm) {
        wsum = 0.0f;
        for (auto e : picked) {
            wsum += router[e];
        }
    }
    std::vector<std::pair<std::uint32_t, float>> out;
    out.reserve(picked.size());
    for (auto e : picked) {
        out.emplace_back(
            e, router[e] / wsum * hp.expert_weights_scale);
    }
    return out;
}

void LlamaModel::moe_ffn(const Layer& lay, Workspace& ws,
                         std::uint32_t layer) const {
    using namespace locus::backend;
    const Ops& op = backend_->ops;
    const std::uint32_t n_ff = hp_.n_ff_exp;

    op.matvec(lay.gate_inp, ws.xb, ws.router);
    const auto picked = moe_select(hp_, lay, ws.router);
    if (MoeStats::enabled()) {
        for (const auto& [e, wgt] : picked) {
            MoeStats::record(layer, e, hp_.n_layers,
                             hp_.n_expert);
        }
    }
    // R8 expert readahead: overlap the SSD reads of every routed
    // expert before the sequential per-expert matmuls fault them
    // in one by one.
    if (readahead_enabled()) {
        for (const auto& [e, wgt] : picked) {
            sys::advise_willneed(lay.gate_exps.expert(e).data,
                                 lay.gate_exps.expert_bytes);
            sys::advise_willneed(lay.up_exps.expert(e).data,
                                 lay.up_exps.expert_bytes);
            sys::advise_willneed(lay.down_exps.expert(e).data,
                                 lay.down_exps.expert_bytes);
        }
    }
    std::fill(ws.moe_acc.begin(), ws.moe_acc.end(), 0.0f);
    auto swiglu_into_acc = [&](const Mat& wg, const Mat& wu,
                               const Mat& wd, std::uint32_t ff,
                               float wgt) {
        matvec_mt(op, wg, ws.xb, {ws.gate.data(), ff});
        matvec_mt(op, wu, ws.xb, {ws.up.data(), ff});
        silu_mul({ws.gate.data(), ff}, {ws.up.data(), ff},
                 {ws.gate.data(), ff});
        matvec_mt(op, wd, {ws.gate.data(), ff}, ws.xb2);
        for (std::uint32_t i = 0; i < hp_.n_embd; ++i) {
            ws.moe_acc[i] += wgt * ws.xb2[i];
        }
    };
    for (const auto& [e, wgt] : picked) {
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

}  // namespace locus::model
