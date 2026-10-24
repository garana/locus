// DeepSeek-V4.1-Flash (deepseek41), CPU single-token decode. Mirrors
// llama.cpp's draft graph (ggml-org/llama.cpp#30817, src/models/
// deepseek41.cpp); docs/models/deepseek41.md walks through it.
//
// - Hyper-connections: the residual is 4 streams of n_embd. Each
//   sublayer reads a weighted collapse of them and writes back through
//   a Sinkhorn-normalized 4x4 mix. V4.1 is lagged: an attention block
//   collapses with the previous FFN's pre-mix (layer 0: stream 0), an
//   FFN with its own layer's attention pre-mix.
// - Attention: q LoRA, one 512-wide MQA kv (K = V), rope on the last
//   64 dims, sinks in the softmax, inverse rope on the output, then a
//   grouped output LoRA. Every layer attends to a window of the last
//   `window` tokens; compressed layers also attend to top-k rows of a
//   shared compressed history (kv-source layers build one row per
//   `ratio` tokens), picked by the indexer of their top-k source.
// - Engram (some layers): n-gram hashed rows of a huge table, gated per
//   stream, added to every stream before the layer's mixes.
// - MoE: sqrt(softplus) router, selection bias, top-k, normalized and
//   scaled; experts and shared expert use silu(min(g, L)) * clamp(u).
//
// Paged-cache layout (kv_dim = head_dim + idx_dim = 640, F32): layers
// 0..n_layers-1 hold each layer's roped kv at its position (K half,
// head_dim floats; the window reads the last `window` of them). One
// extra cache layer per kv-source layer holds its compressed rows (K
// half: row b at position b, latent then index key) and the
// compressor kv state
// (V half at the token's position); the compressor score state sits
// in the source layer's own V half, and layer 0's V half keeps each
// position's token id for the Engram hash.

#include "locus/model/deepseek41.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/model/gguf_load.hpp"

namespace locus::model::deepseek41 {

namespace {

using namespace load_util;
using backend::Mat;

/** @returns Array key `key` as unsigned values (u32 / u64 / i32). */
std::vector<std::uint64_t> uint_array(const gguf::GgufFile& g,
                                      const std::string& key) {
    const auto* a = g.get_array(key);
    if (a == nullptr) {
        throw gguf::Error("deepseek41: missing " + key);
    }
    std::vector<std::uint64_t> out;
    out.reserve(a->size());
    for (const auto& v : *a) {
        out.push_back(v.type == gguf::ValueType::kInt32 ||
                              v.type == gguf::ValueType::kInt64
                          ? static_cast<std::uint64_t>(v.i)
                          : v.u);
    }
    return out;
}

std::vector<float> float_array(const gguf::GgufFile& g,
                               const std::string& key, std::size_t n) {
    const auto* a = g.get_array(key);
    std::vector<float> out(n, 0.0f);
    if (a != nullptr) {
        for (std::size_t i = 0; i < n && i < a->size(); ++i) {
            out[i] = static_cast<float>((*a)[i].f);
        }
    }
    return out;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/** Weightless RMS norm, out of place. */
void rms_unit(std::span<const float> x, float eps, std::span<float> out) {
    float ss = 0.0f;
    for (float v : x) {
        ss += v * v;
    }
    const float s = 1.0f / std::sqrt(ss / static_cast<float>(x.size()) + eps);
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] * s;
    }
}

/** Rope parameters of layer l (window-only vs compressed layers). */
struct Rope {
    float base;
    backend::Yarn yarn;
};

Rope rope_of(const Hparams& hp, std::uint32_t l) {
    const auto& d = hp.ds41;
    if (d.ratio[l] == 0) {
        return {hp.rope_freq_base, backend::Yarn{}};
    }
    // The model's own attn_factor cancels ggml's yarn mscale.
    backend::Yarn y = hp.yarn;
    y.mscale = 1.0f;
    return {d.compress_rope_base, y};
}

/** Ropes the last rope_dims of each `width`-wide head (interleaved
 * pairs); inverse rotates by -theta. */
void rope_tail(std::span<float> x, std::uint32_t heads, std::uint32_t width,
               std::uint32_t rope_dims, std::uint32_t pos, const Rope& r,
               bool inverse = false) {
    for (std::uint32_t h = 0; h < heads; ++h) {
        std::span<float> t =
            x.subspan(std::size_t{h} * width + (width - rope_dims), rope_dims);
        // R(-theta)(a, b) = flip(R(theta)(a, -b)) on the odd lane.
        if (inverse) {
            for (std::size_t i = 1; i < t.size(); i += 2) {
                t[i] = -t[i];
            }
        }
        backend::rope_norm_yarn(t, 1, rope_dims, pos, r.base, r.yarn);
        if (inverse) {
            for (std::size_t i = 1; i < t.size(); i += 2) {
                t[i] = -t[i];
            }
        }
    }
}

/** One sublayer's mHC mixes. */
struct Mixes {
    float pre[4], post[4];
    float comb[16];  // [dst + 4 * src]
};

/** Per-token scratch (one decode at a time per thread). */
struct Scratch {
    std::vector<float> X, Xres, flat, h, a, q, qr, kv, ow, o, oa, f, y;
    std::vector<float> gate, up, router, sel, logits_k, score;
    std::vector<float> iq, iw, comp, sk, ss, erow, ekv, tmp;
    std::vector<std::uint32_t> topk_of_layer_n;  // count per layer
    std::vector<std::vector<std::uint32_t>> topk;  // per index source
    std::vector<std::uint32_t> cand;  // candidate blocks (sorted)
};

Scratch& scratch() {
    thread_local Scratch s;
    return s;
}

Mixes hc_mixes(const Hparams& hp, std::span<const float> X, const Mat& fn,
               std::span<const float> base, std::span<const float> scale,
               const backend::Ops& op, Scratch& s) {
    const auto& d = hp.ds41;
    const std::uint32_t E = hp.n_embd;
    s.flat.resize(std::size_t{d.hc} * E);
    rms_unit(X, hp.rms_eps, s.flat);
    float m[24];
    op.matvec(fn, s.flat, {m, 24});
    Mixes mx{};
    for (int j = 0; j < 4; ++j) {
        mx.pre[j] = sigmoid(m[j] * scale[0] + base[j]) + d.hc_eps;
        mx.post[j] = 2.0f * sigmoid(m[4 + j] * scale[1] + base[4 + j]);
    }
    for (int k = 0; k < 16; ++k) {
        mx.comb[k] = m[8 + k] * scale[2] + base[8 + k];
    }
    sinkhorn(mx.comb, d.hc, d.sinkhorn_iters, d.hc_eps);
    return mx;
}

/** h = sum_k w[k] * X[k]. */
void hc_pre(std::span<const float> X, const float* w, std::uint32_t E,
            std::span<float> h) {
    std::fill(h.begin(), h.end(), 0.0f);
    for (std::uint32_t k = 0; k < 4; ++k) {
        const float* xk = X.data() + std::size_t{k} * E;
        for (std::uint32_t i = 0; i < E; ++i) {
            h[i] += w[k] * xk[i];
        }
    }
}

/** X[dst] = post[dst] * y + sum_src comb[dst + 4 src] * Xres[src]. */
void hc_post(std::span<const float> y, std::span<const float> Xres,
             const Mixes& mx, std::uint32_t E, std::span<float> X) {
    for (std::uint32_t dst = 0; dst < 4; ++dst) {
        float* xd = X.data() + std::size_t{dst} * E;
        for (std::uint32_t i = 0; i < E; ++i) {
            float v = mx.post[dst] * y[i];
            for (std::uint32_t src = 0; src < 4; ++src) {
                v += mx.comb[dst + 4 * src] * Xres[std::size_t{src} * E + i];
            }
            xd[i] = v;
        }
    }
}

/** Engram injection at layer l for the token at pos. */
void engram(const TransformerModel& m, const TransformerModel::Layer& lay,
            std::uint32_t e, kv::PagedKvCache& cache,
            const kv::PagedKvCache::Seq& seq, tok::TokenId token,
            std::uint32_t pos, Scratch& s) {
    const Hparams& hp = m.hparams();
    const auto& d = hp.ds41;
    const backend::Ops& op = m.active_backend().ops;
    const std::uint32_t E = hp.n_embd;
    const std::int64_t dead = -1;
    auto cid = [&](std::int64_t t) -> std::int64_t {
        return t < 0 || t >= static_cast<std::int64_t>(d.token_map.size())
                   ? dead
                   : d.token_map[static_cast<std::size_t>(t)];
    };
    const std::int64_t pad = d.token_map[d.engram_pad];
    std::vector<std::int64_t> toks(d.ngram);
    toks[0] = cid(token);
    bool blocked = toks[0] == dead;
    if (blocked) {
        toks[0] = pad;
    }
    for (std::uint32_t k = 1; k < d.ngram; ++k) {
        std::int64_t c = dead;
        if (pos >= k) {
            // Token ids live in layer 0's V half (see the header).
            c = cid(static_cast<std::int64_t>(cache.v(seq, 0, pos - k)[0]));
        }
        blocked = blocked || c == dead;
        toks[k] = blocked ? pad : c;
    }
    const std::uint32_t ncols = (d.ngram - 1) * d.ngram_heads;
    std::vector<std::uint32_t> rows(ncols);
    engram_rows(d, e, toks, rows);
    s.erow.resize(std::size_t{ncols} * d.engram_dim);
    for (std::uint32_t c = 0; c < ncols; ++c) {
        op.dequant_row(lay.engram_table, rows[c],
                       {s.erow.data() + std::size_t{c} * d.engram_dim,
                        d.engram_dim});
    }
    s.ekv.resize(std::size_t{d.hc + 1} * E);
    matvec_mt(op, lay.engram_kv, s.erow, s.ekv);
    const std::span<const float> val(s.ekv.data() + std::size_t{d.hc} * E, E);
    s.tmp.resize(2 * std::size_t{E});
    std::span<float> xn(s.tmp.data(), E), kn(s.tmp.data() + E, E);
    std::vector<float> g(E);
    for (std::uint32_t h = 0; h < d.hc; ++h) {
        std::span<float> xh(s.X.data() + std::size_t{h} * E, E);
        rms_unit(xh, hp.rms_eps, xn);
        rms_unit({s.ekv.data() + std::size_t{h} * E, E}, hp.rms_eps, kn);
        op.dequant_row(lay.engram_gate, h, g);
        float dot = 0.0f;
        for (std::uint32_t i = 0; i < E; ++i) {
            dot += xn[i] * kn[i] * g[i];
        }
        dot /= std::sqrt(static_cast<float>(E));
        const float mag = std::sqrt(std::max(std::fabs(dot), 1e-6f));
        const float gate = sigmoid(dot < 0.0f ? -mag : mag);
        for (std::uint32_t i = 0; i < E; ++i) {
            xh[i] += gate * val[i];
        }
    }
}

/** Indices of the k largest scores (ties: lower index first). */
std::vector<std::uint32_t> top_k(std::span<const float> score,
                                 std::uint32_t k) {
    std::vector<std::uint32_t> idx(score.size());
    std::iota(idx.begin(), idx.end(), 0u);
    k = std::min<std::uint32_t>(k, static_cast<std::uint32_t>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                          return score[a] > score[b] ||
                                 (score[a] == score[b] && a < b);
                      });
    idx.resize(k);
    return idx;
}

/** Layer l's attention on h (attn_norm output) at pos; writes the
 * 5120-wide branch output into s.a. */
void attention_layer(const TransformerModel& m,
                     const TransformerModel::Layer& lay,
                     kv::PagedKvCache& cache, kv::PagedKvCache::Seq& seq,
                     std::uint32_t l, std::uint32_t pos, Scratch& s) {
    const Hparams& hp = m.hparams();
    const auto& d = hp.ds41;
    const backend::Ops& op = m.active_backend().ops;
    const std::uint32_t H = hp.n_heads, Dh = hp.head_dim, R = d.rope_dims;
    const Rope rope = rope_of(hp, l);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // q LoRA and the single MQA kv row (K = V).
    s.qr.resize(lay.wq_a.rows);
    matvec_mt(op, lay.wq_a, s.h, s.qr);
    backend::rmsnorm(s.qr, lay.q_a_norm, hp.rms_eps, s.qr);
    s.q.resize(std::size_t{H} * Dh);
    matvec_mt(op, lay.wq_b, s.qr, s.q);
    rope_tail(s.q, H, Dh, R, pos, rope);
    s.kv.resize(Dh);
    matvec_mt(op, lay.wkv_a, s.h, s.kv);
    backend::rmsnorm(s.kv, lay.kv_a_norm, hp.rms_eps, s.kv);
    rope_tail(s.kv, 1, Dh, R, pos, rope);
    std::copy(s.kv.begin(), s.kv.end(), cache.k(seq, l, pos));

    // Compressed rows: kv-source layers build one per `ratio` tokens.
    const std::uint32_t r = d.ratio[l];
    std::uint32_t n_vis = 0;
    if (r > 0) {
        const auto src = static_cast<std::uint32_t>(d.kv_src[l]);
        const auto slot = static_cast<std::uint32_t>(d.comp_slot[src]);
        n_vis = (pos + 1) / r;
        if (src == l) {
            float* skv = cache.v(seq, slot, pos);
            float* ssc = cache.v(seq, l, pos);
            s.sk.resize(Dh);
            matvec_mt(op, lay.comp_kv, s.h, s.sk);
            std::copy(s.sk.begin(), s.sk.end(), skv);
            if (r > 1) {
                s.ss.resize(Dh);
                matvec_mt(op, lay.comp_gate, s.h, s.ss);
                std::copy(s.ss.begin(), s.ss.end(), ssc);
            } else {
                std::fill(ssc, ssc + Dh, 0.0f);
            }
            if ((pos + 1) % r == 0) {
                const std::uint32_t b = pos / r;
                const std::uint32_t t0 = pos + 1 - r;
                s.comp.assign(Dh, 0.0f);
                for (std::uint32_t c = 0; c < Dh; ++c) {
                    float mx = neg_inf;
                    for (std::uint32_t t = t0; t <= pos; ++t) {
                        mx = std::max(mx, cache.v(seq, l, t)[c]);
                    }
                    float z = 0.0f, acc = 0.0f;
                    for (std::uint32_t t = t0; t <= pos; ++t) {
                        const float w = std::exp(cache.v(seq, l, t)[c] - mx);
                        z += w;
                        acc += w * cache.v(seq, slot, t)[c];
                    }
                    s.comp[c] = acc / z;
                }
                backend::rmsnorm(s.comp, lay.comp_norm, hp.rms_eps, s.comp);
                float* row = cache.k(seq, slot, b);
                std::span<float> ik(row + Dh, d.idx_dim);
                matvec_mt(op, lay.idx_k, s.comp, ik);
                backend::rmsnorm(ik, lay.idx_k_norm, hp.rms_eps, ik);
                rope_tail(ik, 1, d.idx_dim, R, b * r, rope);
                rope_tail(s.comp, 1, Dh, R, b * r, rope);
                std::copy(s.comp.begin(), s.comp.end(), row);
            }
        }

        // Indexer (top-k sources): score every visible row.
        if (d.topk_src[l] == static_cast<std::int32_t>(l)) {
            const std::uint32_t Hi = d.idx_heads, Di = d.idx_dim;
            s.iq.resize(std::size_t{Hi} * Di);
            matvec_mt(op, lay.idx_q_b, s.qr, s.iq);
            rope_tail(s.iq, Hi, Di, R, pos, rope);
            s.iw.resize(Hi);
            matvec_mt(op, lay.idx_proj, s.h, s.iw);
            const float iws = 1.0f / std::sqrt(static_cast<float>(Di * Hi));
            for (float& v : s.iw) {
                v *= iws;
            }
            s.score.assign(n_vis, 0.0f);
            for (std::uint32_t j = 0; j < n_vis; ++j) {
                const float* ik = cache.k(seq, slot, j) + Dh;
                float sc = 0.0f;
                for (std::uint32_t hh = 0; hh < Hi; ++hh) {
                    const float* qh = s.iq.data() + std::size_t{hh} * Di;
                    float dot = 0.0f;
                    for (std::uint32_t i = 0; i < Di; ++i) {
                        dot += ik[i] * qh[i];
                    }
                    sc += std::max(dot, 0.0f) * s.iw[hh];
                }
                s.score[j] = sc;
            }
            const auto cs = d.cand_src;
            if (cs >= 0 && l == static_cast<std::uint32_t>(cs) && n_vis > 0) {
                // Level 1: best blocks by their max score; the block
                // of the newest row is always kept.
                const std::uint32_t bs = d.cand_block_size;
                const std::uint32_t n_blk = (n_vis + bs - 1) / bs;
                std::vector<float> bmax(n_blk, neg_inf);
                for (std::uint32_t j = 0; j < n_vis; ++j) {
                    bmax[j / bs] = std::max(bmax[j / bs], s.score[j]);
                }
                bmax[(n_vis - 1) / bs] += 1e30f;
                s.cand = top_k(bmax, d.cand_blocks);
                std::sort(s.cand.begin(), s.cand.end());
            } else if (cs >= 0 && l > static_cast<std::uint32_t>(cs)) {
                // Level 2: only rows inside the candidate blocks.
                const std::uint32_t bs = d.cand_block_size;
                for (std::uint32_t j = 0; j < n_vis; ++j) {
                    if (!std::binary_search(s.cand.begin(), s.cand.end(),
                                            j / bs)) {
                        s.score[j] = neg_inf;
                    }
                }
            }
            auto& sel = s.topk[l];
            sel = top_k(s.score, d.idx_top_k);
            // Rows masked at level 2 never attend.
            sel.erase(std::remove_if(sel.begin(), sel.end(),
                                     [&](std::uint32_t j) {
                                         return s.score[j] == neg_inf;
                                     }),
                      sel.end());
        }
    }

    // Joint softmax over the window and the selected compressed rows,
    // with a per-head sink logit in the denominator; V = K.
    const std::uint32_t w0 = pos + 1 > d.window ? pos + 1 - d.window : 0;
    std::vector<const float*> keys;
    keys.reserve(pos + 1 - w0 + d.idx_top_k);
    for (std::uint32_t t = w0; t <= pos; ++t) {
        keys.push_back(cache.k(seq, l, t));
    }
    if (r > 0) {
        const auto slot =
            static_cast<std::uint32_t>(d.comp_slot[d.kv_src[l]]);
        for (std::uint32_t j : s.topk[static_cast<std::size_t>(
                 d.topk_src[l])]) {
            if (j < n_vis) {
                keys.push_back(cache.k(seq, slot, j));
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    s.logits_k.resize(keys.size());
    s.o.assign(std::size_t{H} * Dh, 0.0f);
    for (std::uint32_t hd = 0; hd < H; ++hd) {
        const float* qh = s.q.data() + std::size_t{hd} * Dh;
        float mx = lay.attn_sinks[hd];
        for (std::size_t k = 0; k < keys.size(); ++k) {
            float dot = 0.0f;
            for (std::uint32_t i = 0; i < Dh; ++i) {
                dot += qh[i] * keys[k][i];
            }
            s.logits_k[k] = dot * scale;
            mx = std::max(mx, s.logits_k[k]);
        }
        float z = std::exp(lay.attn_sinks[hd] - mx);
        for (float& v : s.logits_k) {
            v = std::exp(v - mx);
            z += v;
        }
        float* oh = s.o.data() + std::size_t{hd} * Dh;
        for (std::size_t k = 0; k < keys.size(); ++k) {
            const float w = s.logits_k[k] / z;
            for (std::uint32_t i = 0; i < Dh; ++i) {
                oh[i] += w * keys[k][i];
            }
        }
    }
    rope_tail(s.o, H, Dh, R, pos, rope, /*inverse=*/true);

    // Grouped output LoRA: group g maps its heads' 4096 values to
    // out_rank, then attn_output_b maps all groups to n_embd.
    const std::uint32_t G = d.out_groups, rank = d.out_rank;
    const std::size_t gin = std::size_t{H} * Dh / G;
    s.oa.resize(std::size_t{G} * rank);
    const std::size_t rb = backend::mat_row_bytes(lay.attn_out_a);
    for (std::uint32_t g = 0; g < G; ++g) {
        Mat wg = lay.attn_out_a;
        wg.data += std::size_t{g} * rank * rb;
        wg.rows = rank;
        matvec_mt(op, wg, {s.o.data() + g * gin, gin},
                  {s.oa.data() + std::size_t{g} * rank, rank});
    }
    s.a.resize(hp.n_embd);
    matvec_mt(op, lay.attn_out_b, s.oa, s.a);
}

/** One clamped SwiGLU expert on s.h: out = W_d silu(min(g, L)) * clamp(u). */
void swiglu_clamped(const backend::Ops& op, const Mat& wg, const Mat& wu,
                    const Mat& wd, float limit, Scratch& s,
                    std::span<float> out) {
    s.gate.resize(wg.rows);
    s.up.resize(wu.rows);
    matvec_mt(op, wg, s.h, s.gate);
    matvec_mt(op, wu, s.h, s.up);
    for (std::size_t i = 0; i < s.gate.size(); ++i) {
        const float g = limit > 0.0f ? std::min(s.gate[i], limit) : s.gate[i];
        const float u =
            limit > 0.0f ? std::clamp(s.up[i], -limit, limit) : s.up[i];
        s.gate[i] = g / (1.0f + std::exp(-g)) * u;
    }
    matvec_mt(op, wd, s.gate, out);
}

/** MoE on s.h into s.f (routed experts plus the shared expert). */
void moe_layer(const Hparams& hp, const TransformerModel::Layer& lay,
               std::uint32_t l, const backend::Ops& op, Scratch& s) {
    const auto& d = hp.ds41;
    const std::uint32_t E = hp.n_embd;
    s.router.resize(hp.n_expert);
    op.matvec(lay.gate_inp, s.h, s.router);
    s.sel.resize(hp.n_expert);
    for (std::uint32_t e = 0; e < hp.n_expert; ++e) {
        const float x = s.router[e];
        const float sp = x > 20.0f ? x : std::log1p(std::exp(x));
        s.router[e] = std::sqrt(sp);
        s.sel[e] = s.router[e] +
                   (lay.exp_probs_b.empty() ? 0.0f : lay.exp_probs_b[e]);
    }
    const auto picked = top_k(s.sel, hp.n_expert_used);
    float wsum = 0.0f;
    for (auto e : picked) {
        wsum += s.router[e];
    }
    s.f.assign(E, 0.0f);
    s.y.resize(E);
    for (auto e : picked) {
        const float w = s.router[e] / (wsum + 1e-20f) * hp.expert_weights_scale;
        swiglu_clamped(op, lay.gate_exps.expert(e), lay.up_exps.expert(e),
                       lay.down_exps.expert(e), d.clamp_exp[l], s, s.y);
        for (std::uint32_t i = 0; i < E; ++i) {
            s.f[i] += w * s.y[i];
        }
    }
    if (lay.gate_shexp.rows > 0) {
        swiglu_clamped(op, lay.gate_shexp, lay.up_shexp, lay.down_shexp,
                       d.clamp_shexp[l], s, s.y);
        for (std::uint32_t i = 0; i < E; ++i) {
            s.f[i] += s.y[i];
        }
    }
}

}  // namespace

void sinkhorn(std::span<float> m, std::uint32_t hc, std::uint32_t iters,
              float eps) {
    auto at = [&](std::uint32_t dst, std::uint32_t src) -> float& {
        return m[dst + hc * src];
    };
    for (std::uint32_t src = 0; src < hc; ++src) {
        float mx = -std::numeric_limits<float>::infinity();
        for (std::uint32_t dst = 0; dst < hc; ++dst) {
            mx = std::max(mx, at(dst, src));
        }
        float z = 0.0f;
        for (std::uint32_t dst = 0; dst < hc; ++dst) {
            at(dst, src) = std::exp(at(dst, src) - mx);
            z += at(dst, src);
        }
        for (std::uint32_t dst = 0; dst < hc; ++dst) {
            at(dst, src) = at(dst, src) / z + eps;
        }
    }
    auto norm_src = [&] {  // each dst row sums to ~1 over src
        for (std::uint32_t dst = 0; dst < hc; ++dst) {
            float z = 0.0f;
            for (std::uint32_t src = 0; src < hc; ++src) {
                z += at(dst, src);
            }
            for (std::uint32_t src = 0; src < hc; ++src) {
                at(dst, src) /= z + eps;
            }
        }
    };
    auto norm_dst = [&] {  // each src column sums to ~1 over dst
        for (std::uint32_t src = 0; src < hc; ++src) {
            float z = 0.0f;
            for (std::uint32_t dst = 0; dst < hc; ++dst) {
                z += at(dst, src);
            }
            for (std::uint32_t dst = 0; dst < hc; ++dst) {
                at(dst, src) /= z + eps;
            }
        }
    };
    norm_src();
    for (std::uint32_t i = 1; i < iters; ++i) {
        norm_dst();
        norm_src();
    }
}

void engram_rows(const Hparams::Ds41& d, std::uint32_t e,
                 std::span<const std::int64_t> toks,
                 std::span<std::uint32_t> rows) {
    const std::uint32_t N = d.ngram, NH = d.ngram_heads;
    const std::size_t ncols = std::size_t{N - 1} * NH;
    const std::uint64_t* mult = d.engram_mult.data() + std::size_t{e} * N;
    const std::uint64_t* prime = d.engram_size.data() + e * ncols;
    const std::uint64_t* off = d.engram_off.data() + e * ncols;
    std::uint64_t rolling = static_cast<std::uint64_t>(toks[0]) * mult[0];
    for (std::uint32_t o = 1; o < N; ++o) {
        rolling ^= static_cast<std::uint64_t>(toks[o]) * mult[o];
        for (std::uint32_t h = 0; h < NH; ++h) {
            const std::size_t col = std::size_t{o - 1} * NH + h;
            rows[col] =
                static_cast<std::uint32_t>(rolling % prime[col] + off[col]);
        }
    }
}

void load_hparams(const gguf::GgufFile& g, const std::string& p,
                  Hparams& hp) {
    auto& d = hp.ds41;
    hp.arch = Arch::kLlama;
    hp.custom_layer = true;
    hp.head_dim = need_uint(g, p + "attention.key_length");
    if (g.get_uint(p + "attention.value_length").value_or(hp.head_dim) !=
            hp.head_dim ||
        hp.n_kv_heads != 1) {
        throw gguf::Error("deepseek41: expects one kv head with K = V");
    }
    hp.n_ff_exp = need_uint(g, p + "expert_feed_forward_length");
    hp.n_expert_shared = static_cast<std::uint32_t>(
        g.get_uint(p + "expert_shared_count").value_or(0));
    if (g.get_uint(p + "expert_gating_func").value_or(0) != 4) {
        throw gguf::Error("deepseek41: expects sqrt-softplus gating");
    }
    hp.expert_weights_norm =
        g.get_bool(p + "expert_weights_norm").value_or(true);
    hp.expert_weights_scale = get_float(g, p + "expert_weights_scale", 1.0f);
    // Compressed layers rope with yarn (factor, original context).
    const float factor = get_float(g, p + "rope.scaling.factor", 1.0f);
    hp.yarn.freq_scale = 1.0f / factor;
    hp.yarn.n_ctx_orig = static_cast<std::uint32_t>(
        g.get_uint(p + "rope.scaling.original_context_length").value_or(0));
    hp.yarn.beta_fast = get_float(g, p + "rope.scaling.yarn_beta_fast", 32.0f);
    hp.yarn.beta_slow = get_float(g, p + "rope.scaling.yarn_beta_slow", 1.0f);
    hp.yarn.mscale = 1.0f;
    hp.kq_scale = 1.0f / std::sqrt(static_cast<float>(hp.head_dim));

    const std::uint32_t L = hp.n_layers;
    d.rope_dims = need_uint(g, p + "rope.dimension_count");
    d.window = need_uint(g, p + "attention.sliding_window");
    d.compress_rope_base =
        get_float(g, p + "attention.compress_rope_freq_base", 160000.0f);
    const auto ratios = uint_array(g, p + "attention.compress_ratios");
    if (ratios.size() < L) {
        throw gguf::Error("deepseek41: short compress_ratios");
    }
    d.ratio.assign(ratios.begin(), ratios.begin() + L);
    const auto kv_sources = uint_array(g, p + "attention.kv_source_layers");
    const auto idx_sources =
        uint_array(g, p + "attention.indexer.source_layers");
    d.kv_src.assign(L, -1);
    d.topk_src.assign(L, -1);
    d.comp_slot.assign(L, -1);
    std::int32_t last_kv = -1, last_idx = -1;
    std::uint32_t slots = 0;
    for (std::uint32_t l = 0; l < L; ++l) {
        if (std::find(kv_sources.begin(), kv_sources.end(), l) !=
            kv_sources.end()) {
            last_kv = static_cast<std::int32_t>(l);
            d.comp_slot[l] = static_cast<std::int32_t>(L + slots++);
        }
        if (std::find(idx_sources.begin(), idx_sources.end(), l) !=
            idx_sources.end()) {
            last_idx = static_cast<std::int32_t>(l);
        }
        if (d.ratio[l] == 0) {
            continue;
        }
        if (d.ratio[l] > 2 || last_kv < 0 || last_idx < last_kv ||
            d.ratio[static_cast<std::size_t>(last_kv)] != d.ratio[l] ||
            d.ratio[static_cast<std::size_t>(last_idx)] != d.ratio[l]) {
            throw gguf::Error("deepseek41: unsupported compression layout");
        }
        d.kv_src[l] = last_kv;
        d.topk_src[l] = last_idx;
    }
    d.idx_heads = need_uint(g, p + "attention.indexer.head_count");
    d.idx_dim = need_uint(g, p + "attention.indexer.key_length");
    d.idx_top_k = need_uint(g, p + "attention.indexer.top_k");
    const auto* cs = g.find(p + "attention.indexer.candidate_source_layer");
    d.cand_src = cs == nullptr ? -1 : static_cast<std::int32_t>(cs->i);
    d.cand_blocks = static_cast<std::uint32_t>(
        g.get_uint(p + "attention.indexer.candidate_top_k_blocks").value_or(0));
    d.cand_block_size = static_cast<std::uint32_t>(
        g.get_uint(p + "attention.indexer.candidate_block_size").value_or(1));
    d.out_groups = need_uint(g, p + "attention.output_group_count");
    d.out_rank = need_uint(g, p + "attention.output_lora_rank");
    d.hc = need_uint(g, p + "hyper_connection.count");
    if (d.hc != 4) {
        throw gguf::Error("deepseek41: expects 4 hyper-connection streams");
    }
    d.sinkhorn_iters = need_uint(g, p + "hyper_connection.sinkhorn_iterations");
    d.hc_eps = get_float(g, p + "hyper_connection.epsilon", 1e-6f);
    d.clamp_exp = float_array(g, p + "swiglu_clamp_exp", L);
    d.clamp_shexp = g.get_array(p + "swiglu_clamp_shexp") != nullptr
                        ? float_array(g, p + "swiglu_clamp_shexp", L)
                        : d.clamp_exp;

    const auto ple = uint_array(g, p + "ple.layers");
    d.engram_layers.assign(ple.begin(), ple.end());
    d.ngram = need_uint(g, p + "ple.ngram_size");
    d.ngram_heads = need_uint(g, p + "ple.heads_per_ngram");
    d.engram_dim = need_uint(g, p + "embedding_length_per_layer_input");
    d.engram_mult = uint_array(g, p + "ple.layer_multipliers");
    d.engram_off = uint_array(g, p + "ple.head_offsets");
    d.engram_size = uint_array(g, p + "ple.head_vocab_sizes");
    const std::size_t ncols = std::size_t{d.ngram - 1} * d.ngram_heads;
    if (d.engram_mult.size() != d.engram_layers.size() * d.ngram ||
        d.engram_off.size() != d.engram_layers.size() * ncols ||
        d.engram_size.size() != d.engram_off.size()) {
        throw gguf::Error("deepseek41: inconsistent engram tables");
    }
    d.engram_pad = need_uint(g, p + "engram.pad_token_id");
    const auto tm = uint_array(g, p + "engram.token_map");
    d.token_map.clear();
    for (auto v : tm) {
        d.token_map.push_back(static_cast<std::int32_t>(v));
    }
    if (d.engram_pad >= d.token_map.size()) {
        throw gguf::Error("deepseek41: engram pad id out of range");
    }
}

void load_layer(const gguf::GgufFile& g, const std::string& bp,
                const Hparams& hp, TransformerModel::Layer& lay) {
    const auto& d = hp.ds41;
    const std::uint32_t E = hp.n_embd, Dh = hp.head_dim, H = hp.n_heads;
    const std::uint32_t l =
        static_cast<std::uint32_t>(std::stoul(bp.substr(4)));  // "blk.N."
    const auto* qa = g.find_tensor(bp + "attn_q_a.weight");
    if (qa == nullptr) {
        throw gguf::Error("missing tensor: " + bp + "attn_q_a.weight");
    }
    const auto q_rank = static_cast<std::uint32_t>(qa->ne[1]);
    lay.wq_a = need_mat(g, bp + "attn_q_a.weight", E, q_rank);
    lay.q_a_norm = need_vec(g, bp + "attn_q_a_norm.weight", q_rank);
    lay.wq_b = need_mat(g, bp + "attn_q_b.weight", q_rank, H * Dh);
    lay.wkv_a = need_mat(g, bp + "attn_kv.weight", E, Dh);
    lay.kv_a_norm = need_vec(g, bp + "attn_kv_a_norm.weight", Dh);
    lay.attn_sinks = need_vec(g, bp + "attn_sinks.weight", H);
    lay.attn_out_a = need_mat(g, bp + "attn_output_a.weight",
                              H * Dh / d.out_groups, d.out_groups * d.out_rank);
    lay.attn_out_b =
        need_mat(g, bp + "attn_output_b.weight", d.out_groups * d.out_rank, E);
    const std::uint32_t mixes = 2 * d.hc + d.hc * d.hc;
    lay.hc_attn_fn = need_mat(g, bp + "hc_attn_fn.weight", d.hc * E, mixes);
    lay.hc_attn_base = need_vec(g, bp + "hc_attn_base.weight", mixes);
    lay.hc_attn_scale = need_vec(g, bp + "hc_attn_scale.weight", 3);
    lay.hc_ffn_fn = need_mat(g, bp + "hc_ffn_fn.weight", d.hc * E, mixes);
    lay.hc_ffn_base = need_vec(g, bp + "hc_ffn_base.weight", mixes);
    lay.hc_ffn_scale = need_vec(g, bp + "hc_ffn_scale.weight", 3);
    if (d.kv_src[l] == static_cast<std::int32_t>(l)) {
        lay.comp_kv = need_mat(g, bp + "attn_compressor_kv.weight", E, Dh);
        lay.comp_norm = need_vec(g, bp + "attn_compressor_norm.weight", Dh);
        if (d.ratio[l] > 1) {
            lay.comp_gate =
                need_mat(g, bp + "attn_compressor_gate.weight", E, Dh);
        }
        lay.idx_k = need_mat(g, bp + "indexer.attn_k.weight", Dh, d.idx_dim);
        lay.idx_k_norm = need_vec(g, bp + "indexer.k_norm.weight", d.idx_dim);
    }
    if (d.topk_src[l] == static_cast<std::int32_t>(l)) {
        lay.idx_q_b = need_mat(g, bp + "indexer.attn_q_b.weight", q_rank,
                               d.idx_heads * d.idx_dim);
        lay.idx_proj = need_mat(g, bp + "indexer.proj.weight", E, d.idx_heads);
    }
    const auto e = std::find(d.engram_layers.begin(), d.engram_layers.end(), l);
    if (e != d.engram_layers.end()) {
        const std::uint32_t ncols = (d.ngram - 1) * d.ngram_heads;
        lay.engram_kv = need_mat(g, bp + "engram_kv.weight",
                                 ncols * d.engram_dim, (d.hc + 1) * E);
        lay.engram_gate = need_mat(g, bp + "engram_gate.weight", E, d.hc);
        const auto* t = g.find_tensor("per_layer_token_embd.weight");
        if (t == nullptr) {
            throw gguf::Error("missing tensor: per_layer_token_embd.weight");
        }
        lay.engram_table = need_mat(g, "per_layer_token_embd.weight",
                                    d.engram_dim,
                                    static_cast<std::uint32_t>(t->ne[1]));
        // Every bucket of this layer must fall inside the table.
        const auto ei = static_cast<std::size_t>(e - d.engram_layers.begin());
        for (std::size_t c = 0; c < ncols; ++c) {
            const std::size_t k = ei * ncols + c;
            if (d.engram_size[k] < 2 ||
                d.engram_off[k] + d.engram_size[k] > lay.engram_table.rows) {
                throw gguf::Error("deepseek41: engram bucket out of range");
            }
        }
    }
}

void attention(const TransformerModel&, const TransformerModel::Layer&,
               kv::PagedKvCache&, kv::PagedKvCache::Seq&,
               TransformerModel::Workspace&, std::uint32_t, std::uint32_t) {
    throw std::logic_error("deepseek41: attention runs in its forward");
}

std::uint32_t kv_dim(const Hparams& hp) {
    // A compressed row: the latent (head_dim) then its index key.
    return hp.head_dim + hp.ds41.idx_dim;
}

std::uint32_t cache_layers(const Hparams& hp) {
    std::uint32_t slots = 0;
    for (auto s : hp.ds41.comp_slot) {
        slots += s >= 0 ? 1u : 0u;
    }
    return hp.n_layers + slots;
}

void forward(const TransformerModel& m, tok::TokenId token,
             kv::PagedKvCache& cache, kv::PagedKvCache::Seq& seq,
             TransformerModel::Workspace& ws, std::span<float> logits) {
    (void)ws;
    if (cache.quantized()) {
        throw std::invalid_argument("deepseek41: needs an F32 KV cache");
    }
    const Hparams& hp = m.hparams();
    const auto& d = hp.ds41;
    const backend::Ops& op = m.active_backend().ops;
    const std::uint32_t E = hp.n_embd, L = hp.n_layers;
    const std::uint32_t pos = seq.n_tokens;
    Scratch& s = scratch();
    s.topk.assign(L, {});
    s.cand.clear();

    // Embedding: every stream starts as the token embedding; the id is
    // kept for later positions' Engram hashes.
    s.h.resize(E);
    op.dequant_row(m.embedding(), static_cast<std::uint32_t>(token), s.h);
    s.X.resize(std::size_t{d.hc} * E);
    for (std::uint32_t k = 0; k < d.hc; ++k) {
        std::copy(s.h.begin(), s.h.end(), s.X.begin() + std::size_t{k} * E);
    }
    cache.v(seq, 0, pos)[0] = static_cast<float>(token);

    float pre_prev[4] = {1.0f, 0.0f, 0.0f, 0.0f};  // layer 0: stream 0
    for (std::uint32_t l = 0; l < L; ++l) {
        const auto& lay = m.layers()[l];
        const auto e =
            std::find(d.engram_layers.begin(), d.engram_layers.end(), l);
        if (e != d.engram_layers.end()) {
            engram(m, lay, static_cast<std::uint32_t>(e - d.engram_layers.begin()),
                   cache, seq, token, pos, s);
        }
        // Attention: own post / comb mixes, previous FFN's pre-mix.
        s.Xres = s.X;
        const Mixes ma = hc_mixes(hp, s.X, lay.hc_attn_fn, lay.hc_attn_base,
                                  lay.hc_attn_scale, op, s);
        hc_pre(s.X, pre_prev, E, s.h);
        backend::rmsnorm(s.h, lay.attn_norm, hp.rms_eps, s.h);
        attention_layer(m, lay, cache, seq, l, pos, s);
        hc_post(s.a, s.Xres, ma, E, s.X);

        // FFN: own post / comb mixes, this layer's attention pre-mix.
        s.Xres = s.X;
        const Mixes mf = hc_mixes(hp, s.X, lay.hc_ffn_fn, lay.hc_ffn_base,
                                  lay.hc_ffn_scale, op, s);
        std::copy(mf.pre, mf.pre + 4, pre_prev);
        hc_pre(s.X, ma.pre, E, s.h);
        backend::rmsnorm(s.h, lay.ffn_norm, hp.rms_eps, s.h);
        moe_layer(hp, lay, l, op, s);
        hc_post(s.f, s.Xres, mf, E, s.X);
    }

    hc_pre(s.X, pre_prev, E, s.h);
    backend::rmsnorm(s.h, m.output_norm(), hp.rms_eps, s.h);
    matvec_mt(op, m.output_weight(), s.h, logits);
}

}  // namespace locus::model::deepseek41
