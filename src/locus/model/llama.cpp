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

/**
 * How many row-slices to split a `rows`-row matvec across: the
 * thread-pool width, capped by LOCUS_THREADS and by keeping at
 * least kMinRowsPerSlice rows per slice so dispatch overhead never
 * dominates. Returns <= 1 when the work should run inline.
 */
std::size_t mt_slices(std::uint32_t rows) {
    constexpr std::uint32_t kMinRowsPerSlice = 64;
    std::size_t t = sys::ThreadPool::instance().parallelism();
    if (const char* env = std::getenv("LOCUS_THREADS")) {
        const long v = std::atol(env);
        if (v >= 1) {
            t = std::min<std::size_t>(
                t, static_cast<std::size_t>(v));
        }
    }
    return std::min<std::size_t>(t, rows / kMinRowsPerSlice);
}

}  // namespace

void matvec_mt(const backend::Ops& op, const Mat& w,
               std::span<const float> x, std::span<float> out) {
    // A backend whose matvec is not re-entrant (Vulkan's single
    // VulkanContext singleton) must run inline: parallel calls
    // crash the driver. This path is hit when a GPU full-forward
    // bails to the per-op fallback on an unshadered weight type.
    if (!op.mt_safe) {
        op.matvec(w, x, out);
        return;
    }
    const std::size_t t = mt_slices(w.rows);
    if (t <= 1) {
        op.matvec(w, x, out);
        return;
    }
    sys::ThreadPool::instance().parallel_for(t, [&](std::size_t i) {
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
    m.file_backed_ = g.file_backed();
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
                    hp.n_ff_shexp > 0
                        ? hp.n_ff_shexp
                        : hp.n_ff_exp * hp.n_expert_shared;
                lay.gate_shexp =
                    need_mat(g, bp + "ffn_gate_shexp.weight",
                             hp.n_embd, sh);
                lay.up_shexp =
                    need_mat(g, bp + "ffn_up_shexp.weight",
                             hp.n_embd, sh);
                lay.down_shexp =
                    need_mat(g, bp + "ffn_down_shexp.weight",
                             sh, hp.n_embd);
                // qwen2moe gates the shared expert by sigmoid of a
                // 1-logit projection; absent on deepseek2 (ungated).
                if (g.find_tensor(bp + "ffn_gate_inp_shexp.weight") !=
                    nullptr) {
                    lay.gate_inp_shexp = need_mat(
                        g, bp + "ffn_gate_inp_shexp.weight",
                        hp.n_embd, 1);
                }
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
         hp_.n_ff_exp * hp_.n_expert_shared, hp_.n_ff_shexp});
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
    std::uint32_t n_blocks, kv::KvType kv_type) const {
    kv::PagedKvCache::Geometry geom;
    geom.n_layers = hp_.n_layers;
    geom.kv_dim = spec_->kv_dim(hp_);
    geom.block_tokens = 16;
    geom.kv_type = kv_type;
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
    // GPU-mapped KV pool when the backend provides one (alloc_kv is
    // set only for Vulkan today; it hands back unified-memory float*).
    if (backend_->ops.alloc_kv != nullptr) {
        if (kv_type == kv::KvType::kF32) {
            float* storage = backend_->ops.alloc_kv(
                kv::PagedKvCache::pool_floats(geom));
            if (storage != nullptr) {
                return kv::PagedKvCache(geom, storage);
            }
        } else {
            // R14 #45: quantized GPU-mapped KV pool. Size the float
            // allocation to cover pool_bytes and reinterpret to the
            // byte pool the Vulkan attention shaders quantize into
            // and read back (block-of-32 Q8/Q4 layout).
            const std::size_t bytes =
                kv::PagedKvCache::pool_bytes(geom);
            float* storage =
                backend_->ops.alloc_kv((bytes + 3) / 4);
            if (storage != nullptr) {
                return kv::PagedKvCache(
                    geom, reinterpret_cast<std::uint8_t*>(storage));
            }
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
            // The same one-step-ahead schedule feeds both tiers:
            // madvise (SSD -> page cache) and, on backends with
            // a weight pager, op.prefetch (host -> device).
            if (l + 1 < hp_.n_layers) {
                advise_layer_statics(layers_[l + 1]);
                if (op.prefetch != nullptr) {
                    for_each_static_mat(
                        layers_[l + 1],
                        [&](const backend::Mat& m) {
                            if (m.data != nullptr &&
                                m.rows > 0) {
                                op.prefetch(m);
                            }
                        });
                }
            } else {
                advise_mat(out_w_);
                if (op.prefetch != nullptr) {
                    op.prefetch(out_w_);
                }
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

void LlamaModel::embed(std::span<const tok::TokenId> tokens,
                       kv::PagedKvCache& cache,
                       kv::PagedKvCache::Seq& seq, Workspace& ws,
                       std::span<float> out) const {
    if (tokens.empty()) {
        throw std::invalid_argument("embed: empty input");
    }
    if (out.size() != hp_.n_embd) {
        throw std::invalid_argument("embed: out must be n_embd");
    }
    if (backend_->name == "vulkan") {
        throw std::invalid_argument(
            "embed: needs a CPU/CUDA backend");
    }
    // Run the sequence through the layer stack; forward() leaves the
    // last token's final-normed hidden in ws.xb (the LM head reads
    // it), which is the embedding under last-token pooling.
    std::vector<float> logits(hp_.n_vocab);
    for (tok::TokenId t : tokens) {
        if (!cache.ensure_capacity(seq, 1)) {
            throw std::runtime_error("embed: cache exhausted");
        }
        forward(t, cache, seq, ws, logits);
    }
    std::copy_n(ws.xb.data(), hp_.n_embd, out.data());
}

bool LlamaModel::supports_batch() const {
    // forward_batch batches the FFN through op.matvec and reuses
    // the per-token attention, so every CPU/CUDA arch (llama /
    // deepseek2 / glm-dsa, dense or MoE) is supported. The Vulkan
    // backend has its own full forward (vulkan_forward) that this
    // CPU-driver path would bypass, so it opts out.
    return backend_->name != "vulkan";
}

void LlamaModel::forward_batch(std::span<const tok::TokenId> toks,
                               kv::PagedKvCache& cache,
                               kv::PagedKvCache::Seq& seq,
                               Workspace& ws,
                               std::span<float> logits,
                               bool all_logits) const {
    using namespace locus::backend;
    const auto n = static_cast<std::uint32_t>(toks.size());
    if (n == 0) {
        throw std::invalid_argument("forward_batch: empty batch");
    }
    if (!supports_batch()) {
        throw std::invalid_argument(
            "forward_batch: unsupported backend");
    }
    const std::uint32_t base = seq.n_tokens;
    for (auto t : toks) {
        if (t < 0 ||
            static_cast<std::uint32_t>(t) >= hp_.n_vocab) {
            throw std::invalid_argument("token id out of vocab");
        }
    }
    if (base + n > hp_.n_ctx || base + n > cache.capacity(seq)) {
        throw std::invalid_argument(
            "forward_batch: seq capacity not ensured");
    }
    const Ops& op = backend_->ops;
    const std::uint32_t E = hp_.n_embd;
    const std::uint32_t ff = hp_.n_ff;

    // Per-token residual stream x, plus batched dense-FFN scratch.
    std::vector<float> x(static_cast<std::size_t>(n) * E);
    std::vector<float> xbf(static_cast<std::size_t>(n) * E);
    std::vector<float> xb2(static_cast<std::size_t>(n) * E);
    std::vector<float> gate(static_cast<std::size_t>(n) * ff);
    std::vector<float> up(static_cast<std::size_t>(n) * ff);

    for (std::uint32_t t = 0; t < n; ++t) {
        op.dequant_row(embd_, static_cast<std::uint32_t>(toks[t]),
                       {x.data() + static_cast<std::size_t>(t) * E,
                        E});
    }

    for (std::uint32_t l = 0; l < hp_.n_layers; ++l) {
        const Layer& lay = layers_[l];
        // Attention per token via the shared arch implementation
        // (reusing it keeps every arch byte-identical to the
        // sequential path). The attention projections are the only
        // weights not yet batched here.
        for (std::uint32_t t = 0; t < n; ++t) {
            const std::size_t o = static_cast<std::size_t>(t) * E;
            rmsnorm({x.data() + o, E}, lay.attn_norm, hp_.rms_eps,
                    ws.xb);
            spec_->attention(*this, lay, cache, seq, ws, l,
                             base + t);
            matvec_mt(op, lay.wo, ws.out, ws.xb2);
            for (std::uint32_t i = 0; i < E; ++i) {
                x[o + i] += ws.xb2[i];
            }
        }
        if (!lay.is_moe()) {
            // Dense FFN: weight-stationary batched matvecs.
            for (std::uint32_t t = 0; t < n; ++t) {
                const std::size_t o =
                    static_cast<std::size_t>(t) * E;
                rmsnorm({x.data() + o, E}, lay.ffn_norm,
                        hp_.rms_eps, {xbf.data() + o, E});
            }
            matvec_batch(op, lay.w_gate, xbf, gate, n);
            matvec_batch(op, lay.w_up, xbf, up, n);
            for (std::uint32_t t = 0; t < n; ++t) {
                const std::size_t o =
                    static_cast<std::size_t>(t) * ff;
                silu_mul({gate.data() + o, ff}, {up.data() + o, ff},
                         {gate.data() + o, ff});
            }
            matvec_batch(op, lay.w_down, gate, xb2, n);
            for (std::uint32_t i = 0;
                 i < static_cast<std::size_t>(n) * E; ++i) {
                x[i] += xb2[i];
            }
        } else {
            // MoE: ffn-norm all tokens, then apply the routed
            // experts weight-stationary (each read once across the
            // tokens that picked it). Byte-identical to per token.
            for (std::uint32_t t = 0; t < n; ++t) {
                const std::size_t o =
                    static_cast<std::size_t>(t) * E;
                rmsnorm({x.data() + o, E}, lay.ffn_norm,
                        hp_.rms_eps, {xbf.data() + o, E});
            }
            moe_ffn_batch(lay, l, xbf, x, n, ws);
        }
    }

    // Prefill needs only the last token's logits; speculative verify
    // needs every position's (logits is then n * n_vocab).
    if (all_logits) {
        const std::uint32_t V = hp_.n_vocab;
        for (std::uint32_t t = 0; t < n; ++t) {
            rmsnorm({x.data() + static_cast<std::size_t>(t) * E, E},
                    out_norm_, hp_.rms_eps, ws.xb);
            matvec_mt(op, out_w_, ws.xb,
                      logits.subspan(static_cast<std::size_t>(t) * V,
                                     V));
        }
    } else {
        const std::size_t last = static_cast<std::size_t>(n - 1) * E;
        rmsnorm({x.data() + last, E}, out_norm_, hp_.rms_eps, ws.xb);
        matvec_mt(op, out_w_, ws.xb, logits);
    }
    seq.n_tokens = base + n;
}

void LlamaModel::forward_batch_decode(
    std::span<const tok::TokenId> toks, kv::PagedKvCache& cache,
    std::span<kv::PagedKvCache::Seq* const> seqs, Workspace& ws,
    std::span<float> logits) const {
    using namespace locus::backend;
    const auto n = static_cast<std::uint32_t>(toks.size());
    if (n == 0) {
        throw std::invalid_argument("forward_batch_decode: empty");
    }
    if (seqs.size() != n) {
        throw std::invalid_argument(
            "forward_batch_decode: seqs/tokens size mismatch");
    }
    if (!supports_batch()) {
        throw std::invalid_argument(
            "forward_batch_decode: unsupported backend");
    }
    const Ops& op = backend_->ops;
    const std::uint32_t E = hp_.n_embd;
    const std::uint32_t ff = hp_.n_ff;
    const std::uint32_t V = hp_.n_vocab;

    // Each token is an independent sequence at its own position.
    std::vector<std::uint32_t> pos(n);
    for (std::uint32_t t = 0; t < n; ++t) {
        if (toks[t] < 0 ||
            static_cast<std::uint32_t>(toks[t]) >= hp_.n_vocab) {
            throw std::invalid_argument("token id out of vocab");
        }
        pos[t] = seqs[t]->n_tokens;
        if (pos[t] + 1 > hp_.n_ctx ||
            pos[t] + 1 > cache.capacity(*seqs[t])) {
            throw std::invalid_argument(
                "forward_batch_decode: seq capacity not ensured");
        }
    }

    std::vector<float> x(static_cast<std::size_t>(n) * E);
    std::vector<float> xbf(static_cast<std::size_t>(n) * E);
    std::vector<float> xb2(static_cast<std::size_t>(n) * E);
    std::vector<float> gate(static_cast<std::size_t>(n) * ff);
    std::vector<float> up(static_cast<std::size_t>(n) * ff);

    for (std::uint32_t t = 0; t < n; ++t) {
        op.dequant_row(embd_, static_cast<std::uint32_t>(toks[t]),
                       {x.data() + static_cast<std::size_t>(t) * E,
                        E});
    }

    for (std::uint32_t l = 0; l < hp_.n_layers; ++l) {
        const Layer& lay = layers_[l];
        // Attention: each token against ITS OWN sequence's cache.
        for (std::uint32_t t = 0; t < n; ++t) {
            const std::size_t o = static_cast<std::size_t>(t) * E;
            rmsnorm({x.data() + o, E}, lay.attn_norm, hp_.rms_eps,
                    ws.xb);
            spec_->attention(*this, lay, cache, *seqs[t], ws, l,
                             pos[t]);
            matvec_mt(op, lay.wo, ws.out, ws.xb2);
            for (std::uint32_t i = 0; i < E; ++i) {
                x[o + i] += ws.xb2[i];
            }
        }
        for (std::uint32_t t = 0; t < n; ++t) {
            const std::size_t o = static_cast<std::size_t>(t) * E;
            rmsnorm({x.data() + o, E}, lay.ffn_norm, hp_.rms_eps,
                    {xbf.data() + o, E});
        }
        if (!lay.is_moe()) {
            matvec_batch(op, lay.w_gate, xbf, gate, n);
            matvec_batch(op, lay.w_up, xbf, up, n);
            for (std::uint32_t t = 0; t < n; ++t) {
                const std::size_t o =
                    static_cast<std::size_t>(t) * ff;
                silu_mul({gate.data() + o, ff}, {up.data() + o, ff},
                         {gate.data() + o, ff});
            }
            matvec_batch(op, lay.w_down, gate, xb2, n);
            for (std::uint32_t i = 0;
                 i < static_cast<std::size_t>(n) * E; ++i) {
                x[i] += xb2[i];
            }
        } else {
            moe_ffn_batch(lay, l, xbf, x, n, ws);
        }
    }

    // Every decode token needs logits.
    for (std::uint32_t t = 0; t < n; ++t) {
        const std::size_t o = static_cast<std::size_t>(t) * E;
        rmsnorm({x.data() + o, E}, out_norm_, hp_.rms_eps, ws.xb);
        matvec_mt(op, out_w_, ws.xb,
                  logits.subspan(static_cast<std::size_t>(t) * V,
                                 V));
    }
    for (std::uint32_t t = 0; t < n; ++t) {
        seqs[t]->n_tokens = pos[t] + 1;
    }
}

namespace {

/** Cached LOCUS_BATCH_DEQUANT toggle (read once). */
bool batch_dequant_enabled() {
    static const bool on =
        std::getenv("LOCUS_BATCH_DEQUANT") != nullptr;
    return on;
}

/**
 * Runs body(r0, r1) over row-slices of a rows-row matvec: threaded
 * across mt_slices() when the backend is re-entrant, else inline.
 * Each slice carries the full n-token batch, so the weight rows in
 * that slice stay hot across all n tokens (weight-stationary) while
 * the slices themselves fan out over cores.
 */
template <typename Body>
void for_row_slices(const backend::Ops& op, std::uint32_t rows,
                    Body&& body) {
    const std::size_t nslice = op.mt_safe ? mt_slices(rows) : 0;
    if (nslice <= 1) {
        body(0u, rows);
        return;
    }
    sys::ThreadPool::instance().parallel_for(
        nslice, [&](std::size_t i) {
            const std::uint32_t r0 = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(rows) * i / nslice);
            const std::uint32_t r1 = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(rows) * (i + 1) /
                nslice);
            body(r0, r1);
        });
}

}  // namespace

void matvec_batch_deq(const backend::Ops& op, const Mat& w,
                      std::span<const float> x_batch,
                      std::span<float> out_batch, std::uint32_t n) {
    const std::uint32_t xc = w.cols;
    const std::uint32_t oc = w.rows;
    for_row_slices(op, oc, [&](std::uint32_t r0, std::uint32_t r1) {
        std::vector<float> row(xc);
        for (std::uint32_t r = r0; r < r1; ++r) {
            op.dequant_row(w, r, row);
            for (std::uint32_t t = 0; t < n; ++t) {
                const float* x = x_batch.data() +
                                 static_cast<std::size_t>(t) * xc;
                float acc = 0.0f;
                for (std::uint32_t c = 0; c < xc; ++c) {
                    acc += row[c] * x[c];
                }
                out_batch[static_cast<std::size_t>(t) * oc + r] =
                    acc;
            }
        }
    });
}

void matvec_batch(const backend::Ops& op, const Mat& w,
                  std::span<const float> x_batch,
                  std::span<float> out_batch, std::uint32_t n) {
    if (batch_dequant_enabled()) {
        matvec_batch_deq(op, w, x_batch, out_batch, n);
        return;
    }
    const std::uint32_t xc = w.cols;
    const std::uint32_t oc = w.rows;
    if (op.matvec_batch != nullptr) {
        // R11: the backend's register-blocked kernel reads each
        // weight block once and reuses it across all n tokens. Its
        // output is row-major ([row r, token t] at r*n + t), so a
        // row-slice is a contiguous span we can thread. Then
        // transpose back to the token-major layout downstream wants.
        std::vector<float> rt(static_cast<std::size_t>(oc) * n);
        if (op.batch_self_parallel) {
            // A GPU kernel grids over all rows in one launch: hand it
            // the whole matrix rather than fanning slices across CPU
            // threads (which would issue one launch per slice).
            op.matvec_batch(w, x_batch, rt, n);
        } else {
            for_row_slices(
                op, oc, [&](std::uint32_t r0, std::uint32_t r1) {
                    const Mat sub = backend::mat_rows(w, r0, r1 - r0);
                    op.matvec_batch(
                        sub, x_batch,
                        {rt.data() +
                             static_cast<std::size_t>(r0) * n,
                         static_cast<std::size_t>(r1 - r0) * n},
                        n);
                });
        }
        for (std::uint32_t r = 0; r < oc; ++r) {
            for (std::uint32_t t = 0; t < n; ++t) {
                out_batch[static_cast<std::size_t>(t) * oc + r] =
                    rt[static_cast<std::size_t>(r) * n + t];
            }
        }
        return;
    }
    for_row_slices(op, oc, [&](std::uint32_t r0, std::uint32_t r1) {
        const Mat sub = backend::mat_rows(w, r0, r1 - r0);
        for (std::uint32_t t = 0; t < n; ++t) {
            op.matvec(
                sub,
                x_batch.subspan(static_cast<std::size_t>(t) * xc,
                                xc),
                out_batch.subspan(
                    static_cast<std::size_t>(t) * oc + r0,
                    r1 - r0));
        }
    });
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
            // Pager tier: expert(e) yields the same .data the
            // matmuls below pass to matvec -- the page key.
            if (op.prefetch != nullptr) {
                op.prefetch(lay.gate_exps.expert(e));
                op.prefetch(lay.up_exps.expert(e));
                op.prefetch(lay.down_exps.expert(e));
            }
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
    // Weight window (opt-in): drop this layer's routed experts
    // right after use. Routing rarely repeats an expert on the
    // next token, and statics/shared experts are never dropped,
    // so the re-read cost is small while the resident set stays
    // flat -- streamed models stop building memory pressure.
    // file_backed_ gate: DONTNEED on an in-memory (anonymous)
    // image would DISCARD the weights, not just evict them.
    if (file_backed_ &&
        std::getenv("LOCUS_WEIGHT_WINDOW") != nullptr) {
        for (const auto& [e, wgt] : picked) {
            sys::advise_dontneed(lay.gate_exps.expert(e).data,
                                 lay.gate_exps.expert_bytes);
            sys::advise_dontneed(lay.up_exps.expert(e).data,
                                 lay.up_exps.expert_bytes);
            sys::advise_dontneed(lay.down_exps.expert(e).data,
                                 lay.down_exps.expert_bytes);
        }
    }
    if (hp_.n_expert_shared > 0) {
        const std::uint32_t shexp_ff =
            hp_.n_ff_shexp > 0 ? hp_.n_ff_shexp
                               : n_ff * hp_.n_expert_shared;
        // qwen2moe gates the shared output by sigmoid(w_gate . x);
        // deepseek2 (no gate tensor) keeps the ungated 1.0 scale.
        float g = 1.0f;
        if (lay.gate_inp_shexp.rows > 0) {
            float logit = 0.0f;
            op.matvec(lay.gate_inp_shexp, ws.xb, {&logit, 1});
            g = 1.0f / (1.0f + std::exp(-logit));
        }
        swiglu_into_acc(lay.gate_shexp, lay.up_shexp,
                        lay.down_shexp, shexp_ff, g);
    }
}

void LlamaModel::moe_ffn_batch(const Layer& lay,
                               std::uint32_t layer,
                               const std::vector<float>& xbf,
                               std::vector<float>& x,
                               std::uint32_t n, Workspace& ws) const {
    using namespace locus::backend;
    const Ops& op = backend_->ops;
    const std::uint32_t E = hp_.n_embd;
    const std::uint32_t ff = hp_.n_ff_exp;
    const std::uint32_t k = hp_.n_expert_used;

    // Phase 1: route each token (byte-identical to moe_ffn).
    std::vector<std::vector<std::pair<std::uint32_t, float>>>
        picked(n);
    for (std::uint32_t t = 0; t < n; ++t) {
        op.matvec(
            lay.gate_inp,
            {xbf.data() + static_cast<std::size_t>(t) * E, E},
            ws.router);
        picked[t] = moe_select(hp_, lay, ws.router);
        if (MoeStats::enabled()) {
            for (const auto& [e, wgt] : picked[t]) {
                MoeStats::record(layer, e, hp_.n_layers,
                                 hp_.n_expert);
            }
        }
    }

    // Group (token, slot) by expert so each expert's weights are
    // read once across the tokens that picked it.
    struct Use {
        std::uint32_t e, t, s;
    };
    std::vector<Use> uses;
    uses.reserve(static_cast<std::size_t>(n) * k);
    for (std::uint32_t t = 0; t < n; ++t) {
        for (std::uint32_t s = 0;
             s < static_cast<std::uint32_t>(picked[t].size());
             ++s) {
            uses.push_back({picked[t][s].first, t, s});
        }
    }
    std::sort(uses.begin(), uses.end(),
              [](const Use& a, const Use& b) { return a.e < b.e; });

    // Readahead the union once (first appearance of each expert).
    if (readahead_enabled()) {
        std::uint32_t prev = UINT32_MAX;
        for (const auto& u : uses) {
            if (u.e == prev) {
                continue;
            }
            prev = u.e;
            const Mat g = lay.gate_exps.expert(u.e);
            const Mat up = lay.up_exps.expert(u.e);
            const Mat d = lay.down_exps.expert(u.e);
            sys::advise_willneed(g.data, lay.gate_exps.expert_bytes);
            sys::advise_willneed(up.data, lay.up_exps.expert_bytes);
            sys::advise_willneed(d.data, lay.down_exps.expert_bytes);
            if (op.prefetch != nullptr) {
                op.prefetch(g);
                op.prefetch(up);
                op.prefetch(d);
            }
        }
    }

    // Phase 3: weight-stationary swiglu -> res[(t*k + s)*E].
    std::vector<float> res(static_cast<std::size_t>(n) * k * E);
    std::vector<float> down(E);
    auto swiglu = [&](const Mat& wg, const Mat& wu, const Mat& wd,
                      std::uint32_t ffw, const float* xin,
                      float* dout) {
        matvec_mt(op, wg, {xin, E}, {ws.gate.data(), ffw});
        matvec_mt(op, wu, {xin, E}, {ws.up.data(), ffw});
        silu_mul({ws.gate.data(), ffw}, {ws.up.data(), ffw},
                 {ws.gate.data(), ffw});
        matvec_mt(op, wd, {ws.gate.data(), ffw}, {dout, E});
    };
    std::uint32_t cur = UINT32_MAX;
    Mat wg, wu, wd;
    for (const auto& u : uses) {
        if (u.e != cur) {
            cur = u.e;
            wg = lay.gate_exps.expert(u.e);
            wu = lay.up_exps.expert(u.e);
            wd = lay.down_exps.expert(u.e);
        }
        swiglu(wg, wu, wd, ff,
               xbf.data() + static_cast<std::size_t>(u.t) * E,
               down.data());
        const float wgt = picked[u.t][u.s].second;
        float* r = res.data() +
                   (static_cast<std::size_t>(u.t) * k + u.s) * E;
        for (std::uint32_t i = 0; i < E; ++i) {
            r[i] = wgt * down[i];
        }
    }

    // Weight window on the union (opt-in), after routed use.
    if (file_backed_ && std::getenv("LOCUS_WEIGHT_WINDOW")) {
        std::uint32_t prev = UINT32_MAX;
        for (const auto& u : uses) {
            if (u.e == prev) {
                continue;
            }
            prev = u.e;
            sys::advise_dontneed(lay.gate_exps.expert(u.e).data,
                                 lay.gate_exps.expert_bytes);
            sys::advise_dontneed(lay.up_exps.expert(u.e).data,
                                 lay.up_exps.expert_bytes);
            sys::advise_dontneed(lay.down_exps.expert(u.e).data,
                                 lay.down_exps.expert_bytes);
        }
    }

    // Phase 4: per-token mixture in moe_select order, then shared,
    // then into the residual -- byte-identical to moe_ffn's
    // moe_acc accumulation followed by x += moe_acc.
    std::vector<float> acc(E);
    const std::uint32_t ffs =
        hp_.n_ff_shexp > 0 ? hp_.n_ff_shexp
                           : ff * hp_.n_expert_shared;
    for (std::uint32_t t = 0; t < n; ++t) {
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (std::uint32_t s = 0;
             s < static_cast<std::uint32_t>(picked[t].size());
             ++s) {
            const float* r =
                res.data() +
                (static_cast<std::size_t>(t) * k + s) * E;
            for (std::uint32_t i = 0; i < E; ++i) {
                acc[i] += r[i];
            }
        }
        if (hp_.n_expert_shared > 0) {
            const float* xt_in =
                xbf.data() + static_cast<std::size_t>(t) * E;
            float g = 1.0f;
            if (lay.gate_inp_shexp.rows > 0) {
                float logit = 0.0f;
                op.matvec(lay.gate_inp_shexp, {xt_in, E},
                          {&logit, 1});
                g = 1.0f / (1.0f + std::exp(-logit));
            }
            swiglu(lay.gate_shexp, lay.up_shexp, lay.down_shexp,
                   ffs, xt_in, down.data());
            for (std::uint32_t i = 0; i < E; ++i) {
                acc[i] += g * down[i];
            }
        }
        float* xt = x.data() + static_cast<std::size_t>(t) * E;
        for (std::uint32_t i = 0; i < E; ++i) {
            xt[i] += acc[i];
        }
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
