// Metal forward of a plain llama model (i#866): one token, every layer
// on the GPU in one command batch, reading and writing the KV pool in
// place (a shared Metal buffer, Ops.alloc_kv).

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "locus/backend/metal/context.hpp"
#include "locus/backend/metal/kv_pool.hpp"
#include "locus/backend/metal/matvec.hpp"
#include "locus/model/arch.hpp"
#include "locus/model/transformer.hpp"

namespace locus::model {

using backend::metal::Context;

/** The model's weights in Metal buffers, by host pointer; freed with
 *  the model. */
struct MetalWeights {
    std::unordered_map<const void*, Context::Buffer> bufs;
    std::unordered_map<const void*, Context::Wrapped> maps;

    ~MetalWeights() {
        std::lock_guard lk(backend::metal::device_mutex());
        for (auto& [p, b] : bufs) {
            Context::get().free(b);
        }
        for (auto& [p, w] : maps) {
            Context::get().free(w.buf);
        }
    }

    /** A matrix's bytes on the device: a buffer and their offset. */
    struct Ref {
        const Context::Buffer* buf;
        std::uint64_t off;
    };

    /**
     * @returns The `bytes` bytes from p (a tensor in the GGUF mapping),
     *     wrapped without a copy on first use (Context::wrap; i#868): a
     *     model's matrices take no memory beyond the mapping.
     */
    Ref map(Context& ctx, const void* p, std::size_t bytes) {
        auto it = maps.find(p);
        if (it == maps.end()) {
            it = maps.emplace(p, ctx.wrap(p, bytes)).first;
        }
        return {&it->second.buf, it->second.offset};
    }

    /** @returns The buffer holding `bytes` bytes from p, uploaded on
     *  first use. */
    const Context::Buffer& get(Context& ctx, const void* p,
                               std::size_t bytes) {
        const auto it = bufs.find(p);
        if (it != bufs.end()) {
            return it->second;
        }
        return bufs
            .emplace(p, ctx.upload({static_cast<const std::byte*>(p), bytes}))
            .first->second;
    }
};

namespace {

std::atomic<std::uint64_t> g_metal_tokens{0};

}  // namespace

std::uint64_t TransformerModel::metal_forward_tokens() {
    return g_metal_tokens.load();
}

namespace {

/** Scratch buffers of the Metal forward (grow-only; device_mutex). */
struct Scratch {
    Context::Buffer x, xb, q, att, gate, up, logits, cs, scores, blocks;
    // MoE: the router logits (and the shared expert's gate logit after
    // them), and one expert's output.
    Context::Buffer router, eout;

    static void grow(Context& ctx, Context::Buffer& b, std::size_t bytes) {
        if (b.handle == nullptr || b.bytes < bytes) {
            ctx.free(b);
            b = ctx.alloc(bytes);
        }
    }
};

Scratch& scratch() {
    static Scratch s;
    return s;
}

template <typename T>
std::span<const std::byte> bytes_of(const T& v) {
    return std::as_bytes(std::span<const T>(&v, 1));
}

// The kernels' parameter blocks (kernels.metal).
struct NormParams {
    std::uint32_t n;
    float eps;
};
struct RopeParams {
    std::uint64_t off;
    std::uint32_t n_heads, head_dim, neox;
};
struct BiasParams {
    std::uint64_t off;
    std::uint32_t n, n_tok, stride;
};
struct AttnParams {
    std::uint64_t block_floats, layer_off, v_off;
    std::uint32_t n_pos, block_tokens, n_heads, n_kv_heads, head_dim;
    float kq_scale;
};
struct SiluParams {
    std::uint32_t n;
};
struct AxpyParams {
    std::uint32_t n;
    float a;
};

constexpr std::uint32_t kNormGroup = 256;
// attn_fused (kernels.metal): its threadgroup, and the positions its
// threadgroup memory holds.
constexpr std::uint32_t kAttnGroup = 256;
constexpr std::uint32_t kAttnMaxPos = 4096;

}  // namespace

bool TransformerModel::metal_forward_supported() const {
    using namespace backend;
    static const bool off = [] {
        const char* e = std::getenv("LOCUS_METAL_FORWARD");
        return e != nullptr && std::string_view(e) == "0";
    }();
    // The plain llama attention, or qwen2's (q / k / v biases, NEOX
    // RoPE; not qwen2vl's M-RoPE, whose text positions are not the
    // sequence's).
    const bool attn_ok =
        spec_->plain_attention ||
        (spec_->qwen2_attention && hp_.rope_sections[0] == 0);
    if (off || backend_->name != "metal" || !attn_ok ||
        hp_.custom_layer || spec_->layer != nullptr ||
        spec_->forward != nullptr || layer_begin_ != 0 ||
        layer_end_ != hp_.n_layers || hp_.norm_type != NormType::kRmsNorm ||
        hp_.embd_norm || hp_.n_kv_heads == 0 || out_w_.rows != hp_.n_vocab ||
        hp_.head_dim % 4 != 0 || hp_.n_heads % hp_.n_kv_heads != 0 ||
        !Context::usable()) {
        return false;
    }
    const auto ok_w = [](const Mat& m) {
        return metal::matvec_shape_ok(m.type, m.cols);
    };
    // MoE (i#868): routed experts in the forward, routing on the CPU;
    // not the hybrid or another host's experts.
    if (moe_hybrid_on() || remote_experts_ != nullptr) {
        for (const Layer& lay : layers_) {
            if (lay.is_moe()) {
                return false;
            }
        }
    }
    const auto ok_moe = [&](const Layer& lay) {
        if (!lay.is_moe()) {
            return ok_w(lay.w_gate) && ok_w(lay.w_up) && ok_w(lay.w_down);
        }
        const bool shared = hp_.n_expert_shared > 0;
        return ok_w(lay.gate_inp) && ok_w(lay.gate_exps.base) &&
               ok_w(lay.up_exps.base) && ok_w(lay.down_exps.base) &&
               (!shared || (ok_w(lay.gate_shexp) && ok_w(lay.up_shexp) &&
                            ok_w(lay.down_shexp))) &&
               (lay.gate_inp_shexp.rows == 0 || ok_w(lay.gate_inp_shexp));
    };
    for (const Layer& lay : layers_) {
        if (!lay.has_separate_qkv(hp_) ||
            (lay.has_qkv_bias() && !spec_->qwen2_attention) ||
            lay.has_qk_norm() ||
            !lay.post_attn_norm.empty() || !lay.post_ffw_norm.empty() ||
            !ok_w(lay.wq) || !ok_w(lay.wk) || !ok_w(lay.wv) || !ok_w(lay.wo) ||
            !ok_moe(lay)) {
            return false;
        }
    }
    if (!ok_w(out_w_)) {
        return false;
    }
    return true;
}

bool TransformerModel::metal_forward(tok::TokenId token, const float* embd,
                                     kv::PagedKvCache::Seq& seq,
                                     std::uint32_t pos,
                                     kv::PagedKvCache& cache, Workspace& ws,
                                     std::span<float> logits,
                                     LogitRows rows_out) const {
    using namespace backend;
    if (!metal_forward_supported() || cache.quantized()) {
        return false;
    }
    if (embd == nullptr &&
        (token < 0 || static_cast<std::uint32_t>(token) >= hp_.n_vocab)) {
        return false;  // the host path reports it
    }
    if (pos + 1 > hp_.n_ctx || pos >= cache.capacity(seq)) {
        return false;
    }
    const std::uint32_t E = hp_.n_embd, hd = hp_.head_dim,
                        H = hp_.n_heads, KV = hp_.n_kv_heads,
                        ff = hp_.n_ff, V = hp_.n_vocab, L = hp_.n_layers;
    const std::size_t qd = std::size_t{H} * hd;
    if (rows_out.end != 0 &&
        (rows_out.begin >= rows_out.end || rows_out.end > V)) {
        return false;  // the host path reports it
    }
    const std::uint32_t lo = rows_out.end == 0 ? 0 : rows_out.begin;
    const std::uint32_t R = (rows_out.end == 0 ? V : rows_out.end) - lo;
    const std::uint32_t n_pos = pos + 1;
    const std::uint32_t bt = cache.geometry().block_tokens;
    // Short contexts take the one-kernel attention: its scores fit in
    // threadgroup memory and the group splits evenly over head_dim.
    // LOCUS_METAL_ATTN_PASSES=3 keeps the three passes (tests).
    const char* passes = std::getenv("LOCUS_METAL_ATTN_PASSES");
    const bool fused_attn =
        n_pos <= kAttnMaxPos && hd <= kAttnGroup && kAttnGroup % hd == 0 &&
        (passes == nullptr || std::string_view(passes) != "3");

    std::lock_guard lk(metal::device_mutex());
    const Context::Buffer* pool = metal::kv_buffer(cache.pool_data());
    if (pool == nullptr) {
        return false;  // not a Metal pool
    }
    try {
        Context& ctx = Context::get();
        if (!metal_weights_) {
            metal_weights_ = std::make_shared<MetalWeights>();
        }
        MetalWeights& mw = *metal_weights_;
        const auto wbuf = [&](const Mat& m) {
            return mw.map(ctx, m.data,
                          metal::matvec_weight_bytes(m.type, m.rows, m.cols));
        };
        const auto fbuf =
            [&](std::span<const float> v) -> const Context::Buffer& {
            return mw.get(ctx, v.data(), v.size_bytes());
        };

        Scratch& s = scratch();
        const std::uint32_t n_blocks = (n_pos + bt - 1) / bt;
        Scratch::grow(ctx, s.x, E * 4);
        Scratch::grow(ctx, s.xb, E * 4);
        Scratch::grow(ctx, s.q, qd * 4);
        Scratch::grow(ctx, s.att, qd * 4);
        // The FFN scratch takes the widest of the dense, routed and
        // shared FFNs.
        const std::uint32_t shexp_ff =
            hp_.n_ff_shexp > 0 ? hp_.n_ff_shexp
                               : hp_.n_ff_exp * hp_.n_expert_shared;
        const std::size_t ffw =
            std::max<std::size_t>({ff, hp_.n_ff_exp, shexp_ff});
        Scratch::grow(ctx, s.gate, ffw * 4);
        Scratch::grow(ctx, s.up, ffw * 4);
        Scratch::grow(ctx, s.router, (std::size_t{hp_.n_expert} + 1) * 4);
        Scratch::grow(ctx, s.eout, std::size_t{E} * 4);
        Scratch::grow(ctx, s.logits, std::size_t{R} * 4);
        Scratch::grow(ctx, s.cs, std::size_t{hd} * 4);
        Scratch::grow(ctx, s.scores, std::size_t{H} * n_pos * 4);
        Scratch::grow(ctx, s.blocks, std::size_t{n_blocks} * 4);

        // Host-side inputs, written straight into the shared buffers:
        // the embedding, RoPE's (cos, sin) pairs and the block table.
        auto* x = static_cast<float*>(s.x.data);
        if (embd != nullptr) {
            std::copy_n(embd, E, x);
        } else {
            ops().dequant_row(embd_, static_cast<std::uint32_t>(token),
                              {x, E});
        }
        rope_norm_table(hd, pos, hp_.rope_freq_base, rope_factors(),
                        {static_cast<float*>(s.cs.data), hd});
        std::copy_n(seq.blocks.begin(), n_blocks,
                    static_cast<std::uint32_t*>(s.blocks.data));

        // This position's K and V rows, as floats into the pool.
        const auto* base = cache.pool_data();
        const auto row_of = [&](std::uint32_t l, bool v) {
            const float* p = v ? cache.v(seq, l, pos) : cache.k(seq, l, pos);
            return static_cast<std::uint64_t>(p - base);
        };

        const NormParams norm{E, hp_.rms_eps};
        const auto rmsnorm = [&](std::span<const float> w) {
            const Context::Arg a[] = {{&s.x, 0}, {&fbuf(w), 0}, {&s.xb, 0}};
            ctx.dispatch("rmsnorm_f32", a, bytes_of(norm), kNormGroup,
                         kNormGroup);
        };
        const auto mv = [&](const Mat& w, const Context::Buffer& in,
                            const Context::Buffer& out,
                            std::uint64_t out_off, bool acc) {
            const auto r = wbuf(w);
            metal::record_matvec(ctx, w.type, w.rows, w.cols, *r.buf, r.off,
                                 in, 0, out, out_off, acc);
        };
        // A routed or shared expert e of a 3-D tensor (one upload of the
        // whole tensor, expert e at its byte offset): out += wgt * down
        // (silu(gate x) * up x).
        const auto expert_buf = [&](const ExpertMat& em) {
            return mw.map(ctx, em.base.data,
                          em.expert_bytes * std::size_t{hp_.n_expert});
        };
        const auto swiglu_add = [&](const Mat& g, const Context::Buffer& gb,
                                    std::uint64_t g_off, const Mat& u,
                                    const Context::Buffer& ub,
                                    std::uint64_t u_off, const Mat& d,
                                    const Context::Buffer& db,
                                    std::uint64_t d_off, float wgt) {
            metal::record_matvec(ctx, g.type, g.rows, g.cols, gb, g_off, s.xb,
                                 0, s.gate, 0, false);
            ctx.overlap_next();
            metal::record_matvec(ctx, u.type, u.rows, u.cols, ub, u_off, s.xb,
                                 0, s.up, 0, false);
            const SiluParams sp{g.rows};
            const Context::Arg su[] = {{&s.gate, 0}, {&s.up, 0}};
            ctx.dispatch("silu_mul", su, bytes_of(sp), g.rows, 256);
            metal::record_matvec(ctx, d.type, d.rows, d.cols, db, d_off,
                                 s.gate, 0, s.eout, 0, false);
            const AxpyParams ap{E, wgt};
            const Context::Arg aa[] = {{&s.x, 0}, {&s.eout, 0}};
            ctx.dispatch("axpy", aa, bytes_of(ap), E, 256);
        };
        // A MoE layer's FFN on ws.xb: the router (and the shared
        // expert's gate) on the GPU, the batch submitted so the CPU can
        // pick the experts (moe_select, as the CPU forward), then a new
        // batch runs them into the residual.
        const auto moe = [&](const Layer& lay) {
            mv(lay.gate_inp, s.xb, s.router, 0, false);
            const bool gated = lay.gate_inp_shexp.rows > 0;
            if (gated) {
                ctx.overlap_next();
                mv(lay.gate_inp_shexp, s.xb, s.router, hp_.n_expert, false);
            }
            ctx.end_batch();
            const auto* rl = static_cast<const float*>(s.router.data);
            std::vector<float> router(rl, rl + hp_.n_expert);
            const auto picked = moe_select(hp_, lay, router);
            const float shg =
                gated ? 1.0f / (1.0f + std::exp(-rl[hp_.n_expert])) : 1.0f;
            ctx.begin_batch();
            const auto gr = expert_buf(lay.gate_exps);
            const auto ur = expert_buf(lay.up_exps);
            const auto dr = expert_buf(lay.down_exps);
            for (const auto& [e, wgt] : picked) {
                swiglu_add(lay.gate_exps.base, *gr.buf,
                           gr.off + lay.gate_exps.expert_bytes * e,
                           lay.up_exps.base, *ur.buf,
                           ur.off + lay.up_exps.expert_bytes * e,
                           lay.down_exps.base, *dr.buf,
                           dr.off + lay.down_exps.expert_bytes * e, wgt);
            }
            if (hp_.n_expert_shared > 0) {
                const auto sg = wbuf(lay.gate_shexp);
                const auto su = wbuf(lay.up_shexp);
                const auto sd = wbuf(lay.down_shexp);
                swiglu_add(lay.gate_shexp, *sg.buf, sg.off, lay.up_shexp,
                           *su.buf, su.off, lay.down_shexp, *sd.buf, sd.off,
                           shg);
            }
        };
        const std::uint32_t neox = spec_->qwen2_attention ? 1u : 0u;
        // v[off + t * stride ..] += bias, for n_tok rows.
        const auto bias = [&](const Context::Buffer& v, std::uint64_t off,
                              std::span<const float> b, std::uint32_t n_tok,
                              std::uint32_t stride) {
            if (b.empty()) {
                return;
            }
            const BiasParams bp{off, static_cast<std::uint32_t>(b.size()),
                                n_tok, stride};
            const Context::Arg a[] = {{&v, 0}, {&fbuf(b), 0}};
            ctx.dispatch("add_bias", a, bytes_of(bp), n_tok * bp.n, 256);
        };
        const auto rope = [&](const Context::Buffer& b, std::uint64_t off,
                              std::uint32_t heads) {
            const RopeParams rp{off, heads, hd, neox};
            const Context::Arg a[] = {{&b, 0}, {&s.cs, 0}};
            ctx.dispatch("rope_norm", a, bytes_of(rp), heads * hd / 2, 64);
        };

        ctx.begin_batch();
        try {
            for (std::uint32_t l = 0; l < L; ++l) {
                const Layer& lay = layers_[l];
                const std::uint64_t krow = row_of(l, false);
                const std::uint64_t vrow = row_of(l, true);
                rmsnorm(lay.attn_norm);
                // q, k and v read xb and write apart: they may overlap,
                // and so may the two RoPEs.
                mv(lay.wq, s.xb, s.q, 0, false);
                ctx.overlap_next();
                mv(lay.wk, s.xb, *pool, krow, false);
                ctx.overlap_next();
                mv(lay.wv, s.xb, *pool, vrow, false);
                if (lay.has_qkv_bias()) {
                    // qwen2: the biases, which may overlap, then RoPE.
                    bias(s.q, 0, lay.wq_bias, 1, 0);
                    ctx.overlap_next();
                    bias(*pool, krow, lay.wk_bias, 1, 0);
                    ctx.overlap_next();
                    bias(*pool, vrow, lay.wv_bias, 1, 0);
                }
                rope(s.q, 0, H);
                ctx.overlap_next();
                rope(*pool, krow, KV);
                const AttnParams ap{cache.block_floats(),
                                    cache.layer_offset(l),
                                    cache.layer_floats() / 2,
                                    n_pos,
                                    bt,
                                    H,
                                    KV,
                                    hd,
                                    hp_.kq_scale};
                if (fused_attn) {
                    // One kernel, one threadgroup a head.
                    const Context::Arg fa[] = {
                        {&s.q, 0}, {pool, 0}, {&s.blocks, 0}, {&s.att, 0}};
                    ctx.dispatch("attn_fused", fa, bytes_of(ap),
                                 H * kAttnGroup, kAttnGroup);
                } else {
                    const Context::Arg sa[] = {{&s.q, 0},
                                               {pool, 0},
                                               {&s.blocks, 0},
                                               {&s.scores, 0}};
                    ctx.dispatch("attn_scores", sa, bytes_of(ap), H * n_pos,
                                 64);
                    // One threadgroup a head (H groups of kNormGroup).
                    const Context::Arg sm[] = {{&s.scores, 0}};
                    ctx.dispatch("attn_softmax", sm, bytes_of(ap),
                                 H * kNormGroup, kNormGroup);
                    const Context::Arg va[] = {
                        {&s.scores, 0}, {pool, 0}, {&s.blocks, 0}, {&s.att, 0}};
                    ctx.dispatch("attn_values", va, bytes_of(ap), H * hd, 64);
                }
                mv(lay.wo, s.att, s.x, 0, true);
                rmsnorm(lay.ffn_norm);
                if (lay.is_moe()) {
                    moe(lay);
                    continue;
                }
                mv(lay.w_gate, s.xb, s.gate, 0, false);
                ctx.overlap_next();  // gate and up, likewise
                mv(lay.w_up, s.xb, s.up, 0, false);
                const SiluParams sp{ff};
                const Context::Arg su[] = {{&s.gate, 0}, {&s.up, 0}};
                ctx.dispatch("silu_mul", su, bytes_of(sp), ff, 256);
                mv(lay.w_down, s.gate, s.x, 0, true);
            }
            rmsnorm(out_norm_);
            const std::uint64_t row_bytes =
                metal::matvec_weight_bytes(out_w_.type, 1, out_w_.cols);
            const auto hr = wbuf(out_w_);
            metal::record_matvec(ctx, out_w_.type, R, out_w_.cols, *hr.buf,
                                 hr.off + lo * row_bytes, s.xb, 0, s.logits,
                                 0, false);
        } catch (...) {
            if (ctx.batch_open()) {  // MoE ends and reopens it
                ctx.end_batch();
            }
            throw;
        }
        ctx.end_batch();

        const auto* lg = static_cast<const float*>(s.logits.data);
        const float ninf = -std::numeric_limits<float>::infinity();
        std::fill(logits.begin(), logits.begin() + lo, ninf);
        std::copy_n(lg, R, logits.begin() + lo);
        std::fill(logits.begin() + lo + R, logits.begin() + V, ninf);
        scale_logits(logits.subspan(lo, R));
        std::copy_n(static_cast<const float*>(s.xb.data), E, ws.xb.begin());
        g_metal_tokens.fetch_add(1);
    } catch (const std::exception&) {
        return false;  // a device error: the host path runs the token
    }
    return true;
}

namespace {

// The prefill kernels' parameter blocks (kernels.metal).
struct RowNormParams {
    std::uint32_t n, stride;
    float eps;
};
struct RopeRowsParams {
    std::uint64_t off;
    std::uint32_t n_tok, n_heads, head_dim, stride, neox;
};
struct ScatterParams {
    std::uint64_t v_off;
    std::uint32_t n_tok, kv_dim;
};
struct AttnRowsParams {
    AttnParams a;
    std::uint32_t base, q_stride;
};

/** The prefill forward's scratch (grow-only; device_mutex). */
struct BatchScratch {
    Context::Buffer x, xb, q, k, v, att, gate, up, cs, rows, blocks, head;
};

BatchScratch& batch_scratch() {
    static BatchScratch s;
    return s;
}

constexpr std::uint32_t kPrefillChunk = 256;
// Below this many tokens a chunk runs a GPU forward a token: no matmul
// gain at that size, and short chunks (a speculative verify) keep the
// per-token forward's bits.
constexpr std::uint32_t kPrefillMin = 32;

}  // namespace

bool TransformerModel::metal_forward_batch(
    std::span<const tok::TokenId> toks, std::span<const float* const> embd,
    kv::PagedKvCache::Seq& seq, kv::PagedKvCache& cache, Workspace& ws,
    std::span<float> logits, bool all_logits) const {
    using namespace backend;
    const auto n = static_cast<std::uint32_t>(toks.size());
    const std::uint32_t base = seq.n_tokens;
    const std::uint32_t E = hp_.n_embd, hd = hp_.head_dim, H = hp_.n_heads,
                        KV = hp_.n_kv_heads, ff = hp_.n_ff, V = hp_.n_vocab,
                        L = hp_.n_layers;
    // MoE layers sync on their routing every token: a chunk runs the
    // per-token forward.
    const bool any_moe = std::any_of(
        layers_.begin(), layers_.end(),
        [](const Layer& lay) { return lay.is_moe(); });
    if (n < kPrefillMin || any_moe || !metal_forward_supported() ||
        cache.quantized() ||
        base + n > kAttnMaxPos || base + n > hp_.n_ctx ||
        base + n > cache.capacity(seq) || hd > kAttnGroup ||
        kAttnGroup % hd != 0 || (!embd.empty() && embd.size() != n) ||
        logits.size() < std::size_t{all_logits ? n : 1} * V) {
        return false;
    }
    for (std::uint32_t t = 0; t < n; ++t) {
        const bool embedded = !embd.empty() && embd[t] != nullptr;
        if (!embedded &&
            (toks[t] < 0 || static_cast<std::uint32_t>(toks[t]) >= V)) {
            return false;  // the host path reports it
        }
    }
    const std::size_t qd = std::size_t{H} * hd, kvd = std::size_t{KV} * hd;
    // The tiled attention takes head widths of 32 to 128 in steps of 32
    // (LOCUS_METAL_ATTN_TILED=0: attn_rows, for tests).
    const char* tiled_env = std::getenv("LOCUS_METAL_ATTN_TILED");
    const bool tiled_attn =
        hd % 32 == 0 && hd <= 128 &&
        (tiled_env == nullptr || std::string_view(tiled_env) != "0");
    const std::uint32_t bt = cache.geometry().block_tokens;

    std::lock_guard lk(metal::device_mutex());
    const Context::Buffer* pool = metal::kv_buffer(cache.pool_data());
    if (pool == nullptr) {
        return false;
    }
    try {
        Context& ctx = Context::get();
        if (!metal_weights_) {
            metal_weights_ = std::make_shared<MetalWeights>();
        }
        MetalWeights& mw = *metal_weights_;
        const auto wbuf = [&](const Mat& m) {
            return mw.map(ctx, m.data,
                          metal::matvec_weight_bytes(m.type, m.rows, m.cols));
        };
        const auto fbuf =
            [&](std::span<const float> v) -> const Context::Buffer& {
            return mw.get(ctx, v.data(), v.size_bytes());
        };
        const std::uint32_t neox = spec_->qwen2_attention ? 1u : 0u;
        // v[t * stride ..] += bias, for n_tok rows (qwen2's q / k / v).
        const auto bias = [&](const Context::Buffer& v, std::uint64_t off,
                              std::span<const float> b, std::uint32_t n_tok,
                              std::uint32_t stride) {
            if (b.empty()) {
                return;
            }
            const BiasParams bp{off, static_cast<std::uint32_t>(b.size()),
                                n_tok, stride};
            const Context::Arg a[] = {{&v, 0}, {&fbuf(b), 0}};
            ctx.dispatch("add_bias", a, bytes_of(bp), n_tok * bp.n, 256);
        };
        BatchScratch& s = batch_scratch();
        const std::size_t C = std::min(n, kPrefillChunk);
        const std::uint32_t n_blocks = (base + n + bt - 1) / bt;
        Scratch::grow(ctx, s.x, C * E * 4);
        Scratch::grow(ctx, s.xb, C * E * 4);
        Scratch::grow(ctx, s.q, C * qd * 4);
        Scratch::grow(ctx, s.k, C * kvd * 4);
        Scratch::grow(ctx, s.v, C * kvd * 4);
        Scratch::grow(ctx, s.att, C * qd * 4);
        Scratch::grow(ctx, s.gate, C * ff * 4);
        Scratch::grow(ctx, s.up, C * ff * 4);
        Scratch::grow(ctx, s.cs, C * hd * 4);
        Scratch::grow(ctx, s.rows, C * L * 8);
        Scratch::grow(ctx, s.blocks, std::size_t{n_blocks} * 4);
        // With all_logits, a sub-chunk's head rows: C * V floats (131 MB
        // for a 128k vocab at C = 256).
        Scratch::grow(ctx, s.head,
                      std::size_t{all_logits ? C : 1} * V * 4);
        std::copy_n(seq.blocks.begin(), n_blocks,
                    static_cast<std::uint32_t*>(s.blocks.data));
        const float* pool_base = cache.pool_data();

        // out[t] (+)= w in[t] for m token rows: one matmul, or a matvec a
        // token (independent rows: they may overlap).
        const auto mm = [&](const Mat& w, const Context::Buffer& in,
                            std::uint32_t in_stride,
                            const Context::Buffer& out,
                            std::uint32_t out_stride, bool acc,
                            std::uint32_t m) {
            const auto r = wbuf(w);
            if (metal::record_matmul(ctx, w.type, w.rows, w.cols, m, *r.buf,
                                     r.off, in, 0, in_stride, out, 0,
                                     out_stride, acc)) {
                return;
            }
            for (std::uint32_t t = 0; t < m; ++t) {
                if (t > 0) {
                    ctx.overlap_next();
                }
                metal::record_matvec(ctx, w.type, w.rows, w.cols, *r.buf,
                                     r.off, in, t * in_stride, out,
                                     std::uint64_t{t} * out_stride, acc);
            }
        };
        const auto norm_rows = [&](std::span<const float> w,
                                   std::uint32_t m) {
            const RowNormParams np{E, E, hp_.rms_eps};
            const Context::Arg a[] = {{&s.x, 0}, {&fbuf(w), 0}, {&s.xb, 0}};
            ctx.dispatch("rmsnorm_rows", a, bytes_of(np), m * kNormGroup,
                         kNormGroup);
        };

        for (std::uint32_t c0 = 0; c0 < n; c0 += kPrefillChunk) {
            const std::uint32_t m = std::min(kPrefillChunk, n - c0);
            const bool last_chunk = c0 + m == n;
            auto* x = static_cast<float*>(s.x.data);
            auto* cs = static_cast<float*>(s.cs.data);
            auto* rows = static_cast<std::uint64_t*>(s.rows.data);
            for (std::uint32_t t = 0; t < m; ++t) {
                const std::uint32_t i = c0 + t, pos = base + i;
                const float* e = embd.empty() ? nullptr : embd[i];
                if (e != nullptr) {
                    std::copy_n(e, E, x + std::size_t{t} * E);
                } else {
                    ops().dequant_row(embd_,
                                      static_cast<std::uint32_t>(toks[i]),
                                      {x + std::size_t{t} * E, E});
                }
                rope_norm_table(hd, pos, hp_.rope_freq_base, rope_factors(),
                                {cs + std::size_t{t} * hd, hd});
                for (std::uint32_t l = 0; l < L; ++l) {
                    rows[std::size_t{l} * m + t] =
                        static_cast<std::uint64_t>(cache.k(seq, l, pos) -
                                                   pool_base);
                }
            }
            ctx.begin_batch();
            try {
                for (std::uint32_t l = 0; l < L; ++l) {
                    const Layer& lay = layers_[l];
                    norm_rows(lay.attn_norm, m);
                    mm(lay.wq, s.xb, E, s.q, qd, false, m);
                    ctx.overlap_next();
                    mm(lay.wk, s.xb, E, s.k, kvd, false, m);
                    ctx.overlap_next();
                    mm(lay.wv, s.xb, E, s.v, kvd, false, m);
                    if (lay.has_qkv_bias()) {
                        bias(s.q, 0, lay.wq_bias, m,
                             static_cast<std::uint32_t>(qd));
                        ctx.overlap_next();
                        bias(s.k, 0, lay.wk_bias, m,
                             static_cast<std::uint32_t>(kvd));
                        ctx.overlap_next();
                        bias(s.v, 0, lay.wv_bias, m,
                             static_cast<std::uint32_t>(kvd));
                    }
                    const RopeRowsParams rq{0, m, H, hd,
                                            static_cast<std::uint32_t>(qd),
                                            neox};
                    const Context::Arg ra[] = {{&s.q, 0}, {&s.cs, 0}};
                    ctx.dispatch("rope_rows", ra, bytes_of(rq),
                                 m * H * hd / 2, 64);
                    ctx.overlap_next();
                    const RopeRowsParams rk{0, m, KV, hd,
                                            static_cast<std::uint32_t>(kvd),
                                            neox};
                    const Context::Arg rka[] = {{&s.k, 0}, {&s.cs, 0}};
                    ctx.dispatch("rope_rows", rka, bytes_of(rk),
                                 m * KV * hd / 2, 64);
                    const ScatterParams sp{cache.layer_floats() / 2, m,
                                           static_cast<std::uint32_t>(kvd)};
                    const Context::Arg sa[] = {
                        {&s.k, 0},
                        {&s.v, 0},
                        {pool, 0},
                        {&s.rows, std::size_t{l} * m * 8}};
                    ctx.dispatch("kv_scatter", sa, bytes_of(sp),
                                 m * static_cast<std::uint32_t>(kvd), 256);
                    // n_pos carries the sub-chunk's token count.
                    const AttnRowsParams ap{
                        {cache.block_floats(), cache.layer_offset(l),
                         cache.layer_floats() / 2, m, bt, H, KV, hd,
                         hp_.kq_scale},
                        base + c0,
                        static_cast<std::uint32_t>(qd)};
                    const Context::Arg aa[] = {
                        {&s.q, 0}, {pool, 0}, {&s.blocks, 0}, {&s.att, 0}};
                    if (tiled_attn) {
                        // 8 queries of a head a threadgroup, K / V staged.
                        ctx.dispatch("attn_rows_tiled", aa, bytes_of(ap),
                                     ((m + 7) / 8) * H * 256, 256);
                    } else {
                        ctx.dispatch("attn_rows", aa, bytes_of(ap),
                                     m * H * kAttnGroup, kAttnGroup);
                    }
                    mm(lay.wo, s.att, static_cast<std::uint32_t>(qd), s.x, E,
                       true, m);
                    norm_rows(lay.ffn_norm, m);
                    mm(lay.w_gate, s.xb, E, s.gate, ff, false, m);
                    ctx.overlap_next();
                    mm(lay.w_up, s.xb, E, s.up, ff, false, m);
                    const SiluParams sp2{m * ff};
                    const Context::Arg su[] = {{&s.gate, 0}, {&s.up, 0}};
                    ctx.dispatch("silu_mul", su, bytes_of(sp2), m * ff, 256);
                    mm(lay.w_down, s.gate, ff, s.x, E, true, m);
                }
                if (all_logits || last_chunk) {
                    norm_rows(out_norm_, m);
                    const std::uint32_t t0 = all_logits ? 0 : m - 1;
                    for (std::uint32_t t = t0; t < m; ++t) {
                        if (t > t0) {
                            ctx.overlap_next();
                        }
                        const auto hr = wbuf(out_w_);
                        metal::record_matvec(
                            ctx, out_w_.type, V, out_w_.cols, *hr.buf, hr.off,
                            s.xb, t * E, s.head, std::uint64_t{t - t0} * V,
                            false);
                    }
                }
            } catch (...) {
                ctx.end_batch();
                throw;
            }
            ctx.end_batch();
            const auto* head = static_cast<const float*>(s.head.data);
            if (all_logits) {
                std::copy_n(head, std::size_t{m} * V,
                            logits.begin() + std::size_t{c0} * V);
            } else if (last_chunk) {
                std::copy_n(head, V, logits.begin());
            }
            if (last_chunk) {
                std::copy_n(static_cast<const float*>(s.xb.data) +
                                std::size_t{m - 1} * E,
                            E, ws.xb.begin());
            }
        }
        for (std::uint32_t t = 0; t < (all_logits ? n : 1); ++t) {
            scale_logits(logits.subspan(std::size_t{t} * V, V));
        }
        seq.n_tokens = base + n;
        g_metal_tokens.fetch_add(n);
    } catch (const std::exception&) {
        return false;  // a device error: the caller runs the tokens again
    }
    return true;
}

}  // namespace locus::model
