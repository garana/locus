// Metal forward of a plain llama model (i#866): one token, every layer
// on the GPU in one command batch, reading and writing the KV pool in
// place (a shared Metal buffer, Ops.alloc_kv).

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
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
    /** A wrapped tensor and how many bytes from its pointer it covers. */
    struct Map {
        Context::Wrapped w;
        std::size_t bytes = 0;
    };
    std::unordered_map<const void*, Map> maps;
    /** Wraps and copies a longer request at the same pointer replaced:
     *  kept, as batches recorded earlier in this forward may use them,
     *  and freed with the model. */
    std::vector<Context::Buffer> retired;
    /** Whether the weights are the GGUF file's mapping (map() wraps
     *  them), or a heap buffer (map() copies them). */
    bool file_backed = true;

    ~MetalWeights() {
        std::lock_guard lk(backend::metal::device_mutex());
        for (auto& [p, b] : bufs) {
            Context::get().free(b);
        }
        for (auto& [p, m] : maps) {
            Context::get().free(m.w.buf);
        }
        for (auto& b : retired) {
            Context::get().free(b);
        }
    }

    /** A matrix's bytes on the device: a buffer and their offset. */
    struct Ref {
        const Context::Buffer* buf;
        std::uint64_t off;
    };

    /**
     * @returns The `bytes` bytes from p (a tensor of the model), on the
     *     device: wrapped without a copy on first use when the weights
     *     are the GGUF file's mapping (Context::wrap; i#868: read-only,
     *     PROT_READ pages, so the GPU sees what the CPU does), else
     *     copied (a GGUF parsed from a heap buffer, whose page rounding
     *     would reach past the allocation). A later request for more
     *     bytes at the same pointer wraps (or copies) the longer range.
     */
    Ref map(Context& ctx, const void* p, std::size_t bytes) {
        if (!file_backed) {
            return {&get(ctx, p, bytes), 0};
        }
        auto it = maps.find(p);
        if (it != maps.end() && it->second.bytes < bytes) {
            retired.push_back(it->second.w.buf);
            maps.erase(it);
            it = maps.end();
        }
        if (it == maps.end()) {
            it = maps.emplace(p, Map{ctx.wrap(p, bytes), bytes}).first;
        }
        return {&it->second.w.buf, it->second.w.offset};
    }

    /** @returns The buffer holding `bytes` bytes from p, uploaded on
     *  first use. */
    const Context::Buffer& get(Context& ctx, const void* p,
                               std::size_t bytes) {
        auto it = bufs.find(p);
        if (it != bufs.end() && it->second.bytes < bytes) {
            retired.push_back(it->second);  // a longer view: copy again
            bufs.erase(it);
            it = bufs.end();
        }
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
std::atomic<std::uint64_t> g_metal_multi_tokens{0};

}  // namespace

std::uint64_t TransformerModel::metal_forward_tokens() {
    return g_metal_tokens.load();
}

std::uint64_t TransformerModel::metal_multi_tokens() {
    return g_metal_multi_tokens.load();
}

bool TransformerModel::metal_batch_invariant() const {
    const int set = backend::metal::batch_invariant_setting();
    if (set >= 0) {
        return set == 1;
    }
    const auto kq = [](const backend::Mat& m) {
        return m.type == gguf::TensorType::kQ4_K ||
               m.type == gguf::TensorType::kQ6_K;
    };
    if (kq(out_w_)) {
        return false;
    }
    for (const Layer& lay : layers_) {
        for (const backend::Mat* m : {&lay.wq, &lay.wk, &lay.wv, &lay.wo,
                                      &lay.w_gate, &lay.w_up, &lay.w_down}) {
            if (kq(*m)) {
                return false;
            }
        }
    }
    return true;
}

namespace {

/** Scratch buffers of the Metal forward (grow-only; device_mutex). */
struct Scratch {
    Context::Buffer x, xb, q, att, gate, up, logits, cs, scores, blocks;
    Context::Buffer parts;  // attn_split's per-part sums
    // A half pool (kv-type f16): this position's K and V rows as floats,
    // and each layer's K row offset for kv_scatter_f16.
    Context::Buffer kf, vf, krows;
    // MoE: the router logits (and the shared expert's gate logit after
    // them), and one expert's output.
    Context::Buffer router, eout, route;

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

/** @returns whether the GPU forward takes this cache's KV type: f32, or
 *  f16 (--kv-type f16: the f32 layout in halves). */
bool kv_type_ok(const kv::PagedKvCache& c) {
    const kv::KvType t = c.geometry().kv_type;
    return t == kv::KvType::kF32 || t == kv::KvType::kF16;
}

/** @returns whether the pool holds halves (the kernels' _f16 forms). */
bool kv_half(const kv::PagedKvCache& c) {
    return c.geometry().kv_type == kv::KvType::kF16;
}

/** @returns the Metal buffer of the cache's pool, or nullptr. */
const Context::Buffer* kv_pool_buffer(const kv::PagedKvCache& c) {
    return backend::metal::kv_buffer(
        kv_half(c) ? reinterpret_cast<const float*>(c.qpool_data())
                   : c.pool_data());
}

/** @returns the element offset of (seq, layer, pos)'s K row in the pool
 *  (its V row is layer_floats() / 2 further), as kernels.metal's kv_row
 *  computes it; an f32 or f16 pool alike. */
std::uint64_t kv_row_off(const kv::PagedKvCache& c,
                         const kv::PagedKvCache::Seq& seq,
                         std::uint32_t layer, std::uint32_t pos) {
    const auto& g = c.geometry();
    return std::uint64_t{seq.blocks[pos / g.block_tokens]} *
               c.block_floats() +
           c.layer_offset(layer) +
           std::uint64_t{pos % g.block_tokens} * g.kv_dim;
}

/** @returns the kernel `name`, or its _f16 form for a half pool. */
std::string kv_kernel(std::string_view name, bool half) {
    std::string k(name);
    if (half) {
        k += "_f16";
    }
    return k;
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
struct HeadNormParams {
    std::uint64_t off;
    std::uint32_t n_tok, n_heads, head_dim, stride;
    float eps;
};
struct HeadNormRopeParams {
    HeadNormParams n;
    std::uint32_t neox;
};
struct AttnParams {
    std::uint64_t block_floats, layer_off, v_off;
    std::uint32_t n_pos, block_tokens, n_heads, n_kv_heads, head_dim;
    float kq_scale;
};
struct AttnSplitParams {
    AttnParams a;
    std::uint32_t n_parts;
};
struct SiluParams {
    std::uint32_t n;
};
struct AxpyParams {
    std::uint32_t n;
    float a;
};
struct AxpyRouteParams {
    std::uint32_t n, slot;
};
struct MoeSumParams {
    std::uint32_t n, k, n_tok;
};
struct SharedAddParams {
    std::uint32_t n, n_tok, gated, gate_at;
};
struct RouteParams {
    std::uint32_t n_expert, k, softmax, bias_on_logit, has_bias, norm;
    float scale;
    std::uint32_t shared_gate, n_tok, logit_stride;
};

constexpr std::uint32_t kNormGroup = 256;
// attn_fused (kernels.metal): its threadgroup, and the positions its
// threadgroup memory holds.
constexpr std::uint32_t kAttnGroup = 256;
constexpr std::uint32_t kAttnMaxPos = 4096;

}  // namespace

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
    Context::Buffer router, route, eout, idx;  // MoE layers
};

BatchScratch& batch_scratch() {
    static BatchScratch s;
    return s;
}

constexpr std::uint32_t kPrefillChunk = 256;
constexpr std::uint32_t kPrefillChunkStep = 2048;
// Below this many tokens an all_logits chunk (a speculative verify)
// runs a GPU forward a token, so it keeps the per-token forward's bits.
// A prompt chunk takes the batch from 2 tokens: a forward a token costs
// a decode step each (about 4.4 ms on Llama 3.2 1B, M4 Max), the batch
// 17 to 24 ms for 4 to 29 tokens.
constexpr std::uint32_t kPrefillMin = 32;

}  // namespace

namespace {

namespace metal = backend::metal;
using Mat = backend::Mat;
using Layer = TransformerModel::Layer;
using ExpertMat = TransformerModel::ExpertMat;

/** The model's Metal weights, made on first use. */
MetalWeights& weights_of(std::shared_ptr<MetalWeights>& p, bool file_backed) {
    if (!p) {
        p = std::make_shared<MetalWeights>();
        p->file_backed = file_backed;
    }
    return *p;
}

/**
 * Records body into a new batch and runs it. On an exception the batch
 * still open (a MoE layer may end and reopen it) is abandoned, never
 * submitted: its recording stopped part way, so its work must not run.
 * A batch a MoE layer ended before was already waited for.
 */
template <typename F>
void run_batch(Context& ctx, F&& body) {
    ctx.begin_batch();
    try {
        body();
    } catch (...) {
        ctx.abandon_batch();
        throw;
    }
    ctx.end_batch();
}

/**
 * What metal_forward and metal_forward_batch record alike (i#1037):
 * the model's tensors on the device, and the per-row dispatches both
 * make over buffers of token rows.
 */
struct MetalRec {
    Context& ctx;
    MetalWeights& mw;
    const Hparams& hp;
    std::uint32_t hd;
    /** RoPE's NEOX pairing (qwen2's and Qwen3's attention). */
    std::uint32_t neox;

    /** @returns m's bytes on the device. */
    MetalWeights::Ref w(const Mat& m) const {
        return mw.map(ctx, m.data,
                      metal::matvec_weight_bytes(m.type, m.rows, m.cols));
    }

    /** @returns v (a norm weight, a bias) on the device. */
    const Context::Buffer& f(std::span<const float> v) const {
        return mw.get(ctx, v.data(), v.size_bytes());
    }

    /** @returns A 3-D expert tensor on the device: one upload of the
     *  whole tensor, expert e at its byte offset. */
    MetalWeights::Ref experts(const ExpertMat& em) const {
        return mw.map(ctx, em.base.data,
                      em.expert_bytes * std::size_t{hp.n_expert});
    }

    /** qwen3's per-head RMS norm on n_tok rows of `heads` heads. */
    void head_norm(const Context::Buffer& v, std::uint64_t off,
                   std::uint32_t heads, std::span<const float> wt,
                   std::uint32_t n_tok, std::uint32_t stride) const {
        const HeadNormParams np{off, n_tok, heads, hd, stride,
                                hp.rms_eps};
        const Context::Arg a[] = {{&v, 0}, {&f(wt), 0}};
        ctx.dispatch("head_rmsnorm", a, bytes_of(np), n_tok * heads * 32,
                     32);
    }

    /** v[off + t * stride ..] += b, for n_tok rows (qwen2's q / k / v). */
    void bias(const Context::Buffer& v, std::uint64_t off,
              std::span<const float> b, std::uint32_t n_tok,
              std::uint32_t stride) const {
        if (b.empty()) {
            return;
        }
        const BiasParams bp{off, static_cast<std::uint32_t>(b.size()),
                            n_tok, stride};
        const Context::Arg a[] = {{&v, 0}, {&f(b), 0}};
        ctx.dispatch("add_bias", a, bytes_of(bp), n_tok * bp.n, 256);
    }

    /** moe_route over n_tok tokens' router logits in `router` (a token
     *  every logit_stride floats; 0 for one), into `route`. */
    void route(const Layer& lay, bool gated, std::uint32_t n_tok,
               std::uint32_t logit_stride, std::uint32_t group,
               const Context::Buffer& router,
               const Context::Buffer& route) const {
        const bool has_bias = !lay.exp_probs_b.empty();
        const RouteParams rp{
            hp.n_expert,
            hp.n_expert_used,
            hp.gating == GatingFunc::kSoftmax ? 1u : 0u,
            hp.bias_on_logit ? 1u : 0u,
            has_bias ? 1u : 0u,
            hp.expert_weights_norm ? 1u : 0u,
            hp.expert_weights_scale,
            gated ? 1u : 0u,
            n_tok,
            logit_stride};
        const Context::Arg ra[] = {
            {&router, 0},
            {has_bias ? &f(lay.exp_probs_b) : &router, 0},
            {&route, 0}};
        ctx.dispatch("moe_route", ra, bytes_of(rp), n_tok * group, group);
    }
};

/** metal_forward's recording: one token's rows in Scratch. */
struct DecodeRec : MetalRec {
    Scratch& s;
    std::uint32_t E;
    /** Routing on the GPU (moe_route), else the host's pick. */
    bool gpu_route;
    /** moe_route's threadgroup: a thread an expert, whole simdgroups. */
    std::uint32_t route_group;

    /** out[out_off ..] (+)= wt in. */
    void mv(const Mat& wt, const Context::Buffer& in,
            const Context::Buffer& out, std::uint64_t out_off,
            bool acc) const {
        const auto r = w(wt);
        metal::record_matvec(ctx, wt.type, wt.rows, wt.cols, *r.buf, r.off,
                             in, 0, out, out_off, acc);
    }

    /** xb = rmsnorm(x) * wt. */
    void rmsnorm(std::span<const float> wt) const {
        const NormParams norm{E, hp.rms_eps};
        const Context::Arg a[] = {{&s.x, 0}, {&f(wt), 0}, {&s.xb, 0}};
        ctx.dispatch("rmsnorm_f32", a, bytes_of(norm), kNormGroup,
                     kNormGroup);
    }

    /** RoPE on `heads` heads at off in b, by the (cos, sin) table. */
    void rope(const Context::Buffer& b, std::uint64_t off,
              std::uint32_t heads) const {
        const RopeParams rp{off, heads, hd, neox};
        const Context::Arg a[] = {{&b, 0}, {&s.cs, 0}};
        ctx.dispatch("rope_norm", a, bytes_of(rp), heads * hd / 2, 64);
    }

    /** A routed or shared expert: x += wgt * down (silu(gate xb) * up
     *  xb), the weight at route[wgt_slot] on the GPU when given. */
    void swiglu_add(const Mat& g, const Context::Buffer& gb,
                    std::uint64_t g_off, const Mat& u,
                    const Context::Buffer& ub, std::uint64_t u_off,
                    const Mat& d, const Context::Buffer& db,
                    std::uint64_t d_off, float wgt,
                    std::int64_t wgt_slot = -1) const {
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
        if (wgt_slot >= 0) {
            // The weight is on the GPU, at route[wgt_slot].
            const AxpyRouteParams ap{E,
                                     static_cast<std::uint32_t>(wgt_slot)};
            const Context::Arg aa[] = {
                {&s.x, 0}, {&s.eout, 0}, {&s.route, 0}};
            ctx.dispatch("axpy_routed", aa, bytes_of(ap), E, 256);
            return;
        }
        const AxpyParams ap{E, wgt};
        const Context::Arg aa[] = {{&s.x, 0}, {&s.eout, 0}};
        ctx.dispatch("axpy", aa, bytes_of(ap), E, 256);
    }

    /**
     * A MoE layer's FFN on xb: the router (and the shared expert's gate)
     * on the GPU, then moe_route and the k experts at once; or, off the
     * GPU route, the batch submitted so the CPU picks the experts
     * (moe_select, as the CPU forward) and a new batch runs them.
     */
    void moe(const Layer& lay) const {
        mv(lay.gate_inp, s.xb, s.router, 0, false);
        const bool gated = lay.gate_inp_shexp.rows > 0;
        if (gated) {
            ctx.overlap_next();
            mv(lay.gate_inp_shexp, s.xb, s.router, hp.n_expert, false);
        }
        if (gpu_route) {
            const std::uint32_t k = hp.n_expert_used;
            route(lay, gated, 1, 0, route_group, s.router, s.route);
            // The k experts at once: gate and up for every slot (x
            // shared), SiLU over the k rows, down slot by slot into
            // k residual rows, then their weighted sum.
            const auto gr = experts(lay.gate_exps);
            const auto ur = experts(lay.up_exps);
            const auto dr = experts(lay.down_exps);
            const Mat& g = lay.gate_exps.base;
            const Mat& u = lay.up_exps.base;
            const Mat& d = lay.down_exps.base;
            metal::record_matvec_slots(
                ctx, g.type, g.rows, g.cols, *gr.buf, gr.off,
                lay.gate_exps.expert_bytes, s.route, 0, k, s.xb, 0, 0,
                s.gate, 0, g.rows);
            ctx.overlap_next();
            metal::record_matvec_slots(
                ctx, u.type, u.rows, u.cols, *ur.buf, ur.off,
                lay.up_exps.expert_bytes, s.route, 0, k, s.xb, 0, 0,
                s.up, 0, u.rows);
            const SiluParams sp{k * g.rows};
            const Context::Arg su[] = {{&s.gate, 0}, {&s.up, 0}};
            ctx.dispatch("silu_mul", su, bytes_of(sp), k * g.rows, 256);
            metal::record_matvec_slots(
                ctx, d.type, d.rows, d.cols, *dr.buf, dr.off,
                lay.down_exps.expert_bytes, s.route, 0, k, s.gate, 0,
                g.rows, s.eout, 0, E);
            const MoeSumParams ms{E, k, 1};
            const Context::Arg ma[] = {
                {&s.x, 0}, {&s.eout, 0}, {&s.route, 0}};
            ctx.dispatch("moe_sum", ma, bytes_of(ms), E, 256);
            if (hp.n_expert_shared > 0) {
                const auto sg = w(lay.gate_shexp);
                const auto su = w(lay.up_shexp);
                const auto sd = w(lay.down_shexp);
                // The shared expert by the gate at route[2 k] (or 1).
                swiglu_add(lay.gate_shexp, *sg.buf, sg.off, lay.up_shexp,
                           *su.buf, su.off, lay.down_shexp, *sd.buf,
                           sd.off, 1.0f,
                           gated ? std::int64_t{2} * k : -1);
            }
            return;
        }
        ctx.end_batch();
        const auto* rl = static_cast<const float*>(s.router.data);
        std::vector<float> router(rl, rl + hp.n_expert);
        const auto picked = moe_select(hp, lay, router);
        const float shg =
            gated ? 1.0f / (1.0f + std::exp(-rl[hp.n_expert])) : 1.0f;
        ctx.begin_batch();
        const auto gr = experts(lay.gate_exps);
        const auto ur = experts(lay.up_exps);
        const auto dr = experts(lay.down_exps);
        for (const auto& [e, wgt] : picked) {
            swiglu_add(lay.gate_exps.base, *gr.buf,
                       gr.off + lay.gate_exps.expert_bytes * e,
                       lay.up_exps.base, *ur.buf,
                       ur.off + lay.up_exps.expert_bytes * e,
                       lay.down_exps.base, *dr.buf,
                       dr.off + lay.down_exps.expert_bytes * e, wgt);
        }
        if (hp.n_expert_shared > 0) {
            const auto sg = w(lay.gate_shexp);
            const auto su = w(lay.up_shexp);
            const auto sd = w(lay.down_shexp);
            swiglu_add(lay.gate_shexp, *sg.buf, sg.off, lay.up_shexp,
                       *su.buf, su.off, lay.down_shexp, *sd.buf, sd.off,
                       shg);
        }
    }
};

/** Grows metal_forward_batch's scratch for sub-chunks of C tokens over
 *  n_blocks KV blocks (all_logits: C head rows, else 1). */
void grow_batch(Context& ctx, BatchScratch& s, const Hparams& hp,
                std::span<const Layer> layers, std::size_t C, bool any_moe,
                std::uint32_t n_blocks, bool all_logits) {
    const std::uint32_t E = hp.n_embd, hd = hp.head_dim, ff = hp.n_ff,
                        V = hp.n_vocab, L = hp.n_layers;
    const std::size_t qd = std::size_t{hp.n_heads} * hd,
                      kvd = std::size_t{hp.n_kv_heads} * hd;
    Scratch::grow(ctx, s.x, C * E * 4);
    Scratch::grow(ctx, s.xb, C * E * 4);
    Scratch::grow(ctx, s.q, C * qd * 4);
    Scratch::grow(ctx, s.k, C * kvd * 4);
    Scratch::grow(ctx, s.v, C * kvd * 4);
    Scratch::grow(ctx, s.att, C * qd * 4);
    // The FFN's widest rows: dense, a token's k experts, or the
    // shared expert.
    const std::uint32_t k_used = any_moe ? hp.n_expert_used : 0;
    std::size_t ffw = ff;
    for (const Layer& lay : layers) {
        if (lay.is_moe()) {
            ffw = std::max<std::size_t>(
                ffw, std::size_t{k_used} * lay.gate_exps.base.rows);
            ffw = std::max<std::size_t>(ffw, lay.gate_shexp.rows);
        }
    }
    Scratch::grow(ctx, s.gate, C * ffw * 4);
    Scratch::grow(ctx, s.up, C * ffw * 4);
    const std::uint32_t NR = hp.n_expert + 1;  // a token's logits
    if (any_moe) {
        Scratch::grow(ctx, s.router, C * NR * 4);
        Scratch::grow(ctx, s.route, C * (2 * std::size_t{k_used} + 1) * 4);
        Scratch::grow(ctx, s.eout,
                      C * std::max<std::size_t>(k_used, 1) * E * 4);
        const auto slots = static_cast<std::uint32_t>(C * k_used);
        Scratch::grow(ctx, s.idx,
                      (slots + 3 * std::size_t{metal::moe_max_tiles(
                                       slots, hp.n_expert)}) *
                          4);
    }
    Scratch::grow(ctx, s.cs, C * hd * 4);
    Scratch::grow(ctx, s.rows, C * L * 8);
    Scratch::grow(ctx, s.blocks, std::size_t{n_blocks} * 4);
    // With all_logits, a sub-chunk's head rows: C * V floats (131 MB
    // for a 128k vocab at C = 256).
    Scratch::grow(ctx, s.head,
                  std::size_t{all_logits ? C : 1} * V * 4);
}

/** metal_forward_batch's recording: m token rows in BatchScratch. */
struct BatchRec : MetalRec {
    BatchScratch& s;
    std::uint32_t E;
    /** A token's router logits: the experts', then the shared gate's. */
    std::uint32_t NR;
    /** moe_route's threadgroup: a thread an expert, whole simdgroups. */
    std::uint32_t route_group;
    /** The experts as grouped matmuls, else a matvec slot each. */
    bool moe_mm;

    /** out[t] (+)= wt in[t] for m token rows: one matmul, or a matvec a
     *  token (independent rows: they may overlap). */
    void mm(const Mat& wt, const Context::Buffer& in,
            std::uint32_t in_stride, const Context::Buffer& out,
            std::uint32_t out_stride, bool acc, std::uint32_t m,
            std::uint64_t out_off = 0) const {
        const auto r = w(wt);
        // Up to 8 rows: the decode's columns kernel reads each weight
        // block once for all of them, where the matmul computes a
        // whole 32-token tile (i#1074; Q8_0, and Q6_K from 4 rows).
        // LOCUS_METAL_PREFILL_COLS=0 keeps the matmul (tests, benches).
        static const bool cols_on = [] {
            const char* e = std::getenv("LOCUS_METAL_PREFILL_COLS");
            return e == nullptr || std::string_view(e) != "0";
        }();
        if (cols_on && m <= 8 &&
            metal::record_matvec_cols(ctx, wt.type, wt.rows, wt.cols, *r.buf,
                                      r.off, in, 0, in_stride, out, out_off,
                                      out_stride, m, acc, nullptr, 0.0f,
                                      true)) {
            return;
        }
        // Without a columns kernel (Q4_K), up to 3 rows run faster as a
        // matvec each than as a 32-token tile (M4 Max, Llama 1B Q4_K_M:
        // 3 tokens 14.1 to 9.1 ms; at 4 to 7 it is about level).
        // LOCUS_METAL_PREFILL_MV_ROWS=N moves the cut (benches).
        static const std::uint32_t mv_rows = [] {
            const char* e = std::getenv("LOCUS_METAL_PREFILL_MV_ROWS");
            return e != nullptr ? static_cast<std::uint32_t>(std::atoi(e))
                                : 3u;
        }();
        if (!(cols_on && m <= mv_rows) &&
            metal::record_matmul(ctx, wt.type, wt.rows, wt.cols, m, *r.buf,
                                 r.off, in, 0, in_stride, out, out_off,
                                 out_stride, acc)) {
            return;
        }
        for (std::uint32_t t = 0; t < m; ++t) {
            if (t > 0) {
                ctx.overlap_next();
            }
            metal::record_matvec(ctx, wt.type, wt.rows, wt.cols, *r.buf,
                                 r.off, in, t * in_stride, out,
                                 out_off + std::uint64_t{t} * out_stride,
                                 acc);
        }
    }

    /** xb[t] = rmsnorm(x[t]) * wt for m rows. */
    void norm_rows(std::span<const float> wt, std::uint32_t m) const {
        const RowNormParams np{E, E, hp.rms_eps};
        const Context::Arg a[] = {{&s.x, 0}, {&f(wt), 0}, {&s.xb, 0}};
        ctx.dispatch("rmsnorm_rows", a, bytes_of(np), m * kNormGroup,
                     kNormGroup);
    }

    /**
     * A MoE layer's FFN on m rows of xb into x: the router logits as a
     * matmul, moe_route a token (token-major route: ids, then weights,
     * then shared gates), each (token, pick) a matvec slot or a grouped
     * matmul reading its token's row, the k outputs summed by their
     * weights, then the shared expert by its gate (or 1).
     */
    void moe_rows(const Layer& lay, std::uint32_t m) const {
        const std::uint32_t k = hp.n_expert_used;
        mm(lay.gate_inp, s.xb, E, s.router, NR, false, m);
        const bool gated = lay.gate_inp_shexp.rows > 0;
        if (gated) {
            ctx.overlap_next();
            mm(lay.gate_inp_shexp, s.xb, E, s.router, NR, false, m,
               hp.n_expert);
        }
        route(lay, gated, m, NR, route_group, s.router, s.route);
        const auto gr = experts(lay.gate_exps);
        const auto ur = experts(lay.up_exps);
        const auto dr = experts(lay.down_exps);
        const Mat& g = lay.gate_exps.base;
        const Mat& u = lay.up_exps.base;
        const std::uint32_t slots = m * k;
        if (moe_mm) {
            metal::record_moe_tiles(ctx, s.route, slots, hp.n_expert,
                                    s.idx);
        }
        // Slot s (token s / k) through expert route[s]: a grouped
        // matmul, else a matvec slot each.
        const auto run_experts = [&](const ExpertMat& em,
                                     const MetalWeights::Ref& r,
                                     const Context::Buffer& in,
                                     std::uint32_t in_stride,
                                     std::uint32_t div,
                                     const Context::Buffer& out,
                                     std::uint32_t out_stride) {
            const Mat& wm = em.base;
            if (moe_mm &&
                metal::record_matmul_id(
                    ctx, wm.type, wm.rows, wm.cols, slots, hp.n_expert,
                    *r.buf, r.off, em.expert_bytes, in, in_stride, div,
                    out, out_stride, s.idx)) {
                return;
            }
            metal::record_matvec_slots(ctx, wm.type, wm.rows, wm.cols,
                                       *r.buf, r.off, em.expert_bytes,
                                       s.route, 0, slots, in, 0,
                                       in_stride, out, 0, out_stride,
                                       div);
        };
        run_experts(lay.gate_exps, gr, s.xb, E, k, s.gate, g.rows);
        ctx.overlap_next();
        run_experts(lay.up_exps, ur, s.xb, E, k, s.up, u.rows);
        const SiluParams sp{slots * g.rows};
        const Context::Arg su[] = {{&s.gate, 0}, {&s.up, 0}};
        ctx.dispatch("silu_mul", su, bytes_of(sp), slots * g.rows, 256);
        run_experts(lay.down_exps, dr, s.gate, g.rows, 1, s.eout, E);
        const MoeSumParams ms{E, k, m};
        const Context::Arg ma[] = {{&s.x, 0}, {&s.eout, 0}, {&s.route, 0}};
        ctx.dispatch("moe_sum", ma, bytes_of(ms), m * E, 256);
        if (hp.n_expert_shared == 0) {
            return;
        }
        const std::uint32_t sf = lay.gate_shexp.rows;
        mm(lay.gate_shexp, s.xb, E, s.gate, sf, false, m);
        ctx.overlap_next();
        mm(lay.up_shexp, s.xb, E, s.up, sf, false, m);
        const SiluParams ssp{m * sf};
        ctx.dispatch("silu_mul", su, bytes_of(ssp), m * sf, 256);
        mm(lay.down_shexp, s.gate, sf, s.eout, E, false, m);
        const SharedAddParams sa{E, m, gated ? 1u : 0u, 2 * m * k};
        const Context::Arg sha[] = {
            {&s.x, 0}, {&s.eout, 0}, {&s.route, 0}};
        ctx.dispatch("shared_add", sha, bytes_of(sa), m * E, 256);
    }
};

}  // namespace

std::string TransformerModel::metal_forward_why_not() const {
    using namespace backend;
    static const bool off = [] {
        const char* e = std::getenv("LOCUS_METAL_FORWARD");
        return e != nullptr && std::string_view(e) == "0";
    }();
    // The plain llama attention, qwen2's (q / k / v biases, NEOX RoPE;
    // not qwen2vl's M-RoPE, whose text positions are not the
    // sequence's), or Qwen3's (per-head q / k norm, NEOX RoPE): qwen3moe
    // and dense qwen3 (Qwen3Guard, i#1218).
    const bool qwen3 = spec_->name == "qwen3moe" || spec_->qwen3_attention;
    const std::string arch(spec_->name);
    if (off) return "LOCUS_METAL_FORWARD=0";
    if (backend_->name != "metal") return "the backend is not metal";
    if (!Context::usable()) return "no usable Metal device";
    const bool attn_ok =
        spec_->plain_attention || qwen3 ||
        (spec_->qwen2_attention && hp_.rope_sections[0] == 0);
    if (!attn_ok) {
        return spec_->qwen2_attention ? arch + "'s M-RoPE"
                                      : arch + "'s attention";
    }
    if (hp_.custom_layer || spec_->layer != nullptr ||
        spec_->forward != nullptr) {
        return arch + "'s own layer or forward";
    }
    if (layer_begin_ != 0 || layer_end_ != hp_.n_layers) {
        return "a pipeline stage's slice of the layers";
    }
    if (hp_.norm_type != NormType::kRmsNorm) return "a norm other than RMS";
    if (hp_.embd_norm) return "a norm on the embeddings";
    if (hp_.n_kv_heads == 0 || hp_.n_heads % hp_.n_kv_heads != 0) {
        return "its head counts";
    }
    if (out_w_.rows != hp_.n_vocab) return "an output matrix not one row "
                                           "a token";
    if (hp_.head_dim % 4 != 0) return "a head width not a multiple of 4";
    // A weight type or shape without a Metal kernel: the first found.
    std::string bad;
    const auto ok_w = [&](const Mat& m) {
        if (metal::matvec_shape_ok(m.type, m.cols)) return true;
        if (bad.empty()) {
            bad = "weights of type " +
                  std::string(gguf::tensor_type_name(m.type)) +
                  " (no Metal kernel)";
        }
        return false;
    };
    // MoE (i#868): routed experts in the forward, routing on the CPU;
    // not the hybrid or another host's experts.
    if (moe_hybrid_on() || remote_experts_ != nullptr) {
        for (const Layer& lay : layers_) {
            if (lay.is_moe()) {
                return moe_hybrid_on() ? "the MoE hybrid"
                                       : "experts on another host";
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
        if (!lay.has_separate_qkv(hp_)) return "a fused q / k / v matrix";
        if (lay.has_qkv_bias() && !spec_->qwen2_attention) {
            return "q / k / v biases outside qwen2's attention";
        }
        if (lay.has_qk_norm() && !qwen3) {
            return "q / k norms outside Qwen3's attention";
        }
        if (!lay.post_attn_norm.empty() || !lay.post_ffw_norm.empty()) {
            return "post-attention or post-FFN norms";
        }
        if (!ok_w(lay.wq) || !ok_w(lay.wk) || !ok_w(lay.wv) ||
            !ok_w(lay.wo) || !ok_moe(lay)) {
            return bad;
        }
    }
    if (!ok_w(out_w_)) return bad;
    return "";
}

bool TransformerModel::metal_forward_supported() const {
    return metal_forward_why_not().empty();
}

std::uint32_t TransformerModel::metal_neox() const {
    return spec_->qwen2_attention || spec_->name == "qwen3moe" ||
                   spec_->qwen3_attention
               ? 1u
               : 0u;
}

bool TransformerModel::metal_forward(tok::TokenId token, const float* embd,
                                     kv::PagedKvCache::Seq& seq,
                                     std::uint32_t pos,
                                     kv::PagedKvCache& cache, Workspace& ws,
                                     std::span<float> logits,
                                     LogitRows rows_out) const {
    using namespace backend;
    if (!metal_forward_supported() || !kv_type_ok(cache)) {
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
    // Past 512 positions, split-K: a threadgroup a (head, 256
    // positions), so a long context fills the GPU, not one threadgroup
    // a head (LOCUS_METAL_ATTN_SPLIT=0: off, for tests).
    const char* split_env = std::getenv("LOCUS_METAL_ATTN_SPLIT");
    const bool split_attn =
        n_pos > 2 * kAttnGroup && hd <= kAttnGroup &&
        kAttnGroup % hd == 0 &&
        (passes == nullptr || std::string_view(passes) != "3") &&
        (split_env == nullptr || std::string_view(split_env) != "0");
    const std::uint32_t n_parts = (n_pos + kAttnGroup - 1) / kAttnGroup;

    std::lock_guard lk(metal::device_mutex());
    const Context::Buffer* pool = kv_pool_buffer(cache);
    if (pool == nullptr) {
        return false;  // not a Metal pool
    }
    const bool half = kv_half(cache);
    try {
        Context& ctx = Context::get();
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
        // (The routed experts' scratch holds all k picks at once.)
        const std::size_t ffw = std::max<std::size_t>(
            {ff, std::size_t{hp_.n_ff_exp} * hp_.n_expert_used, shexp_ff});
        Scratch::grow(ctx, s.gate, ffw * 4);
        Scratch::grow(ctx, s.up, ffw * 4);
        Scratch::grow(ctx, s.router, (std::size_t{hp_.n_expert} + 1) * 4);
        Scratch::grow(ctx, s.route,
                      (2 * std::size_t{hp_.n_expert_used} + 1) * 4);
        Scratch::grow(ctx, s.eout,
                      std::size_t{E} * std::max(1u, hp_.n_expert_used) * 4);
        Scratch::grow(ctx, s.logits, std::size_t{R} * 4);
        Scratch::grow(ctx, s.cs, std::size_t{hd} * 4);
        Scratch::grow(ctx, s.scores, std::size_t{H} * n_pos * 4);
        Scratch::grow(ctx, s.parts,
                      std::size_t{H} * n_parts * (std::size_t{hd} + 4) * 4);
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

        // This position's K and V rows: element offsets into the pool.
        const auto row_of = [&](std::uint32_t l, bool v) {
            return kv_row_off(cache, seq, l, pos) +
                   (v ? cache.layer_floats() / 2 : 0);
        };
        // An f32 pool takes the K and V rows straight from their
        // matvecs; a half pool gets them as floats in kf / vf (bias,
        // norm and RoPE there), then rounded in by kv_scatter_f16.
        if (half) {
            const std::size_t kvd_b = std::size_t{KV} * hd * 4;
            Scratch::grow(ctx, s.kf, kvd_b);
            Scratch::grow(ctx, s.vf, kvd_b);
            Scratch::grow(ctx, s.krows, std::size_t{L} * 8);
            auto* kr = static_cast<std::uint64_t*>(s.krows.data);
            for (std::uint32_t l = 0; l < L; ++l) {
                kr[l] = row_of(l, false);
            }
        }

        // Routing on the GPU (moe_route) when the kernel covers the
        // arch's selection: no group-limited routing, at most 1024
        // experts. LOCUS_METAL_MOE_ROUTE=cpu keeps the host pick (tests).
        const char* route_env = std::getenv("LOCUS_METAL_MOE_ROUTE");
        const DecodeRec r{
            {ctx, weights_of(metal_weights_, file_backed_), hp_, hd,
             metal_neox()},
            s,
            E,
            hp_.n_group <= 1 && hp_.n_expert <= 1024 &&
                (route_env == nullptr ||
                 std::string_view(route_env) != "cpu"),
            std::max(32u, (hp_.n_expert + 31) / 32 * 32)};
        // A matvec with the steps of f in it (i#1064): false when its
        // type lacks one.
        const auto mvf = [&](const Mat& w, const Context::Buffer& in,
                             const Context::Buffer& out,
                             std::uint64_t out_off, const metal::MvFuse& f) {
            const auto wb = r.w(w);
            return metal::record_matvec_fused(ctx, w.type, w.rows, w.cols,
                                              *wb.buf, wb.off, in, 0, out,
                                              out_off, f);
        };
        // mvf where the matvec must be recorded (no fusion, or a norm
        // fold fold_ok allowed): a false would drop it silently, so it
        // throws, and the host path runs the token.
        const auto mvf_must = [&](const Mat& w, const Context::Buffer& in,
                                  const Context::Buffer& out,
                                  std::uint64_t out_off,
                                  const metal::MvFuse& f) {
            if (!mvf(w, in, out, out_off, f)) {
                throw std::logic_error("metal decode: a matvec not recorded");
            }
        };
        // The RMSNorm folded into the matvecs that read it, where each
        // of them takes it: by default on Apple GPU family 9 and up,
        // where it gains; family 8 (the M2) decodes Q8_0 12 to 24%
        // slower with it (i#1081). LOCUS_METAL_NORM_FOLD=1 or 0 forces
        // it on or off (0: rmsnorm_f32 first).
        const char* nf_env = std::getenv("LOCUS_METAL_NORM_FOLD");
        const bool norm_fold =
            nf_env != nullptr ? std::string_view(nf_env) != "0"
                              : ctx.apple_family() >= 9;
        const auto fold_ok = [&](std::initializer_list<const Mat*> ms) {
            return norm_fold &&
                   std::all_of(ms.begin(), ms.end(), [](const Mat* m) {
                       return metal::matvec_prenorm_ok(m->type);
                   });
        };
        // LOCUS_METAL_ROPE_EPILOGUE=0: always the separate rope_norm
        // (tests, benches).
        const char* re_env = std::getenv("LOCUS_METAL_ROPE_EPILOGUE");
        const bool rope_epilogue =
            r.neox == 0 &&
            (re_env == nullptr || std::string_view(re_env) != "0");
        // head_rmsnorm then rope_norm as one dispatch (i#1064);
        // LOCUS_METAL_NORM_ROPE=0 keeps the two (tests, benches).
        const char* nr_env = std::getenv("LOCUS_METAL_NORM_ROPE");
        const bool norm_rope =
            nr_env == nullptr || std::string_view(nr_env) != "0";
        const auto head_norm_rope = [&](const Context::Buffer& v,
                                        std::uint64_t off,
                                        std::uint32_t heads,
                                        std::span<const float> w) {
            const HeadNormRopeParams hp{{off, 1, heads, hd, 0, hp_.rms_eps},
                                        r.neox};
            const Context::Arg a[] = {{&v, 0}, {&r.f(w), 0}, {&s.cs, 0}};
            ctx.dispatch("head_norm_rope", a, bytes_of(hp), heads * 32, 32);
        };

        run_batch(ctx, [&] {
            for (std::uint32_t l = 0; l < L; ++l) {
                const Layer& lay = layers_[l];
                const std::uint64_t krow = row_of(l, false);
                const std::uint64_t vrow = row_of(l, true);
                // Where K and V are written: the pool rows, or kf / vf.
                const Context::Buffer& kdst = half ? s.kf : *pool;
                const Context::Buffer& vdst = half ? s.vf : *pool;
                const std::uint64_t koff = half ? 0 : krow;
                const std::uint64_t voff = half ? 0 : vrow;
                // The attention's RMSNorm in the q / k / v matvecs when
                // all three take it, else rmsnorm_f32 into xb first.
                const bool fold_attn = fold_ok({&lay.wq, &lay.wk, &lay.wv});
                metal::MvFuse nf;
                if (fold_attn) {
                    nf.norm_w = &r.f(lay.attn_norm);
                    nf.eps = hp_.rms_eps;
                } else {
                    r.rmsnorm(lay.attn_norm);
                }
                const Context::Buffer& qin = fold_attn ? s.x : s.xb;
                // q, k and v read the same input and write apart: they
                // may overlap, and so may the two RoPEs. RoPE straight
                // after q and k (no biases, no q / k norm, interleaved
                // pairs) goes in their matvecs' store.
                const bool fuse = rope_epilogue && !lay.has_qkv_bias() &&
                                  !lay.has_qk_norm();
                metal::MvFuse nr = nf;
                nr.rope_cs = &s.cs;
                nr.rope_hd = hd;
                const bool rq = fuse && mvf(lay.wq, qin, s.q, 0, nr);
                if (!rq) {
                    mvf_must(lay.wq, qin, s.q, 0, nf);
                }
                ctx.overlap_next();
                const bool rk = fuse && mvf(lay.wk, qin, kdst, koff, nr);
                if (!rk) {
                    mvf_must(lay.wk, qin, kdst, koff, nf);
                }
                ctx.overlap_next();
                mvf_must(lay.wv, qin, vdst, voff, nf);
                if (lay.has_qkv_bias()) {
                    // qwen2: the biases, which may overlap, then RoPE.
                    r.bias(s.q, 0, lay.wq_bias, 1, 0);
                    ctx.overlap_next();
                    r.bias(kdst, koff, lay.wk_bias, 1, 0);
                    ctx.overlap_next();
                    r.bias(vdst, voff, lay.wv_bias, 1, 0);
                }
                const bool qk_norm_rope = lay.has_qk_norm() && norm_rope;
                if (qk_norm_rope) {
                    // qwen3: the per-head q / k norms with their RoPE, one
                    // dispatch each (overlapping).
                    head_norm_rope(s.q, 0, H, lay.attn_q_norm);
                    ctx.overlap_next();
                    head_norm_rope(kdst, koff, KV, lay.attn_k_norm);
                } else if (lay.has_qk_norm()) {
                    // qwen3: the per-head q / k norms (overlapping).
                    r.head_norm(s.q, 0, H, lay.attn_q_norm, 1, 0);
                    ctx.overlap_next();
                    r.head_norm(kdst, koff, KV, lay.attn_k_norm, 1, 0);
                }
                if (!rq && !qk_norm_rope) {
                    r.rope(s.q, 0, H);
                }
                if (!rk && !qk_norm_rope) {
                    if (!rq) {
                        ctx.overlap_next();
                    }
                    r.rope(kdst, koff, KV);
                }
                if (half) {
                    // K and V, done, rounded into the pool's halves.
                    const ScatterParams sp{cache.layer_floats() / 2, 1,
                                           KV * hd};
                    const Context::Arg sa[] = {{&s.kf, 0},
                                               {&s.vf, 0},
                                               {pool, 0},
                                               {&s.krows, std::size_t{l} * 8}};
                    ctx.dispatch("kv_scatter_f16", sa, bytes_of(sp), KV * hd,
                                 256);
                }
                const AttnParams ap{cache.block_floats(),
                                    cache.layer_offset(l),
                                    cache.layer_floats() / 2,
                                    n_pos,
                                    bt,
                                    H,
                                    KV,
                                    hd,
                                    hp_.kq_scale};
                if (split_attn) {
                    // A threadgroup a (head, 256 positions), merged.
                    const AttnSplitParams spp{ap, n_parts};
                    const Context::Arg pa[] = {
                        {&s.q, 0}, {pool, 0}, {&s.blocks, 0}, {&s.parts, 0}};
                    ctx.dispatch(kv_kernel("attn_split", half), pa,
                                 bytes_of(spp), H * n_parts * kAttnGroup,
                                 kAttnGroup);
                    const Context::Arg ma[] = {{&s.parts, 0}, {&s.att, 0}};
                    ctx.dispatch("attn_merge", ma, bytes_of(spp), H * hd, 64);
                } else if (fused_attn) {
                    // One kernel, one threadgroup a head.
                    const Context::Arg fa[] = {
                        {&s.q, 0}, {pool, 0}, {&s.blocks, 0}, {&s.att, 0}};
                    ctx.dispatch(kv_kernel("attn_fused", half), fa,
                                 bytes_of(ap), H * kAttnGroup, kAttnGroup);
                } else {
                    const Context::Arg sa[] = {{&s.q, 0},
                                               {pool, 0},
                                               {&s.blocks, 0},
                                               {&s.scores, 0}};
                    ctx.dispatch(kv_kernel("attn_scores", half), sa,
                                 bytes_of(ap), H * n_pos, 64);
                    // One threadgroup a head (H groups of kNormGroup).
                    const Context::Arg sm[] = {{&s.scores, 0}};
                    ctx.dispatch("attn_softmax", sm, bytes_of(ap),
                                 H * kNormGroup, kNormGroup);
                    const Context::Arg va[] = {
                        {&s.scores, 0}, {pool, 0}, {&s.blocks, 0}, {&s.att, 0}};
                    ctx.dispatch(kv_kernel("attn_values", half), va,
                                 bytes_of(ap), H * hd, 64);
                }
                r.mv(lay.wo, s.att, s.x, 0, true);
                // A dense FFN's RMSNorm in gate / up likewise; a MoE layer
                // reads xb in many places, so it keeps rmsnorm_f32.
                const bool fold_ffn =
                    !lay.is_moe() && fold_ok({&lay.w_gate, &lay.w_up});
                metal::MvFuse ff_nf;
                if (fold_ffn) {
                    ff_nf.norm_w = &r.f(lay.ffn_norm);
                    ff_nf.eps = hp_.rms_eps;
                } else {
                    r.rmsnorm(lay.ffn_norm);
                }
                if (lay.is_moe()) {
                    r.moe(lay);
                    continue;
                }
                const Context::Buffer& fin = fold_ffn ? s.x : s.xb;
                mvf_must(lay.w_gate, fin, s.gate, 0, ff_nf);
                ctx.overlap_next();  // gate and up, likewise
                mvf_must(lay.w_up, fin, s.up, 0, ff_nf);
                const SiluParams sp{ff};
                const Context::Arg su[] = {{&s.gate, 0}, {&s.up, 0}};
                ctx.dispatch("silu_mul", su, bytes_of(sp), ff, 256);
                r.mv(lay.w_down, s.gate, s.x, 0, true);
            }
            r.rmsnorm(out_norm_);
            const std::uint64_t row_bytes =
                metal::matvec_weight_bytes(out_w_.type, 1, out_w_.cols);
            const auto hr = r.w(out_w_);
            metal::record_matvec(ctx, out_w_.type, R, out_w_.cols, *hr.buf,
                                 hr.off + lo * row_bytes, s.xb, 0, s.logits,
                                 0, false);
        });

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

/** metal_forward_multi's scratch: N rows of each (grow-only). */
struct MultiScratch {
    Context::Buffer x, xb, q, k, v, att, gate, up, logits, cs, blocks,
        rows, scores, parts;
};

MultiScratch& multi_scratch() {
    static MultiScratch s;
    return s;
}

}  // namespace

bool TransformerModel::metal_forward_multi(
    std::span<const tok::TokenId> toks,
    std::span<kv::PagedKvCache::Seq* const> seqs, kv::PagedKvCache& cache,
    std::span<float> logits) const {
    using namespace backend;
    const auto n = static_cast<std::uint32_t>(toks.size());
    const std::uint32_t E = hp_.n_embd, hd = hp_.head_dim, H = hp_.n_heads,
                        KV = hp_.n_kv_heads, ff = hp_.n_ff, V = hp_.n_vocab,
                        L = hp_.n_layers;
    const char* multi_env = std::getenv("LOCUS_METAL_MULTI");
    if ((multi_env != nullptr && std::string_view(multi_env) == "0") ||
        n == 0 || n > kMultiMax || seqs.size() != n ||
        !metal_forward_supported() || !kv_type_ok(cache) ||
        logits.size() < std::size_t{n} * V) {
        return false;
    }
    // Dense layers without per-head q / k norms: a MoE row's picks
    // differ per sequence, and the norms have no row form here yet.
    for (const Layer& lay : layers_) {
        if (lay.is_moe() || lay.has_qk_norm()) {
            return false;
        }
    }
    std::vector<std::uint32_t> pos(n);
    std::uint32_t most = 0;
    for (std::uint32_t c = 0; c < n; ++c) {
        if (toks[c] < 0 || static_cast<std::uint32_t>(toks[c]) >= V) {
            return false;  // the host path reports it
        }
        pos[c] = seqs[c]->n_tokens;
        if (pos[c] + 1 > hp_.n_ctx || pos[c] >= cache.capacity(*seqs[c])) {
            return false;
        }
        most = std::max(most, pos[c] + 1);
    }
    const std::size_t qd = std::size_t{H} * hd, kvd = std::size_t{KV} * hd;
    const std::uint32_t bt = cache.geometry().block_tokens;
    const char* passes = std::getenv("LOCUS_METAL_ATTN_PASSES");
    const bool three = passes != nullptr && std::string_view(passes) == "3";
    const bool head_fits = hd <= kAttnGroup && kAttnGroup % hd == 0;
    const char* split_env = std::getenv("LOCUS_METAL_ATTN_SPLIT");
    const bool split_ok =
        split_env == nullptr || std::string_view(split_env) != "0";
    const std::uint32_t max_parts = (most + kAttnGroup - 1) / kAttnGroup;
    // Each sequence's block table, back to back.
    std::vector<std::uint32_t> blk_at(n);
    std::uint32_t n_blk = 0;
    for (std::uint32_t c = 0; c < n; ++c) {
        blk_at[c] = n_blk;
        n_blk += (pos[c] + 1 + bt - 1) / bt;
    }

    std::lock_guard lk(metal::device_mutex());
    const Context::Buffer* pool = kv_pool_buffer(cache);
    if (pool == nullptr) {
        return false;  // not a Metal pool
    }
    const bool half = kv_half(cache);
    try {
        Context& ctx = Context::get();
        MultiScratch& s = multi_scratch();
        const auto grow = [&](Context::Buffer& b, std::size_t bytes) {
            Scratch::grow(ctx, b, bytes);
        };
        grow(s.x, std::size_t{n} * E * 4);
        grow(s.xb, std::size_t{n} * E * 4);
        grow(s.q, n * qd * 4);
        grow(s.att, n * qd * 4);
        grow(s.k, n * kvd * 4);
        grow(s.v, n * kvd * 4);
        grow(s.gate, std::size_t{n} * ff * 4);
        grow(s.up, std::size_t{n} * ff * 4);
        grow(s.logits, std::size_t{n} * V * 4);
        grow(s.cs, std::size_t{n} * hd * 4);
        grow(s.blocks, std::size_t{n_blk} * 4);
        grow(s.rows, std::size_t{n} * L * 8);
        // One slice of scores and of parts a sequence: the sequences'
        // attentions overlap.
        const std::size_t score_bytes = std::size_t{H} * most * 4;
        const std::size_t part_bytes =
            std::size_t{H} * max_parts * (std::size_t{hd} + 4) * 4;
        grow(s.scores, n * score_bytes);
        grow(s.parts, n * part_bytes);

        auto* x = static_cast<float*>(s.x.data);
        auto* cs = static_cast<float*>(s.cs.data);
        auto* blocks = static_cast<std::uint32_t*>(s.blocks.data);
        auto* rows = static_cast<std::uint64_t*>(s.rows.data);
        for (std::uint32_t c = 0; c < n; ++c) {
            ops().dequant_row(embd_, static_cast<std::uint32_t>(toks[c]),
                              {x + std::size_t{c} * E, E});
            rope_norm_table(hd, pos[c], hp_.rope_freq_base, rope_factors(),
                            {cs + std::size_t{c} * hd, hd});
            const auto& sb = seqs[c]->blocks;
            std::copy_n(sb.begin(), (pos[c] + 1 + bt - 1) / bt,
                        blocks + blk_at[c]);
            // Row (l, c): sequence c's K row at its position in layer l.
            for (std::uint32_t l = 0; l < L; ++l) {
                rows[std::size_t{l} * n + c] =
                    kv_row_off(cache, *seqs[c], l, pos[c]);
            }
        }

        const MetalRec r{ctx, weights_of(metal_weights_, file_backed_), hp_,
                         hd, metal_neox()};
        // Q6_K's columns kernel changes its bits with n, so only when
        // the decode need not be batch-invariant.
        const bool kquant = !metal_batch_invariant();
        // out[c] (+)= wt in[c] for the n rows: one matvec over the
        // columns (each weight read once), else one a row; norm, when
        // set, as the matvec's RMSNorm prologue (in is then the norm's
        // input). Either way a row's bits do not depend on n.
        const auto mm = [&](const Mat& wt, const Context::Buffer& in,
                            std::uint32_t in_stride,
                            const Context::Buffer& out,
                            std::uint64_t out_off,
                            std::uint32_t out_stride, bool acc,
                            const Context::Buffer* norm = nullptr) {
            const auto wb = r.w(wt);
            if (n == 1) {
                // One row: the matvec itself. The Q8_0 columns kernel
                // gives a column mv_q8_0's (and mv_q8_0_pn's) bits, so
                // this is the bits a batched step gives the row, at the
                // matvec's speed (the columns kernel's one column ran
                // a lone Q8_0 decode at half speed on an M2 Pro).
                if (norm == nullptr) {
                    metal::record_matvec(ctx, wt.type, wt.rows, wt.cols,
                                         *wb.buf, wb.off, in, 0, out,
                                         out_off, acc);
                    return;
                }
                // A folded norm feeds q / k / v and gate / up, which
                // never accumulate (the fused matvec stores).
                metal::MvFuse nf;
                nf.norm_w = norm;
                nf.eps = hp_.rms_eps;
                if (acc ||
                    !metal::record_matvec_fused(ctx, wt.type, wt.rows,
                                                wt.cols, *wb.buf, wb.off, in,
                                                0, out, out_off, nf)) {
                    throw std::logic_error(
                        "metal multi decode: a matvec not recorded");
                }
                return;
            }
            // Q4_K and Q6_K: the 8-wide matmul dequantizes each weight
            // block once for all n rows, at a cost flat in n (i#1191; M4
            // Max, Q4_K 8192 x 2048: 64 us against 87 for 4 matvecs and
            // 174 for 8; at 2048 x 2048, 42 against 38 and 74). So from
            // 4 rows on tall matrices and 6 on the others. Its bits
            // differ from the matvec's, so not in the batch-invariant
            // mode. LOCUS_METAL_MM8=0 keeps the matvecs (benches).
            static const bool mm8_on = [] {
                const char* e = std::getenv("LOCUS_METAL_MM8");
                return e == nullptr || std::string_view(e) != "0";
            }();
            const std::uint32_t mm8_from = wt.rows >= 4096 ? 4 : 6;
            if (mm8_on && kquant && norm == nullptr && n >= mm8_from &&
                (wt.type == gguf::TensorType::kQ4_K ||
                 wt.type == gguf::TensorType::kQ6_K) &&
                metal::record_matmul8(ctx, wt.type, wt.rows, wt.cols, n,
                                      *wb.buf, wb.off, in, 0, in_stride, out,
                                      out_off, out_stride, acc)) {
                return;
            }
            if (metal::record_matvec_cols(ctx, wt.type, wt.rows, wt.cols,
                                          *wb.buf, wb.off, in, 0, in_stride,
                                          out, out_off, out_stride, n, acc,
                                          norm, hp_.rms_eps, kquant)) {
                return;
            }
            if (norm != nullptr) {
                // fold_ok gave a type with no columns kernel a norm.
                throw std::logic_error(
                    "metal multi decode: a folded matvec not recorded");
            }
            for (std::uint32_t c = 0; c < n; ++c) {
                if (c > 0) {
                    ctx.overlap_next();  // independent rows
                }
                metal::record_matvec(
                    ctx, wt.type, wt.rows, wt.cols, *wb.buf, wb.off, in,
                    c * in_stride, out,
                    out_off + std::uint64_t{c} * out_stride, acc);
            }
        };
        // Fold a norm into the matvecs that read it where every one of
        // them is Q8_0 (the columns kernel's prologue), for any n, so
        // the choice and the bits do not depend on n. metal_forward's
        // gate otherwise (it folds F32 too).
        const char* nf_env = std::getenv("LOCUS_METAL_NORM_FOLD");
        const bool norm_fold =
            nf_env != nullptr ? std::string_view(nf_env) != "0"
                              : ctx.apple_family() >= 9;
        const auto fold_ok = [&](std::initializer_list<const Mat*> ms) {
            return norm_fold &&
                   std::all_of(ms.begin(), ms.end(), [](const Mat* m) {
                       return m->type == gguf::TensorType::kQ8_0;
                   });
        };
        const auto norm_rows = [&](std::span<const float> wt) {
            const RowNormParams np{E, E, hp_.rms_eps};
            const Context::Arg a[] = {{&s.x, 0}, {&r.f(wt), 0}, {&s.xb, 0}};
            ctx.dispatch("rmsnorm_rows", a, bytes_of(np), n * kNormGroup,
                         kNormGroup);
        };
        const auto rope_row = [&](const Context::Buffer& b, std::uint64_t off,
                                  std::uint32_t heads, std::uint32_t c) {
            const RopeParams rp{off, heads, hd, r.neox};
            const Context::Arg a[] = {
                {&b, 0}, {&s.cs, std::size_t{c} * hd * 4}};
            ctx.dispatch("rope_norm", a, bytes_of(rp), heads * hd / 2, 64);
        };

        run_batch(ctx, [&] {
            for (std::uint32_t l = 0; l < L; ++l) {
                const Layer& lay = layers_[l];
                const bool fold_attn = fold_ok({&lay.wq, &lay.wk, &lay.wv});
                const Context::Buffer* an = nullptr;
                if (fold_attn) {
                    an = &r.f(lay.attn_norm);
                } else {
                    norm_rows(lay.attn_norm);
                }
                const Context::Buffer& qin = fold_attn ? s.x : s.xb;
                // A lone row writes its K and V straight into its blocks
                // (as metal_forward); rows write theirs to k / v, then
                // kv_scatter copies each into its own sequence's. The
                // same kernels either way, so the same bits.
                // A half pool takes them through kv_scatter_f16 always.
                const bool direct = n == 1 && !half;
                const std::uint64_t krow = rows[std::size_t{l} * n];
                const std::uint64_t vrow =
                    krow + cache.layer_floats() / 2;
                const Context::Buffer& kb = direct ? *pool : s.k;
                const Context::Buffer& vb = direct ? *pool : s.v;
                const std::uint64_t ko = direct ? krow : 0,
                                    vo = direct ? vrow : 0;
                mm(lay.wq, qin, E, s.q, 0, static_cast<std::uint32_t>(qd),
                   false, an);
                ctx.overlap_next();
                mm(lay.wk, qin, E, kb, ko, static_cast<std::uint32_t>(kvd),
                   false, an);
                ctx.overlap_next();
                mm(lay.wv, qin, E, vb, vo, static_cast<std::uint32_t>(kvd),
                   false, an);
                if (lay.has_qkv_bias()) {
                    r.bias(s.q, 0, lay.wq_bias, n,
                           static_cast<std::uint32_t>(qd));
                    ctx.overlap_next();
                    r.bias(kb, ko, lay.wk_bias, n,
                           static_cast<std::uint32_t>(kvd));
                    ctx.overlap_next();
                    r.bias(vb, vo, lay.wv_bias, n,
                           static_cast<std::uint32_t>(kvd));
                }
                // Each row's RoPE at its own position (independent rows).
                for (std::uint32_t c = 0; c < n; ++c) {
                    if (c > 0) ctx.overlap_next();
                    rope_row(s.q, c * qd, H, c);
                    ctx.overlap_next();
                    rope_row(kb, ko + c * kvd, KV, c);
                }
                if (!direct) {
                    // Each row's K and V into its own sequence's blocks.
                    const ScatterParams sp{cache.layer_floats() / 2, n,
                                           static_cast<std::uint32_t>(kvd)};
                    const Context::Arg sa[] = {
                        {&s.k, 0}, {&s.v, 0}, {pool, 0},
                        {&s.rows, std::size_t{l} * n * 8}};
                    ctx.dispatch(kv_kernel("kv_scatter", half), sa,
                                 bytes_of(sp),
                                 static_cast<std::uint32_t>(n * kvd), 256);
                }
                // Each sequence's attention over its own blocks, as
                // metal_forward picks it for its length.
                for (std::uint32_t c = 0; c < n; ++c) {
                    const std::uint32_t np = pos[c] + 1;
                    const AttnParams ap{cache.block_floats(),
                                        cache.layer_offset(l),
                                        cache.layer_floats() / 2,
                                        np,
                                        bt,
                                        H,
                                        KV,
                                        hd,
                                        hp_.kq_scale};
                    const std::size_t qo = c * qd * 4, bo = blk_at[c] * 4,
                                      so = c * score_bytes,
                                      po = c * part_bytes;
                    if (c > 0) ctx.overlap_next();  // independent rows
                    if (!three && head_fits && split_ok &&
                        np > 2 * kAttnGroup) {
                        const std::uint32_t parts =
                            (np + kAttnGroup - 1) / kAttnGroup;
                        const AttnSplitParams spp{ap, parts};
                        const Context::Arg pa[] = {{&s.q, qo},
                                                   {pool, 0},
                                                   {&s.blocks, bo},
                                                   {&s.parts, po}};
                        ctx.dispatch(kv_kernel("attn_split", half), pa,
                                     bytes_of(spp), H * parts * kAttnGroup,
                                     kAttnGroup);
                        const Context::Arg ma[] = {{&s.parts, po},
                                                   {&s.att, qo}};
                        ctx.dispatch("attn_merge", ma, bytes_of(spp), H * hd,
                                     64);
                    } else if (!three && head_fits && np <= kAttnMaxPos) {
                        const Context::Arg fa[] = {{&s.q, qo},
                                                   {pool, 0},
                                                   {&s.blocks, bo},
                                                   {&s.att, qo}};
                        ctx.dispatch(kv_kernel("attn_fused", half), fa,
                                     bytes_of(ap), H * kAttnGroup,
                                     kAttnGroup);
                    } else {
                        const Context::Arg sca[] = {{&s.q, qo},
                                                    {pool, 0},
                                                    {&s.blocks, bo},
                                                    {&s.scores, so}};
                        ctx.dispatch(kv_kernel("attn_scores", half), sca,
                                     bytes_of(ap), H * np, 64);
                        const Context::Arg sm[] = {{&s.scores, so}};
                        ctx.dispatch("attn_softmax", sm, bytes_of(ap),
                                     H * kNormGroup, kNormGroup);
                        const Context::Arg va[] = {{&s.scores, so},
                                                   {pool, 0},
                                                   {&s.blocks, bo},
                                                   {&s.att, qo}};
                        ctx.dispatch(kv_kernel("attn_values", half), va,
                                     bytes_of(ap), H * hd, 64);
                    }
                }
                mm(lay.wo, s.att, static_cast<std::uint32_t>(qd), s.x, 0, E,
                   true);
                const bool fold_ffn = fold_ok({&lay.w_gate, &lay.w_up});
                const Context::Buffer* fn = nullptr;
                if (fold_ffn) {
                    fn = &r.f(lay.ffn_norm);
                } else {
                    norm_rows(lay.ffn_norm);
                }
                const Context::Buffer& fin = fold_ffn ? s.x : s.xb;
                mm(lay.w_gate, fin, E, s.gate, 0, ff, false, fn);
                ctx.overlap_next();
                mm(lay.w_up, fin, E, s.up, 0, ff, false, fn);
                const SiluParams sp2{n * ff};
                const Context::Arg su[] = {{&s.gate, 0}, {&s.up, 0}};
                ctx.dispatch("silu_mul", su, bytes_of(sp2), n * ff, 256);
                mm(lay.w_down, s.gate, ff, s.x, 0, E, true);
            }
            norm_rows(out_norm_);
            mm(out_w_, s.xb, E, s.logits, 0, V, false);
        });

        const auto* lg = static_cast<const float*>(s.logits.data);
        for (std::uint32_t c = 0; c < n; ++c) {
            const auto row = logits.subspan(std::size_t{c} * V, V);
            std::copy_n(lg + std::size_t{c} * V, V, row.begin());
            scale_logits(row);
            seqs[c]->n_tokens = pos[c] + 1;
        }
        g_metal_tokens.fetch_add(n);
        g_metal_multi_tokens.fetch_add(n);
    } catch (const std::exception&) {
        return false;  // a device error: the host path runs the tokens
    }
    return true;
}

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
    // MoE layers route on the GPU (moe_route a token); a selection it
    // does not cover (group-limited, over 1024 experts, or
    // LOCUS_METAL_MOE_ROUTE=cpu) runs the per-token forward.
    const bool any_moe = std::any_of(
        layers_.begin(), layers_.end(),
        [](const Layer& lay) { return lay.is_moe(); });
    const char* route_env = std::getenv("LOCUS_METAL_MOE_ROUTE");
    const bool gpu_route =
        hp_.n_group <= 1 && hp_.n_expert <= 1024 &&
        (route_env == nullptr || std::string_view(route_env) != "cpu");
    // moe_route's threadgroup: a thread an expert, whole simdgroups.
    const std::uint32_t route_group =
        std::max(32u, (hp_.n_expert + 31) / 32 * 32);
    if (n < (all_logits ? kPrefillMin : 2) || (any_moe && !gpu_route) ||
        !metal_forward_supported() ||
        !kv_type_ok(cache) ||
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
    // Where the tiled attention runs, its simdgroup-matrix form
    // (LOCUS_METAL_ATTN_MMA=0: the tiled one, for tests).
    const char* mma_env = std::getenv("LOCUS_METAL_ATTN_MMA");
    const bool mma_attn =
        tiled_attn &&
        (mma_env == nullptr || std::string_view(mma_env) != "0");
    const std::uint32_t bt = cache.geometry().block_tokens;

    std::lock_guard lk(metal::device_mutex());
    const Context::Buffer* pool = kv_pool_buffer(cache);
    if (pool == nullptr) {
        return false;
    }
    const bool half = kv_half(cache);
    try {
        Context& ctx = Context::get();
        BatchScratch& s = batch_scratch();
        // The sub-chunk is the engine's whole step: the attention gets
        // more threadgroups and each MoE expert's matmul tiles more of
        // its slots (all logits keep 256: their head rows are C * V
        // floats).
        const std::uint32_t chunk =
            all_logits ? kPrefillChunk : kPrefillChunkStep;
        const std::size_t C = std::min(n, chunk);
        const std::uint32_t n_blocks = (base + n + bt - 1) / bt;
        grow_batch(ctx, s, hp_, layers_, C, any_moe, n_blocks, all_logits);
        const std::uint32_t NR = hp_.n_expert + 1;  // a token's logits
        std::copy_n(seq.blocks.begin(), n_blocks,
                    static_cast<std::uint32_t*>(s.blocks.data));
        // The experts as grouped matmuls (LOCUS_METAL_MOE_MM=0: a matvec
        // slot each, for tests).
        const char* mm_env = std::getenv("LOCUS_METAL_MOE_MM");
        const std::uint32_t neox = metal_neox();
        const BatchRec r{
            {ctx, weights_of(metal_weights_, file_backed_), hp_, hd, neox},
            s,
            E,
            NR,
            route_group,
            mm_env == nullptr || std::string_view(mm_env) != "0"};

        for (std::uint32_t c0 = 0; c0 < n; c0 += chunk) {
            const std::uint32_t m = std::min(chunk, n - c0);
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
                        kv_row_off(cache, seq, l, pos);
                }
            }
            run_batch(ctx, [&] {
                for (std::uint32_t l = 0; l < L; ++l) {
                    const Layer& lay = layers_[l];
                    r.norm_rows(lay.attn_norm, m);
                    r.mm(lay.wq, s.xb, E, s.q, qd, false, m);
                    ctx.overlap_next();
                    r.mm(lay.wk, s.xb, E, s.k, kvd, false, m);
                    ctx.overlap_next();
                    r.mm(lay.wv, s.xb, E, s.v, kvd, false, m);
                    if (lay.has_qkv_bias()) {
                        r.bias(s.q, 0, lay.wq_bias, m,
                             static_cast<std::uint32_t>(qd));
                        ctx.overlap_next();
                        r.bias(s.k, 0, lay.wk_bias, m,
                             static_cast<std::uint32_t>(kvd));
                        ctx.overlap_next();
                        r.bias(s.v, 0, lay.wv_bias, m,
                             static_cast<std::uint32_t>(kvd));
                    }
                    if (lay.has_qk_norm()) {
                        r.head_norm(s.q, 0, H, lay.attn_q_norm, m,
                                  static_cast<std::uint32_t>(qd));
                        ctx.overlap_next();
                        r.head_norm(s.k, 0, KV, lay.attn_k_norm, m,
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
                    ctx.dispatch(kv_kernel("kv_scatter", half), sa,
                                 bytes_of(sp),
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
                    if (mma_attn) {
                        // 64 queries of a head a threadgroup, on the
                        // simdgroup matrices.
                        ctx.dispatch(kv_kernel("attn_rows_mma", half), aa,
                                     bytes_of(ap), ((m + 63) / 64) * H * 256,
                                     256);
                    } else if (tiled_attn) {
                        // 8 queries of a head a threadgroup, K / V staged.
                        ctx.dispatch(kv_kernel("attn_rows_tiled", half), aa,
                                     bytes_of(ap), ((m + 7) / 8) * H * 256,
                                     256);
                    } else {
                        ctx.dispatch(kv_kernel("attn_rows", half), aa,
                                     bytes_of(ap), m * H * kAttnGroup,
                                     kAttnGroup);
                    }
                    r.mm(lay.wo, s.att, static_cast<std::uint32_t>(qd), s.x, E,
                       true, m);
                    r.norm_rows(lay.ffn_norm, m);
                    if (lay.is_moe()) {
                        r.moe_rows(lay, m);
                        continue;
                    }
                    r.mm(lay.w_gate, s.xb, E, s.gate, ff, false, m);
                    ctx.overlap_next();
                    r.mm(lay.w_up, s.xb, E, s.up, ff, false, m);
                    const SiluParams sp2{m * ff};
                    const Context::Arg su[] = {{&s.gate, 0}, {&s.up, 0}};
                    ctx.dispatch("silu_mul", su, bytes_of(sp2), m * ff, 256);
                    r.mm(lay.w_down, s.gate, ff, s.x, E, true, m);
                }
                if (all_logits || last_chunk) {
                    r.norm_rows(out_norm_, m);
                    const std::uint32_t t0 = all_logits ? 0 : m - 1;
                    for (std::uint32_t t = t0; t < m; ++t) {
                        if (t > t0) {
                            ctx.overlap_next();
                        }
                        const auto hr = r.w(out_w_);
                        metal::record_matvec(
                            ctx, out_w_.type, V, out_w_.cols, *hr.buf, hr.off,
                            s.xb, t * E, s.head, std::uint64_t{t - t0} * V,
                            false);
                    }
                }
            });
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
