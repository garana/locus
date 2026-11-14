// Metal forward of a plain llama model (i#866): one token, every layer
// on the GPU in one command batch, reading and writing the KV pool in
// place (a shared Metal buffer, Ops.alloc_kv).

#include <algorithm>
#include <atomic>
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

    ~MetalWeights() {
        std::lock_guard lk(backend::metal::device_mutex());
        for (auto& [p, b] : bufs) {
            Context::get().free(b);
        }
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
    std::uint32_t n_heads, head_dim;
};
struct AttnParams {
    std::uint64_t block_floats, layer_off, v_off;
    std::uint32_t n_pos, block_tokens, n_heads, n_kv_heads, head_dim;
    float kq_scale;
};
struct SiluParams {
    std::uint32_t n;
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
    if (off || backend_->name != "metal" || !spec_->plain_attention ||
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
    for (const Layer& lay : layers_) {
        if (lay.is_moe() || !lay.has_separate_qkv(hp_) ||
            lay.has_qkv_bias() || lay.has_qk_norm() ||
            !lay.post_attn_norm.empty() || !lay.post_ffw_norm.empty() ||
            !ok_w(lay.wq) || !ok_w(lay.wk) || !ok_w(lay.wv) || !ok_w(lay.wo) ||
            !ok_w(lay.w_gate) || !ok_w(lay.w_up) || !ok_w(lay.w_down)) {
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
        const auto wbuf = [&](const Mat& m) -> const Context::Buffer& {
            return mw.get(ctx, m.data,
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
        Scratch::grow(ctx, s.gate, std::size_t{ff} * 4);
        Scratch::grow(ctx, s.up, std::size_t{ff} * 4);
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
            metal::record_matvec(ctx, w.type, w.rows, w.cols, wbuf(w), 0, in,
                                 0, out, out_off, acc);
        };
        const auto rope = [&](const Context::Buffer& b, std::uint64_t off,
                              std::uint32_t heads) {
            const RopeParams rp{off, heads, hd};
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
            metal::record_matvec(ctx, out_w_.type, R, out_w_.cols,
                                 wbuf(out_w_), lo * row_bytes, s.xb, 0,
                                 s.logits, 0, false);
        } catch (...) {
            ctx.end_batch();
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

}  // namespace locus::model
