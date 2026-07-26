#include <algorithm>
#include <stdexcept>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/backend/vulkan/weight_pool.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/model/moe_stats.hpp"

namespace locus::backend {

namespace {

using vk::Kernel;
using vk::VulkanContext;

std::uint32_t fbits(float v) {
    return std::bit_cast<std::uint32_t>(v);
}

/** @returns On-disk bytes of one weight matrix, 4-byte padded. */
std::size_t weight_bytes(const Mat& w) {
    std::size_t b;
    switch (w.type) {
        case gguf::TensorType::kF32:
            b = static_cast<std::size_t>(w.rows) * w.cols * 4;
            break;
        case gguf::TensorType::kF16:
            b = static_cast<std::size_t>(w.rows) * w.cols * 2;
            break;
        case gguf::TensorType::kQ8_0:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 32) * 34;
            break;
        case gguf::TensorType::kQ4_0:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 32) * 18;
            break;
        case gguf::TensorType::kQ5_0:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 32) * 22;
            break;
        case gguf::TensorType::kQ4_K:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 256) * 144;
            break;
        case gguf::TensorType::kQ5_K:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 256) * 176;
            break;
        case gguf::TensorType::kQ6_K:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 256) * 210;
            break;
        case gguf::TensorType::kQ2_K:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 256) * 84;
            break;
        default:
            return 0;  // no GPU kernel for this type
    }
    return (b + 3) & ~std::size_t{3};
}

/** @returns The matvec kernel for w, or kCount_ if unsupported. */
Kernel matvec_kernel(const Mat& w) {
    switch (w.type) {
        case gguf::TensorType::kF32: return Kernel::kMatvecF32;
        case gguf::TensorType::kF16: return Kernel::kMatvecF16;
        case gguf::TensorType::kQ8_0: return Kernel::kMatvecQ8_0;
        case gguf::TensorType::kQ4_0: return Kernel::kMatvecQ4_0;
        case gguf::TensorType::kQ5_0: return Kernel::kMatvecQ5_0;
        case gguf::TensorType::kQ4_K: return Kernel::kMatvecQ4_K;
        case gguf::TensorType::kQ5_K: return Kernel::kMatvecQ5_K;
        case gguf::TensorType::kQ6_K: return Kernel::kMatvecQ6_K;
        case gguf::TensorType::kQ2_K: return Kernel::kMatvecQ2_K;
        default: return Kernel::kCount_;
    }
}

/** Converts a byte offset into shader w_off units for w's type. */
std::uint32_t w_off_units(const Mat& w, std::uint64_t bytes) {
    switch (w.type) {
        case gguf::TensorType::kF32:
            return static_cast<std::uint32_t>(bytes / 4);
        case gguf::TensorType::kF16:
            return static_cast<std::uint32_t>(bytes / 2);
        default:
            return static_cast<std::uint32_t>(bytes);
    }
}

using WeightPool = WeightPoolT<VulkanContext::Buffer>;

/**
 * Process-lifetime GPU state: one context, a weight pager for
 * weights and norm vectors (keyed by their mmap pointers, stable
 * for the model's lifetime), grow-only activation scratch, and
 * the GPU-mapped KV pools handed to PagedKvCache.
 *
 * Single-threaded by contract: ops are called only from the
 * engine/model thread.
 */
struct State {
    VulkanContext ctx;
    WeightPool pool;
    std::unordered_map<const void*, VulkanContext::Buffer>
        kv_pools;

    State() {
        pool.create = [this](std::size_t n) {
            return ctx.create_buffer(n);
        };
        pool.destroy = [this](VulkanContext::Buffer b) {
            ctx.destroy_buffer(b);
        };
    }
    ~State() { pool.report(); }

    /** Activation buffers, sized on first use per model dims. */
    VulkanContext::Buffer x{}, xb{}, q{}, gate{}, up{}, attout{},
        att{}, logits{}, table{}, ones{}, kv_a{}, q_abs{},
        latent{}, q_a{};
    std::size_t x_n = 0, xb_n = 0, q_n = 0, gate_n = 0, up_n = 0,
                attout_n = 0, att_n = 0, logits_n = 0,
                table_n = 0, ones_n = 0, kv_a_n = 0, q_abs_n = 0,
                latent_n = 0, q_a_n = 0;

    VulkanContext::Buffer upload(const void* ptr,
                                 std::size_t bytes) {
        return pool.acquire(
            ptr, bytes, [&](VulkanContext::Buffer b) {
                ctx.write_buffer(
                    b, {static_cast<const std::byte*>(ptr),
                        bytes});
            });
    }

    /** Uploads w dequantized to F32 (any CPU-supported type). */
    VulkanContext::Buffer upload_f32(const Mat& w) {
        const std::size_t bytes =
            static_cast<std::size_t>(w.rows) * w.cols * 4;
        return pool.acquire(
            w.data, bytes, [&](VulkanContext::Buffer b) {
                std::vector<float> host(
                    static_cast<std::size_t>(w.rows) * w.cols);
                for (std::uint32_t r = 0; r < w.rows; ++r) {
                    dequant_row(
                        w, r,
                        {host.data() +
                             static_cast<std::size_t>(r) * w.cols,
                         w.cols});
                }
                ctx.write_buffer(
                    b, std::as_bytes(
                           std::span<const float>(host)));
            });
    }

    void grow(VulkanContext::Buffer& b, std::size_t& cap,
              std::size_t n_floats) {
        if (n_floats > cap) {
            if (cap != 0) {
                ctx.destroy_buffer(b);
            }
            b = ctx.create_buffer(n_floats * 4);
            cap = n_floats;
        }
    }

    /** Rope divisor buffer: factors when scaled, else 1.0s. */
    VulkanContext::Buffer rope_divisors(
        std::span<const float> factors, std::uint32_t half_dim) {
        if (!factors.empty()) {
            return upload(factors.data(), factors.size() * 4);
        }
        if (half_dim > ones_n) {
            if (ones_n != 0) {
                ctx.destroy_buffer(ones);
            }
            ones = ctx.create_buffer(half_dim * 4);
            auto* p = static_cast<float*>(ctx.mapped(ones));
            for (std::uint32_t i = 0; i < half_dim; ++i) {
                p[i] = 1.0f;
            }
            ones_n = half_dim;
        }
        return ones;
    }
};

State& state() {
    static State s;
    return s;
}

/** Records one matvec dispatch into the open batch. */
void rec_matvec(State& s, const Mat& w, VulkanContext::Buffer x,
                VulkanContext::Buffer out,
                std::uint32_t out_offset, bool accumulate,
                std::uint64_t w_byte_off = 0,
                std::uint32_t x_off = 0, float scale = 1.0f,
                VulkanContext::Buffer* whole = nullptr) {
    const VulkanContext::Buffer wb =
        whole != nullptr ? *whole
                         : s.upload(w.data, weight_bytes(w));
    const VulkanContext::Buffer bufs[] = {wb, x, out};
    const std::uint32_t push[] = {w.rows,
                                  w.cols,
                                  out_offset,
                                  accumulate ? 1u : 0u,
                                  w_off_units(w, w_byte_off),
                                  x_off,
                                  fbits(scale)};
    s.ctx.dispatch(matvec_kernel(w), bufs, push,
                   (w.rows + 63) / 64);
}

}  // namespace

bool vulkan_backend_usable() {
    static const bool ok = VulkanContext::available();
    return ok;
}

float* vulkan_alloc_kv(std::size_t n_floats) {
    State& s = state();
    auto buf = s.ctx.create_buffer(n_floats * 4);
    void* map = s.ctx.mapped(buf);
    std::memset(map, 0, n_floats * 4);
    s.kv_pools.emplace(map, buf);
    return static_cast<float*>(map);
}

void matvec_vulkan(const Mat& w, std::span<const float> x,
                   std::span<float> out) {
    if (matvec_kernel(w) == Kernel::kCount_) {
        matvec(w, x, out);  // no shader for this weight type
        return;
    }
    State& s = state();
    auto wb = s.upload(w.data, weight_bytes(w));
    s.grow(s.x, s.x_n, x.size());
    s.grow(s.xb, s.xb_n, out.size());
    s.ctx.write_buffer(s.x, std::as_bytes(x));
    s.ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {wb, s.x, s.xb};
    const std::uint32_t push[] = {w.rows, w.cols, 0, 0, 0, 0,
                                  fbits(1.0f)};
    s.ctx.dispatch(matvec_kernel(w), bufs, push,
                   (w.rows + 63) / 64);
    s.ctx.end_batch();
    s.pool.on_batch_end();
    s.ctx.read_buffer(s.xb, std::as_writable_bytes(out));
}

bool vulkan_forward(const model::LlamaModel& m, tok::TokenId token,
                    kv::PagedKvCache& cache,
                    kv::PagedKvCache::Seq& seq,
                    std::span<float> logits) {
    const auto& hp = m.hparams();
    State& s = state();
    auto pool_it = s.kv_pools.find(cache.pool_data());
    if (pool_it == s.kv_pools.end()) {
        return false;  // cache pool is not GPU-mapped
    }
    if (hp.idx_top_k > 0) {
        // DSA indexer models: the 704-float cache row and the
        // top-k selection are CPU-only for now.
        return false;
    }
    const bool mla = hp.arch == model::Arch::kDeepseek2;
    for (const auto& l : m.layers()) {
        if (matvec_kernel(l.wo) == Kernel::kCount_) {
            return false;
        }
        auto ok = [](const Mat& w) {
            return w.rows == 0 ||
                   matvec_kernel(w) != Kernel::kCount_;
        };
        if (!ok(l.wq) || !ok(l.wq_a) || !ok(l.wq_b) ||
            !ok(l.wk) || !ok(l.wv) || !ok(l.w_gate) ||
            !ok(l.w_up) || !ok(l.w_down) || !ok(l.gate_inp) ||
            !ok(l.gate_exps.base) || !ok(l.up_exps.base) ||
            !ok(l.down_exps.base) || !ok(l.gate_shexp) ||
            !ok(l.up_shexp) || !ok(l.down_shexp) ||
            !ok(l.wkv_a) || !ok(l.wk_b) || !ok(l.wv_b)) {
            return false;
        }
        // Fused wkv_b goes through the F32-dequant path.
    }
    if (matvec_kernel(m.output_weight()) == Kernel::kCount_) {
        return false;
    }

    if (seq.n_tokens >= hp.n_ctx ||
        seq.n_tokens >= cache.capacity(seq)) {
        throw std::invalid_argument("seq capacity not ensured");
    }
    const std::uint32_t pos = seq.n_tokens;
    const auto& geom = cache.geometry();
    const std::uint32_t kv_dim = geom.kv_dim;
    const std::uint32_t layer_stride =
        geom.block_tokens * kv_dim * 2;
    const std::uint32_t block_stride =
        hp.n_layers * layer_stride;

    const std::uint32_t rank = hp.kv_lora_rank;
    const std::uint32_t nope = hp.qk_nope_dim;
    const std::uint32_t rope_d = hp.qk_rope_dim;
    const std::uint32_t vd = hp.v_head_dim;
    const std::uint32_t qk = nope + rope_d;

    s.grow(s.x, s.x_n, hp.n_embd);
    const std::uint32_t ff_max = std::max(
        {hp.n_ff, hp.n_ff_exp,
         hp.n_ff_exp * hp.n_expert_shared});
    s.grow(s.xb, s.xb_n,
           std::max<std::size_t>(hp.n_embd, ff_max));
    s.grow(s.q, s.q_n,
           mla ? std::size_t{hp.n_heads} * qk : hp.n_embd);
    s.grow(s.gate, s.gate_n, ff_max);
    s.grow(s.up, s.up_n, ff_max);
    s.grow(s.attout, s.attout_n,
           mla ? std::size_t{hp.n_heads} * vd : hp.n_embd);
    s.grow(s.att, s.att_n,
           static_cast<std::size_t>(hp.n_heads) * hp.n_ctx);
    s.grow(s.logits, s.logits_n, hp.n_vocab);
    s.grow(s.table, s.table_n, cache.total_blocks());
    if (mla) {
        s.grow(s.kv_a, s.kv_a_n, rank + rope_d);
        s.grow(s.q_abs, s.q_abs_n,
               static_cast<std::size_t>(hp.n_heads) * rank);
        s.grow(s.latent, s.latent_n,
               static_cast<std::size_t>(hp.n_heads) * rank);
        s.grow(s.q_a, s.q_a_n,
               std::max<std::size_t>(hp.q_lora_rank, 1));
    }

    // CPU side: embedding lookup and this token's block table.
    std::vector<float> x_host(hp.n_embd);
    dequant_row(m.embedding(), static_cast<std::uint32_t>(token),
                x_host);
    s.ctx.write_buffer(
        s.x, std::as_bytes(std::span<const float>(x_host)));
    s.ctx.write_buffer(
        s.table,
        std::as_bytes(std::span<const kv::BlockId>(seq.blocks)));

    const auto pool = pool_it->second;
    const std::uint32_t eps_bits = fbits(hp.rms_eps);
    const std::uint32_t base_bits = fbits(hp.rope_freq_base);
    float corr_lo = 0.0f, corr_hi = 0.0f;
    if (mla) {
        yarn_corr_range(rope_d, hp.rope_freq_base, hp.yarn,
                        corr_lo, corr_hi);
    }
    const std::uint32_t fs_bits = fbits(hp.yarn.freq_scale);
    const std::uint32_t lo_bits = fbits(corr_lo);
    const std::uint32_t hi_bits = fbits(corr_hi);
    const std::uint32_t block = seq.blocks[pos / geom.block_tokens];
    const std::uint32_t in_block = pos % geom.block_tokens;
    const std::uint32_t group =
        hp.n_kv_heads > 0 ? hp.n_heads / hp.n_kv_heads : 1;

    auto norm = [&](std::span<const float> w,
                    VulkanContext::Buffer in,
                    VulkanContext::Buffer out,
                    std::uint32_t n, std::uint32_t out_off) {
        const VulkanContext::Buffer bufs[] = {
            in, s.upload(w.data(), w.size() * 4), out};
        const std::uint32_t push[] = {n, eps_bits, out_off};
        s.ctx.dispatch(Kernel::kRmsNorm, bufs, push, 1);
    };
    auto swiglu = [&](const Mat& wg, const Mat& wu, const Mat& wd,
                      std::uint32_t ff, float wgt,
                      std::uint64_t off_g, std::uint64_t off_u,
                      std::uint64_t off_d,
                      VulkanContext::Buffer* whole_g,
                      VulkanContext::Buffer* whole_u,
                      VulkanContext::Buffer* whole_d) {
        Mat g2 = wg, u2 = wu, d2 = wd;
        g2.rows = ff;
        u2.rows = ff;
        d2.cols = ff;
        rec_matvec(s, g2, s.xb, s.gate, 0, false, off_g, 0, 1.0f,
                   whole_g);
        rec_matvec(s, u2, s.xb, s.up, 0, false, off_u, 0, 1.0f,
                   whole_u);
        const VulkanContext::Buffer bufs[] = {s.gate, s.up,
                                              s.gate};
        const std::uint32_t push[] = {ff};
        s.ctx.dispatch(Kernel::kSiluMul, bufs, push,
                       (ff + 63) / 64);
        rec_matvec(s, d2, s.gate, s.x, 0, true, off_d, 0, wgt,
                   whole_d);
    };

    std::vector<float> xb_host(hp.n_embd);
    std::vector<float> router(hp.n_expert);

    s.ctx.begin_batch();
    for (std::uint32_t l = 0; l < hp.n_layers; ++l) {
        const auto& lay = m.layers()[l];
        const std::uint32_t row_off = block * block_stride +
                                      l * layer_stride +
                                      in_block * kv_dim;

        norm(lay.attn_norm, s.x, s.xb, hp.n_embd, 0);
        if (!mla) {
            const VulkanContext::Buffer divisors =
                s.rope_divisors(m.rope_factors(),
                                hp.head_dim / 2);
            rec_matvec(s, lay.wq, s.xb, s.q, 0, false);
            rec_matvec(s, lay.wk, s.xb, pool, row_off, false);
            rec_matvec(s, lay.wv, s.xb, pool,
                       row_off + layer_stride / 2, false);
            {
                const VulkanContext::Buffer bufs[] = {s.q,
                                                      divisors};
                const std::uint32_t push[] = {
                    hp.n_heads, hp.head_dim, pos, 0, hp.head_dim,
                    base_bits,  fbits(1.0f), 0,   0};
                s.ctx.dispatch(Kernel::kRope, bufs, push,
                               hp.n_heads);
            }
            {
                const VulkanContext::Buffer bufs[] = {pool,
                                                      divisors};
                const std::uint32_t push[] = {
                    hp.n_kv_heads, hp.head_dim, pos,
                    row_off,       hp.head_dim, base_bits,
                    fbits(1.0f),   0,           0};
                s.ctx.dispatch(Kernel::kRope, bufs, push,
                               hp.n_kv_heads);
            }
            {
                const VulkanContext::Buffer bufs[] = {
                    s.q, pool, s.table, s.att, s.attout};
                const std::uint32_t push[] = {
                    hp.n_heads,   group,
                    hp.head_dim,  kv_dim,
                    pos + 1,      geom.block_tokens,
                    block_stride, l * layer_stride,
                    layer_stride / 2};
                s.ctx.dispatch(Kernel::kAttnPaged, bufs, push,
                               hp.n_heads);
            }
        } else {
            const bool split = lay.wk_b.rows > 0;
            VulkanContext::Buffer wkv_b32{};
            if (!split) {
                wkv_b32 = s.upload_f32(lay.wkv_b);
            }
            const VulkanContext::Buffer divisors =
                s.rope_divisors({}, rope_d / 2);
            if (hp.q_lora_rank > 0) {
                rec_matvec(s, lay.wq_a, s.xb, s.q_a, 0, false);
                norm(lay.q_a_norm, s.q_a, s.q_a,
                     hp.q_lora_rank, 0);
                rec_matvec(s, lay.wq_b, s.q_a, s.q, 0, false);
            } else {
                rec_matvec(s, lay.wq, s.xb, s.q, 0, false);
            }
            rec_matvec(s, lay.wkv_a, s.xb, s.kv_a, 0, false);
            norm(lay.kv_a_norm, s.kv_a, pool, rank, row_off);
            s.ctx.copy_buffer(s.kv_a, rank * 4, pool,
                              (row_off + rank) * 4, rope_d * 4);
            {
                const VulkanContext::Buffer bufs[] = {pool,
                                                      divisors};
                const std::uint32_t push[] = {
                    1,       rope_d,  pos,     row_off + rank,
                    0,       base_bits, fs_bits, lo_bits,
                    hi_bits};
                s.ctx.dispatch(Kernel::kRope, bufs, push, 1);
            }
            {
                const VulkanContext::Buffer bufs[] = {s.q,
                                                      divisors};
                const std::uint32_t push[] = {
                    hp.n_heads, rope_d,  pos,     nope,
                    qk,         base_bits, fs_bits, lo_bits,
                    hi_bits};
                s.ctx.dispatch(Kernel::kRope, bufs, push,
                               hp.n_heads);
            }
            for (std::uint32_t h = 0; h < hp.n_heads; ++h) {
                if (split) {
                    // Pre-transposed k_b: plain matvec per head.
                    Mat kb = lay.wk_b;
                    kb.rows = rank;
                    rec_matvec(
                        s, kb, s.q, s.q_abs, h * rank, false,
                        std::uint64_t{h} * rank *
                            mat_row_bytes(lay.wk_b),
                        h * qk, 1.0f);
                } else {
                    const VulkanContext::Buffer bufs[] = {
                        wkv_b32, s.q, s.q_abs};
                    const std::uint32_t push[] = {
                        nope, rank, (h * (nope + vd)) * rank,
                        h * qk, h * rank};
                    s.ctx.dispatch(Kernel::kMatvecT, bufs, push,
                                   (rank + 63) / 64);
                }
            }
            {
                const VulkanContext::Buffer bufs[] = {
                    s.q,   s.q_abs, pool,
                    s.table, s.att,   s.latent};
                const std::uint32_t push[] = {
                    hp.n_heads,   rank,
                    nope,         rope_d,
                    pos + 1,      geom.block_tokens,
                    block_stride, l * layer_stride,
                    fbits(hp.kq_scale)};
                s.ctx.dispatch(Kernel::kAttnMla, bufs, push,
                               hp.n_heads);
            }
            for (std::uint32_t h = 0; h < hp.n_heads; ++h) {
                if (split) {
                    Mat vb = lay.wv_b;
                    vb.rows = vd;
                    rec_matvec(
                        s, vb, s.latent, s.attout, h * vd,
                        false,
                        std::uint64_t{h} * vd *
                            mat_row_bytes(lay.wv_b),
                        h * rank, 1.0f);
                } else {
                    Mat uv{gguf::TensorType::kF32, nullptr, vd,
                           rank};
                    rec_matvec(
                        s, uv, s.latent, s.attout, h * vd,
                        false,
                        std::uint64_t{h * (nope + vd) + nope} *
                            rank * 4,
                        h * rank, 1.0f, &wkv_b32);
                }
            }
        }
        rec_matvec(s, lay.wo, s.attout, s.x, 0, true);

        norm(lay.ffn_norm, s.x, s.xb, hp.n_embd, 0);
        if (!lay.is_moe()) {
            swiglu(lay.w_gate, lay.w_up, lay.w_down, hp.n_ff,
                   1.0f, 0, 0, 0, nullptr, nullptr, nullptr);
        } else {
            // Router runs on the CPU: sync the batch, read the
            // normed activations, gate, then keep recording.
            s.ctx.end_batch();
            s.pool.on_batch_end();
            s.ctx.read_buffer(
                s.xb,
                std::as_writable_bytes(std::span<float>(
                    xb_host.data(), hp.n_embd)));
            matvec(lay.gate_inp,
                   {xb_host.data(), hp.n_embd}, router);
            const auto picked =
                model::moe_select(hp, lay, router);
            if (model::MoeStats::enabled()) {
                for (const auto& [e, wgt] : picked) {
                    model::MoeStats::record(l, e, hp.n_layers,
                                            hp.n_expert);
                }
            }
            s.ctx.begin_batch();
            // Upload only the ROUTED experts, each to its own
            // pooled buffer keyed by that expert's mmap pointer, so
            // the pager caches/evicts per expert across tokens.
            // Uploading all n_expert inflated MoE residency by
            // n_expert/n_expert_used (e.g. 64/6) and OOMs big
            // models on small hosts; only the picked experts are
            // ever read by the swiglu dispatches below.
            const auto up_expert =
                [&](const model::LlamaModel::ExpertMat& em,
                    std::uint32_t e) {
                    return s.upload(
                        em.expert(e).data,
                        (em.expert_bytes + 3) & ~std::uint64_t{3});
                };
            for (const auto& [e, wgt] : picked) {
                VulkanContext::Buffer bg =
                    up_expert(lay.gate_exps, e);
                VulkanContext::Buffer bu =
                    up_expert(lay.up_exps, e);
                VulkanContext::Buffer bd =
                    up_expert(lay.down_exps, e);
                swiglu(lay.gate_exps.expert(e),
                       lay.up_exps.expert(e),
                       lay.down_exps.expert(e), hp.n_ff_exp, wgt, 0,
                       0, 0, &bg, &bu, &bd);
            }
            if (hp.n_expert_shared > 0) {
                swiglu(lay.gate_shexp, lay.up_shexp,
                       lay.down_shexp,
                       hp.n_ff_exp * hp.n_expert_shared, 1.0f, 0,
                       0, 0, nullptr, nullptr, nullptr);
            }
        }
    }
    norm(m.output_norm(), s.x, s.xb, hp.n_embd, 0);
    rec_matvec(s, m.output_weight(), s.xb, s.logits, 0, false);
    s.ctx.end_batch();
    s.pool.on_batch_end();

    s.ctx.read_buffer(s.logits, std::as_writable_bytes(logits));
    seq.n_tokens = pos + 1;
    return true;
}

}  // namespace locus::backend
