#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "iq_grids.h"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/backend/vulkan/limits.hpp"
#include "locus/backend/vulkan/mem_fit.hpp"
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

/** @returns On-disk bytes of one weight matrix (exact). */
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
        case gguf::TensorType::kIQ2_XXS:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 256) * 66;
            break;
        case gguf::TensorType::kIQ3_XXS:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 256) * 98;
            break;
        case gguf::TensorType::kIQ4_XS:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 256) * 136;
            break;
        default:
            return 0;  // no GPU kernel for this type
    }
    return b;
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
        case gguf::TensorType::kIQ2_XXS:
            return Kernel::kMatvecIQ2_XXS;
        case gguf::TensorType::kIQ3_XXS:
            return Kernel::kMatvecIQ3_XXS;
        case gguf::TensorType::kIQ4_XS:
            return Kernel::kMatvecIQ4_XS;
        default: return Kernel::kCount_;
    }
}

/** @returns The batched (*_b) matvec kernel for w, or kCount_. */
Kernel batched_kernel(const Mat& w) {
    switch (w.type) {
        case gguf::TensorType::kF32: return Kernel::kMatvecF32B;
        case gguf::TensorType::kF16: return Kernel::kMatvecF16B;
        case gguf::TensorType::kQ8_0: return Kernel::kMatvecQ8_0B;
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
    /** Per-model GPU-routing verdict (memory fit), decided once. */
    std::unordered_map<const void*, bool> moe_gpu_fits;

    State() {
        pool.create = [this](std::size_t n) {
            return ctx.create_buffer(n);
        };
        pool.destroy = [this](VulkanContext::Buffer b) {
            ctx.destroy_buffer(b);
        };
    }
    ~State() {
        pool.report();
        ctx.report_profile();
    }

    /** Activation buffers, sized on first use per model dims. */
    VulkanContext::Buffer x{}, xb{}, q{}, gate{}, up{}, attout{},
        att{}, logits{}, table{}, ones{}, kv_a{}, q_abs{},
        latent{}, q_a{}, router{}, route{};
    std::size_t x_n = 0, xb_n = 0, q_n = 0, gate_n = 0, up_n = 0,
                attout_n = 0, att_n = 0, logits_n = 0,
                table_n = 0, ones_n = 0, kv_a_n = 0, q_abs_n = 0,
                latent_n = 0, q_a_n = 0, router_n = 0, route_n = 0;

    /**
     * Uploads a whole stacked expert tensor (every expert) as one
     * buffer, keyed by the ExpertMat itself: its base pointer is
     * also expert 0's key on the per-expert (CPU-routed) path.
     */
    VulkanContext::Buffer upload_experts(
        const model::TransformerModel::ExpertMat& em,
        std::uint32_t n_expert) {
        return upload_keyed(
            &em, em.base.data,
            static_cast<std::size_t>(em.expert_bytes) * n_expert);
    }

    VulkanContext::Buffer upload(const void* ptr,
                                 std::size_t bytes) {
        return upload_keyed(ptr, ptr, bytes);
    }

    /**
     * Uploads exactly `bytes` from src into a pooled buffer keyed by
     * `key`, allocated gpu_alloc_bytes(bytes) with the tail zeroed
     * (never reads past the tensor on the host or the buffer on the
     * GPU).
     */
    VulkanContext::Buffer upload_keyed(const void* key, const void* src,
                                       std::size_t bytes) {
        const std::size_t alloc = vk::gpu_alloc_bytes(bytes);
        return pool.acquire(key, alloc, [&](VulkanContext::Buffer b) {
            auto* dst = static_cast<std::byte*>(ctx.mapped(b));
            std::memcpy(dst, src, bytes);
            std::memset(dst + bytes, 0, alloc - bytes);
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

    /**
     * Uploads MLA's fused wkv_b dequantized to F32 with each head's
     * W_uk block transposed to rank x nope (row-major), so the
     * batched absorption q_abs = W_uk^T q_nope is a plain matmul.
     * Keyed one byte into the tensor: no weight starts there, and
     * upload_f32 already owns the tensor's own address.
     */
    VulkanContext::Buffer upload_kb_t(const Mat& w, std::uint32_t heads,
                                      std::uint32_t nope, std::uint32_t vd,
                                      std::uint32_t rank) {
        const std::size_t n =
            static_cast<std::size_t>(heads) * rank * nope;
        return pool.acquire(
            static_cast<const std::byte*>(w.data) + 1, n * 4,
            [&](VulkanContext::Buffer b) {
                std::vector<float> host(n);
                std::vector<float> row(rank);
                for (std::uint32_t h = 0; h < heads; ++h) {
                    for (std::uint32_t j = 0; j < nope; ++j) {
                        dequant_row(w, h * (nope + vd) + j, row);
                        for (std::uint32_t i = 0; i < rank; ++i) {
                            host[(static_cast<std::size_t>(h) * rank +
                                  i) * nope + j] = row[i];
                        }
                    }
                }
                ctx.write_buffer(
                    b, std::as_bytes(std::span<const float>(host)));
            });
    }

    /** Scratch for the batched (prefill) forward, per chunk. */
    enum Chunk {
        kCX,       // residual rows, n x n_embd
        kCXb,      // normed rows
        kCQ,       // queries
        kCK,       // llama: roped K rows before the pool scatter
        kCV,
        kCGate,    // FFN gate (routed: one row per token, slot)
        kCUp,
        kCAttOut,  // attention output rows (wo input)
        kCAtt,     // attention scores, n x heads x n_pos
        kCQa,      // MLA: q_a rows (q_lora_rank > 0)
        kCKva,     // MLA: kv_a rows, normed + roped in place
        kCQAbs,    // MLA: absorbed queries, n x heads x rank
        kCLat,     // MLA: attention latents, n x heads x rank
        kCRouter,  // MoE: logits + selection scratch, n x 2E
        kCRoute,   // MoE: ids + weights, n x 2k
        kCIds,     // MoE: per-expert pair lists (moe_group)
        kCDown,    // MoE: routed down outputs, n*k x n_embd
        kCLogits,
        kChunkCount_
    };
    VulkanContext::Buffer chunk[kChunkCount_]{};
    std::size_t chunk_n[kChunkCount_]{};

    /** @returns Chunk buffer c, grown to at least n_floats. */
    VulkanContext::Buffer chunk_buf(Chunk c, std::size_t n_floats) {
        grow(chunk[c], chunk_n[c], std::max<std::size_t>(n_floats, 1));
        return chunk[c];
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

    // IQ grid+signs tables (SSBO 3), uploaded once: the grid then
    // ksigns_iq2xs (128 B) at the grid's end. iq2xxs_grid is 2048 B
    // (256 u64), iq3xxs_grid is 1024 B (256 u32).
    VulkanContext::Buffer iq2xxs_tbl{}, iq3xxs_tbl{};
    bool iq2xxs_tbl_ready = false, iq3xxs_tbl_ready = false;

    template <typename Grid>
    VulkanContext::Buffer grid_table(VulkanContext::Buffer& buf,
                                     bool& ready, const Grid& grid) {
        if (!ready) {
            const std::size_t gsz = sizeof(grid);
            const std::size_t ssz = sizeof(::ksigns_iq2xs);
            buf = ctx.create_buffer(gsz + ssz);
            auto* p = static_cast<std::byte*>(ctx.mapped(buf));
            std::memcpy(p, &grid[0], gsz);
            std::memcpy(p + gsz, ::ksigns_iq2xs, ssz);
            ready = true;
        }
        return buf;
    }

    /** The extra SSBO an IQ kernel binds (grid+signs), or null. */
    VulkanContext::Buffer grid_for(Kernel k) {
        if (k == Kernel::kMatvecIQ2_XXS) {
            return grid_table(iq2xxs_tbl, iq2xxs_tbl_ready,
                              ::iq2xxs_grid);
        }
        if (k == Kernel::kMatvecIQ3_XXS) {
            return grid_table(iq3xxs_tbl, iq3xxs_tbl_ready,
                              ::iq3xxs_grid);
        }
        return {};
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
    const Kernel k = matvec_kernel(w);
    const VulkanContext::Buffer grid = s.grid_for(k);
    const VulkanContext::Buffer bufs[] = {wb, x, out, grid};
    const std::size_t nbuf = grid.impl != nullptr ? 4u : 3u;
    const std::uint32_t push[] = {w.rows,
                                  w.cols,
                                  out_offset,
                                  accumulate ? 1u : 0u,
                                  w_off_units(w, w_byte_off),
                                  x_off,
                                  fbits(scale)};
    s.ctx.dispatch(k, {bufs, nbuf}, push,
                   VulkanContext::matvec_groups(k, w.rows));
}

/** Batched matvec mode bits (matvec*.comp, -DBATCHED). */
constexpr std::uint32_t kRouteIds = 1;  // slot -> route[slot] id
constexpr std::uint32_t kSlotSum = 2;   // weighted sum of slots

/**
 * Records one batched matvec: `slots` matrices of w's shape in
 * `wb`, slot k at byte offset w_byte_off + id * w_stride_bytes
 * reading x at x_off + k * x_stride. Split mode (no kSlotSum)
 * runs a workgroup row per slot writing out_offset + k *
 * out_stride; kSlotSum sums the slots, weighted by the route
 * buffer, into one output row. `route` must be bound even when
 * unused (pass any buffer).
 */
void rec_matvec_b(State& s, const Mat& w, VulkanContext::Buffer wb,
                  VulkanContext::Buffer x, VulkanContext::Buffer out,
                  std::uint32_t out_offset, bool accumulate,
                  std::uint64_t w_byte_off, std::uint32_t x_off,
                  std::uint64_t w_stride_bytes, std::uint32_t x_stride,
                  std::uint32_t out_stride, std::uint32_t slots,
                  std::uint32_t mode, VulkanContext::Buffer route) {
    const Kernel k = batched_kernel(w);
    const VulkanContext::Buffer bufs[] = {wb, x, out, route};
    const std::uint32_t push[] = {w.rows,
                                  w.cols,
                                  out_offset,
                                  accumulate ? 1u : 0u,
                                  w_off_units(w, w_byte_off),
                                  x_off,
                                  fbits(1.0f),
                                  w_off_units(w, w_stride_bytes),
                                  x_stride,
                                  out_stride,
                                  slots,
                                  mode};
    s.ctx.dispatch(k, bufs, push, VulkanContext::matvec_groups(k, w.rows),
                   (mode & kSlotSum) != 0 ? 1u : slots);
}

/** matmul.comp tile sizes: weight rows and tokens per workgroup
 * (the MOE_ID variant's token tile is narrower). */
constexpr std::uint32_t kMatmulRows = 64;
constexpr std::uint32_t kMatmulToks = 64;
constexpr std::uint32_t kMatmulToksId = 16;

/** @returns The prefill matmul kernel for w, or kCount_. */
Kernel matmul_kernel(const Mat& w) {
    switch (w.type) {
        case gguf::TensorType::kQ8_0: return Kernel::kMatmulQ8_0;
        case gguf::TensorType::kF32: return Kernel::kMatmulF32;
        default: return Kernel::kCount_;
    }
}

/** Token-row layout of one prefill matmul operand set. */
struct MatmulRows {
    std::uint32_t n_tok;
    std::uint32_t x_off = 0, x_stride = 0;
    std::uint32_t out_off = 0, out_stride = 0;
    /** Batches (MLA heads): each shifts W / X / out by these. */
    std::uint32_t n_batch = 1;
    std::uint64_t w_bstride_bytes = 0;
    std::uint32_t x_bstride = 0, out_bstride = 0;
};

/**
 * Records out[t] (+)= scale * W x[t] for every token row t (and
 * batch b) in one matmul dispatch, reading each weight tile once per
 * kMatmulToks tokens. Weight types without a matmul kernel fall back
 * to one matvec per (batch, token): correct, just not amortized.
 */
void rec_matmul(State& s, const Mat& w, VulkanContext::Buffer wb,
                VulkanContext::Buffer x, VulkanContext::Buffer out,
                const MatmulRows& r, bool accumulate,
                std::uint64_t w_byte_off = 0, float scale = 1.0f) {
    const Kernel k = matmul_kernel(w);
    if (k == Kernel::kCount_) {
        for (std::uint32_t b = 0; b < r.n_batch; ++b) {
            for (std::uint32_t t = 0; t < r.n_tok; ++t) {
                rec_matvec(s, w, x, out,
                           r.out_off + b * r.out_bstride +
                               t * r.out_stride,
                           accumulate, w_byte_off + b * r.w_bstride_bytes,
                           r.x_off + b * r.x_bstride + t * r.x_stride,
                           scale, &wb);
            }
        }
        return;
    }
    const std::uint32_t tiles = (r.n_tok + kMatmulToks - 1) / kMatmulToks;
    const VulkanContext::Buffer bufs[] = {wb, x, out};
    const std::uint32_t push[] = {w.rows,
                                  w.cols,
                                  r.n_tok,
                                  accumulate ? 1u : 0u,
                                  w_off_units(w, w_byte_off),
                                  r.x_off,
                                  r.x_stride,
                                  r.out_off,
                                  r.out_stride,
                                  fbits(scale),
                                  w_off_units(w, r.w_bstride_bytes),
                                  r.x_bstride,
                                  r.out_bstride,
                                  tiles,
                                  1,
                                  0};
    s.ctx.dispatch(k, bufs, push, (w.rows + kMatmulRows - 1) / kMatmulRows,
                   tiles * r.n_batch);
}

/**
 * Records the routed-expert matmul for a prefill chunk: every
 * expert's weights (stacked in wb, expert_bytes apart) applied once
 * to the (token, slot) pairs moe_group listed for it in ids. Pair p
 * reads X row p / x_div and writes output row p. Q8_0 only.
 */
void rec_matmul_id(State& s, const Mat& w, VulkanContext::Buffer wb,
                   VulkanContext::Buffer x, VulkanContext::Buffer out,
                   VulkanContext::Buffer ids, std::uint32_t n_tok,
                   std::uint32_t n_expert, std::uint64_t expert_bytes,
                   std::uint32_t x_stride, std::uint32_t x_div,
                   std::uint32_t out_stride) {
    const std::uint32_t tiles =
        (n_tok + kMatmulToksId - 1) / kMatmulToksId;
    const VulkanContext::Buffer bufs[] = {wb, x, out, ids};
    const std::uint32_t push[] = {w.rows,
                                  w.cols,
                                  n_tok,
                                  0,
                                  0,
                                  0,
                                  x_stride,
                                  0,
                                  out_stride,
                                  fbits(1.0f),
                                  static_cast<std::uint32_t>(expert_bytes),
                                  0,
                                  0,
                                  tiles,
                                  x_div,
                                  n_tok};
    s.ctx.dispatch(Kernel::kMatmulQ8_0Id, bufs, push,
                   (w.rows + kMatmulRows - 1) / kMatmulRows,
                   tiles * n_expert);
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

void vulkan_reset_state() {
    State& s = state();
    // Weight pool: every model's weights are uploaded once and kept
    // resident (unbounded by default) -- the unbounded accumulator
    // across a multi-model test process. Free them all.
    for (auto& [ptr, e] : s.pool.map_) {
        s.pool.destroy(e.buf);
    }
    s.pool.map_.clear();
    for (auto& b : s.pool.transient_) {
        s.pool.destroy(b);
    }
    s.pool.transient_.clear();
    s.pool.used_ = 0;
    // GPU-mapped KV pools: one per cache, never freed otherwise.
    for (auto& [map, buf] : s.kv_pools) {
        s.ctx.destroy_buffer(buf);
    }
    s.kv_pools.clear();
    s.moe_gpu_fits.clear();
    // Activation scratch: bounded, but release it too so a large
    // model's buffers are not held across the rest of the run; each
    // regrows on next use.
    auto drop = [&](VulkanContext::Buffer& b, std::size_t& cap) {
        if (cap != 0) {
            s.ctx.destroy_buffer(b);
            b = VulkanContext::Buffer{};
            cap = 0;
        }
    };
    drop(s.x, s.x_n);
    drop(s.xb, s.xb_n);
    drop(s.q, s.q_n);
    drop(s.gate, s.gate_n);
    drop(s.up, s.up_n);
    drop(s.attout, s.attout_n);
    drop(s.att, s.att_n);
    drop(s.logits, s.logits_n);
    drop(s.table, s.table_n);
    drop(s.ones, s.ones_n);
    drop(s.kv_a, s.kv_a_n);
    drop(s.q_abs, s.q_abs_n);
    drop(s.latent, s.latent_n);
    drop(s.q_a, s.q_a_n);
    drop(s.router, s.router_n);
    drop(s.route, s.route_n);
    for (int c = 0; c < State::kChunkCount_; ++c) {
        drop(s.chunk[c], s.chunk_n[c]);
    }
}

void matvec_vulkan(const Mat& w, std::span<const float> x,
                   std::span<float> out) {
    if (matvec_kernel(w) == Kernel::kCount_) {
        matvec(w, x, out);  // no shader for this weight type
        return;
    }
    State& s = state();
    const Kernel k = matvec_kernel(w);
    auto wb = s.upload(w.data, weight_bytes(w));
    s.grow(s.x, s.x_n, x.size());
    s.grow(s.xb, s.xb_n, out.size());
    s.ctx.write_buffer(s.x, std::as_bytes(x));
    const VulkanContext::Buffer grid = s.grid_for(k);
    s.ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {wb, s.x, s.xb, grid};
    const std::size_t nbuf = grid.impl != nullptr ? 4u : 3u;
    const std::uint32_t push[] = {w.rows, w.cols, 0, 0, 0, 0,
                                  fbits(1.0f)};
    s.ctx.dispatch(k, {bufs, nbuf}, push,
                   VulkanContext::matvec_groups(k, w.rows));
    s.ctx.end_batch();
    s.pool.on_batch_end();
    s.ctx.read_buffer(s.xb, std::as_writable_bytes(out));
}

namespace {

/**
 * Whether a MoE layer can route on the GPU (moe_route + batched
 * expert matvecs, no host sync). Needs every expert resident, so
 * not under a LOCUS_GPU_POOL_MB budget, and only when all of the
 * model's expert tensors fit in 80% of the device's available
 * memory (decided once per model, before any is uploaded); not
 * while LOCUS_MOE_STATS wants host-side picks; byte offsets must
 * fit the shaders' uint. LOCUS_VK_CPU_ROUTE=1 forces the
 * host-routed path.
 */
bool gpu_route_ok(State& s, const model::TransformerModel& m,
                  const model::TransformerModel::Layer& lay) {
    const auto& hp = m.hparams();
    static const bool forced_cpu =
        std::getenv("LOCUS_VK_CPU_ROUTE") != nullptr;
    if (forced_cpu || model::MoeStats::enabled()) {
        return false;
    }
    if (s.pool.budget_ == 0) {
        s.pool.init_budget();
    }
    if (s.pool.budget_ != std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    if (hp.n_expert_used > 16 || hp.n_expert_used == 0 ||
        hp.n_group > 32 ||
        (hp.n_group > 1 && hp.n_expert % hp.n_group != 0)) {
        return false;
    }
    for (const auto* em : {&lay.gate_exps, &lay.up_exps, &lay.down_exps}) {
        if (batched_kernel(em->base) == Kernel::kCount_ ||
            em->expert_bytes * hp.n_expert >
                std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
    }
    auto fit = s.moe_gpu_fits.find(&m);
    if (fit == s.moe_gpu_fits.end()) {
        std::uint64_t experts = 0;
        for (const auto& l : m.layers()) {
            if (l.is_moe()) {
                experts += (l.gate_exps.expert_bytes +
                            l.up_exps.expert_bytes +
                            l.down_exps.expert_bytes) *
                           hp.n_expert;
            }
        }
        const std::uint64_t avail = s.ctx.memory_available();
        fit = s.moe_gpu_fits.emplace(&m, vk::moe_experts_fit(experts, avail))
                  .first;
        if (!fit->second) {
            std::fprintf(stderr,
                         "[vulkan] MoE experts (%llu MB) exceed 80%% of "
                         "available device memory (%llu MB): routing "
                         "on the host\n",
                         static_cast<unsigned long long>(experts >> 20),
                         static_cast<unsigned long long>(avail >> 20));
        }
    }
    return fit->second;
}

/** @returns true when every weight the GPU forward reads has a
 * matvec kernel (fused wkv_b goes through the F32-dequant path). */
bool gpu_weights_ok(const model::TransformerModel& m) {
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
    }
    return matvec_kernel(m.output_weight()) != Kernel::kCount_;
}

}  // namespace

bool vulkan_forward(const model::TransformerModel& m, tok::TokenId token,
                    kv::PagedKvCache& cache,
                    kv::PagedKvCache::Seq& seq,
                    std::span<float> logits) {
    const auto& hp = m.hparams();
    State& s = state();
    const bool quant = cache.quantized();
    auto pool_it = s.kv_pools.find(
        quant ? static_cast<const void*>(cache.qpool_data())
              : static_cast<const void*>(cache.pool_data()));
    if (pool_it == s.kv_pools.end()) {
        return false;  // cache pool is not GPU-mapped
    }
    if (hp.idx_top_k > 0) {
        // DSA indexer models: the 704-float cache row and the
        // top-k selection are CPU-only for now.
        return false;
    }
    const bool mla = hp.arch == model::Arch::kDeepseek2;
    // The attention kernels stage per-head slices in a fixed shared
    // array (vk::kAttnPartMax floats): wider heads, or a larger MLA
    // latent rank, take the CPU path instead of overrunning it.
    if (!vk::attn_width_fits(hp.head_dim) ||
        (mla && !vk::attn_width_fits(hp.kv_lora_rank))) {
        return false;
    }
    if (quant && mla) {
        // MLA caches a single latent row (no separate K/V); its
        // quantized read is a #45 follow-up. Fall back to CPU.
        return false;
    }
    if (!gpu_weights_ok(m)) {
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
    // Derive the block stride from the CACHE's layer count, not the
    // model's: under slice-only loading (DESIGN.md R15+) they differ,
    // and this stride must match the pool the offsets below index into.
    // This full-forward path assumes a base-0 full-height cache (the
    // loop below runs absolute layer l as the row index), which holds
    // because Vulkan only runs the whole model in one process; a sliced
    // cache would need l - geom.layer_base here and in the loop.
    assert(geom.layer_base == 0 &&
           "vulkan_forward assumes a base-0 full-height KV cache");
    const std::uint32_t block_stride =
        geom.n_layers * layer_stride;
    // Quantized byte strides (R14 #45); unused on the F32 path.
    const std::uint32_t q_row_b =
        quant ? static_cast<std::uint32_t>(cache.q_row_bytes()) : 0;
    const std::uint32_t q_layer_b =
        quant ? static_cast<std::uint32_t>(cache.q_layer_stride())
              : 0;
    const std::uint32_t q_block_b =
        quant ? static_cast<std::uint32_t>(cache.q_block_stride())
              : 0;
    const std::uint32_t kv_code =
        geom.kv_type == kv::KvType::kQ8 ? 1u : 2u;

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
    // The non-MLA quantized-KV path reuses gate/up as F32 scratch
    // for this token's K and V rows (kv_dim elements each), so they
    // must be at least kv_dim wide, not just ff_max. Every real
    // transformer has n_ff >= n_embd >= kv_dim so this is normally
    // just ff_max, but a model with an inverted ratio would
    // otherwise overrun the FFN scratch with a K/V write -- size
    // defensively.
    // GPU-routed MoE keeps all n_expert_used experts' gate/up rows
    // side by side.
    const std::size_t ff_kv = std::max<std::size_t>(
        {ff_max, kv_dim,
         std::size_t{hp.n_expert_used} * hp.n_ff_exp});
    s.grow(s.gate, s.gate_n, ff_kv);
    s.grow(s.up, s.up_n, ff_kv);
    s.grow(s.attout, s.attout_n,
           mla ? std::size_t{hp.n_heads} * vd : hp.n_embd);
    s.grow(s.att, s.att_n,
           static_cast<std::size_t>(hp.n_heads) * hp.n_ctx);
    s.grow(s.logits, s.logits_n, hp.n_vocab);
    s.grow(s.table, s.table_n, cache.total_blocks());
    if (hp.n_expert > 0) {
        s.grow(s.router, s.router_n, std::size_t{hp.n_expert} * 2);
        s.grow(s.route, s.route_n,
               std::max<std::size_t>(hp.n_expert_used * 2, 1));
    }
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
            {
                const VulkanContext::Buffer bufs[] = {s.q,
                                                      divisors};
                const std::uint32_t push[] = {
                    hp.n_heads, hp.head_dim, pos, 0, hp.head_dim,
                    base_bits,  fbits(1.0f), 0,   0};
                s.ctx.dispatch(Kernel::kRope, bufs, push,
                               hp.n_heads);
            }
            if (!quant) {
                rec_matvec(s, lay.wk, s.xb, pool, row_off, false);
                rec_matvec(s, lay.wv, s.xb, pool,
                           row_off + layer_stride / 2, false);
                {
                    const VulkanContext::Buffer bufs[] = {
                        pool, divisors};
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
                // R14 #45: K/V into F32 scratch (reuse the FFN
                // gate/up buffers -- free until this layer's FFN),
                // rope K, quantize-store into the byte pool, then
                // attention decodes the quantized rows in-shader.
                rec_matvec(s, lay.wk, s.xb, s.gate, 0, false);
                rec_matvec(s, lay.wv, s.xb, s.up, 0, false);
                {
                    const VulkanContext::Buffer bufs[] = {
                        s.gate, divisors};
                    const std::uint32_t push[] = {
                        hp.n_kv_heads, hp.head_dim, pos,
                        0,             hp.head_dim, base_bits,
                        fbits(1.0f),   0,           0};
                    s.ctx.dispatch(Kernel::kRope, bufs, push,
                                   hp.n_kv_heads);
                }
                const std::uint32_t dst_k = block * q_block_b +
                                            l * q_layer_b +
                                            in_block * q_row_b;
                const std::uint32_t dst_v =
                    dst_k + geom.block_tokens * q_row_b;
                {
                    const VulkanContext::Buffer bufs[] = {s.gate,
                                                          pool};
                    const std::uint32_t push[] = {kv_dim, dst_k,
                                                  kv_code, 0};
                    s.ctx.dispatch(Kernel::kKvStoreQ, bufs, push,
                                   1);
                }
                {
                    const VulkanContext::Buffer bufs[] = {s.up,
                                                          pool};
                    const std::uint32_t push[] = {kv_dim, dst_v,
                                                  kv_code, 0};
                    s.ctx.dispatch(Kernel::kKvStoreQ, bufs, push,
                                   1);
                }
                {
                    const VulkanContext::Buffer bufs[] = {
                        s.q, pool, s.table, s.att, s.attout};
                    const std::uint32_t push[] = {
                        hp.n_heads,   group,
                        hp.head_dim,  kv_dim,
                        pos + 1,      geom.block_tokens,
                        q_block_b,    l * q_layer_b,
                        geom.block_tokens * q_row_b,
                        q_row_b,      kv_code};
                    s.ctx.dispatch(Kernel::kAttnPagedQ, bufs, push,
                                   hp.n_heads);
                }
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
            // Per-head weight absorption, all heads per dispatch
            // where a batched kernel exists (head h = slot h).
            if (!split) {
                const VulkanContext::Buffer bufs[] = {wkv_b32, s.q,
                                                      s.q_abs};
                const std::uint32_t push[] = {
                    nope, rank, 0, 0, 0, (nope + vd) * rank, qk,
                    rank};
                s.ctx.dispatch(Kernel::kMatvecT, bufs, push,
                               (rank + 63) / 64, hp.n_heads);
            } else if (batched_kernel(lay.wk_b) != Kernel::kCount_) {
                // Pre-transposed k_b: one rank-row matvec per head.
                Mat kb = lay.wk_b;
                kb.rows = rank;
                rec_matvec_b(
                    s, kb, s.upload(lay.wk_b.data, weight_bytes(lay.wk_b)),
                    s.q, s.q_abs, 0, false, 0, 0,
                    std::uint64_t{rank} * mat_row_bytes(lay.wk_b), qk,
                    rank, hp.n_heads, 0, s.q);
            } else {
                for (std::uint32_t h = 0; h < hp.n_heads; ++h) {
                    Mat kb = lay.wk_b;
                    kb.rows = rank;
                    rec_matvec(
                        s, kb, s.q, s.q_abs, h * rank, false,
                        std::uint64_t{h} * rank *
                            mat_row_bytes(lay.wk_b),
                        h * qk, 1.0f);
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
            if (!split) {
                // W_uv rows of the dequantized fused wkv_b, head h
                // at element (h * (nope + vd) + nope) * rank.
                Mat uv{gguf::TensorType::kF32, nullptr, vd, rank};
                rec_matvec_b(s, uv, wkv_b32, s.latent, s.attout, 0,
                             false, std::uint64_t{nope} * rank * 4, 0,
                             std::uint64_t{nope + vd} * rank * 4, rank,
                             vd, hp.n_heads, 0, s.latent);
            } else if (batched_kernel(lay.wv_b) != Kernel::kCount_) {
                Mat vb = lay.wv_b;
                vb.rows = vd;
                rec_matvec_b(
                    s, vb, s.upload(lay.wv_b.data, weight_bytes(lay.wv_b)),
                    s.latent, s.attout, 0, false, 0, 0,
                    std::uint64_t{vd} * mat_row_bytes(lay.wv_b), rank,
                    vd, hp.n_heads, 0, s.latent);
            } else {
                for (std::uint32_t h = 0; h < hp.n_heads; ++h) {
                    Mat vb = lay.wv_b;
                    vb.rows = vd;
                    rec_matvec(
                        s, vb, s.latent, s.attout, h * vd,
                        false,
                        std::uint64_t{h} * vd *
                            mat_row_bytes(lay.wv_b),
                        h * rank, 1.0f);
                }
            }
        }
        rec_matvec(s, lay.wo, s.attout, s.x, 0, true);

        norm(lay.ffn_norm, s.x, s.xb, hp.n_embd, 0);
        if (!lay.is_moe()) {
            swiglu(lay.w_gate, lay.w_up, lay.w_down, hp.n_ff,
                   1.0f, 0, 0, 0, nullptr, nullptr, nullptr);
        } else if (gpu_route_ok(s, m, lay)) {
            // Router, top-k and the routed experts all on the GPU
            // in the open batch: moe_route writes the picked ids
            // and weights, the batched matvecs read them.
            const std::uint32_t k_used = hp.n_expert_used;
            const std::uint32_t ff = hp.n_ff_exp;
            rec_matvec(s, lay.gate_inp, s.xb, s.router, 0, false);
            {
                const bool has_bias = !lay.exp_probs_b.empty();
                const VulkanContext::Buffer bufs[] = {
                    s.router,
                    has_bias ? s.upload(lay.exp_probs_b.data(),
                                        lay.exp_probs_b.size() * 4)
                             : s.router,
                    s.route};
                const std::uint32_t push[] = {
                    hp.n_expert,
                    k_used,
                    hp.gating == model::GatingFunc::kSigmoid ? 1u : 0u,
                    has_bias ? 1u : 0u,
                    hp.n_group,
                    hp.n_group_used,
                    hp.expert_weights_norm ? 1u : 0u,
                    fbits(hp.expert_weights_scale)};
                s.ctx.dispatch(Kernel::kMoeRoute, bufs, push, 1);
            }
            Mat g = lay.gate_exps.expert(0), u = lay.up_exps.expert(0),
                d = lay.down_exps.expert(0);
            g.rows = ff;
            u.rows = ff;
            d.cols = ff;
            rec_matvec_b(s, g, s.upload_experts(lay.gate_exps, hp.n_expert),
                         s.xb, s.gate, 0, false, 0, 0,
                         lay.gate_exps.expert_bytes, 0, ff, k_used,
                         kRouteIds, s.route);
            rec_matvec_b(s, u, s.upload_experts(lay.up_exps, hp.n_expert),
                         s.xb, s.up, 0, false, 0, 0,
                         lay.up_exps.expert_bytes, 0, ff, k_used,
                         kRouteIds, s.route);
            {
                const VulkanContext::Buffer bufs[] = {s.gate, s.up,
                                                      s.gate};
                const std::uint32_t push[] = {k_used * ff};
                s.ctx.dispatch(Kernel::kSiluMul, bufs, push,
                               (k_used * ff + 63) / 64);
            }
            rec_matvec_b(s, d, s.upload_experts(lay.down_exps, hp.n_expert),
                         s.gate, s.x, 0, true, 0, 0,
                         lay.down_exps.expert_bytes, ff, 0, k_used,
                         kRouteIds | kSlotSum, s.route);
            if (hp.n_expert_shared > 0) {
                swiglu(lay.gate_shexp, lay.up_shexp, lay.down_shexp,
                       hp.n_ff_exp * hp.n_expert_shared, 1.0f, 0, 0, 0,
                       nullptr, nullptr, nullptr);
            }
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
                [&](const model::TransformerModel::ExpertMat& em,
                    std::uint32_t e) {
                    return s.upload(em.expert(e).data,
                                    em.expert_bytes);
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
    s.ctx.note_token();
    seq.n_tokens = pos + 1;
    return true;
}

bool vulkan_forward_batch(const model::TransformerModel& m,
                          std::span<const tok::TokenId> tokens,
                          kv::PagedKvCache& cache,
                          kv::PagedKvCache::Seq& seq,
                          std::span<float> logits, bool all_logits) {
    using C = State::Chunk;
    const auto& hp = m.hparams();
    State& s = state();
    const auto n = static_cast<std::uint32_t>(tokens.size());
    if (n == 0 || cache.quantized() || hp.idx_top_k > 0 ||
        !gpu_weights_ok(m)) {
        return false;
    }
    auto pool_it =
        s.kv_pools.find(static_cast<const void*>(cache.pool_data()));
    if (pool_it == s.kv_pools.end()) {
        return false;  // cache pool is not GPU-mapped
    }
    const bool mla = hp.arch == model::Arch::kDeepseek2;
    const auto& geom = cache.geometry();
    const std::uint32_t kv_dim = geom.kv_dim;
    const std::uint32_t rank = hp.kv_lora_rank;
    const std::uint32_t nope = hp.qk_nope_dim;
    const std::uint32_t rope_d = hp.qk_rope_dim;
    const std::uint32_t vd = hp.v_head_dim;
    const std::uint32_t qk = nope + rope_d;
    const std::uint32_t kvw = rank + rope_d;  // MLA latent row
    if (mla && kv_dim != kvw) {
        return false;
    }
    // Routed experts need GPU routing (all experts resident) and the
    // Q8_0 id-matmul; moe_group handles up to 256 experts.
    for (const auto& lay : m.layers()) {
        if (lay.is_moe() &&
            (!gpu_route_ok(s, m, lay) || hp.n_expert > 256 ||
             lay.gate_exps.base.type != gguf::TensorType::kQ8_0 ||
             lay.up_exps.base.type != gguf::TensorType::kQ8_0 ||
             lay.down_exps.base.type != gguf::TensorType::kQ8_0)) {
            return false;
        }
    }
    const std::uint32_t base = seq.n_tokens;
    if (base + n > hp.n_ctx || base + n > cache.capacity(seq)) {
        throw std::invalid_argument("seq capacity not ensured");
    }
    assert(geom.layer_base == 0 &&
           "vulkan_forward_batch assumes a base-0 full-height KV cache");

    const std::uint32_t E = hp.n_embd;
    const std::uint32_t H = hp.n_heads;
    const std::uint32_t layer_stride = geom.block_tokens * kv_dim * 2;
    const std::uint32_t block_stride = geom.n_layers * layer_stride;
    const std::uint32_t q_dim = mla ? H * qk : H * hp.head_dim;
    const std::uint32_t att_dim = mla ? H * vd : H * hp.head_dim;
    const std::uint32_t n_pos = base + n;  // last token's span
    const std::uint32_t k_used = hp.n_expert_used;
    const std::uint32_t ff_dense =
        std::max(hp.n_ff, hp.n_ff_exp * hp.n_expert_shared);
    const std::size_t ff_rows = std::max<std::size_t>(
        std::size_t{n} * ff_dense,
        std::size_t{n} * k_used * hp.n_ff_exp);
    const std::size_t n_logits =
        all_logits ? std::size_t{n} * hp.n_vocab : hp.n_vocab;
    if (logits.size() != n_logits) {
        throw std::invalid_argument(
            "vulkan_forward_batch: logits span size");
    }

    const auto bx = s.chunk_buf(C::kCX, std::size_t{n} * E);
    const auto bxb = s.chunk_buf(C::kCXb, std::size_t{n} * E);
    const auto bq = s.chunk_buf(C::kCQ, std::size_t{n} * q_dim);
    const auto bgate = s.chunk_buf(C::kCGate, ff_rows);
    const auto bup = s.chunk_buf(C::kCUp, ff_rows);
    const auto batt_out = s.chunk_buf(C::kCAttOut, std::size_t{n} * att_dim);
    const auto batt =
        s.chunk_buf(C::kCAtt, std::size_t{n} * H * n_pos);
    const auto blogits = s.chunk_buf(C::kCLogits, n_logits);
    s.grow(s.table, s.table_n, cache.total_blocks());

    // CPU side: the chunk's embedding rows and the block table.
    std::vector<float> x_host(std::size_t{n} * E);
    for (std::uint32_t t = 0; t < n; ++t) {
        dequant_row(m.embedding(), static_cast<std::uint32_t>(tokens[t]),
                    {x_host.data() + std::size_t{t} * E, E});
    }
    s.ctx.write_buffer(bx,
                       std::as_bytes(std::span<const float>(x_host)));
    s.ctx.write_buffer(
        s.table,
        std::as_bytes(std::span<const kv::BlockId>(seq.blocks)));

    const auto pool = pool_it->second;
    const std::uint32_t eps_bits = fbits(hp.rms_eps);
    const std::uint32_t base_bits = fbits(hp.rope_freq_base);
    float corr_lo = 0.0f, corr_hi = 0.0f;
    if (mla) {
        yarn_corr_range(rope_d, hp.rope_freq_base, hp.yarn, corr_lo,
                        corr_hi);
    }
    const std::uint32_t fs_bits = fbits(hp.yarn.freq_scale);
    const std::uint32_t lo_bits = fbits(corr_lo);
    const std::uint32_t hi_bits = fbits(corr_hi);
    const std::uint32_t group =
        hp.n_kv_heads > 0 ? hp.n_heads / hp.n_kv_heads : 1;

    // Row-wise helpers over the chunk's n token rows.
    auto norm_rows = [&](std::span<const float> w, VulkanContext::Buffer in,
                         VulkanContext::Buffer out, std::uint32_t width,
                         std::uint32_t in_off, std::uint32_t in_stride,
                         std::uint32_t out_stride, std::uint32_t rows) {
        const VulkanContext::Buffer bufs[] = {
            in, s.upload(w.data(), w.size() * 4), out};
        const std::uint32_t push[] = {width, eps_bits, 0,
                                      in_off, in_stride, out_stride};
        s.ctx.dispatch(Kernel::kRmsNormB, bufs, push, 1, rows);
    };
    auto rope_rows = [&](VulkanContext::Buffer buf,
                         VulkanContext::Buffer divisors,
                         std::uint32_t heads, std::uint32_t dim,
                         std::uint32_t offset, std::uint32_t stride,
                         std::uint32_t row_stride, bool yarn) {
        const VulkanContext::Buffer bufs[] = {buf, divisors};
        const std::uint32_t push[] = {
            heads,  dim,
            base,   offset,
            stride, base_bits,
            yarn ? fs_bits : fbits(1.0f),
            yarn ? lo_bits : 0u,
            yarn ? hi_bits : 0u,
            row_stride};
        s.ctx.dispatch(Kernel::kRopeB, bufs, push, heads, n);
    };
    auto scatter_rows = [&](VulkanContext::Buffer src, std::uint32_t width,
                            std::uint32_t dst_off) {
        const VulkanContext::Buffer bufs[] = {src, pool, s.table};
        const std::uint32_t push[] = {width,        base,
                                      geom.block_tokens, block_stride,
                                      kv_dim,       dst_off,
                                      0,            width};
        s.ctx.dispatch(Kernel::kKvScatter, bufs, push, 1, n);
    };
    auto silu_rows = [&](std::size_t count) {
        const VulkanContext::Buffer bufs[] = {bgate, bup, bgate};
        const std::uint32_t push[] = {static_cast<std::uint32_t>(count)};
        s.ctx.dispatch(Kernel::kSiluMul, bufs, push,
                       static_cast<std::uint32_t>(std::min<std::size_t>(
                           (count + 63) / 64, 65535)));
    };
    // Dense SwiGLU over the normed rows in bxb, accumulated into bx.
    auto swiglu_rows = [&](const Mat& wg, const Mat& wu, const Mat& wd,
                           std::uint32_t ff) {
        Mat g2 = wg, u2 = wu, d2 = wd;
        g2.rows = ff;
        u2.rows = ff;
        d2.cols = ff;
        const MatmulRows in{n, 0, E, 0, ff};
        rec_matmul(s, g2, s.upload(wg.data, weight_bytes(g2)), bxb, bgate,
                   in, false);
        rec_matmul(s, u2, s.upload(wu.data, weight_bytes(u2)), bxb, bup,
                   in, false);
        silu_rows(std::size_t{n} * ff);
        rec_matmul(s, d2, s.upload(wd.data, weight_bytes(wd)), bgate, bx,
                   MatmulRows{n, 0, ff, 0, E}, true);
    };

    s.ctx.begin_batch();
    for (std::uint32_t l = 0; l < hp.n_layers; ++l) {
        const auto& lay = m.layers()[l];
        const std::uint32_t layer_off = l * layer_stride;

        norm_rows(lay.attn_norm, bx, bxb, E, 0, E, E, n);
        if (!mla) {
            const std::uint32_t hd = hp.head_dim;
            const auto divisors =
                s.rope_divisors(m.rope_factors(), hd / 2);
            const auto bk = s.chunk_buf(C::kCK, std::size_t{n} * kv_dim);
            const auto bv = s.chunk_buf(C::kCV, std::size_t{n} * kv_dim);
            rec_matmul(s, lay.wq,
                       s.upload(lay.wq.data, weight_bytes(lay.wq)), bxb,
                       bq, MatmulRows{n, 0, E, 0, q_dim}, false);
            rec_matmul(s, lay.wk, s.upload(lay.wk.data, weight_bytes(lay.wk)),
                       bxb, bk, MatmulRows{n, 0, E, 0, kv_dim}, false);
            rec_matmul(s, lay.wv, s.upload(lay.wv.data, weight_bytes(lay.wv)),
                       bxb, bv, MatmulRows{n, 0, E, 0, kv_dim}, false);
            rope_rows(bq, divisors, H, hd, 0, hd, q_dim, false);
            rope_rows(bk, divisors, hp.n_kv_heads, hd, 0, hd, kv_dim,
                      false);
            scatter_rows(bk, kv_dim, layer_off);
            scatter_rows(bv, kv_dim, layer_off + layer_stride / 2);
            const VulkanContext::Buffer bufs[] = {bq, pool, s.table, batt,
                                                  batt_out};
            const std::uint32_t push[] = {
                H,           group,           hd,
                kv_dim,      base + 1,        geom.block_tokens,
                block_stride, layer_off,      layer_stride / 2,
                q_dim,       n_pos};
            s.ctx.dispatch(Kernel::kAttnPagedB, bufs, push, H, n);
        } else {
            const bool split = lay.wk_b.rows > 0;
            const auto divisors = s.rope_divisors({}, rope_d / 2);
            const auto bkva = s.chunk_buf(C::kCKva, std::size_t{n} * kvw);
            const auto bqabs =
                s.chunk_buf(C::kCQAbs, std::size_t{n} * H * rank);
            const auto blat =
                s.chunk_buf(C::kCLat, std::size_t{n} * H * rank);
            if (hp.q_lora_rank > 0) {
                const std::uint32_t qr = hp.q_lora_rank;
                const auto bqa = s.chunk_buf(C::kCQa, std::size_t{n} * qr);
                rec_matmul(s, lay.wq_a,
                           s.upload(lay.wq_a.data, weight_bytes(lay.wq_a)),
                           bxb, bqa, MatmulRows{n, 0, E, 0, qr}, false);
                norm_rows(lay.q_a_norm, bqa, bqa, qr, 0, qr, qr, n);
                rec_matmul(s, lay.wq_b,
                           s.upload(lay.wq_b.data, weight_bytes(lay.wq_b)),
                           bqa, bq, MatmulRows{n, 0, qr, 0, q_dim}, false);
            } else {
                rec_matmul(s, lay.wq,
                           s.upload(lay.wq.data, weight_bytes(lay.wq)),
                           bxb, bq, MatmulRows{n, 0, E, 0, q_dim}, false);
            }
            // Latent rows: rms-norm the c_kv part and rope the k_pe
            // part in place, then scatter the rows into the pool.
            rec_matmul(s, lay.wkv_a,
                       s.upload(lay.wkv_a.data, weight_bytes(lay.wkv_a)),
                       bxb, bkva, MatmulRows{n, 0, E, 0, kvw}, false);
            norm_rows(lay.kv_a_norm, bkva, bkva, rank, 0, kvw, kvw, n);
            rope_rows(bkva, divisors, 1, rope_d, rank, 0, kvw, true);
            scatter_rows(bkva, kvw, layer_off);
            rope_rows(bq, divisors, H, rope_d, nope, qk, q_dim, true);

            // Per-head absorption q_abs = W_uk^T q_nope, all heads
            // and tokens in one matmul (head = batch).
            MatmulRows absorb{n, 0, H * qk, 0, H * rank};
            absorb.n_batch = H;
            absorb.x_bstride = qk;
            absorb.out_bstride = rank;
            if (!split) {
                const Mat kbt{gguf::TensorType::kF32, nullptr, rank, nope};
                absorb.w_bstride_bytes = std::uint64_t{rank} * nope * 4;
                rec_matmul(s, kbt,
                           s.upload_kb_t(lay.wkv_b, H, nope, vd, rank),
                           bq, bqabs, absorb, false);
            } else {
                Mat kb = lay.wk_b;
                kb.rows = rank;
                absorb.w_bstride_bytes =
                    std::uint64_t{rank} * mat_row_bytes(lay.wk_b);
                rec_matmul(s, kb,
                           s.upload(lay.wk_b.data, weight_bytes(lay.wk_b)),
                           bq, bqabs, absorb, false);
            }
            {
                const VulkanContext::Buffer bufs[] = {bq,   bqabs,   pool,
                                                      s.table, batt, blat};
                const std::uint32_t push[] = {
                    H,            rank,
                    nope,         rope_d,
                    base + 1,     geom.block_tokens,
                    block_stride, layer_off,
                    fbits(hp.kq_scale), q_dim,
                    H * rank,     n_pos};
                s.ctx.dispatch(Kernel::kAttnMlaB, bufs, push, H, n);
            }
            // Per-head W_uv projection of the latents.
            MatmulRows uv_rows{n, 0, H * rank, 0, att_dim};
            uv_rows.n_batch = H;
            uv_rows.x_bstride = rank;
            uv_rows.out_bstride = vd;
            if (!split) {
                const Mat uv{gguf::TensorType::kF32, nullptr, vd, rank};
                uv_rows.w_bstride_bytes = std::uint64_t{nope + vd} * rank * 4;
                rec_matmul(s, uv, s.upload_f32(lay.wkv_b), blat, batt_out,
                           uv_rows, false, std::uint64_t{nope} * rank * 4);
            } else {
                Mat vb = lay.wv_b;
                vb.rows = vd;
                uv_rows.w_bstride_bytes =
                    std::uint64_t{vd} * mat_row_bytes(lay.wv_b);
                rec_matmul(s, vb,
                           s.upload(lay.wv_b.data, weight_bytes(lay.wv_b)),
                           blat, batt_out, uv_rows, false);
            }
        }
        rec_matmul(s, lay.wo, s.upload(lay.wo.data, weight_bytes(lay.wo)),
                   batt_out, bx, MatmulRows{n, 0, att_dim, 0, E}, true);

        norm_rows(lay.ffn_norm, bx, bxb, E, 0, E, E, n);
        if (!lay.is_moe()) {
            swiglu_rows(lay.w_gate, lay.w_up, lay.w_down, hp.n_ff);
            continue;
        }
        // MoE: route every token on the GPU, group the (token, slot)
        // pairs by expert, run each expert once over its pairs, then
        // add the weighted slots back into the residual.
        const std::uint32_t ne = hp.n_expert;
        const std::uint32_t ff = hp.n_ff_exp;
        const auto brouter =
            s.chunk_buf(C::kCRouter, std::size_t{n} * 2 * ne);
        const auto broute =
            s.chunk_buf(C::kCRoute, std::size_t{n} * 2 * k_used);
        const auto bids =
            s.chunk_buf(C::kCIds, ne + std::size_t{ne} * n);
        const auto bdown =
            s.chunk_buf(C::kCDown, std::size_t{n} * k_used * E);
        rec_matmul(s, lay.gate_inp,
                   s.upload(lay.gate_inp.data, weight_bytes(lay.gate_inp)),
                   bxb, brouter, MatmulRows{n, 0, E, 0, 2 * ne}, false);
        {
            const bool has_bias = !lay.exp_probs_b.empty();
            const VulkanContext::Buffer bufs[] = {
                brouter,
                has_bias ? s.upload(lay.exp_probs_b.data(),
                                    lay.exp_probs_b.size() * 4)
                         : brouter,
                broute};
            const std::uint32_t push[] = {
                ne,
                k_used,
                hp.gating == model::GatingFunc::kSigmoid ? 1u : 0u,
                has_bias ? 1u : 0u,
                hp.n_group,
                hp.n_group_used,
                hp.expert_weights_norm ? 1u : 0u,
                fbits(hp.expert_weights_scale),
                2 * ne,
                2 * k_used};
            s.ctx.dispatch(Kernel::kMoeRouteB, bufs, push, 1, n);
        }
        {
            const VulkanContext::Buffer bufs[] = {broute, bids};
            const std::uint32_t push[] = {n, k_used, ne, n};
            s.ctx.dispatch(Kernel::kMoeGroup, bufs, push, 1);
        }
        Mat g = lay.gate_exps.expert(0), u = lay.up_exps.expert(0),
            d = lay.down_exps.expert(0);
        g.rows = ff;
        u.rows = ff;
        d.cols = ff;
        rec_matmul_id(s, g, s.upload_experts(lay.gate_exps, ne), bxb, bgate,
                      bids, n, ne, lay.gate_exps.expert_bytes, E, k_used,
                      ff);
        rec_matmul_id(s, u, s.upload_experts(lay.up_exps, ne), bxb, bup,
                      bids, n, ne, lay.up_exps.expert_bytes, E, k_used, ff);
        silu_rows(std::size_t{n} * k_used * ff);
        rec_matmul_id(s, d, s.upload_experts(lay.down_exps, ne), bgate,
                      bdown, bids, n, ne, lay.down_exps.expert_bytes, ff, 1,
                      E);
        {
            const VulkanContext::Buffer bufs[] = {bdown, broute, bx};
            const std::uint32_t push[] = {E, k_used};
            s.ctx.dispatch(Kernel::kMoeCombine, bufs, push, (E + 255) / 256,
                           n);
        }
        if (hp.n_expert_shared > 0) {
            swiglu_rows(lay.gate_shexp, lay.up_shexp, lay.down_shexp,
                        hp.n_ff_exp * hp.n_expert_shared);
        }
    }

    // Prefill needs only the last row's logits; speculative verify
    // needs every row's.
    const Mat& out_w = m.output_weight();
    if (all_logits) {
        norm_rows(m.output_norm(), bx, bxb, E, 0, E, E, n);
        rec_matmul(s, out_w, s.upload(out_w.data, weight_bytes(out_w)), bxb,
                   blogits, MatmulRows{n, 0, E, 0, hp.n_vocab}, false);
    } else {
        norm_rows(m.output_norm(), bx, bxb, E, (n - 1) * E, 0, 0, 1);
        rec_matvec(s, out_w, bxb, blogits, 0, false);
    }
    s.ctx.end_batch();
    s.pool.on_batch_end();

    s.ctx.read_buffer(blogits, std::as_writable_bytes(logits));
    for (std::uint32_t t = 0; t < n; ++t) {
        s.ctx.note_token();
    }
    seq.n_tokens = base + n;
    return true;
}

}  // namespace locus::backend
