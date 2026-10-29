#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "iq_grids.h"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/backend/vulkan/limits.hpp"
#include "locus/backend/vulkan/mem_fit.hpp"
#include "locus/backend/vulkan/q8_layout.hpp"
#include "locus/backend/vulkan/weight_pool.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/model/moe_stats.hpp"
#include "locus/sys/mmap_file.hpp"

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
        case gguf::TensorType::kBF16:
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
        case gguf::TensorType::kBF16: return Kernel::kMatvecBF16;
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
        case gguf::TensorType::kBF16: return Kernel::kMatvecBF16B;
        case gguf::TensorType::kQ8_0: return Kernel::kMatvecQ8_0B;
        case gguf::TensorType::kQ4_K: return Kernel::kMatvecQ4_KB;
        case gguf::TensorType::kQ6_K: return Kernel::kMatvecQ6_KB;
        case gguf::TensorType::kQ5_0: return Kernel::kMatvecQ5_0B;
        default: return Kernel::kCount_;
    }
}

/** Converts a byte offset into shader w_off units for w's type. */
std::uint32_t w_off_units(const Mat& w, std::uint64_t bytes) {
    switch (w.type) {
        case gguf::TensorType::kF32:
            return static_cast<std::uint32_t>(bytes / 4);
        case gguf::TensorType::kF16:
        case gguf::TensorType::kBF16:
            return static_cast<std::uint32_t>(bytes / 2);
        default:
            return static_cast<std::uint32_t>(bytes);
    }
}

using WeightPool = WeightPoolT<VulkanContext::Buffer>;

void on_unmap(const void* p, std::size_t n);

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
    /** --prefill-f16: matmuls take their f16-staging variants. */
    bool prefill_f16 = false;
    /** Per-model GPU-routing verdict (memory fit), decided once. */
    std::unordered_map<const void*, bool> moe_gpu_fits;

    State() {
        pool.create = [this](std::size_t n) {
            return ctx.create_buffer(n);
        };
        pool.destroy = [this](VulkanContext::Buffer b) {
            ctx.destroy_buffer(b);
        };
        // A model file unmapped drops the buffers over its pages.
        sys::set_unmap_hook(&on_unmap);
    }
    ~State() {
        sys::set_unmap_hook(nullptr);
        pool.report();
        ctx.report_profile();
        // Free every GPU buffer while ctx (and its VkDevice) is still
        // alive: ctx is the first member, so it is destroyed last. The
        // tools (locus-run / locus-server) keep this singleton for the
        // whole process, so without this the device was destroyed with
        // buffers still live (VUID-vkDestroyDevice-device-00378, i#124).
        free_buffers();
    }

    /**
     * Destroys every GPU buffer this owns (weight pool, KV pools,
     * activation and chunk scratch, buffers retired mid-batch, and the
     * IQ grid tables), leaving ctx untouched. Called both to reset
     * between models in a multi-model process and from ~State to tear
     * the buffers down before ctx destroys the VkDevice.
     */
    void free_buffers() {
        // Weight pool: every model's weights are uploaded once and kept
        // resident (unbounded by default) -- the unbounded accumulator
        // across a multi-model test process. Free them all.
        for (auto& [ptr, e] : pool.map_) {
            pool.destroy(e.buf);
        }
        pool.map_.clear();
        for (auto& b : pool.transient_) {
            pool.destroy(b);
        }
        pool.transient_.clear();
        pool.used_ = 0;
        // GPU-mapped KV pools: one per cache, never freed otherwise.
        for (auto& [map, buf] : kv_pools) {
            ctx.destroy_buffer(buf);
        }
        kv_pools.clear();
        moe_gpu_fits.clear();
        // Buffers retired by grow() mid-batch; end_batch frees these,
        // but free them here too in case a batch never closed.
        for (auto b : retired) {
            ctx.destroy_buffer(b);
        }
        retired.clear();
        // Activation scratch: bounded, but release it too so a large
        // model's buffers are not held across the rest of the run; each
        // regrows on next use.
        auto drop = [&](VulkanContext::Buffer& b, std::size_t& cap) {
            if (cap != 0) {
                ctx.destroy_buffer(b);
                b = VulkanContext::Buffer{};
                cap = 0;
            }
        };
        drop(x, x_n);
        drop(xb, xb_n);
        drop(img_q, img_q_n);
        drop(img_k, img_k_n);
        drop(img_v, img_v_n);
        drop(img_s, img_s_n);
        drop(img_o, img_o_n);
        drop(q, q_n);
        drop(gate, gate_n);
        drop(up, up_n);
        drop(attout, attout_n);
        drop(att, att_n);
        drop(logits, logits_n);
        drop(table, table_n);
        drop(ones, ones_n);
        drop(kv_a, kv_a_n);
        drop(q_abs, q_abs_n);
        drop(latent, latent_n);
        drop(q_a, q_a_n);
        drop(router, router_n);
        drop(route, route_n);
        drop(attn_part, attn_part_n);
        for (int c = 0; c < kChunkCount_; ++c) {
            drop(chunk[c], chunk_n[c]);
        }
        for (std::size_t i = 0; i < kMultiMax; ++i) {
            drop(mtable[i], mtable_n[i]);
        }
        // IQ grid+signs tables: uploaded once, resident; re-created
        // lazily after a reset (the ready flag drives grid_table).
        auto drop_tbl = [&](VulkanContext::Buffer& b, bool& ready) {
            if (ready) {
                ctx.destroy_buffer(b);
                b = VulkanContext::Buffer{};
                ready = false;
            }
        };
        drop_tbl(iq2xxs_tbl, iq2xxs_tbl_ready);
        drop_tbl(iq3xxs_tbl, iq3xxs_tbl_ready);
    }

    /** Image-model attention (vulkan_attention): Q, per-head K, per-head
     *  V^T, scores and output. Scratch rewritten every call, never in
     *  the weight pool. */
    VulkanContext::Buffer img_q{}, img_k{}, img_v{}, img_s{}, img_o{};
    std::size_t img_q_n = 0, img_k_n = 0, img_v_n = 0, img_s_n = 0,
                img_o_n = 0;

    /** Activation buffers, sized on first use per model dims. */
    VulkanContext::Buffer x{}, xb{}, q{}, gate{}, up{}, attout{},
        att{}, logits{}, table{}, ones{}, kv_a{}, q_abs{},
        latent{}, q_a{}, router{}, route{}, attn_part{};
    std::size_t x_n = 0, xb_n = 0, q_n = 0, gate_n = 0, up_n = 0,
                attout_n = 0, att_n = 0, logits_n = 0,
                table_n = 0, ones_n = 0, kv_a_n = 0, q_abs_n = 0,
                latent_n = 0, q_a_n = 0, router_n = 0, route_n = 0,
                attn_part_n = 0;

    /**
     * Uploads a whole stacked expert tensor (every expert) as one
     * buffer, keyed by the ExpertMat itself: its base pointer is
     * also expert 0's key on the per-expert (CPU-routed) path.
     */
    VulkanContext::Buffer upload_experts(
        const model::TransformerModel::ExpertMat& em,
        std::uint32_t n_expert) {
        const std::uint64_t bytes = em.expert_bytes * n_expert;
        return upload_rows(&em, em.base, bytes);
    }

    /**
     * Uploads the first w.rows rows of weight matrix w, keyed by
     * w.data. Q8_0 rows go up in the split GPU layout
     * (q8_layout.hpp), other types as stored.
     */
    VulkanContext::Buffer upload_w(const Mat& w) {
        return upload_rows(w.data, w, weight_bytes(w));
    }

    /**
     * Uploads same-width Q8_0 matrices stacked by rows (each row in
     * the split GPU layout) as one buffer, keyed three bytes into the
     * first one's tensor (upload_kb_t and upload_f32 own +1 and +2).
     * Matrix i starts at the sum of the earlier weight_bytes.
     */
    VulkanContext::Buffer upload_cat(std::span<const Mat* const> ms) {
        std::uint64_t total = 0;
        for (const Mat* w : ms) {
            total += weight_bytes(*w);
        }
        const std::size_t n = static_cast<std::size_t>(total);
        const std::size_t alloc = vk::gpu_alloc_bytes(n);
        return pool.acquire(
            static_cast<const std::byte*>(ms[0]->data) + 3, alloc,
            [&](VulkanContext::Buffer b) {
                auto* dst = static_cast<std::byte*>(ctx.mapped(b));
                for (const Mat* w : ms) {
                    vk::q8_0_split_rows(
                        static_cast<const std::byte*>(w->data), dst,
                        w->rows, w->cols);
                    dst += weight_bytes(*w);
                }
                std::memset(dst, 0, alloc - n);
            });
    }

    /**
     * Uploads exactly `bytes` of w's rows from w.data under pool key
     * `key`, repacking whole Q8_0 rows, into a gpu_alloc_bytes(bytes)
     * buffer with the tail zeroed. Rows stored as the GPU reads them
     * (every type but Q8_0) of at least kImportMin bytes, inside a
     * model file's mapping, are imported in place instead where the
     * device can (#317): the mmap'd weights are not copied. Only
     * tensors whose size is already a multiple of 16 import, so an
     * imported buffer is exactly the tensor, with no padding tail that
     * would hold the next tensor's bytes instead of zeros. The unmap
     * hook drops imports before their mapping goes away.
     */
    VulkanContext::Buffer upload_rows(const void* key, const Mat& w,
                                      std::uint64_t bytes) {
        const std::size_t n = static_cast<std::size_t>(bytes);
        const std::size_t alloc = vk::gpu_alloc_bytes(n);
        constexpr std::size_t kImportMin = std::size_t{64} << 10;
        if (w.type != gguf::TensorType::kQ8_0 && n >= kImportMin &&
            alloc == n && ctx.host_import() &&
            sys::in_file_mapping(w.data, n)) {
            auto b = pool.acquire_external(
                key, w.data, [&]() -> std::optional<VulkanContext::Buffer> {
                    auto ib = ctx.import_host(w.data, alloc, n);
                    if (ib.impl == nullptr) {
                        return std::nullopt;
                    }
                    return ib;
                });
            if (b) {
                return *b;
            }
        }
        return pool.acquire(
            key, alloc,
            [&](VulkanContext::Buffer b) {
                const auto* src = static_cast<const std::byte*>(w.data);
                auto* dst = static_cast<std::byte*>(ctx.mapped(b));
                if (w.type == gguf::TensorType::kQ8_0) {
                    const std::size_t row_bytes = (w.cols / 32) * 34;
                    vk::q8_0_split_rows(src, dst, n / row_bytes, w.cols);
                } else {
                    std::memcpy(dst, src, n);
                }
                std::memset(dst + n, 0, alloc - n);
            },
            w.data);
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

    /**
     * Uploads w dequantized to F32 (any CPU-supported type), keyed
     * apart from w.data (upload_w's key for the same matrix as
     * stored) -- like upload_kb_t's +1 key.
     */
    VulkanContext::Buffer upload_f32(const Mat& w) {
        const std::size_t bytes =
            static_cast<std::size_t>(w.rows) * w.cols * 4;
        return pool.acquire(
            static_cast<const std::byte*>(w.data) + 2, bytes,
            [&](VulkanContext::Buffer b) {
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
        kCSplit,   // split-K partial planes (decode attention)
        kChunkCount_
    };
    VulkanContext::Buffer chunk[kChunkCount_]{};
    std::size_t chunk_n[kChunkCount_]{};

    /** vulkan_forward_multi: one block table per sequence. */
    static constexpr std::size_t kMultiMax = 8;
    VulkanContext::Buffer mtable[kMultiMax]{};
    std::size_t mtable_n[kMultiMax]{};

    /** @returns Chunk buffer c, grown to at least n_floats. */
    VulkanContext::Buffer chunk_buf(Chunk c, std::size_t n_floats) {
        grow(chunk[c], chunk_n[c], std::max<std::size_t>(n_floats, 1));
        return chunk[c];
    }

    /**
     * Grows scratch buffer b to n_floats. The old buffer is retired,
     * not destroyed: dispatches already recorded in the open batch
     * may still reference it (e.g. a later, larger split-K plane in
     * the same batch). end_batch() frees retired buffers.
     */
    void grow(VulkanContext::Buffer& b, std::size_t& cap,
              std::size_t n_floats) {
        if (n_floats > cap) {
            if (cap != 0) {
                retired.push_back(b);
            }
            b = ctx.create_buffer(n_floats * 4);
            cap = n_floats;
        }
    }

    std::vector<VulkanContext::Buffer> retired;

    /**
     * Drops the open batch unsubmitted and releases what it pinned,
     * as end_batch would (an exception escaped mid-forward).
     */
    void abandon_batch() {
        ctx.abandon_batch();
        pool.on_batch_end();
        for (auto b : retired) {
            ctx.destroy_buffer(b);
        }
        retired.clear();
    }

    /**
     * Abandons the batch still open when the scope unwinds, so an
     * exception escaping a forward never leaves a batch open with its
     * weights pinned (on_unmap would then abort when the model
     * unloads). A no-op once end_batch has run.
     */
    struct BatchScope {
        State& s;
        explicit BatchScope(State& st) : s(st) {}
        BatchScope(const BatchScope&) = delete;
        BatchScope& operator=(const BatchScope&) = delete;
        ~BatchScope() {
            if (s.ctx.batch_open()) {
                s.abandon_batch();
            }
        }
    };

    /** Submits the open batch, then releases what it no longer needs. */
    void end_batch() {
        ctx.end_batch();
        pool.on_batch_end();
        for (auto b : retired) {
            ctx.destroy_buffer(b);
        }
        retired.clear();
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

/**
 * Serializes every exported entry point that touches state() (i#268).
 * The server reaches the backend from the engine thread and from HTTP
 * threads (image, vision, embeddings, audio) at once, and State's
 * scratch buffers, open command batch and weight pool are shared.
 * Recursive, so an entry point may reach another through the model's
 * ops while holding it.
 */
std::recursive_mutex& state_mu();

/**
 * sys unmap hook (set while State lives): a model file's mapping is
 * going away, so drop every pool buffer over its pages (#317).
 */
void on_unmap(const void* p, std::size_t n) {
    std::lock_guard lk(state_mu());
    State& s = state();
    if (s.ctx.batch_open()) {
        // A recorded batch still references these weights; freeing
        // them under it is undefined. Runs from ~MappedFile: no throw.
        std::fprintf(stderr,
                     "[vulkan] model file unmapped while a batch still "
                     "uses its weights\n");
        std::abort();
    }
    s.pool.release_range(p, n);
}

std::recursive_mutex& state_mu() {
    static std::recursive_mutex mu;
    return mu;
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
                         : s.upload_w(w);
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

/**
 * @returns Whether ms (2 or 3 matrices) can run as one
 * rec_matvec_seg dispatch: all Q8_0 of one width.
 */
bool seg_ok(std::span<const Mat* const> ms) {
    for (const Mat* w : ms) {
        if (w->type != gguf::TensorType::kQ8_0 ||
            w->cols != ms[0]->cols) {
            return false;
        }
    }
    return ms.size() == 2 || ms.size() == 3;
}

/**
 * Epilogue rope for rec_matvec_seg (matvec_q8_0.comp SEGMENTS): rows
 * of ms[0] (seg 0) and ms[1] (seg 1) are heads of hs rows whose rd
 * rows from ro on rotate at pos, rope.comp style. rd 0 = no rope;
 * hs, ro and rd must be even.
 */
struct SegRope {
    std::uint32_t hs0 = 1, ro0 = 0, rd0 = 0;
    std::uint32_t hs1 = 1, ro1 = 0, rd1 = 0;
    std::uint32_t pos = 0;
    std::uint32_t base_bits = 0, fs_bits = 0, lo_bits = 0, hi_bits = 0;
    VulkanContext::Buffer divisors{};
};

/**
 * Records the matvecs of 2 or 3 matrices reading x (seg_ok) as one
 * dispatch over their stacked rows (State::upload_cat): ms[0]'s rows
 * to out at out_off, ms[1]'s to out2 at off1, ms[2]'s (if any) to
 * out2 at off2, roped per `rope` (default: none).
 */
void rec_matvec_seg(State& s, std::span<const Mat* const> ms,
                    VulkanContext::Buffer x, VulkanContext::Buffer out,
                    std::uint32_t out_off, VulkanContext::Buffer out2,
                    std::uint32_t off1, std::uint32_t off2,
                    const SegRope& rope = {}) {
    const std::uint32_t seg1 = ms[0]->rows;
    const std::uint32_t seg2 = seg1 + ms[1]->rows;
    // The epilogue's pairing assumptions (vk::seg_rope_fits, segment 1
    // starting on an even row) hold for every caller by construction;
    // check them so a new caller fails loudly instead of roping the
    // wrong pairs. accumulate is always 0 here: rope then write.
    if ((rope.rd0 != 0 || rope.rd1 != 0) &&
        (seg1 % 2 != 0 ||
         !vk::seg_rope_fits(seg1, rope.hs0, rope.ro0, rope.rd0) ||
         !vk::seg_rope_fits(ms[1]->rows, rope.hs1, rope.ro1, rope.rd1))) {
        throw std::logic_error("matvec_q8_0_seg: rope spec breaks the "
                               "epilogue's pairing invariants");
    }
    const std::uint32_t rows =
        seg2 + (ms.size() > 2 ? ms[2]->rows : 0);
    const VulkanContext::Buffer bufs[] = {
        s.upload_cat(ms), x, out, out2,
        rope.divisors.impl != nullptr ? rope.divisors : x};
    const std::uint32_t push[] = {
        rows,         ms[0]->cols, out_off,     0,
        0,            0,           fbits(1.0f), seg1,
        off1,         seg2,        off2,        rope.hs0,
        rope.ro0,     rope.rd0,    rope.hs1,    rope.ro1,
        rope.rd1,     rope.pos,    rope.base_bits, rope.fs_bits,
        rope.lo_bits, rope.hi_bits};
    s.ctx.dispatch(Kernel::kMatvecQ8_0Seg, bufs, push,
                   VulkanContext::matvec_groups(Kernel::kMatvecQ8_0Seg,
                                                rows));
}

/**
 * @returns How many of q, k, v (in that order) can run as one
 * rec_matvec_q4k_seg dispatch: q and k Q4_K of one width (2), and v
 * too when it matches them (3); 0 otherwise.
 */
std::size_t q4k_seg_count(const Mat& q, const Mat& k, const Mat& v) {
    const auto q4 = gguf::TensorType::kQ4_K;
    if (q.type != q4 || k.type != q4 || k.cols != q.cols) {
        return 0;
    }
    return v.type == q4 && v.cols == q.cols ? 3 : 2;
}

/**
 * Records the matvecs of 2 or 3 Q4_K matrices reading x
 * (q4k_seg_count) as one matvec_q4_k_seg dispatch, each read from its
 * own upload: ms[0]'s rows to out at out_off, ms[1]'s to out2 at off1,
 * ms[2]'s (if any) to out2 at off2.
 */
void rec_matvec_q4k_seg(State& s, std::span<const Mat* const> ms,
                        VulkanContext::Buffer x, VulkanContext::Buffer out,
                        std::uint32_t out_off, VulkanContext::Buffer out2,
                        std::uint32_t off1, std::uint32_t off2) {
    const std::uint32_t seg1 = ms[0]->rows;
    const std::uint32_t seg2 = seg1 + ms[1]->rows;
    const std::uint32_t rows = seg2 + (ms.size() > 2 ? ms[2]->rows : 0);
    const VulkanContext::Buffer w1 = s.upload_w(*ms[1]);
    const VulkanContext::Buffer bufs[] = {
        s.upload_w(*ms[0]), x, out, out2, w1,
        ms.size() > 2 ? s.upload_w(*ms[2]) : w1};
    const std::uint32_t push[] = {rows, ms[0]->cols, out_off, 0, 0, 0,
                                  fbits(1.0f), seg1, off1, seg2, off2};
    s.ctx.dispatch(Kernel::kMatvecQ4_KSeg, bufs, push,
                   VulkanContext::matvec_groups(Kernel::kMatvecQ4_KSeg,
                                                rows));
}

/**
 * @returns Whether a GPU-routed MoE layer's shared experts can ride
 * the routed dispatches (matvec_q8_0_glu_sh_b / _sh_b): all Q8_0,
 * gate / up as wide as the routed expert g, and n_expert_shared
 * slots of n_ff_exp rows each, d being the routed down slice.
 */
bool moe_shared_ok(const model::Hparams& hp,
                   const model::TransformerModel::Layer& lay,
                   const Mat& g, const Mat& d) {
    const auto q8 = gguf::TensorType::kQ8_0;
    const std::uint64_t ff_sh =
        std::uint64_t{hp.n_ff_exp} * hp.n_expert_shared;
    // A gated shared expert (qwen2moe) needs its sigmoid scale; the
    // fused routed + shared kernels have none.
    return hp.n_expert_shared > 0 && hp.n_ff_shexp == 0 &&
           lay.gate_inp_shexp.rows == 0 && g.type == q8 && d.type == q8 &&
           lay.up_exps.expert(0).type == q8 &&
           lay.gate_exps.expert_bytes == lay.up_exps.expert_bytes &&
           lay.gate_shexp.type == q8 && lay.up_shexp.type == q8 &&
           lay.down_shexp.type == q8 && lay.gate_shexp.cols == g.cols &&
           lay.up_shexp.cols == g.cols && lay.gate_shexp.rows >= ff_sh &&
           lay.up_shexp.rows >= ff_sh && lay.down_shexp.cols == ff_sh &&
           lay.down_shexp.rows == d.rows && ff_sh % 32 == 0;
}

/** @returns The shared-expert FFN width (n_ff_shexp when it differs
 * from the routed width: qwen2moe), as the CPU forward uses it. */
std::uint32_t shexp_ff(const model::Hparams& hp) {
    return hp.n_ff_shexp > 0 ? hp.n_ff_shexp
                             : hp.n_ff_exp * hp.n_expert_shared;
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
 * unused (pass any buffer). row_stride (F32/F16 only; 0 = w.cols)
 * lets w.cols be a column chunk of wider rows.
 */
void rec_matvec_b(State& s, const Mat& w, VulkanContext::Buffer wb,
                  VulkanContext::Buffer x, VulkanContext::Buffer out,
                  std::uint32_t out_offset, bool accumulate,
                  std::uint64_t w_byte_off, std::uint32_t x_off,
                  std::uint64_t w_stride_bytes, std::uint32_t x_stride,
                  std::uint32_t out_stride, std::uint32_t slots,
                  std::uint32_t mode, VulkanContext::Buffer route,
                  std::uint32_t row_stride = 0) {
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
                                  mode,
                                  row_stride};
    s.ctx.dispatch(k, bufs, push, VulkanContext::matvec_groups(k, w.rows),
                   (mode & kSlotSum) != 0 ? 1u : slots);
}

/** Token rows up to which a batched forward takes matvec-style
 * kernels (matvec_q8_0_n.comp NT; MoE pair matvecs) over matmuls. */
constexpr std::uint32_t kSmallBatch = 8;

/** matmul.comp tile sizes: weight rows and tokens per workgroup
 * (the MOE_ID variant's token tile is narrower). */
constexpr std::uint32_t kMatmulRows = VulkanContext::kMatmulTileRows;
constexpr std::uint32_t kMatmulToks = VulkanContext::kMatmulTileToks;
constexpr std::uint32_t kMatmulToksId = VulkanContext::kMatmulTileToks;

/** @returns k's f16-staging variant under --prefill-f16, else k
 * (the K-quant matmuls have none: they always stage f32). */
Kernel staged(const State& s, Kernel k) {
    if (!s.prefill_f16) {
        return k;
    }
    switch (k) {
        case Kernel::kMatmulQ8_0: return Kernel::kMatmulQ8_0H;
        case Kernel::kMatmulF32: return Kernel::kMatmulF32H;
        case Kernel::kMatmulQ8_0Id: return Kernel::kMatmulQ8_0IdH;
        case Kernel::kMatmulPagedF32: return Kernel::kMatmulPagedF32H;
        case Kernel::kMatmulPagedTF32: return Kernel::kMatmulPagedTF32H;
        default: return k;
    }
}

/** @returns The prefill matmul kernel for w, or kCount_. */
Kernel matmul_kernel(const Mat& w) {
    switch (w.type) {
        case gguf::TensorType::kQ8_0: return Kernel::kMatmulQ8_0;
        case gguf::TensorType::kQ4_K: return Kernel::kMatmulQ4_K;
        case gguf::TensorType::kQ6_K: return Kernel::kMatmulQ6_K;
        case gguf::TensorType::kQ5_0: return Kernel::kMatmulQ5_0;
        case gguf::TensorType::kF32: return Kernel::kMatmulF32;
        default: return Kernel::kCount_;
    }
}

/** @returns The routed-expert (MOE_ID) matmul kernel for w, or
 * kCount_. */
Kernel matmul_id_kernel(const Mat& w) {
    switch (w.type) {
        case gguf::TensorType::kQ8_0: return Kernel::kMatmulQ8_0Id;
        case gguf::TensorType::kQ4_K: return Kernel::kMatmulQ4_KId;
        case gguf::TensorType::kQ5_0: return Kernel::kMatmulQ5_0Id;
        case gguf::TensorType::kQ6_K: return Kernel::kMatmulQ6_KId;
        default: return Kernel::kCount_;
    }
}

/** @returns The short-batch (matvec_q8_0_n family) kernel for w over
 * n_tok <= kSmallBatch tokens, or kCount_. 16-bit weights take the
 * NT 2 build for up to 2 tokens (matvec_q8_0_n.comp). */
Kernel small_batch_kernel(const Mat& w, std::uint32_t n_tok) {
    const bool two = n_tok <= 2;
    switch (w.type) {
        case gguf::TensorType::kQ8_0: return Kernel::kMatvecQ8_0N;
        case gguf::TensorType::kF32: return Kernel::kMatvecF32N;
        case gguf::TensorType::kF16:
            return two ? Kernel::kMatvecF16N2 : Kernel::kMatvecF16N;
        case gguf::TensorType::kBF16:
            return two ? Kernel::kMatvecBF16N2 : Kernel::kMatvecBF16N;
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
    /** Batches sharing one weight slice (W moves every w_bdiv). */
    std::uint32_t w_bdiv = 1;
    /** Two-level token rows (GQA): token t is (t / div, t % div),
     * the inner index stepping X / out by these. */
    std::uint32_t div = 1, x_inner = 0, out_inner = 0;
};

/**
 * Records out[t] (+)= scale * W x[t] for every token row t (and
 * batch b) in one matmul dispatch, reading each weight tile once per
 * kMatmulToks tokens. Weight types without a matmul kernel fall back
 * to one matvec per (batch, token): correct, just not amortized.
 *
 * A grid under kMinGroups workgroups (few weight rows, e.g. MLA's
 * kv_a, or a short chunk) splits the K range over more of them, as
 * rec_paged_matmul does, when the output is one tight plane (single
 * batch, out_off 0, rows apart) that sum_splits can add up.
 */
void rec_matmul(State& s, const Mat& w, VulkanContext::Buffer wb,
                VulkanContext::Buffer x, VulkanContext::Buffer out,
                const MatmulRows& r, bool accumulate,
                std::uint64_t w_byte_off = 0, float scale = 1.0f) {
    // A short batch (a speculative verify, a tiny chunk, the CFG pair of
    // vulkan_forward_multi): one weight read for all its tokens
    // (matvec_*_n, bandwidth-bound) instead of a mostly-padding matmul
    // tile, or, for F16 / BF16 (no matmul kernel), one matvec per token.
    const Kernel small = small_batch_kernel(w, r.n_tok);
    if (small != Kernel::kCount_ && r.n_tok <= kSmallBatch && r.div == 1 &&
        r.w_bdiv == 1) {
        const VulkanContext::Buffer bufs[] = {wb, x, out};
        const std::uint32_t push[] = {w.rows,
                                      w.cols,
                                      r.out_off,
                                      accumulate ? 1u : 0u,
                                      w_off_units(w, w_byte_off),
                                      r.x_off,
                                      fbits(scale),
                                      r.n_tok,
                                      r.x_stride,
                                      r.out_stride,
                                      w_off_units(w, r.w_bstride_bytes),
                                      r.x_bstride,
                                      r.out_bstride};
        s.ctx.dispatch(small, bufs, push,
                       VulkanContext::matvec_groups(small, w.rows),
                       r.n_batch);
        return;
    }
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
    constexpr std::uint32_t kMinGroups = 256;
    constexpr std::uint32_t kSplitCols = 256;
    const std::uint32_t row_tiles = (w.rows + kMatmulRows - 1) / kMatmulRows;
    const std::uint32_t groups = row_tiles * tiles * r.n_batch;
    const bool tight = r.n_batch == 1 && r.out_off == 0 &&
                       r.out_stride == w.rows && r.div == 1;
    const std::uint32_t k_splits =
        !tight || groups >= kMinGroups
            ? 1
            : std::clamp<std::uint32_t>((kMinGroups + groups - 1) / groups,
                                        1, std::max(w.cols / kSplitCols, 1u));
    const std::uint32_t plane = r.n_tok * w.rows;
    const VulkanContext::Buffer dst =
        k_splits > 1
            ? s.chunk_buf(State::kCSplit, std::size_t{k_splits} * plane)
            : out;
    const VulkanContext::Buffer bufs[] = {wb, x, dst};
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
                                  r.div,
                                  0,
                                  r.w_bdiv,
                                  0,
                                  0,
                                  0,
                                  k_splits,
                                  plane,
                                  r.x_inner,
                                  r.out_inner};
    s.ctx.dispatch(staged(s, k), bufs, push, row_tiles,
                   tiles * r.n_batch * k_splits);
    if (k_splits > 1) {
        const VulkanContext::Buffer sbufs[] = {dst, out};
        const std::uint32_t spush[] = {plane, k_splits, 0,
                                       accumulate ? 1u : 0u};
        s.ctx.dispatch(Kernel::kSumSplits, sbufs, spush,
                       (plane + 255) / 256);
    }
}

/**
 * Records out[out_off + r] += bias[r]: an F32 matvec of the bias
 * column by a constant 1, accumulated (q / k / v biases, qwen2moe,
 * without a dedicated kernel).
 */
void rec_bias(State& s, std::span<const float> bias,
              VulkanContext::Buffer out, std::uint32_t out_off) {
    static const float kOne = 1.0f;
    const Mat b{gguf::TensorType::kF32,
                reinterpret_cast<const std::byte*>(bias.data()),
                static_cast<std::uint32_t>(bias.size()), 1};
    rec_matvec(s, b, s.upload(&kOne, sizeof kOne), out, out_off, true);
}

/**
 * Dispatches kv_scatter (or kv_scatter_f16) with push p in the
 * shader's layout (width, pos0, block_tokens, block_stride, row_w,
 * dst_off, src_off, src_stride, src_plane, dst_plane) over a
 * (1, rows, planes) grid. The f16 build first checks its even-pair
 * invariant (vk::kv_scatter_f16_fits).
 * @throws std::logic_error when an f16 scatter's terms are not even.
 */
void rec_kv_scatter(State& s, bool f16,
                    std::span<const VulkanContext::Buffer> bufs,
                    std::span<const std::uint32_t, 10> p,
                    std::uint32_t rows, std::uint32_t planes) {
    if (f16 && !vk::kv_scatter_f16_fits(p[0], p[3], p[4], p[5], p[9])) {
        throw std::logic_error(
            "kv_scatter_f16: width, block_stride, row_w, dst_off and "
            "dst_plane must be even");
    }
    s.ctx.dispatch(f16 ? Kernel::kKvScatterF16 : Kernel::kKvScatter, bufs,
                   p, 1, rows, planes);
}

/**
 * @returns The most MOE_ID matmul token tiles n_tok tokens' n_tok * k
 * pairs can fill across n_expert experts: every expert's last tile
 * may be partial, so ceil(pairs / tile) + n_expert, and never more
 * than n_expert times one expert's worst case.
 */
std::uint32_t moe_tiles_max(std::uint32_t n_tok, std::uint32_t k,
                            std::uint32_t n_expert) {
    const std::uint32_t pairs = n_tok * k;
    return std::min((pairs + kMatmulToksId - 1) / kMatmulToksId + n_expert,
                    n_expert * ((n_tok + kMatmulToksId - 1) / kMatmulToksId));
}

/**
 * Records the routed-expert matmul for a prefill chunk: every
 * expert's weights (stacked in wb, expert_bytes apart) applied once
 * to the (token, slot) pairs moe_group listed for it in ids, one
 * workgroup row per entry of moe_group's tile list (n_tok tokens of
 * k slots each). Pair p reads X row p / x_div and writes output row p.
 * w must have a matmul_id_kernel.
 */
void rec_matmul_id(State& s, const Mat& w, VulkanContext::Buffer wb,
                   VulkanContext::Buffer x, VulkanContext::Buffer out,
                   VulkanContext::Buffer ids, std::uint32_t n_tok,
                   std::uint32_t k, std::uint32_t n_expert,
                   std::uint64_t expert_bytes, std::uint32_t x_stride,
                   std::uint32_t x_div, std::uint32_t out_stride) {
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
                                  n_expert,
                                  x_div,
                                  n_tok,
                                  1,
                                  0,
                                  0,
                                  0,
                                  1,
                                  0,
                                  0,
                                  0};
    s.ctx.dispatch(staged(s, matmul_id_kernel(w)), bufs, push,
                   (w.rows + kMatmulRows - 1) / kMatmulRows,
                   moe_tiles_max(n_tok, k, n_expert));
}

/** The paged KV pool as a matmul weight operand. */
struct PagedPool {
    VulkanContext::Buffer pool, table;
    std::uint32_t block_tokens, block_stride, row_w;
    /** F16 pool (opt-in, #105): same layout, halves for floats. */
    bool f16 = false;
};

/**
 * Records a matmul whose weight is the paged pool (kMatmulPagedF32:
 * row r = cache position r; kMatmulPagedTF32: column c = cache
 * position c), w_off floats into each position's cache row.
 * r.w_bstride_bytes moves w_off per weight slice.
 *
 * When the grid would be under kMinGroups workgroups (a decode
 * token: few output tiles), the K range is split over more of them,
 * each writing a partial plane of `plane` floats -- the output's
 * extent from r.out_off, which must then be 0 -- that sum_splits
 * adds into out.
 */
void rec_paged_matmul(State& s, Kernel k, const PagedPool& pp,
                      std::uint32_t rows, std::uint32_t cols,
                      std::uint32_t w_off, VulkanContext::Buffer x,
                      VulkanContext::Buffer out, const MatmulRows& r,
                      bool accumulate, std::uint32_t plane) {
    // Fewer workgroups than this leave the GPU partly idle; a split
    // keeps at least kSplitCols columns.
    constexpr std::uint32_t kMinGroups = 128;
    constexpr std::uint32_t kSplitCols = 32;
    const std::uint32_t tiles = (r.n_tok + kMatmulToks - 1) / kMatmulToks;
    const std::uint32_t groups =
        (rows + kMatmulRows - 1) / kMatmulRows * tiles * r.n_batch;
    const std::uint32_t k_splits =
        groups >= kMinGroups
            ? 1
            : std::clamp<std::uint32_t>((kMinGroups + groups - 1) / groups,
                                        1, std::max(cols / kSplitCols, 1u));
    const VulkanContext::Buffer dst =
        k_splits > 1
            ? s.chunk_buf(State::kCSplit, std::size_t{k_splits} * plane)
            : out;
    if (pp.f16) {
        k = k == Kernel::kMatmulPagedF32 ? Kernel::kMatmulPagedF16
                                         : Kernel::kMatmulPagedTF16;
    }
    const VulkanContext::Buffer bufs[] = {pp.pool, x, dst, pp.table};
    const std::uint32_t push[] = {
        rows,         cols,
        r.n_tok,      accumulate ? 1u : 0u,
        w_off,        r.x_off,
        r.x_stride,   k_splits > 1 ? 0 : r.out_off,
        r.out_stride, fbits(1.0f),
        static_cast<std::uint32_t>(r.w_bstride_bytes / 4),
        r.x_bstride,  r.out_bstride,
        tiles,        r.div,
        0,            r.w_bdiv,
        pp.block_tokens, pp.block_stride,
        pp.row_w,     k_splits,
        plane,        r.x_inner,
        r.out_inner};
    s.ctx.dispatch(staged(s, k), bufs, push,
                   (rows + kMatmulRows - 1) / kMatmulRows,
                   tiles * r.n_batch * k_splits);
    if (k_splits > 1) {
        // Partial planes start at 0; out_off moves the sum (an
        // attention query sub-block's rows).
        const VulkanContext::Buffer sbufs[] = {dst, out};
        const std::uint32_t spush[] = {plane, k_splits, r.out_off,
                                       accumulate ? 1u : 0u};
        s.ctx.dispatch(Kernel::kSumSplits, sbufs, spush, (plane + 255) / 256);
    }
}

/**
 * Attention score scratch cap, in floats (1 GiB): one buffer stays
 * within maxStorageBufferRange on every device and its indices far
 * below 2^32 (i#313).
 */
constexpr std::size_t kAttnScratchFloats = std::size_t{1} << 28;

/** The cap in use (vulkan_set_attn_scratch_floats; tests only). */
std::atomic<std::size_t> attn_scratch_floats{kAttnScratchFloats};

/** vulkan_attn_block at the cap in use. */
std::uint32_t attn_block(std::uint32_t n, std::uint32_t heads,
                         std::uint32_t n_pos) {
    const std::size_t per = std::size_t{heads} * n_pos;
    return static_cast<std::uint32_t>(std::clamp<std::size_t>(
        attn_scratch_floats.load() / std::max<std::size_t>(per, 1), 1,
        std::max(n, 1u)));
}

/** Records the causal softmax over `rows` score rows (attn_softmax). */
void rec_softmax(State& s, VulkanContext::Buffer scores, std::uint32_t rows,
                 std::uint32_t n_pos, std::uint32_t heads, std::uint32_t base,
                 float scale) {
    const VulkanContext::Buffer bufs[] = {scores};
    const std::uint32_t push[] = {n_pos, heads, base, fbits(scale), rows};
    const std::uint32_t gx = std::min<std::uint32_t>(rows, 65535);
    s.ctx.dispatch(Kernel::kAttnSoftmax, bufs, push, gx,
                   (rows + gx - 1) / gx);
}

/**
 * Records causal GQA attention for n query tokens at positions
 * base .. base + n - 1 of one layer as matmuls over the paged pool:
 * S = K Q^T, masked softmax, O = V^T P, one batch per K/V head whose
 * `group` query heads are extra token rows (row t * group + i), so
 * each K/V tile is read once for all of them. Q and out rows are
 * q_dim floats per token (head h at h * hd). Queries run in
 * sub-blocks of attn_block(n, heads, base + n) tokens, each over the
 * positions it can see; scores holds one sub-block's rows.
 */
/** One attention segment set for rec_attn_decode (see attn_decode.comp). */
struct DecodeAttn {
    VulkanContext::Buffer q1, q2, out;
    std::uint32_t k_off, v_off, kv_stride, group, d1, d2, q1_stride,
        q2_stride, q2_off, dv, out_stride;
    float scale;
};

/**
 * @returns Whether attn_decode_grp can run a: vec4 V loads (dv % 4,
 * 4-float aligned V rows) and dv within vk::kAttnGrpDvMax.
 */
bool grp_ok(const PagedPool& pp, const DecodeAttn& a) {
    return vk::attn_grp_dv_fits(a.dv) && a.v_off % 4 == 0 &&
           a.kv_stride % 4 == 0 && pp.row_w % 4 == 0 &&
           pp.block_stride % 4 == 0;
}

/**
 * Records flash-decoding attention for one query token over n_pos
 * cache positions, plus attn_combine when there is more than one
 * block: attn_decode per (64-position block, head), or with
 * `grouped` (grp_ok) attn_decode_grp per (32-position block, KV
 * head), reading each K/V row once for the KV head's query heads.
 */
void rec_attn_decode(State& s, const PagedPool& pp, std::uint32_t n_pos,
                     std::uint32_t heads, const DecodeAttn& a,
                     bool grouped) {
    // Positions per block: attn_decode_grp.comp / attn_decode.comp CH.
    const std::uint32_t block = grouped ? 32 : 64;
    const std::uint32_t blocks = (n_pos + block - 1) / block;
    VulkanContext::Buffer part = a.out;
    if (blocks > 1) {
        s.grow(s.attn_part, s.attn_part_n,
               std::size_t{blocks} * heads * (a.dv + 2));
        part = s.attn_part;
    }
    const VulkanContext::Buffer bufs[] = {pp.pool, pp.table, a.q1,
                                          a.q2,    a.out,    part};
    const std::uint32_t push[] = {n_pos,       pp.block_tokens,
                                  pp.block_stride, pp.row_w,
                                  a.k_off,     a.v_off,
                                  a.kv_stride, a.group,
                                  a.d1,        a.d2,
                                  a.q1_stride, a.q2_stride,
                                  a.q2_off,    a.dv,
                                  fbits(a.scale), a.out_stride};
    if (grouped) {
        // Workgroup y: KV head x chunks of hg heads (attn_decode_grp
        // HG): 8 for a group that fits, so no duplicate-head lanes.
        const bool hg8 = a.group <= 8;
        const std::uint32_t hg = hg8 ? 8 : 16;
        const std::uint32_t chunks = (a.group + hg - 1) / hg;
        const Kernel k =
            hg8 ? (pp.f16 ? Kernel::kAttnDecodeGrp8F16
                          : Kernel::kAttnDecodeGrp8)
                : (pp.f16 ? Kernel::kAttnDecodeGrpF16
                          : Kernel::kAttnDecodeGrp);
        s.ctx.dispatch(k, bufs, push, blocks, heads / a.group * chunks);
    } else {
        s.ctx.dispatch(
            pp.f16 ? Kernel::kAttnDecodeF16 : Kernel::kAttnDecode, bufs,
            push, blocks, heads);
    }
    if (blocks > 1) {
        const VulkanContext::Buffer cbufs[] = {part, a.out};
        const std::uint32_t cpush[] = {blocks, heads, a.dv, a.out_stride};
        s.ctx.dispatch(Kernel::kAttnCombine, cbufs, cpush,
                       (a.dv + 63) / 64, heads);
    }
}

void rec_attn_gqa(State& s, const PagedPool& pp, VulkanContext::Buffer q,
                  VulkanContext::Buffer scores, VulkanContext::Buffer out,
                  std::uint32_t n, std::uint32_t base, std::uint32_t heads,
                  std::uint32_t group, std::uint32_t hd, std::uint32_t q_dim,
                  std::uint32_t layer_off, std::uint32_t v_offset) {
    const std::uint32_t n_pos = base + n;
    const DecodeAttn da{q,  q,     out, layer_off, layer_off + v_offset,
                        hd, group, hd,  0,         hd,
                        0,  0,     hd,  hd,
                        1.0f / std::sqrt(static_cast<float>(hd))};
    const bool grouped = group > 1 && grp_ok(pp, da);
    if (n == 1 && (grouped || vk::attn_decode_fits(hd))) {
        rec_attn_decode(s, pp, n_pos, heads, da, grouped);
        return;
    }
    const std::uint32_t n_kv = heads / group;
    const std::uint32_t mb = attn_block(n, heads, n_pos);
    for (std::uint32_t t0 = 0; t0 < n; t0 += mb) {
        // Queries t0 .. t0 + m - 1 see positions [0, np).
        const std::uint32_t m = std::min(mb, n - t0);
        const std::uint32_t np = base + t0 + m;
        MatmulRows sr{m * group, t0 * q_dim, q_dim, 0, heads * np};
        sr.n_batch = n_kv;
        sr.x_bstride = group * hd;
        sr.out_bstride = group * np;
        sr.w_bstride_bytes = std::uint64_t{hd} * 4;
        sr.div = group;
        sr.x_inner = hd;
        sr.out_inner = np;
        rec_paged_matmul(s, Kernel::kMatmulPagedF32, pp, np, hd, layer_off,
                         q, scores, sr, false, m * heads * np);
        rec_softmax(s, scores, m * heads, np, heads, base + t0,
                    1.0f / std::sqrt(static_cast<float>(hd)));
        MatmulRows orow{m * group, 0, heads * np, t0 * q_dim, q_dim};
        orow.n_batch = n_kv;
        orow.x_bstride = group * np;
        orow.out_bstride = group * hd;
        orow.w_bstride_bytes = std::uint64_t{hd} * 4;
        orow.div = group;
        orow.x_inner = np;
        orow.out_inner = hd;
        rec_paged_matmul(s, Kernel::kMatmulPagedTF32, pp, hd, np,
                         layer_off + v_offset, scores, out, orow, false,
                         m * q_dim);
    }
}

/**
 * Records causal MLA attention for n query tokens at positions
 * base .. base + n - 1 of one layer. Every head reads the same
 * latent rows, so the n * heads (token, head) pairs are the query
 * rows of one matmul (each latent tile read once for all of them):
 * S = C q_abs^T + K_pe q_pe^T, masked softmax, latent = C^T P.
 * q is [token][head][nope + rope], q_abs and lat [token][head][rank].
 */
void rec_attn_mla(State& s, const PagedPool& pp, VulkanContext::Buffer q,
                  VulkanContext::Buffer q_abs, VulkanContext::Buffer scores,
                  VulkanContext::Buffer lat, std::uint32_t n,
                  std::uint32_t base, std::uint32_t heads, std::uint32_t rank,
                  std::uint32_t nope, std::uint32_t rope_d,
                  std::uint32_t layer_off, float kq_scale) {
    const std::uint32_t n_pos = base + n;
    // Decode: flash-decoding, q_abs . latent + q_pe . k_pe on the
    // one shared latent row (group = heads), with attn_decode_grp
    // reading each row once for 16 heads (M4 Max,
    // deepseek-coder-v2-lite decode at 3k: 64 -> 82 tok/s over the
    // matmul form below). Else attn_decode per head up to ~1k
    // positions (it reads the latent once per head).
    const DecodeAttn da{q_abs, q,    lat,          layer_off, layer_off,
                        0,     heads, rank,        rope_d,    rank,
                        nope + rope_d, nope,       rank,      rank,
                        kq_scale};
    if (n == 1 && grp_ok(pp, da)) {
        rec_attn_decode(s, pp, n_pos, heads, da, true);
        return;
    }
    if (n == 1 && n_pos <= 1024 && vk::attn_decode_fits(rank + rope_d)) {
        rec_attn_decode(s, pp, n_pos, heads, da, false);
        return;
    }
    // Query sub-blocks as rec_attn_gqa: tokens t0 .. t0 + m - 1 are
    // rows t0 * heads .. of q_abs / q / lat and see positions [0, np).
    const std::uint32_t qw = nope + rope_d;
    const std::uint32_t mb = attn_block(n, heads, n_pos);
    for (std::uint32_t t0 = 0; t0 < n; t0 += mb) {
        const std::uint32_t m = std::min(mb, n - t0);
        const std::uint32_t np = base + t0 + m;
        const std::uint32_t rows = m * heads;
        const std::uint32_t r0 = t0 * heads;
        rec_paged_matmul(s, Kernel::kMatmulPagedF32, pp, np, rank,
                         layer_off, q_abs, scores,
                         MatmulRows{rows, r0 * rank, rank, 0, np}, false,
                         rows * np);
        rec_paged_matmul(s, Kernel::kMatmulPagedF32, pp, np, rope_d,
                         layer_off + rank, q, scores,
                         MatmulRows{rows, r0 * qw + nope, qw, 0, np}, true,
                         rows * np);
        rec_softmax(s, scores, rows, np, heads, base + t0, kq_scale);
        rec_paged_matmul(s, Kernel::kMatmulPagedTF32, pp, rank, np,
                         layer_off, scores, lat,
                         MatmulRows{rows, 0, np, r0 * rank, rank}, false,
                         rows * rank);
    }
}

}  // namespace

void vulkan_set_prefill_f16(bool on) {
    // Tools call this unconditionally: on a box without a usable
    // device it must not build the context (that throws).
    if (!vulkan_backend_usable()) {
        if (on) {
            std::fprintf(stderr,
                         "locus: --prefill-f16 ignored: no usable "
                         "Vulkan device\n");
        }
        return;
    }
    std::lock_guard lk(state_mu());
    State& s = state();
    if (on && !s.ctx.has_f16()) {
        std::fprintf(stderr,
                     "locus: --prefill-f16 ignored: the Vulkan device "
                     "has no shaderFloat16 (prefill stays f32)\n");
        on = false;
    }
    s.prefill_f16 = on;
}

std::uint32_t vulkan_attn_block(std::uint32_t n, std::uint32_t heads,
                                std::uint32_t n_pos) {
    return attn_block(n, heads, n_pos);
}

void vulkan_set_attn_scratch_floats(std::size_t floats) {
    attn_scratch_floats.store(floats != 0 ? floats : kAttnScratchFloats);
}

bool vulkan_backend_usable() {
    static const bool ok = VulkanContext::available();
    return ok;
}

float* vulkan_alloc_kv(std::size_t n_floats) {
    std::lock_guard lk(state_mu());
    State& s = state();
    auto buf = s.ctx.create_buffer(n_floats * 4);
    void* map = s.ctx.mapped(buf);
    std::memset(map, 0, n_floats * 4);
    s.kv_pools.emplace(map, buf);
    return static_cast<float*>(map);
}

void vulkan_reset_state() {
    std::lock_guard lk(state_mu());
    state().free_buffers();
}

void matvec_vulkan(const Mat& w, std::span<const float> x,
                   std::span<float> out) {
    if (matvec_kernel(w) == Kernel::kCount_) {
        matvec(w, x, out);  // no shader for this weight type
        return;
    }
    std::lock_guard lk(state_mu());
    State& s = state();
    const Kernel k = matvec_kernel(w);
    auto wb = s.upload_w(w);
    s.grow(s.x, s.x_n, x.size());
    s.grow(s.xb, s.xb_n, out.size());
    s.ctx.write_buffer(s.x, std::as_bytes(x));
    const VulkanContext::Buffer grid = s.grid_for(k);
    s.ctx.begin_batch();
    State::BatchScope batch_scope{s};
    const VulkanContext::Buffer bufs[] = {wb, s.x, s.xb, grid};
    const std::size_t nbuf = grid.impl != nullptr ? 4u : 3u;
    const std::uint32_t push[] = {w.rows, w.cols, 0, 0, 0, 0,
                                  fbits(1.0f)};
    s.ctx.dispatch(k, {bufs, nbuf}, push,
                   VulkanContext::matvec_groups(k, w.rows));
    s.end_batch();
    s.ctx.read_buffer(s.xb, std::as_writable_bytes(out));
}

void matvec_batch_vulkan(const Mat& w, std::span<const float> x,
                         std::span<float> out, std::uint32_t n) {
    if (n == 0) {
        return;
    }
    if (matmul_kernel(w) == Kernel::kCount_ &&
        matvec_kernel(w) == Kernel::kCount_) {
        matvec_batch_scalar(w, x, out, n);  // no shader for this type
        return;
    }
    const std::size_t rows = w.rows;
    std::vector<float> tok(rows * n);
    vulkan_linear_tokens(w, x, n, tok);
    // The GPU writes token-major; the matvec_batch contract is
    // row-major (out[r * n + t]).
    for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t r = 0; r < rows; ++r) {
            out[r * n + t] = tok[t * rows + r];
        }
    }
}

void vulkan_linear_tokens(const Mat& w, std::span<const float> x,
                          std::uint32_t n, std::span<float> out) {
    if (n == 0) {
        return;
    }
    const std::size_t rows = w.rows, cols = w.cols;
    if (x.size() < cols * n || out.size() < rows * n) {
        throw std::invalid_argument("vulkan_linear_tokens: short spans");
    }
    if (matmul_kernel(w) == Kernel::kCount_ &&
        matvec_kernel(w) == Kernel::kCount_) {
        for (std::size_t t = 0; t < n; ++t) {  // no shader for this type
            matvec(w, x.subspan(t * cols, cols), out.subspan(t * rows, rows));
        }
        return;
    }
    std::lock_guard lk(state_mu());
    State& s = state();
    auto wb = s.upload_w(w);
    s.grow(s.x, s.x_n, cols * n);
    s.grow(s.xb, s.xb_n, rows * n);
    s.ctx.write_buffer(s.x, std::as_bytes(x.first(cols * n)));
    s.ctx.begin_batch();
    State::BatchScope batch_scope{s};
    rec_matmul(s, w, wb, s.x, s.xb, MatmulRows{n, 0, w.cols, 0, w.rows},
               false);
    s.end_batch();
    s.ctx.read_buffer(s.xb, std::as_writable_bytes(out.first(rows * n)));
}

const Ops& vulkan_image_ops() {
    // Not the vulkan backend entry's Ops: that one leaves matvec_batch
    // null so the language-model paths keep their own batching. No
    // matvec_q8k: the image models do not use it, and the CPU kernel
    // would run on one thread under self_parallel.
    static const Ops ops{.matvec = &matvec_vulkan,
                         .dequant_row = &dequant_row,
                         .matvec_batch = &matvec_batch_vulkan,
                         .self_parallel = true,
                         .alloc_kv = nullptr,
                         .prefetch = nullptr,
                         .mt_safe = false};
    return ops;
}

void vulkan_attention(std::span<const float> q, std::uint32_t nq,
                      std::span<const float> kh, std::span<const float> vt,
                      std::uint32_t nk, std::uint32_t heads, std::uint32_t hd,
                      std::span<float> out) {
    const std::uint32_t w = heads * hd;
    const std::size_t qn = std::size_t{nq} * w, kn = std::size_t{heads} * nk * hd,
                      sn = std::size_t{heads} * nq * nk;
    if (nq == 0 || nk == 0 || q.size() < qn || kh.size() < kn ||
        vt.size() < kn || out.size() < qn || sn > 0xffffffffu) {
        throw std::invalid_argument("vulkan_attention: bad sizes");
    }
    std::lock_guard lk(state_mu());
    State& s = state();
    s.grow(s.img_q, s.img_q_n, qn);
    s.grow(s.img_k, s.img_k_n, kn);
    s.grow(s.img_v, s.img_v_n, kn);
    s.grow(s.img_s, s.img_s_n, sn);
    s.grow(s.img_o, s.img_o_n, qn);
    s.ctx.write_buffer(s.img_q, std::as_bytes(q.first(qn)));
    s.ctx.write_buffer(s.img_k, std::as_bytes(kh.first(kn)));
    s.ctx.write_buffer(s.img_v, std::as_bytes(vt.first(kn)));
    s.ctx.begin_batch();
    State::BatchScope batch_scope{s};
    // S_h[t][j] = K_h[j] . Q[t][h]: K_h ([nk][hd], head h's slice of
    // img_k) as the weight, Q's head-h columns as the token rows.
    for (std::uint32_t h = 0; h < heads; ++h) {
        const Mat km{gguf::TensorType::kF32, nullptr, nk, hd};
        rec_matmul(s, km, s.img_k, s.img_q, s.img_s,
                   MatmulRows{nq, h * hd, w, h * nq * nk, nk}, false,
                   std::uint64_t{h} * nk * hd * 4);
    }
    // Softmax of every row over all nk keys (no mask): the causal
    // shader with heads = rows and base = nk - 1 makes each row's
    // valid span exactly nk.
    rec_softmax(s, s.img_s, heads * nq, nk, heads * nq, nk - 1,
                1.0f / std::sqrt(static_cast<float>(hd)));
    // O[t][h] = V_h^T P_h[t]: V_h^T ([hd][nk]) as the weight.
    for (std::uint32_t h = 0; h < heads; ++h) {
        const Mat vm{gguf::TensorType::kF32, nullptr, hd, nk};
        rec_matmul(s, vm, s.img_v, s.img_s, s.img_o,
                   MatmulRows{nq, h * nq * nk, nk, h * hd, w}, false,
                   std::uint64_t{h} * hd * nk * 4);
    }
    s.end_batch();
    s.ctx.read_buffer(s.img_o, std::as_writable_bytes(out.first(qn)));
}

bool vulkan_imports_host_memory() {
    if (!vulkan_backend_usable()) {
        return false;
    }
    std::lock_guard lk(state_mu());
    return state().ctx.host_import();
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
/**
 * Logs, once per reason, that vulkan_forward_batch declined: the
 * caller then prefills one token at a time (i#321). Called under
 * state_mu.
 * @param why A string literal naming the reason.
 * @returns false, for `return batch_declined(...)`.
 */
bool batch_declined(const char* why) {
    static std::set<const char*> seen;
    if (seen.insert(why).second) {
        std::fprintf(stderr,
                     "[vulkan] batched prefill off (%s): prompts "
                     "prefill one token at a time\n",
                     why);
    }
    return false;
}

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
        hp.n_group > 32 || hp.n_expert > 256 ||
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
        // Experts imported in place (#317: every type but Q8_0's
        // repacked rows) take no device memory of their own.
        std::uint64_t experts = 0;
        for (const auto& l : m.layers()) {
            if (!l.is_moe()) {
                continue;
            }
            for (const auto* em : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
                if (!s.ctx.host_import() ||
                    em->base.type == gguf::TensorType::kQ8_0) {
                    experts += em->expert_bytes * hp.n_expert;
                }
            }
        }
        const std::uint64_t avail = s.ctx.memory_available();
        fit = s.moe_gpu_fits.emplace(&m, vk::moe_experts_fit(experts, avail))
                  .first;
        if (!fit->second) {
            std::fprintf(stderr,
                         "[vulkan] MoE experts (%llu MB) exceed 80%% of "
                         "available device memory (%llu MB): routing "
                         "on the host (a smaller --ctx or --kv-type "
                         "f16 frees room)\n",
                         static_cast<unsigned long long>(experts >> 20),
                         static_cast<unsigned long long>(avail >> 20));
        }
    }
    return fit->second;
}

/** @returns true when every layer has the attention shape the GPU
 * forward implements and every weight it reads has a matvec kernel
 * (fused wkv_b goes through the F32-dequant path). */
bool gpu_weights_ok(const model::TransformerModel& m) {
    const auto& hp = m.hparams();
    const bool mla = hp.arch == model::Arch::kDeepseek2;
    for (const auto& l : m.layers()) {
        // The full GPU forward implements the llama / deepseek2
        // attention plus per-head q/k norms (qwen3moe); q/k/v biases
        // (qwen2moe) and any other extra take the CPU path.
        // q / k / v biases (qwen2moe) are added after the
        // projections; a lone q or k norm is not a known arch.
        if (l.attn_q_norm.empty() != l.attn_k_norm.empty()) {
            return false;
        }
        // Outside MLA it reads separate q, k and v projections of the
        // GQA shapes; anything else (dbrx's fused qkv held in wq, wk
        // and wv empty) is declined rather than admitted by omission.
        if (!mla && !l.has_separate_qkv(hp)) {
            return false;
        }
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

bool vulkan_hparams_supported(const model::Hparams& hp) {
    // rmsnorm only (no LayerNorm: dbrx). DSA indexer models: the
    // 704-float cache row and the top-k selection are CPU-only for
    // now. The attention kernels stage per-head slices in a fixed
    // shared array (vk::kAttnPartMax floats): wider heads, or a
    // larger MLA latent rank, would overrun it.
    // Archs with their own layer body (muse-spark) run on the CPU
    // single-token forward only.
    const bool mla = hp.arch == model::Arch::kDeepseek2;
    return !hp.custom_layer &&
           hp.norm_type == model::NormType::kRmsNorm &&
           hp.idx_top_k == 0 && vk::attn_width_fits(hp.head_dim) &&
           (!mla || vk::attn_width_fits(hp.kv_lora_rank));
}

bool vulkan_forward_supported(const model::TransformerModel& m) {
    return vulkan_hparams_supported(m.hparams()) && gpu_weights_ok(m);
}

bool vulkan_forward(const model::TransformerModel& m, tok::TokenId token,
                    kv::PagedKvCache& cache,
                    kv::PagedKvCache::Seq& seq,
                    std::span<float> logits, std::uint32_t row_begin,
                    std::uint32_t row_end, std::span<const float> embd_in,
                    std::span<float> normed_out) {
    const auto& hp = m.hparams();
    std::lock_guard lk(state_mu());
    State& s = state();
    const bool quant = cache.quantized();
    auto pool_it = s.kv_pools.find(
        quant ? static_cast<const void*>(cache.qpool_data())
              : static_cast<const void*>(cache.pool_data()));
    if (pool_it == s.kv_pools.end()) {
        return false;  // cache pool is not GPU-mapped
    }
    if (!vulkan_forward_supported(m)) {
        return false;
    }
    const bool mla = hp.arch == model::Arch::kDeepseek2;
    if (quant && mla &&
        cache.geometry().kv_type != kv::KvType::kF16) {
        // MLA caches a single latent row (no separate K/V); its Q8/Q4
        // read is a #45 follow-up. Fall back to CPU.
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
    // F16 keeps the F32 pool layout (strides above count halves);
    // only Q8/Q4 take the block-codec path.
    const bool f16 = geom.kv_type == kv::KvType::kF16;
    const bool qblk = quant && !f16;
    if (qblk && (hp.rope_neox ||
                 std::ranges::any_of(m.layers(), [](const auto& l) {
                     return l.has_qk_norm() || l.has_qkv_bias();
                 }))) {
        // The Q8/Q4 KV store path norms and ropes K inside the store:
        // interleaved pairs only, no per-head norm. Those models keep
        // that combination on the CPU.
        return false;
    }

    const std::uint32_t rank = hp.kv_lora_rank;
    const std::uint32_t nope = hp.qk_nope_dim;
    const std::uint32_t rope_d = hp.qk_rope_dim;
    const std::uint32_t vd = hp.v_head_dim;
    const std::uint32_t qk = nope + rope_d;

    s.grow(s.x, s.x_n, hp.n_embd);
    const std::uint32_t ff_max =
        std::max({hp.n_ff, hp.n_ff_exp, shexp_ff(hp)});
    s.grow(s.xb, s.xb_n,
           std::max<std::size_t>(hp.n_embd, ff_max));
    // GQA q / attention output: n_heads * head_dim, which exceeds
    // n_embd when head_dim != n_embd / n_heads (qwen3moe).
    const std::size_t gqa_q =
        std::max<std::size_t>(hp.n_embd, std::size_t{hp.n_heads} *
                                             hp.head_dim);
    s.grow(s.q, s.q_n, mla ? std::size_t{hp.n_heads} * qk : gqa_q);
    // The non-MLA quantized-KV path reuses gate/up as F32 scratch
    // for this token's K and V rows (kv_dim elements each), so they
    // must be at least kv_dim wide, not just ff_max. Every real
    // transformer has n_ff >= n_embd >= kv_dim so this is normally
    // just ff_max, but a model with an inverted ratio would
    // otherwise overrun the FFN scratch with a K/V write -- size
    // defensively.
    // GPU-routed MoE keeps all n_expert_used experts' gate/up rows
    // side by side, then the shared experts' (moe_shared_ok).
    const std::size_t ff_kv = std::max<std::size_t>(
        {ff_max, std::size_t{2} * kv_dim,
         std::size_t{hp.n_expert_used + hp.n_expert_shared} *
             hp.n_ff_exp});
    s.grow(s.gate, s.gate_n, ff_kv);
    s.grow(s.up, s.up_n, ff_kv);
    s.grow(s.attout, s.attout_n,
           mla ? std::size_t{hp.n_heads} * vd : gqa_q);
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

    // CPU side: embedding lookup (or the given input embedding) and
    // this token's block table.
    std::vector<float> x_host(hp.n_embd);
    if (!embd_in.empty()) {
        if (embd_in.size() != hp.n_embd) {
            throw std::invalid_argument(
                "vulkan_forward: an input embedding must be n_embd");
        }
        std::copy(embd_in.begin(), embd_in.end(), x_host.begin());
    } else {
        dequant_row(m.embedding(), static_cast<std::uint32_t>(token),
                    x_host);
    }
    s.ctx.write_buffer(
        s.x, std::as_bytes(std::span<const float>(x_host)));
    s.ctx.write_buffer(
        s.table,
        std::as_bytes(std::span<const kv::BlockId>(seq.blocks)));

    const auto pool = pool_it->second;
    const PagedPool pp{pool, s.table, geom.block_tokens, block_stride,
                       kv_dim, f16};
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

    // out[out_off..] = rmsnorm(in[0, n)) then in[n, n + tail) as is.
    auto norm = [&](std::span<const float> w,
                    VulkanContext::Buffer in,
                    VulkanContext::Buffer out,
                    std::uint32_t n, std::uint32_t out_off,
                    std::uint32_t tail = 0) {
        const VulkanContext::Buffer bufs[] = {
            in, s.upload(w.data(), w.size() * 4), out};
        const std::uint32_t push[] = {n, eps_bits, out_off, tail};
        s.ctx.dispatch(Kernel::kRmsNorm, bufs, push, 1);
    };
    auto swiglu = [&](const Mat& wg, const Mat& wu, const Mat& wd,
                      std::uint32_t ff, float wgt,
                      std::uint64_t off_g, std::uint64_t off_u,
                      std::uint64_t off_d,
                      VulkanContext::Buffer* whole_g,
                      VulkanContext::Buffer* whole_u,
                      VulkanContext::Buffer* whole_d,
                      const Mat* sh_gate = nullptr) {
        Mat g2 = wg, u2 = wu, d2 = wd;
        g2.rows = ff;
        u2.rows = ff;
        d2.cols = ff;
        if (sh_gate != nullptr) {
            // Gated shared expert (qwen2moe): the activation is also
            // scaled by sigmoid(sh_gate . x), its 1-wide logit (in
            // s.router: routing has consumed it by now).
            rec_matvec(s, *sh_gate, s.xb, s.router, 0, false);
            rec_matvec(s, g2, s.xb, s.gate, 0, false, off_g, 0, 1.0f,
                       whole_g);
            rec_matvec(s, u2, s.xb, s.up, 0, false, off_u, 0, 1.0f,
                       whole_u);
            const VulkanContext::Buffer bufs[] = {s.gate, s.up, s.gate,
                                                  s.router};
            const std::uint32_t push[] = {ff, ff};
            s.ctx.dispatch(Kernel::kSiluMulSg, bufs, push,
                           (ff + 63) / 64);
        } else if (wg.type == gguf::TensorType::kQ8_0 &&
            wu.type == gguf::TensorType::kQ8_0 && wg.cols == wu.cols &&
            off_g == off_u) {
            // Gate, up and silu_mul in one fused dispatch.
            const VulkanContext::Buffer bufs[] = {
                whole_g != nullptr ? *whole_g : s.upload_w(g2), s.xb,
                s.gate, whole_u != nullptr ? *whole_u : s.upload_w(u2)};
            const std::uint32_t push[] = {
                ff, wg.cols, 0, 0, w_off_units(wg, off_g), 0,
                fbits(1.0f)};
            s.ctx.dispatch(Kernel::kMatvecQ8_0Glu, bufs, push,
                           VulkanContext::matvec_groups(
                               Kernel::kMatvecQ8_0Glu, ff));
        } else {
            rec_matvec(s, g2, s.xb, s.gate, 0, false, off_g, 0, 1.0f,
                       whole_g);
            rec_matvec(s, u2, s.xb, s.up, 0, false, off_u, 0, 1.0f,
                       whole_u);
            const VulkanContext::Buffer bufs[] = {s.gate, s.up,
                                                  s.gate};
            const std::uint32_t push[] = {ff};
            s.ctx.dispatch(Kernel::kSiluMul, bufs, push,
                           (ff + 63) / 64);
        }
        rec_matvec(s, d2, s.gate, s.x, 0, true, off_d, 0, wgt,
                   whole_d);
    };

    std::vector<float> xb_host(hp.n_embd);
    std::vector<float> router(hp.n_expert);

    s.ctx.begin_batch();
    State::BatchScope batch_scope{s};
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
            const Mat* qkv[] = {&lay.wq, &lay.wk, &lay.wv};
            const bool fused = !qblk && seg_ok(qkv) &&
                               hp.head_dim % 2 == 0;
            // Q4_K q and k (and v when it is Q4_K too) in one dispatch.
            const std::size_t n_q4k_seg =
                fused || qblk ? 0
                              : q4k_seg_count(lay.wq, lay.wk, lay.wv);
            // The seg epilogue ropes interleaved pairs and has no q/k
            // norm, so NEOX / q-k-norm models (qwen3moe) still stack
            // the projection but norm and rope q and k afterwards.
            const bool qk_norm = lay.has_qk_norm();
            const bool bias = lay.has_qkv_bias();
            // Biases land before rope, so no epilogue rope with them.
            const bool epi_rope =
                fused && !hp.rope_neox && !qk_norm && !bias;
            const Kernel rope_k =
                hp.rope_neox ? Kernel::kRopeNeox : Kernel::kRope;
            // q-k-norm models norm and rope q and K in one dispatch,
            // once K has landed (below).
            const bool qk_fused = !epi_rope && qk_norm && !qblk &&
                                  hp.head_dim <=
                                      VulkanContext::kQkNormRopeHdMax;
            // Per-head RMS norm in place: `heads` rows of head_dim
            // floats from `off`, each scaled by the same weights.
            auto head_norm = [&](VulkanContext::Buffer buf,
                                 std::uint32_t off, std::uint32_t heads,
                                 std::span<const float> w) {
                const VulkanContext::Buffer bufs[] = {
                    buf, s.upload(w.data(), w.size() * 4), buf};
                const std::uint32_t push[] = {
                    hp.head_dim, eps_bits, off, off, hp.head_dim,
                    hp.head_dim};
                s.ctx.dispatch(Kernel::kRmsNormB, bufs, push, 1, heads);
            };
            // F32: K/V rows straight into the pool. F16: into F32
            // scratch (K then V in s.gate, free until the FFN), then
            // packed into the pool. With epi_rope the fused projection
            // ropes q and k in its epilogue; otherwise q and K are
            // normed (qk_norm) and roped where they land, below.
            const VulkanContext::Buffer kv_dst = f16 ? s.gate : pool;
            const std::uint32_t k_at = f16 ? 0 : row_off;
            const std::uint32_t v_at =
                f16 ? kv_dim : row_off + layer_stride / 2;
            if (fused) {
                SegRope rope;  // rd 0: no epilogue rope
                if (epi_rope) {
                    // q and k roped in the epilogue (whole heads).
                    rope.hs0 = rope.rd0 = hp.head_dim;
                    rope.hs1 = rope.rd1 = hp.head_dim;
                    rope.pos = pos;
                    rope.base_bits = base_bits;
                    rope.fs_bits = fbits(1.0f);
                    rope.divisors = divisors;
                }
                rec_matvec_seg(s, qkv, s.xb, s.q, 0, kv_dst, k_at, v_at,
                               rope);
            } else if (n_q4k_seg > 0) {
                rec_matvec_q4k_seg(s, {qkv, n_q4k_seg}, s.xb, s.q, 0,
                                   kv_dst, k_at, v_at);
            } else {
                rec_matvec(s, lay.wq, s.xb, s.q, 0, false);
            }
            if (!lay.wq_bias.empty()) {
                rec_bias(s, lay.wq_bias, s.q, 0);
            }
            if (!epi_rope && !qk_fused) {
                if (qk_norm) {
                    head_norm(s.q, 0, hp.n_heads, lay.attn_q_norm);
                }
                const VulkanContext::Buffer bufs[] = {s.q,
                                                      divisors};
                const std::uint32_t push[] = {
                    hp.n_heads, hp.head_dim, pos, 0, hp.head_dim,
                    base_bits,  fbits(1.0f), 0,   0};
                s.ctx.dispatch(rope_k, bufs, push, hp.n_heads);
            }
            if (!qblk) {
                if (!fused && n_q4k_seg == 0) {
                    rec_matvec(s, lay.wk, s.xb, kv_dst, k_at, false);
                }
                if (!fused && n_q4k_seg < 3) {
                    rec_matvec(s, lay.wv, s.xb, kv_dst, v_at, false);
                }
                if (!lay.wk_bias.empty()) {
                    rec_bias(s, lay.wk_bias, kv_dst, k_at);
                }
                if (!lay.wv_bias.empty()) {
                    rec_bias(s, lay.wv_bias, kv_dst, v_at);
                }
                if (qk_fused) {
                    const VulkanContext::Buffer bufs[] = {
                        s.q, kv_dst,
                        s.upload(lay.attn_q_norm.data(),
                                 lay.attn_q_norm.size() * 4),
                        s.upload(lay.attn_k_norm.data(),
                                 lay.attn_k_norm.size() * 4),
                        divisors};
                    const std::uint32_t push[] = {
                        hp.n_heads, hp.head_dim, eps_bits,
                        0,          k_at,        pos,
                        base_bits,  hp.rope_neox ? 1u : 0u};
                    s.ctx.dispatch(Kernel::kQkNormRope, bufs, push,
                                   hp.n_heads + hp.n_kv_heads);
                } else if (!epi_rope) {
                    if (qk_norm) {
                        head_norm(kv_dst, k_at, hp.n_kv_heads,
                                  lay.attn_k_norm);
                    }
                    const VulkanContext::Buffer bufs[] = {
                        kv_dst, divisors};
                    const std::uint32_t push[] = {
                        hp.n_kv_heads, hp.head_dim, pos,
                        k_at,          hp.head_dim, base_bits,
                        fbits(1.0f),   0,           0};
                    s.ctx.dispatch(rope_k, bufs, push, hp.n_kv_heads);
                }
                if (f16) {
                    // K and V rows (planes kv_dim apart) in one go.
                    const VulkanContext::Buffer bufs[] = {s.gate, pool,
                                                          s.table};
                    const std::uint32_t push[] = {
                        kv_dim, pos,           geom.block_tokens,
                        block_stride, kv_dim,  l * layer_stride,
                        0,      kv_dim,        kv_dim,
                        layer_stride / 2};
                    rec_kv_scatter(s, true, bufs, push, 1, 2);
                }
                rec_attn_gqa(s, pp, s.q, s.att, s.attout, 1, pos,
                             hp.n_heads, group, hp.head_dim,
                             hp.n_heads * hp.head_dim, l * layer_stride,
                             layer_stride / 2);
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
            // A Q8_0 fused wkv_b is read as stored (split rows): W_uk^T
            // by the transposed Q8 matvec, W_uv by the batched one.
            // Other types go through a dequantized F32 copy.
            const bool kv_q8 = !split &&
                               lay.wkv_b.type == gguf::TensorType::kQ8_0 &&
                               rank % 128 == 0;
            VulkanContext::Buffer wkv_b32{}, wkv_bq{};
            if (kv_q8) {
                wkv_bq = s.upload_w(lay.wkv_b);
            } else if (!split) {
                wkv_b32 = s.upload_f32(lay.wkv_b);
            }
            const std::uint32_t kv_row_b =
                kv_q8 ? static_cast<std::uint32_t>(mat_row_bytes(lay.wkv_b))
                      : 0;
            const VulkanContext::Buffer divisors =
                s.rope_divisors({}, rope_d / 2);
            if (hp.q_lora_rank > 0) {
                rec_matvec(s, lay.wq_a, s.xb, s.q_a, 0, false);
                norm(lay.q_a_norm, s.q_a, s.q_a,
                     hp.q_lora_rank, 0);
                rec_matvec(s, lay.wq_b, s.q_a, s.q, 0, false);
                rec_matvec(s, lay.wkv_a, s.xb, s.kv_a, 0, false);
            }
            // q | kv_a in one dispatch, q_pe and k_pe roped in its
            // epilogue (rope widths even).
            const Mat* qkv[] = {&lay.wq, &lay.wkv_a};
            const bool fused = hp.q_lora_rank == 0 && seg_ok(qkv) &&
                               rope_d % 2 == 0 && nope % 2 == 0 &&
                               rank % 2 == 0;
            if (fused) {
                SegRope rope;
                rope.hs0 = qk;
                rope.ro0 = nope;
                rope.rd0 = rope_d;
                rope.hs1 = rank + rope_d;
                rope.ro1 = rank;
                rope.rd1 = rope_d;
                rope.pos = pos;
                rope.base_bits = base_bits;
                rope.fs_bits = fs_bits;
                rope.lo_bits = lo_bits;
                rope.hi_bits = hi_bits;
                rope.divisors = divisors;
                rec_matvec_seg(s, qkv, s.xb, s.q, 0, s.kv_a, 0, 0, rope);
            } else if (hp.q_lora_rank == 0) {
                rec_matvec(s, lay.wq, s.xb, s.q, 0, false);
                rec_matvec(s, lay.wkv_a, s.xb, s.kv_a, 0, false);
            }
            // Latent row [norm(c_kv) | k_pe] in one norm (k_pe as its
            // tail, roped by the fused epilogue or just below): into
            // the pool (F32), or into F32 scratch (s.gate, free until
            // the FFN) and packed into the pool once roped (F16).
            const VulkanContext::Buffer lat_dst = f16 ? s.gate : pool;
            const std::uint32_t lat_at = f16 ? 0 : row_off;
            norm(lay.kv_a_norm, s.kv_a, lat_dst, rank, lat_at, rope_d);
            if (!fused) {
                {
                    const VulkanContext::Buffer bufs[] = {lat_dst,
                                                          divisors};
                    const std::uint32_t push[] = {
                        1,       rope_d,  pos,     lat_at + rank,
                        0,       base_bits, fs_bits, lo_bits,
                        hi_bits};
                    s.ctx.dispatch(Kernel::kRope, bufs, push, 1);
                }
                const VulkanContext::Buffer bufs[] = {s.q,
                                                      divisors};
                const std::uint32_t push[] = {
                    hp.n_heads, rope_d,  pos,     nope,
                    qk,         base_bits, fs_bits, lo_bits,
                    hi_bits};
                s.ctx.dispatch(Kernel::kRope, bufs, push,
                               hp.n_heads);
            }
            if (f16) {
                const VulkanContext::Buffer bufs[] = {s.gate, pool,
                                                      s.table};
                const std::uint32_t push[] = {
                    rank + rope_d, pos,      geom.block_tokens,
                    block_stride,  kv_dim,   l * layer_stride,
                    0,             0,        0,
                    0};
                rec_kv_scatter(s, true, bufs, push, 1, 1);
            }
            // Per-head weight absorption, all heads per dispatch
            // where a batched kernel exists (head h = slot h).
            if (kv_q8) {
                const VulkanContext::Buffer bufs[] = {wkv_bq, s.q,
                                                      s.q_abs};
                const std::uint32_t push[] = {
                    nope, rank, 0, 0, 0, (nope + vd) * kv_row_b, qk,
                    rank};
                s.ctx.dispatch(Kernel::kMatvecTQ8, bufs, push, rank / 128,
                               hp.n_heads);
            } else if (!split) {
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
                    s, kb, s.upload_w(lay.wk_b),
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
            rec_attn_mla(s, pp, s.q, s.q_abs, s.att, s.latent, 1, pos,
                         hp.n_heads, rank, nope, rope_d, l * layer_stride,
                         hp.kq_scale);
            if (kv_q8) {
                // W_uv: head h's vd rows start at row h * (nope + vd)
                // + nope of wkv_b.
                Mat uv{gguf::TensorType::kQ8_0, lay.wkv_b.data, vd, rank};
                rec_matvec_b(s, uv, wkv_bq, s.latent, s.attout, 0, false,
                             std::uint64_t{nope} * kv_row_b, 0,
                             std::uint64_t{nope + vd} * kv_row_b, rank, vd,
                             hp.n_heads, 0, s.latent);
            } else if (!split) {
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
                    s, vb, s.upload_w(lay.wv_b),
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
            // Router logits as split-K partials over column chunks
            // (one batched-matvec slot each, so the small router
            // matrix spreads over many workgroups); moe_route sums
            // the planes. Q8_0 rows are not column-contiguous in the
            // GPU layout, so they take one plane.
            const Mat& gi = lay.gate_inp;
            const std::uint32_t chunk = 128;
            const std::uint32_t splits =
                gi.type != gguf::TensorType::kQ8_0 &&
                        batched_kernel(gi) != Kernel::kCount_ &&
                        gi.cols % chunk == 0
                    ? gi.cols / chunk
                    : 1;
            if (splits > 1) {
                Mat part = gi;
                part.cols = chunk;
                const std::uint64_t elt =
                    gi.type == gguf::TensorType::kF32 ? 4 : 2;
                // Slot y: columns y*chunk.., rows still gi.cols apart.
                rec_matvec_b(s, part, s.upload_w(gi), s.xb, s.router, 0,
                             false, 0, 0, chunk * elt, chunk, hp.n_expert,
                             splits, 0, s.xb, gi.cols);
            } else {
                rec_matvec(s, gi, s.xb, s.router, 0, false);
            }
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
                    fbits(hp.expert_weights_scale),
                    splits};
                s.ctx.dispatch(Kernel::kMoeRoute, bufs, push, 1);
            }
            Mat g = lay.gate_exps.expert(0), u = lay.up_exps.expert(0),
                d = lay.down_exps.expert(0);
            g.rows = ff;
            u.rows = ff;
            d.cols = ff;
            const bool sh = moe_shared_ok(hp, lay, g, d);
            const std::uint32_t n_sh = sh ? hp.n_expert_shared : 0;
            Mat gs = lay.gate_shexp, us = lay.up_shexp,
                ds = lay.down_shexp;
            gs.rows = us.rows = hp.n_ff_exp * hp.n_expert_shared;
            ds.cols = gs.rows;
            if (sh) {
                // Routed and shared experts in one dispatch: slots
                // k_used.. run the shared gate / up, ff rows each.
                const VulkanContext::Buffer bufs[] = {
                    s.upload_experts(lay.gate_exps, hp.n_expert), s.xb,
                    s.gate, s.route,
                    s.upload_experts(lay.up_exps, hp.n_expert),
                    s.upload_w(gs), s.upload_w(us)};
                const std::uint32_t push[] = {
                    ff, g.cols, 0, 0, 0, 0, fbits(1.0f),
                    w_off_units(g, lay.gate_exps.expert_bytes), 0, ff,
                    k_used, kRouteIds, 0,
                    static_cast<std::uint32_t>(ff * mat_row_bytes(gs)),
                    0};
                s.ctx.dispatch(Kernel::kMatvecQ8_0GluShB, bufs, push,
                               VulkanContext::matvec_groups(
                                   Kernel::kMatvecQ8_0GluShB, ff),
                               k_used + n_sh);
            } else if (g.type == gguf::TensorType::kQ8_0 &&
                       u.type == gguf::TensorType::kQ8_0 &&
                       lay.gate_exps.expert_bytes ==
                           lay.up_exps.expert_bytes) {
                // Routed gate, up and silu_mul in one dispatch: slot
                // y = picked expert y, its ff outputs at y * ff.
                const VulkanContext::Buffer bufs[] = {
                    s.upload_experts(lay.gate_exps, hp.n_expert), s.xb,
                    s.gate, s.route,
                    s.upload_experts(lay.up_exps, hp.n_expert)};
                const std::uint32_t push[] = {
                    ff, g.cols, 0, 0, 0, 0, fbits(1.0f),
                    w_off_units(g, lay.gate_exps.expert_bytes), 0, ff,
                    k_used, kRouteIds, 0};
                s.ctx.dispatch(Kernel::kMatvecQ8_0GluB, bufs, push,
                               VulkanContext::matvec_groups(
                                   Kernel::kMatvecQ8_0GluB, ff),
                               k_used);
            } else if (g.type == gguf::TensorType::kQ4_K &&
                       u.type == gguf::TensorType::kQ4_K &&
                       lay.gate_exps.expert_bytes ==
                           lay.up_exps.expert_bytes) {
                // The same fusion over Q4_K experts (matvec_q4_k_glu_b):
                // one dispatch, the activations read once.
                const VulkanContext::Buffer bufs[] = {
                    s.upload_experts(lay.gate_exps, hp.n_expert), s.xb,
                    s.gate, s.route,
                    s.upload_experts(lay.up_exps, hp.n_expert)};
                const std::uint32_t push[] = {
                    ff, g.cols, 0, 0, 0, 0, fbits(1.0f),
                    w_off_units(g, lay.gate_exps.expert_bytes), 0, ff,
                    k_used, kRouteIds, 0};
                s.ctx.dispatch(Kernel::kMatvecQ4_KGluB, bufs, push,
                               VulkanContext::matvec_groups(
                                   Kernel::kMatvecQ4_KGluB, ff),
                               k_used);
            } else {
                rec_matvec_b(s, g,
                             s.upload_experts(lay.gate_exps, hp.n_expert),
                             s.xb, s.gate, 0, false, 0, 0,
                             lay.gate_exps.expert_bytes, 0, ff, k_used,
                             kRouteIds, s.route);
                rec_matvec_b(s, u,
                             s.upload_experts(lay.up_exps, hp.n_expert),
                             s.xb, s.up, 0, false, 0, 0,
                             lay.up_exps.expert_bytes, 0, ff, k_used,
                             kRouteIds, s.route);
                const VulkanContext::Buffer bufs[] = {s.gate, s.up,
                                                      s.gate};
                const std::uint32_t push[] = {k_used * ff};
                s.ctx.dispatch(Kernel::kSiluMul, bufs, push,
                               (k_used * ff + 63) / 64);
            }
            if (sh) {
                // Weighted routed sum plus the shared down row.
                const VulkanContext::Buffer bufs[] = {
                    s.upload_experts(lay.down_exps, hp.n_expert), s.gate,
                    s.x, s.route, s.upload_w(ds)};
                const std::uint32_t push[] = {
                    d.rows, ff, 0, 1, 0, 0, fbits(1.0f),
                    w_off_units(d, lay.down_exps.expert_bytes), ff, 0,
                    k_used, kRouteIds | kSlotSum, 0, ds.cols,
                    k_used * ff};
                s.ctx.dispatch(Kernel::kMatvecQ8_0ShB, bufs, push,
                               VulkanContext::matvec_groups(
                                   Kernel::kMatvecQ8_0ShB, d.rows));
            } else {
                rec_matvec_b(s, d,
                             s.upload_experts(lay.down_exps, hp.n_expert),
                             s.gate, s.x, 0, true, 0, 0,
                             lay.down_exps.expert_bytes, ff, 0, k_used,
                             kRouteIds | kSlotSum, s.route);
            }
            if (hp.n_expert_shared > 0 && !sh) {
                swiglu(lay.gate_shexp, lay.up_shexp, lay.down_shexp,
                       shexp_ff(hp), 1.0f, 0, 0, 0, nullptr, nullptr,
                       nullptr,
                       lay.gate_inp_shexp.rows > 0 ? &lay.gate_inp_shexp
                                                   : nullptr);
            }
        } else {
            // Router runs on the CPU: sync the batch, read the
            // normed activations, gate, then keep recording.
            s.end_batch();
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
                    const Mat m = em.expert(e);
                    return s.upload_rows(m.data, m, em.expert_bytes);
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
                swiglu(lay.gate_shexp, lay.up_shexp, lay.down_shexp,
                       shexp_ff(hp), 1.0f, 0, 0, 0, nullptr, nullptr,
                       nullptr,
                       lay.gate_inp_shexp.rows > 0 ? &lay.gate_inp_shexp
                                                   : nullptr);
            }
        }
    }
    norm(m.output_norm(), s.x, s.xb, hp.n_embd, 0);
    const Mat& out_w = m.output_weight();
    if (row_end == 0 || (row_begin == 0 && row_end == out_w.rows)) {
        rec_matvec(s, out_w, s.xb, s.logits, 0, false);
    } else {
        // Only head rows [row_begin, row_end): the same kernel over a
        // row view of the uploaded weights, written at the same logit
        // offsets, so each computed row is bit-identical to the full
        // head's.
        VulkanContext::Buffer whole = s.upload_w(out_w);
        rec_matvec(s, mat_rows(out_w, row_begin, row_end - row_begin),
                   s.xb, s.logits, row_begin, false,
                   std::uint64_t{row_begin} * mat_row_bytes(out_w), 0, 1.0f,
                   &whole);
    }
    s.end_batch();

    s.ctx.read_buffer(s.logits, std::as_writable_bytes(logits));
    if (!normed_out.empty()) {
        s.ctx.read_buffer(
            s.xb, std::as_writable_bytes(normed_out.first(hp.n_embd)));
    }
    s.ctx.note_token();
    seq.n_tokens = pos + 1;
    return true;
}

bool vulkan_forward_multi(const model::TransformerModel& m,
                          std::span<const model::EmbedStep> steps,
                          kv::PagedKvCache& cache, std::uint32_t row_begin,
                          std::uint32_t row_end) {
    const auto& hp = m.hparams();
    const auto n = static_cast<std::uint32_t>(steps.size());
    // Every weight needs a short-batch kernel: otherwise its projection
    // is a mostly-padding matmul tile or one matvec per row, and two
    // vulkan_forward calls (with their fused kernels) are faster.
    const auto short_ok = [n](const Mat& w) {
        return small_batch_kernel(w, n) != Kernel::kCount_;
    };
    if (n == 0 || n > State::kMultiMax || n > kSmallBatch ||
        cache.quantized() || hp.arch == model::Arch::kDeepseek2 ||
        hp.n_expert > 0 || !vulkan_forward_supported(m) ||
        !short_ok(m.output_weight()) ||
        std::ranges::any_of(m.layers(), [&](const auto& l) {
            return l.is_moe() || l.has_qk_norm() || l.has_qkv_bias() ||
                   !short_ok(l.wq) || !short_ok(l.wk) || !short_ok(l.wv) ||
                   !short_ok(l.wo) || !short_ok(l.w_gate) ||
                   !short_ok(l.w_up) || !short_ok(l.w_down);
        })) {
        return false;
    }
    std::lock_guard lk(state_mu());
    State& s = state();
    auto pool_it =
        s.kv_pools.find(static_cast<const void*>(cache.pool_data()));
    if (pool_it == s.kv_pools.end()) {
        return false;  // cache pool is not GPU-mapped
    }
    for (const auto& st : steps) {
        if (st.seq->n_tokens >= hp.n_ctx ||
            st.seq->n_tokens >= cache.capacity(*st.seq)) {
            throw std::invalid_argument("seq capacity not ensured");
        }
    }
    const auto& geom = cache.geometry();
    assert(geom.layer_base == 0 &&
           "vulkan_forward_multi assumes a base-0 full-height KV cache");
    const std::uint32_t E = hp.n_embd, hd = hp.head_dim, H = hp.n_heads;
    const std::uint32_t ff = hp.n_ff;
    const std::uint32_t kv_dim = geom.kv_dim;
    const std::uint32_t q_dim = H * hd;
    const std::uint32_t layer_stride = geom.block_tokens * kv_dim * 2;
    const std::uint32_t block_stride = geom.n_layers * layer_stride;
    const std::uint32_t group =
        hp.n_kv_heads > 0 ? H / hp.n_kv_heads : 1;
    const std::uint32_t n_rows = row_end - row_begin;

    // Rows of all the sequences side by side: residual, normed, q, K
    // then V, FFN, attention output, head rows.
    using C = State::Chunk;
    const auto bx = s.chunk_buf(C::kCX, std::size_t{n} * E);
    const auto bxb = s.chunk_buf(C::kCXb, std::size_t{n} * E);
    const auto bq = s.chunk_buf(C::kCQ, std::size_t{n} * q_dim);
    const auto bkv = s.chunk_buf(C::kCK, std::size_t{2} * n * kv_dim);
    const auto bgate = s.chunk_buf(C::kCGate, std::size_t{n} * ff);
    const auto bup = s.chunk_buf(C::kCUp, std::size_t{n} * ff);
    const auto batt = s.chunk_buf(C::kCAttOut, std::size_t{n} * q_dim);
    const auto blog = s.chunk_buf(C::kCLogits, std::size_t{n} * n_rows);
    // The decode attention reads q and writes its output at offset 0:
    // one sequence at a time goes through these.
    s.grow(s.q, s.q_n, q_dim);
    s.grow(s.attout, s.attout_n, q_dim);
    s.grow(s.att, s.att_n, static_cast<std::size_t>(H) * hp.n_ctx);

    std::vector<float> x_host(std::size_t{n} * E);
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto& st = steps[i];
        auto row = std::span<float>(x_host).subspan(std::size_t{i} * E, E);
        if (!st.embd_in.empty()) {
            std::copy(st.embd_in.begin(), st.embd_in.end(), row.begin());
        } else {
            dequant_row(m.embedding(), static_cast<std::uint32_t>(st.token),
                        row);
        }
        s.grow(s.mtable[i], s.mtable_n[i], cache.total_blocks());
        s.ctx.write_buffer(s.mtable[i],
                           std::as_bytes(std::span<const kv::BlockId>(
                               st.seq->blocks)));
    }
    s.ctx.write_buffer(bx, std::as_bytes(std::span<const float>(x_host)));

    const auto pool = pool_it->second;
    const std::uint32_t eps_bits = fbits(hp.rms_eps);
    const std::uint32_t base_bits = fbits(hp.rope_freq_base);
    const Kernel rope_k = hp.rope_neox ? Kernel::kRopeNeox : Kernel::kRope;
    const VulkanContext::Buffer divisors =
        s.rope_divisors(m.rope_factors(), hd / 2);

    // Every row of `in` RMS-normed into `out` (rows E apart).
    auto norm_rows = [&](std::span<const float> w, VulkanContext::Buffer in,
                         VulkanContext::Buffer out) {
        const VulkanContext::Buffer bufs[] = {
            in, s.upload(w.data(), w.size() * 4), out};
        const std::uint32_t push[] = {E, eps_bits, 0, 0, E, E};
        s.ctx.dispatch(Kernel::kRmsNormB, bufs, push, 1, n);
    };
    // out row i (+)= w x row i, every row in one weight read.
    auto proj = [&](const Mat& w, VulkanContext::Buffer x,
                    std::uint32_t x_stride, VulkanContext::Buffer out,
                    std::uint32_t out_off, std::uint32_t out_stride,
                    bool acc) {
        rec_matmul(s, w, s.upload_w(w), x, out,
                   MatmulRows{n, 0, x_stride, out_off, out_stride}, acc);
    };
    auto copy = [&](VulkanContext::Buffer src, std::uint32_t src_off,
                    VulkanContext::Buffer dst, std::uint32_t dst_off,
                    std::uint32_t count) {
        const VulkanContext::Buffer bufs[] = {src, dst};
        const std::uint32_t push[] = {count, src_off, dst_off};
        s.ctx.dispatch(Kernel::kCopy, bufs, push, (count + 255) / 256);
    };

    s.ctx.begin_batch();
    State::BatchScope batch_scope{s};
    for (std::uint32_t l = 0; l < hp.n_layers; ++l) {
        const auto& lay = m.layers()[l];
        norm_rows(lay.attn_norm, bx, bxb);
        proj(lay.wq, bxb, E, bq, 0, q_dim, false);
        proj(lay.wk, bxb, E, bkv, 0, kv_dim, false);
        proj(lay.wv, bxb, E, bkv, n * kv_dim, kv_dim, false);
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint32_t pos = steps[i].seq->n_tokens;
            {
                const VulkanContext::Buffer bufs[] = {bq, divisors};
                const std::uint32_t push[] = {
                    H,         hd, pos, i * q_dim, hd, base_bits,
                    fbits(1.0f), 0, 0};
                s.ctx.dispatch(rope_k, bufs, push, H);
            }
            {
                const VulkanContext::Buffer bufs[] = {bkv, divisors};
                const std::uint32_t push[] = {
                    hp.n_kv_heads, hd, pos, i * kv_dim, hd, base_bits,
                    fbits(1.0f),   0,  0};
                s.ctx.dispatch(rope_k, bufs, push, hp.n_kv_heads);
            }
            {
                // This sequence's K and V rows (planes n * kv_dim apart)
                // into its own block, through its own table.
                const VulkanContext::Buffer bufs[] = {bkv, pool,
                                                      s.mtable[i]};
                const std::uint32_t push[] = {
                    kv_dim,       pos,        geom.block_tokens,
                    block_stride, kv_dim,     l * layer_stride,
                    i * kv_dim,   kv_dim,     n * kv_dim,
                    layer_stride / 2};
                s.ctx.dispatch(Kernel::kKvScatter, bufs, push, 1, 1, 2);
            }
            copy(bq, i * q_dim, s.q, 0, q_dim);
            const PagedPool pp{pool, s.mtable[i], geom.block_tokens,
                               block_stride, kv_dim, false};
            rec_attn_gqa(s, pp, s.q, s.att, s.attout, 1, pos, H, group, hd,
                         q_dim, l * layer_stride, layer_stride / 2);
            copy(s.attout, 0, batt, i * q_dim, q_dim);
        }
        proj(lay.wo, batt, q_dim, bx, 0, E, true);

        norm_rows(lay.ffn_norm, bx, bxb);
        proj(lay.w_gate, bxb, E, bgate, 0, ff, false);
        proj(lay.w_up, bxb, E, bup, 0, ff, false);
        {
            const VulkanContext::Buffer bufs[] = {bgate, bup, bgate};
            const std::uint32_t push[] = {n * ff};
            s.ctx.dispatch(Kernel::kSiluMul, bufs, push,
                           (n * ff + 63) / 64);
        }
        proj(lay.w_down, bgate, ff, bx, 0, E, true);
    }
    norm_rows(m.output_norm(), bx, bxb);
    // Head rows [row_begin, row_end) of every sequence: a row view of
    // the uploaded head, as vulkan_forward's partial head.
    const Mat& out_w = m.output_weight();
    rec_matmul(s, mat_rows(out_w, row_begin, n_rows), s.upload_w(out_w),
               bxb, blog, MatmulRows{n, 0, E, 0, n_rows}, false,
               std::uint64_t{row_begin} * mat_row_bytes(out_w));
    s.end_batch();

    std::vector<float> logits_host(std::size_t{n} * n_rows);
    std::vector<float> normed_host(std::size_t{n} * E);
    s.ctx.read_buffer(blog, std::as_writable_bytes(std::span(logits_host)));
    s.ctx.read_buffer(bxb, std::as_writable_bytes(std::span(normed_host)));
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto& st = steps[i];
        std::copy_n(logits_host.begin() + std::size_t{i} * n_rows, n_rows,
                    st.logits.begin() + row_begin);
        std::copy_n(normed_host.begin() + std::size_t{i} * E, E,
                    st.normed_out.begin());
        st.seq->n_tokens += 1;
        s.ctx.note_token();
    }
    return true;
}

bool vulkan_forward_batch(const model::TransformerModel& m,
                          std::span<const tok::TokenId> tokens,
                          kv::PagedKvCache& cache,
                          kv::PagedKvCache::Seq& seq,
                          std::span<float> logits, bool all_logits) {
    using C = State::Chunk;
    const auto& hp = m.hparams();
    std::lock_guard lk(state_mu());
    State& s = state();
    const auto n = static_cast<std::uint32_t>(tokens.size());
    const bool f16 = cache.geometry().kv_type == kv::KvType::kF16;
    const bool mla = hp.arch == model::Arch::kDeepseek2;
    // F16 is the one non-F32 pool the batched path reads.
    if (n == 0 || !vulkan_forward_supported(m)) {
        return false;
    }
    if (cache.quantized() && !f16) {
        return batch_declined("quantized KV cache");
    }
    auto pool_it = s.kv_pools.find(
        f16 ? static_cast<const void*>(cache.qpool_data())
            : static_cast<const void*>(cache.pool_data()));
    if (pool_it == s.kv_pools.end()) {
        return batch_declined("KV pool not GPU-mapped");
    }
    const auto& geom = cache.geometry();
    const std::uint32_t kv_dim = geom.kv_dim;
    const std::uint32_t rank = hp.kv_lora_rank;
    const std::uint32_t nope = hp.qk_nope_dim;
    const std::uint32_t rope_d = hp.qk_rope_dim;
    const std::uint32_t vd = hp.v_head_dim;
    const std::uint32_t qk = nope + rope_d;
    const std::uint32_t kvw = rank + rope_d;  // MLA latent row
    if (mla && kv_dim != kvw) {
        return batch_declined("MLA cache row is not rank + rope");
    }
    // Routed experts need GPU routing (all experts resident) and an
    // id-matmul; moe_group handles up to 256 experts.
    for (const auto& lay : m.layers()) {
        if (!lay.is_moe()) {
            continue;
        }
        if (!gpu_route_ok(s, m, lay) || hp.n_expert > 256) {
            return batch_declined("MoE experts routed on the host");
        }
        if (matmul_id_kernel(lay.gate_exps.base) == Kernel::kCount_ ||
            matmul_id_kernel(lay.up_exps.base) == Kernel::kCount_ ||
            matmul_id_kernel(lay.down_exps.base) == Kernel::kCount_) {
            return batch_declined("no expert matmul for a weight type");
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
    const std::uint32_t ff_dense = std::max(hp.n_ff, shexp_ff(hp));
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
        s.chunk_buf(C::kCAtt, std::size_t{attn_block(n, H, n_pos)} * H *
                                  n_pos);
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
    const PagedPool pp{pool, s.table, geom.block_tokens, block_stride,
                       kv_dim, f16};
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
        // GQA rows of NEOX models (qwen3moe) rotate NEOX pairs; MLA's
        // yarn rope is interleaved.
        s.ctx.dispatch(!mla && hp.rope_neox ? Kernel::kRopeNeoxB
                                             : Kernel::kRopeB,
                       bufs, push, heads, n);
    };
    auto scatter_rows = [&](VulkanContext::Buffer src, std::uint32_t width,
                            std::uint32_t dst_off) {
        const VulkanContext::Buffer bufs[] = {src, pool, s.table};
        const std::uint32_t push[] = {width,        base,
                                      geom.block_tokens, block_stride,
                                      kv_dim,       dst_off,
                                      0,            width,
                                      0,            0};
        rec_kv_scatter(s, f16, bufs, push, n, 1);
    };
    auto silu_rows = [&](std::size_t count) {
        const VulkanContext::Buffer bufs[] = {bgate, bup, bgate};
        const std::uint32_t push[] = {static_cast<std::uint32_t>(count)};
        s.ctx.dispatch(Kernel::kSiluMul, bufs, push,
                       static_cast<std::uint32_t>(std::min<std::size_t>(
                           (count + 63) / 64, 65535)));
    };
    // Dense SwiGLU over the normed rows in bxb, accumulated into bx.
    // sh_gate: qwen2moe's gated shared expert, its activation rows
    // also scaled by sigmoid(sh_gate . x) (one logit per token, in
    // the router buffer, free once routing is done).
    auto swiglu_rows = [&](const Mat& wg, const Mat& wu, const Mat& wd,
                           std::uint32_t ff, const Mat* sh_gate = nullptr) {
        Mat g2 = wg, u2 = wu, d2 = wd;
        g2.rows = ff;
        u2.rows = ff;
        d2.cols = ff;
        const MatmulRows in{n, 0, E, 0, ff};
        rec_matmul(s, g2, s.upload_w(g2), bxb, bgate,
                   in, false);
        rec_matmul(s, u2, s.upload_w(u2), bxb, bup,
                   in, false);
        if (sh_gate != nullptr) {
            const auto blogit = s.chunk_buf(
                C::kCRouter, std::size_t{n} * 2 * hp.n_expert);
            rec_matmul(s, *sh_gate, s.upload_w(*sh_gate), bxb, blogit,
                       MatmulRows{n, 0, E, 0, 1}, false);
            const VulkanContext::Buffer bufs[] = {bgate, bup, bgate,
                                                  blogit};
            const std::uint32_t push[] = {n * ff, ff};
            s.ctx.dispatch(Kernel::kSiluMulSg, bufs, push,
                           std::min<std::uint32_t>((n * ff + 63) / 64,
                                                   65535));
        } else {
            silu_rows(std::size_t{n} * ff);
        }
        rec_matmul(s, d2, s.upload_w(wd), bgate, bx,
                   MatmulRows{n, 0, ff, 0, E}, true);
    };
    // q / k / v biases (qwen2): bias added to each of n rows, in place.
    auto bias_rows = [&](std::span<const float> b, VulkanContext::Buffer o,
                         std::uint32_t dim) {
        if (b.empty()) {
            return;
        }
        const VulkanContext::Buffer bufs[] = {
            s.upload(b.data(), b.size() * 4), o};
        const std::uint32_t push[] = {dim, n, 0, dim};
        s.ctx.dispatch(Kernel::kBiasRows, bufs, push,
                       std::min<std::uint32_t>((n * dim + 255) / 256,
                                               65535));
    };

    s.ctx.begin_batch();
    State::BatchScope batch_scope{s};
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
            // Decode's stacked q|k|v buffer when it has one (no
            // second copy of the weights), else one per matrix.
            const Mat* qkv[] = {&lay.wq, &lay.wk, &lay.wv};
            const bool cat = seg_ok(qkv);
            const auto wcat =
                cat ? s.upload_cat(qkv) : VulkanContext::Buffer{};
            const std::uint64_t k_at = weight_bytes(lay.wq);
            const std::uint64_t v_at = k_at + weight_bytes(lay.wk);
            rec_matmul(s, lay.wq, cat ? wcat : s.upload_w(lay.wq), bxb,
                       bq, MatmulRows{n, 0, E, 0, q_dim}, false);
            rec_matmul(s, lay.wk, cat ? wcat : s.upload_w(lay.wk), bxb,
                       bk, MatmulRows{n, 0, E, 0, kv_dim}, false,
                       cat ? k_at : 0);
            rec_matmul(s, lay.wv, cat ? wcat : s.upload_w(lay.wv), bxb,
                       bv, MatmulRows{n, 0, E, 0, kv_dim}, false,
                       cat ? v_at : 0);
            bias_rows(lay.wq_bias, bq, q_dim);
            bias_rows(lay.wk_bias, bk, kv_dim);
            bias_rows(lay.wv_bias, bv, kv_dim);
            if (lay.has_qk_norm()) {
                // Per-head q / k RMS norm (qwen3moe) before rope: n *
                // heads rows of hd floats, in place.
                norm_rows(lay.attn_q_norm, bq, bq, hd, 0, hd, hd, n * H);
                norm_rows(lay.attn_k_norm, bk, bk, hd, 0, hd, hd,
                          n * hp.n_kv_heads);
            }
            rope_rows(bq, divisors, H, hd, 0, hd, q_dim, false);
            rope_rows(bk, divisors, hp.n_kv_heads, hd, 0, hd, kv_dim,
                      false);
            scatter_rows(bk, kv_dim, layer_off);
            scatter_rows(bv, kv_dim, layer_off + layer_stride / 2);
            rec_attn_gqa(s, pp, bq, batt, batt_out, n, base, H, group, hd,
                         q_dim, layer_off, layer_stride / 2);
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
                           s.upload_w(lay.wq_a),
                           bxb, bqa, MatmulRows{n, 0, E, 0, qr}, false);
                norm_rows(lay.q_a_norm, bqa, bqa, qr, 0, qr, qr, n);
                rec_matmul(s, lay.wq_b,
                           s.upload_w(lay.wq_b),
                           bqa, bq, MatmulRows{n, 0, qr, 0, q_dim}, false);
            }
            // Decode's stacked q|kv_a buffer when it has one.
            const Mat* qkv[] = {&lay.wq, &lay.wkv_a};
            const bool cat = hp.q_lora_rank == 0 && seg_ok(qkv);
            const auto wcat =
                cat ? s.upload_cat(qkv) : VulkanContext::Buffer{};
            if (hp.q_lora_rank == 0) {
                rec_matmul(s, lay.wq, cat ? wcat : s.upload_w(lay.wq),
                           bxb, bq, MatmulRows{n, 0, E, 0, q_dim}, false);
            }
            // Latent rows: rms-norm the c_kv part and rope the k_pe
            // part in place, then scatter the rows into the pool.
            rec_matmul(s, lay.wkv_a,
                       cat ? wcat : s.upload_w(lay.wkv_a), bxb, bkva,
                       MatmulRows{n, 0, E, 0, kvw}, false,
                       cat ? weight_bytes(lay.wq) : 0);
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
                           s.upload_w(lay.wk_b),
                           bq, bqabs, absorb, false);
            }
            rec_attn_mla(s, pp, bq, bqabs, batt, blat, n, base, H, rank, nope,
                         rope_d, layer_off, hp.kq_scale);
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
                           s.upload_w(lay.wv_b),
                           blat, batt_out, uv_rows, false);
            }
        }
        rec_matmul(s, lay.wo, s.upload_w(lay.wo),
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
        // Counts, per-expert lists, then moe_group's tile list.
        const auto bids =
            s.chunk_buf(C::kCIds, ne + std::size_t{ne} * n + 1 +
                                      moe_tiles_max(n, k_used, ne));
        const auto bdown =
            s.chunk_buf(C::kCDown, std::size_t{n} * k_used * E);
        rec_matmul(s, lay.gate_inp,
                   s.upload_w(lay.gate_inp),
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
                1u,
                2 * ne,
                2 * k_used};
            s.ctx.dispatch(Kernel::kMoeRouteB, bufs, push, 1, n);
        }
        Mat g = lay.gate_exps.expert(0), u = lay.up_exps.expert(0),
            d = lay.down_exps.expert(0);
        g.rows = ff;
        u.rows = ff;
        d.cols = ff;
        const auto q8 = gguf::TensorType::kQ8_0;
        if (n <= kSmallBatch && g.type == q8 && u.type == q8 &&
            d.type == q8 &&
            lay.gate_exps.expert_bytes == lay.up_exps.expert_bytes) {
            // A short batch (speculative verify): matvecs per (token,
            // expert) pair read each picked expert once, instead of
            // grouping pairs by expert into mostly-padding matmul
            // tiles. Gate / up / silu per pair, then each token's
            // weighted down sum straight into its residual row.
            const std::uint32_t r_stride = 2 * k_used;
            {
                const VulkanContext::Buffer bufs[] = {
                    s.upload_experts(lay.gate_exps, ne), bxb, bgate,
                    broute, s.upload_experts(lay.up_exps, ne)};
                const std::uint32_t push[] = {
                    ff, g.cols, 0, 0, 0, 0, fbits(1.0f),
                    w_off_units(g, lay.gate_exps.expert_bytes), E, ff,
                    0, 0, 0, r_stride, k_used};
                s.ctx.dispatch(Kernel::kMatvecQ8_0GluP, bufs, push,
                               VulkanContext::matvec_groups(
                                   Kernel::kMatvecQ8_0GluP, ff),
                               n * k_used);
            }
            {
                const VulkanContext::Buffer bufs[] = {
                    s.upload_experts(lay.down_exps, ne), bgate, bx,
                    broute};
                const std::uint32_t push[] = {
                    d.rows, ff, 0, 1, 0, 0, fbits(1.0f),
                    w_off_units(d, lay.down_exps.expert_bytes), ff, E,
                    0, 0, 0, r_stride, k_used};
                s.ctx.dispatch(Kernel::kMatvecQ8_0P, bufs, push,
                               VulkanContext::matvec_groups(
                                   Kernel::kMatvecQ8_0P, d.rows),
                               n);
            }
        } else {
            {
                const VulkanContext::Buffer bufs[] = {broute, bids};
                const std::uint32_t push[] = {n, k_used, ne, n,
                                              kMatmulToksId};
                s.ctx.dispatch(Kernel::kMoeGroup, bufs, push, 1);
            }
            rec_matmul_id(s, g, s.upload_experts(lay.gate_exps, ne), bxb,
                          bgate, bids, n, k_used, ne,
                          lay.gate_exps.expert_bytes, E, k_used, ff);
            rec_matmul_id(s, u, s.upload_experts(lay.up_exps, ne), bxb, bup,
                          bids, n, k_used, ne, lay.up_exps.expert_bytes, E,
                          k_used, ff);
            silu_rows(std::size_t{n} * k_used * ff);
            rec_matmul_id(s, d, s.upload_experts(lay.down_exps, ne), bgate,
                          bdown, bids, n, k_used, ne,
                          lay.down_exps.expert_bytes, ff, 1, E);
            const VulkanContext::Buffer bufs[] = {bdown, broute, bx};
            const std::uint32_t push[] = {E, k_used};
            s.ctx.dispatch(Kernel::kMoeCombine, bufs, push, (E + 255) / 256,
                           n);
        }
        if (hp.n_expert_shared > 0) {
            swiglu_rows(lay.gate_shexp, lay.up_shexp, lay.down_shexp,
                        shexp_ff(hp),
                        lay.gate_inp_shexp.rows > 0 ? &lay.gate_inp_shexp
                                                    : nullptr);
        }
    }

    // Prefill needs only the last row's logits; speculative verify
    // needs every row's.
    const Mat& out_w = m.output_weight();
    if (all_logits) {
        norm_rows(m.output_norm(), bx, bxb, E, 0, E, E, n);
        rec_matmul(s, out_w, s.upload_w(out_w), bxb,
                   blogits, MatmulRows{n, 0, E, 0, hp.n_vocab}, false);
    } else {
        norm_rows(m.output_norm(), bx, bxb, E, (n - 1) * E, 0, 0, 1);
        rec_matvec(s, out_w, bxb, blogits, 0, false);
    }
    s.end_batch();

    s.ctx.read_buffer(blogits, std::as_writable_bytes(logits));
    for (std::uint32_t t = 0; t < n; ++t) {
        s.ctx.note_token();
    }
    seq.n_tokens = base + n;
    return true;
}

}  // namespace locus::backend
