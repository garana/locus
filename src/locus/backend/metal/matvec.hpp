#pragma once

#include <cstdint>
#include <span>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/metal/context.hpp"
#include "locus/backend/registry.hpp"

namespace locus::backend::metal {

/**
 * @returns Whether the Metal backend has a matvec kernel for weights
 *     of type t (F32, F16, BF16, Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K),
 *     for any shape whose columns are a whole number of t's blocks.
 */
bool matvec_supported(gguf::TensorType t);

/** @returns Whether type t has a kernel and `cols` is a whole number of
 *  its blocks. */
bool matvec_shape_ok(gguf::TensorType t, std::uint32_t cols);

/**
 * out = w x on the GPU (i#866): w's bytes are copied once into a
 * shared Metal buffer, cached by their host pointer, and x and out go
 * through scratch buffers. The cache checks type, shape and a
 * fingerprint of 192 of the bytes, so another matrix at a freed one's
 * address is uploaded afresh; one that differs from the old only
 * outside those bytes would still get them, so pass weights that stay
 * put while in use (mmapped tensors). Calls from several threads run
 * one at a time. Weight types without a kernel run the scalar CPU
 * matvec.
 * @throws std::invalid_argument when x or out has the wrong size.
 */
void matvec(const Mat& w, std::span<const float> x, std::span<float> out);

/**
 * A bench hook: records `reps` copies of w's matvec in one batch (each
 * ordered after the one before, as a forward chains them), submits it
 * and waits.
 * @param out The last copy's result.
 * @returns The submit-to-done wall time per copy, in ms.
 * @throws std::invalid_argument for a type without a kernel, reps 0,
 *     or mismatched spans.
 */
double matvec_repeat(const Mat& w, std::span<const float> x,
                     std::span<float> out, std::uint32_t reps);

/**
 * matvec_batch on the GPU (the Ops contract): w applied to n
 * token-major inputs, out row-major (out[r * n + t]). A matmul where
 * the type has one (record_matmul's), else a matvec a token in one
 * batch; other types run the scalar CPU batch. Weights
 * go through matvec()'s cache, so the same lifetime rule holds.
 */
void matvec_batch(const Mat& w, std::span<const float> x,
                  std::span<float> out, std::uint32_t n);

/**
 * @returns The Ops for the image and encoder models' large passes on
 *     Metal (as vulkan_image_ops): matvec and matvec_batch on the GPU,
 *     self-parallel, not thread-safe.
 */
const Ops& image_ops();

/** Frees every cached weight buffer (tests; the next matvec uploads
 *  again). */
void drop_weights();

/**
 * Frees the cached weights whose bytes start in [p, p + n). Every
 * model file's unmap calls it (sys::add_unmap_hook, set with the first
 * upload), so a dead model's copies leave the device whatever model
 * held them. Takes device_mutex(): a file must not be unmapped by a
 * thread that holds it (inside a metal::Seq's scope). A no-op, making
 * nothing, before any Metal context exists.
 */
void release_weights(const void* p, std::size_t n);

/** @returns The bytes of the cached weight buffers (tests). */
std::size_t cached_weight_bytes();

/**
 * Records out[out_off + r] (+)= dot(row r of the matrix, x[x_off ..])
 * for r < rows into ctx's open batch (or runs it alone): the forward's
 * building block. The caller holds device_mutex().
 * @param t The weight type (matvec_supported).
 * @param w The buffer holding the matrix's bytes from w_off on. Q4_K
 *     loads 8 bytes at a time and F32 float4s, so w_off is a multiple
 *     of 8 for Q4_K and 16 for F32 (a whole number of rows is);
 *     Q8_0 and Q6_K read bytes, any offset.
 * @param x_off Floats into x, a multiple of 4.
 * @param out_off Floats into out.
 * @throws std::invalid_argument for a type without a kernel, columns
 *     that are not whole blocks, or a bad x_off.
 */
void record_matvec(Context& ctx, gguf::TensorType t, std::uint32_t rows,
                   std::uint32_t cols, const Context::Buffer& w,
                   std::uint64_t w_off, const Context::Buffer& x,
                   std::uint32_t x_off, const Context::Buffer& out,
                   std::uint64_t out_off, bool accumulate);

/**
 * out[c] (+)= w x[c] for n activation columns at once (i#1184: one
 * decode token from each of n sequences): each weight block is read
 * once for all of them.
 *
 * A column's bits do not depend on n or on the other columns, so a
 * sequence decodes alike alone or in a batch.
 *
 * @param x_stride, out_stride Floats between columns.
 * @param n The columns, 1 to 8.
 * @param norm_w When set, the RMSNorm (weights norm_w, epsilon eps) as
 *     a prologue, each column over its own x (as MvFuse::norm_w).
 * @param kquant Whether Q6_K may take its columns kernel (from 4
 *     columns, where it beats n matvecs). Its bits differ from the
 *     matvec's, so a caller that must give a sequence the same bits at
 *     any n passes false. Q4_K has none (no form beat its matvecs).
 * @returns false when this type or shape has no such kernel (the
 *     caller then runs n matvecs).
 */
bool record_matvec_cols(Context& ctx, gguf::TensorType t,
                        std::uint32_t rows, std::uint32_t cols,
                        const Context::Buffer& w, std::uint64_t w_off,
                        const Context::Buffer& x, std::uint32_t x_off,
                        std::uint32_t x_stride, const Context::Buffer& out,
                        std::uint64_t out_off, std::uint32_t out_stride,
                        std::uint32_t n, bool accumulate,
                        const Context::Buffer* norm_w = nullptr,
                        float eps = 0.0f, bool kquant = false);

/** The steps a matvec can take around its dot (i#1064), so the decode
 *  saves their dispatches and barriers. */
struct MvFuse {
    /** rope_norm in the store: rows 2 i and 2 i + 1 of each
     *  rope_hd-wide head rotate by rope_cs[2 i] (cos) and
     *  rope_cs[2 i + 1] (sin), rope_norm's arithmetic on the same
     *  values, so its bits. Q4_K only (ggml's layout gives a simdgroup
     *  each even row and its pair). */
    const Context::Buffer* rope_cs = nullptr;
    std::uint32_t rope_hd = 0;
    /** RMSNorm as a prologue: x is the norm's input, read as x * norm_w,
     *  its x^2 summed over the row, the dot scaled by 1 / sqrt(mean +
     *  eps). The rounding order differs from rmsnorm_f32 then the
     *  matvec, so the low bits do. matvec_prenorm_ok types only. */
    const Context::Buffer* norm_w = nullptr;
    float eps = 0.0f;
};

/**
 * Batch-invariant Metal decode (i#1184, as Vulkan's #432): when on, a
 * lone sequence also decodes through metal_forward_multi, so a
 * request's logits are bit-identical alone or batched; off, it takes
 * metal_forward. LOCUS_METAL_BATCH_INVARIANT=1 / 0, or this call, sets
 * it; unset, the model decides
 * (TransformerModel::metal_batch_invariant).
 */
void set_batch_invariant(bool on);
/** Drops a set_batch_invariant setting: back to the
 * LOCUS_METAL_BATCH_INVARIANT or the model's default. */
void reset_batch_invariant();
/** @returns 1 or 0 when the call or LOCUS_METAL_BATCH_INVARIANT set
 *  the mode, -1 when neither did. */
int batch_invariant_setting();

/** @returns Whether t's matvec takes MvFuse::norm_w. */
bool matvec_prenorm_ok(gguf::TensorType t);

/**
 * record_matvec, not accumulated, with fuse's steps.
 * @returns false, recording nothing, when t (or the shape) lacks one of
 *     them: the caller then runs the steps itself.
 */
bool record_matvec_fused(Context& ctx, gguf::TensorType t,
                         std::uint32_t rows, std::uint32_t cols,
                         const Context::Buffer& w, std::uint64_t w_off,
                         const Context::Buffer& x, std::uint32_t x_off,
                         const Context::Buffer& out, std::uint64_t out_off,
                         const MvFuse& fuse);

/**
 * Records out[out_off + t * out_stride + r] (+)= dot(row r, x token t)
 * for r < rows and t < n_tok: the prefill matmul, each weight row read
 * once for a tile of 32 tokens. Token t of x starts at x_off + t *
 * x_stride (floats); the products are times scale. The caller holds
 * device_mutex().
 * @returns false, recording nothing, for a type without a matmul
 *     kernel (Q4_K, Q6_K, Q8_0, Q4_0, Q5_0, F32, F16, BF16) or
 *     columns that are not whole k stages: the caller loops
 *     record_matvec.
 */
bool record_matmul(Context& ctx, gguf::TensorType t, std::uint32_t rows,
                   std::uint32_t cols, std::uint32_t n_tok,
                   const Context::Buffer& w, std::uint64_t w_off,
                   const Context::Buffer& x, std::uint64_t x_off,
                   std::uint32_t x_stride, const Context::Buffer& out,
                   std::uint64_t out_off, std::uint32_t out_stride,
                   bool accumulate, float scale = 1.0f);

/**
 * The device copy of w, from the matvec weight cache (uploaded on
 * first use, keyed by its address and a fingerprint of its bytes). The
 * caller holds device_mutex(); the buffer lives until drop_weights().
 * @throws std::invalid_argument for a type without a matvec kernel.
 */
const Context::Buffer& cached_weight(Context& ctx, const Mat& w);

/** The tiles moe_tiles may write for n_slots slots of n_expert. */
std::uint32_t moe_max_tiles(std::uint32_t n_slots, std::uint32_t n_expert);

/**
 * Records moe_tiles: the n_slots slots of route[0 .. n_slots) (expert
 * ids) grouped by expert into idx (n_slots + 3 moe_max_tiles uints),
 * for record_matmul_id. n_expert is at most 1024.
 */
void record_moe_tiles(Context& ctx, const Context::Buffer& route,
                      std::uint32_t n_slots, std::uint32_t n_expert,
                      const Context::Buffer& idx);

/**
 * The grouped matmul (MoE prefill): for each slot s of n_slots, out
 * row s (out_stride floats apart) = expert route[s]'s matrix (at w_off
 * + e * expert_bytes) times x row s / slot_div, each expert's weight
 * rows read once for a tile of 32 of its slots. idx is
 * record_moe_tiles' output.
 * @returns false, recording nothing, where record_matmul would.
 */
bool record_matmul_id(Context& ctx, gguf::TensorType t, std::uint32_t rows,
                      std::uint32_t cols, std::uint32_t n_slots,
                      std::uint32_t n_expert, const Context::Buffer& w,
                      std::uint64_t w_off, std::uint64_t expert_bytes,
                      const Context::Buffer& x, std::uint32_t x_stride,
                      std::uint32_t slot_div, const Context::Buffer& out,
                      std::uint32_t out_stride, const Context::Buffer& idx);

/**
 * record_matvec_routed for n_slots experts in one dispatch (MoE's k
 * picks, i#868): slot s multiplies the expert at route[route_slot +
 * s] (from w_off, expert_bytes apart) by x at x_off + (s /
 * slot_div) * x_slot_stride into out at out_off + s *
 * out_slot_stride. slot_div k lets each token's k picks share its x
 * row (MoE prefill).
 * @throws std::invalid_argument for a type without a kernel, an x
 *     stride that is not a multiple of 4 or a zero slot_div.
 */
void record_matvec_slots(Context& ctx, gguf::TensorType t,
                         std::uint32_t rows, std::uint32_t cols,
                         const Context::Buffer& w, std::uint64_t w_off,
                         std::uint64_t expert_bytes,
                         const Context::Buffer& route,
                         std::uint32_t route_slot, std::uint32_t n_slots,
                         const Context::Buffer& x, std::uint32_t x_off,
                         std::uint32_t x_slot_stride,
                         const Context::Buffer& out, std::uint64_t out_off,
                         std::uint32_t out_slot_stride,
                         std::uint32_t slot_div = 1);

/**
 * record_matvec for a routed expert (MoE on the GPU, i#868): the
 * matrix starts at w_off + route[slot] * expert_bytes, the expert id
 * read on the GPU from the route buffer (moe_route's output), so no
 * host round trip picks it.
 */
void record_matvec_routed(Context& ctx, gguf::TensorType t,
                          std::uint32_t rows, std::uint32_t cols,
                          const Context::Buffer& w, std::uint64_t w_off,
                          std::uint64_t expert_bytes,
                          const Context::Buffer& route, std::uint32_t slot,
                          const Context::Buffer& x, std::uint32_t x_off,
                          const Context::Buffer& out, std::uint64_t out_off,
                          bool accumulate);

/** @returns The bytes of a rows x cols matrix of type t (0 for a type
 *  without a kernel). */
std::size_t matvec_weight_bytes(gguf::TensorType t, std::uint32_t rows,
                                std::uint32_t cols);

}  // namespace locus::backend::metal
