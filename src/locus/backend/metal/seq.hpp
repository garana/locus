#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "locus/backend/cpu_ops.hpp"

namespace locus::backend::metal {

/**
 * VulkanSeq's interface on Metal (i#868): a sequence of GPU ops on
 * resident activations in numbered scratch slots, for the image
 * transformers whose whole step runs on the GPU (FLUX, Qwen-Image,
 * PixArt, the Janus vision tower). Weights go through the matvec
 * weight cache; biases and norm weights are copied per batch, so they
 * may live anywhere.
 *
 * Offsets, strides and sizes count floats; rows are token-major. Ops
 * record into an open batch; flush() submits it and waits. write() and
 * read() flush first, so they see every op recorded before them.
 *
 * Holds the Metal device lock for its lifetime except inside flush()
 * (and write(), read() and a growing reserve(), which flush): between
 * two batches it lets a Metal call waiting on another thread run, so
 * text decode is not held off for a whole image step (i#1057). Create
 * one per step, on one thread, and keep other Metal calls on that
 * thread out of its scope.
 */
class Seq {
  public:
    /** A scratch buffer index, below kSlots. */
    using Slot = std::uint32_t;
    static constexpr Slot kSlots = 20;

    /** @returns Whether a Metal device is usable. */
    static bool available();

    /** Takes the device lock and opens a batch.
     *  @throws std::runtime_error when Metal is not usable. */
    Seq();
    /** Drops an unflushed batch unrun (its results were never read)
     *  and releases the lock; never throws. */
    ~Seq();
    Seq(const Seq&) = delete;
    Seq& operator=(const Seq&) = delete;

    /** Makes slot s hold at least n floats (contents kept only when it
     *  already did). */
    void reserve(Slot s, std::size_t n);
    /** Copies v into slot s at off. */
    void write(Slot s, std::size_t off, std::span<const float> v);
    /** Copies out.size() floats from slot s at off. */
    void read(Slot s, std::size_t off, std::span<float> out);
    /** Submits the recorded ops and waits for them. */
    void flush();

    /** A token-row operand: rows at off + t * stride in slot. */
    struct Rows {
        Slot slot;
        std::uint32_t off = 0, stride = 0;
    };

    /**
     * out[t] = w x[t] + b for n token rows (b may be empty): a matmul
     * where the type has one, else a matvec a row.
     * @throws std::invalid_argument when b is neither empty nor w.rows
     *     long, w's type has no Metal kernel, or the matvec fallback
     *     gets an x offset or stride that is not a multiple of 4.
     */
    void linear(const Mat& w, std::span<const float> b, Rows x, Rows out,
                std::uint32_t n);

    /**
     * out[t] += scale * w x[t] for n token rows (a LoRA's B side, of
     * any rank: a matmul where the shape has one, else a thread an
     * output for F32, F16 and BF16).
     * @throws std::invalid_argument for a type with neither.
     */
    void linear_add(const Mat& w, Rows x, Rows out, std::uint32_t n,
                    float scale);

    /** LayerNorm without affine over n rows of width d, then
     *  out = ln(x) * (1 + scale) + shift from slot mod. */
    void ln_mod(Rows x, Rows out, std::uint32_t n, std::uint32_t d,
                Slot mod, std::uint32_t shift_off, std::uint32_t scale_off,
                float eps);

    /** Per-head RMS norm (hd floats of norm) then RoPE in place: pair
     *  (2 i, 2 i + 1) of token t by the cos / sin rows rope_row0 + t of
     *  slots cos_s and sin_s (hd floats a row). */
    void head_norm_rope(Rows x, std::uint32_t n, std::uint32_t heads,
                        std::uint32_t hd, std::span<const float> norm,
                        Slot cos_s, Slot sin_s, std::uint32_t rope_row0,
                        float eps);

    /**
     * Unmasked multi-head attention, n queries over n keys and values:
     * softmax(q k^T / sqrt(hd)) v into out, on the simdgroup matrices.
     * kh, vt and scores are VulkanSeq's scratch slots, unused here.
     * @throws std::invalid_argument unless hd is a multiple of 8 up to
     *     128.
     */
    void attention(Rows q, Rows k, Rows v, Rows out, std::uint32_t n,
                   std::uint32_t heads, std::uint32_t hd, Slot kh, Slot vt,
                   Slot scores);

    /** attention() with n_q queries over n_kv keys and values. */
    void cross_attention(Rows q, std::uint32_t n_q, Rows k, Rows v,
                         std::uint32_t n_kv, Rows out, std::uint32_t heads,
                         std::uint32_t hd, Slot kh, Slot vt, Slot scores);

    /** torch's tanh GELU in place on n rows of width d. */
    void gelu_tanh(Rows x, std::uint32_t n, std::uint32_t d);

    /** The exact (erf) GELU in place on n rows of width d. */
    void gelu_erf(Rows x, std::uint32_t n, std::uint32_t d);

    /**
     * LayerNorm with weight and bias (d floats each) over n rows.
     * @throws std::invalid_argument when w or b is not d long.
     */
    void layer_norm(Rows x, Rows out, std::uint32_t n, std::uint32_t d,
                    std::span<const float> w, std::span<const float> b,
                    float eps);

    /** x[t][i] += y[t][i] on n rows of width d. */
    void add(Rows x, Rows y, std::uint32_t n, std::uint32_t d);

    /** x[t][i] += gate[i] * y[t][i]; gate at gate_off in slot gate_s. */
    void gated_add(Rows x, Rows y, std::uint32_t n, std::uint32_t d,
                   Slot gate_s, std::uint32_t gate_off);

  private:
    struct Impl;
    std::unique_ptr<Impl> im_;
};

}  // namespace locus::backend::metal
