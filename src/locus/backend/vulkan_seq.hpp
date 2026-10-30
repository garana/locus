#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "locus/backend/cpu_ops.hpp"

namespace locus::backend {

/**
 * A sequence of GPU ops on resident activations, for the image
 * transformers whose whole step runs on the GPU (FLUX, i#372). The
 * activations live in numbered scratch slots between calls, so a step
 * uploads its inputs once and reads back only its output; weights go
 * through the Vulkan weight pool (imported in place from the GGUF
 * mapping where the device can).
 *
 * Offsets, strides and sizes count floats; rows are token-major. Ops
 * record into an open command batch; flush() submits it and waits.
 * write() and read() flush first, so they see every op recorded before
 * them. Biases and norm weights are copied per batch, so they may live
 * anywhere.
 *
 * Holds the Vulkan backend's lock for its lifetime: create one per
 * step, on one thread, and keep other Vulkan calls out of its scope.
 */
class VulkanSeq {
  public:
    /** A scratch buffer index, below kSlots. */
    using Slot = std::uint32_t;
    static constexpr Slot kSlots = 16;

    /** @returns Whether the Vulkan backend is usable. */
    static bool available();

    /** Takes the backend lock and opens a batch.
     *  @throws std::runtime_error when Vulkan is not usable. */
    VulkanSeq();
    /** Submits nothing: drops an unflushed batch, releases the lock. */
    ~VulkanSeq();
    VulkanSeq(const VulkanSeq&) = delete;
    VulkanSeq& operator=(const VulkanSeq&) = delete;

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
     * out[t] = w x[t] + b for n token rows (b may be empty).
     * @throws std::invalid_argument when b is neither empty nor w.rows
     *     long.
     */
    void linear(const Mat& w, std::span<const float> b, Rows x, Rows out,
                std::uint32_t n);

    /**
     * LayerNorm without affine over n rows of width d, then
     * out = ln(x) * (1 + scale) + shift, scale and shift read from slot
     * mod at the given offsets.
     */
    void ln_mod(Rows x, Rows out, std::uint32_t n, std::uint32_t d,
                Slot mod, std::uint32_t shift_off, std::uint32_t scale_off,
                float eps);

    /**
     * Per-head RMS norm (weight norm, hd floats) then RoPE in place on
     * n rows of heads x hd: pair (2 i, 2 i + 1) of token t rotates by
     * the angle whose cosine and sine sit in slots cos_s and sin_s at
     * row rope_row0 + t (hd floats per row).
     */
    void head_norm_rope(Rows x, std::uint32_t n, std::uint32_t heads,
                        std::uint32_t hd, std::span<const float> norm,
                        Slot cos_s, Slot sin_s, std::uint32_t rope_row0,
                        float eps);

    /**
     * Unmasked multi-head attention: n queries over n keys and values,
     * all rows heads x hd, softmax(q k^T / sqrt(hd)) v into out. Slots
     * kh and vt hold the per-head K and V^T planes it builds, scores
     * the score blocks (queries run in blocks within the attention
     * scratch cap).
     */
    void attention(Rows q, Rows k, Rows v, Rows out, std::uint32_t n,
                   std::uint32_t heads, std::uint32_t hd, Slot kh, Slot vt,
                   Slot scores);

    /** torch's tanh GELU in place on n rows of width d. */
    void gelu_tanh(Rows x, std::uint32_t n, std::uint32_t d);

    /** x[t][i] += gate[i] * y[t][i] on n rows of width d; gate at
     *  gate_off in slot gate_s. */
    void gated_add(Rows x, Rows y, std::uint32_t n, std::uint32_t d,
                   Slot gate_s, std::uint32_t gate_off);

  private:
    struct Impl;
    std::unique_ptr<Impl> im_;
};

}  // namespace locus::backend
