#pragma once

#include <cstdint>
#include <memory>
#include <span>

namespace locus::backend::vk {

/** Compute kernels compiled into the binary as SPIR-V. */
enum class Kernel {
    kMatvecF32,
    kMatvecF16,
    kMatvecQ8_0,
    kMatvecQ4_0,
    kMatvecQ5_0,
    kMatvecQ4_K,
    kMatvecQ5_K,
    kMatvecQ6_K,
    kMatvecQ2_K,
    kMatvecIQ2_XXS,
    kMatvecT,
    kRmsNorm,
    kRope,
    kSiluMul,
    kAttnPaged,
    kAttnMla,
    kCount_,
};

/**
 * Synchronous Vulkan compute context: one device, one compute
 * queue, one pipeline per Kernel, SPIR-V embedded at compile time
 * (glslangValidator --vn, per the ../pbw pattern).
 *
 * Work is recorded in batches: begin_batch(), any number of
 * dispatch() calls (a memory barrier separates consecutive
 * dispatches), end_batch() submits once and waits. The full GPU
 * forward records one batch per token, which is what makes the
 * backend usable -- a submit per op was ~20x slower.
 *
 * Buffers are host-visible and coherent; on Apple unified memory
 * that is also the device-fast path. Only available when the
 * build found the Vulkan SDK and a shader compiler
 * (LOCUS_HAS_VULKAN_KERNELS); otherwise construction throws
 * std::runtime_error.
 */
class VulkanContext {
  public:
    /** @returns true when built with kernels and a device exists. */
    static bool available();

    /** @throws std::runtime_error when no usable device. */
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    /**
     * Opaque handle to a device buffer (host-visible; on Apple
     * unified memory this is also device-fast). Buffers live
     * until destroy_buffer or context destruction.
     */
    struct Buffer {
        void* impl = nullptr;
    };

    /** @throws std::runtime_error when allocation fails. */
    Buffer create_buffer(std::size_t bytes);
    void destroy_buffer(Buffer b);
    void write_buffer(Buffer b, std::span<const std::byte> data);
    void read_buffer(Buffer b, std::span<std::byte> out);

    /** @returns The host mapping of b (coherent; UMA-fast). */
    void* mapped(Buffer b);

    /** Opens a command batch; dispatches record until end_batch. */
    void begin_batch();

    /**
     * Records one compute dispatch into the open batch.
     *
     * @param k Kernel to run; buffer count must match its layout.
     * @param push Raw push constants (floats bit-cast to uint32).
     * @param groups_x X workgroup count.
     */
    void dispatch(Kernel k, std::span<const Buffer> buffers,
                  std::span<const std::uint32_t> push,
                  std::uint32_t groups_x);

    /** Submits the batch and waits for completion. */
    void end_batch();

    /**
     * Records a buffer copy into the open batch, fenced against
     * surrounding compute dispatches.
     */
    void copy_buffer(Buffer src, std::size_t src_off, Buffer dst,
                     std::size_t dst_off, std::size_t bytes);

    /**
     * GPU f32 matvec over resident buffers:
     * out[r] = dot(w row r, x). One-dispatch batch (synchronous);
     * persistent weights avoid per-call uploads.
     */
    void matvec_f32(Buffer w, std::uint32_t rows,
                    std::uint32_t cols, Buffer x, Buffer out);

    /**
     * Convenience overload copying host spans through scratch
     * buffers (tests/one-shot use).
     */
    void matvec_f32(std::span<const float> w, std::uint32_t rows,
                    std::uint32_t cols, std::span<const float> x,
                    std::span<float> out);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace locus::backend::vk
