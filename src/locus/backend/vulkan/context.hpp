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
    kMatvecIQ3_XXS,
    kMatvecIQ4_XS,
    kMatvecT,
    kRmsNorm,
    kRope,
    kSiluMul,
    kAttnPagedQ,
    kKvStoreQ,
    /** Batched (-DBATCHED) matvecs: one dispatch, several matrices. */
    kMatvecF32B,
    kMatvecF16B,
    kMatvecQ8_0B,
    kMoeRoute,
    /** Prefill (batched forward over a chunk of tokens). */
    kMatmulQ8_0,
    kMatmulF32,
    kMatmulQ8_0Id,
    kKvScatter,
    kMoeGroup,
    kMoeCombine,
    /** -DBATCHED variants: workgroup y = token row of the chunk. */
    kRmsNormB,
    kRopeB,
    kMoeRouteB,
    /** Attention as matmuls: W read from the paged KV pool. */
    kMatmulPagedF32,
    kMatmulPagedTF32,
    kAttnSoftmax,
    kSumSplits,
    kMatvecTQ8,
    /** Fused SwiGLU Q8_0 matvecs: silu(W x) * (WU x). */
    kMatvecQ8_0Glu,
    kMatvecQ8_0GluB,
    /** Decode (one query token) flash attention and its merge. */
    kAttnDecode,
    kAttnCombine,
    /** Q8_0 matvec over stacked matrices, rows to two outputs. */
    kMatvecQ8_0Seg,
    /** MoE batched matvecs with the shared experts as extra work. */
    kMatvecQ8_0ShB,
    kMatvecQ8_0GluShB,
    /** attn_decode reading each K/V row once for a head group. */
    kAttnDecodeGrp,
    /** Prefill matmuls with f16 operand staging (--prefill-f16). */
    kMatmulQ8_0H,
    kMatmulF32H,
    kMatmulQ8_0IdH,
    kMatmulPagedF32H,
    kMatmulPagedTF32H,
    /** Q8_0 matvec over a short batch of token rows (<= 8). */
    kMatvecQ8_0N,
    /** Short-batch routed MoE: (token, expert) pair matvecs. */
    kMatvecQ8_0P,
    kMatvecQ8_0GluP,
    /** F32 counterpart of kMatvecQ8_0N. */
    kMatvecF32N,
    kCount_,
};

/**
 * Synchronous Vulkan compute context: one device, one compute
 * queue, one pipeline per Kernel, SPIR-V embedded at compile time
 * (glslangValidator --vn, per the ../pbw pattern).
 *
 * Work is recorded in batches: begin_batch(), any number of
 * dispatch() calls (a memory barrier separates consecutive
 * dispatches, except where the driver serializes them anyway --
 * see serial_dispatch()), end_batch() submits once and waits. The full GPU
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

    /**
     * Per-process batch timing, collected when LOCUS_VK_PROFILE is
     * set (GPU time via timestamp queries). host_ms covers
     * begin_batch .. end_batch on the CPU (recording plus any host
     * work between), wait_ms the submit + vkQueueWaitIdle, gpu_ms
     * the device time between the batch's first and last command.
     * With LOCUS_VK_PROFILE=2, kernel_ms splits that per kernel (a
     * timestamp after each dispatch, which the inter-dispatch
     * barrier serializes; costs host time on some drivers).
     */
    struct Profile {
        std::uint64_t tokens = 0;
        std::uint64_t batches = 0;
        std::uint64_t dispatches = 0;
        double host_ms = 0.0;
        double wait_ms = 0.0;
        double gpu_ms = 0.0;
        double kernel_ms[static_cast<int>(Kernel::kCount_)] = {};
        std::uint64_t kernel_n[static_cast<int>(Kernel::kCount_)] =
            {};
    };

    /**
     * Workgroup count covering `rows` output rows of a matvec kernel
     * (kMatvec*, not kMatvecT): row-parallel kernels map 4 rows to a
     * workgroup, the older one-thread-per-row kernels 64.
     *
     * @returns The X group count to pass to dispatch().
     */
    static std::uint32_t matvec_groups(Kernel k, std::uint32_t rows);

    /** @returns true when LOCUS_VK_PROFILE is set. */
    static bool profiling();

    /**
     * @returns The count of Vulkan validation errors seen so far in
     * this process. Nonzero only when LOCUS_VK_VALIDATION enabled the
     * Khronos validation layer (with GPU-assisted validation) at
     * instance creation; 0 otherwise and in a build without kernels.
     * Used by the out-of-bounds (F2) regression test.
     */
    static std::uint32_t validation_errors();

    /**
     * matmul.comp tile: weight rows and tokens per workgroup (the
     * shader's BM = 16 TM and BN = 16 TN; keep them in sync).
     */
    static constexpr std::uint32_t kMatmulTileRows = 128;
    static constexpr std::uint32_t kMatmulTileToks = 64;

    /**
     * matmul.comp columns staged per K step (the shader's BK). The
     * default tile's shared stage is 25088 bytes; on a device whose
     * maxComputeSharedMemorySize cannot fit it, the context builds the
     * matmul pipelines from the BK16 variant instead (12800 bytes),
     * which keeps the GPU prefill path on tight devices (e.g. V3DV,
     * 16384). These must match matmul.comp's BK default and BK16
     * override.
     */
    static constexpr std::uint32_t kMatmulBK = 32;
    static constexpr std::uint32_t kMatmulBKSmall = 16;

    /**
     * Device memory this context can still allocate, per
     * memory_available_from() (mem_fit.hpp): VK_EXT_memory_budget
     * when the device has it, else 3/4 of the heap minus this
     * context's allocations (the heap backing its host-visible
     * buffers).
     *
     * @returns Available bytes (0 when exhausted).
     */
    std::uint64_t memory_available() const;

    /**
     * Whether dispatch() skips the barrier between dispatches
     * because the driver already serializes them (MoltenVK; unless
     * LOCUS_VK_BARRIERS is set).
     *
     * @returns true when barriers are elided.
     */
    bool serial_dispatch() const;

    /**
     * Whether the device does f16 shader arithmetic (shaderFloat16,
     * enabled at device creation): the kernels that need it
     * (kernel_needs_f16) exist only then.
     *
     * @returns true when the f16 kernels can be dispatched.
     */
    bool has_f16() const;

    /** @returns Whether kernel k needs has_f16(). */
    static bool kernel_needs_f16(Kernel k);

    /** Counts one forwarded token toward the profile. */
    void note_token();

    /** @returns The profile collected so far. */
    const Profile& profile() const;

    /** Prints the per-token profile to stderr (when profiling). */
    void report_profile() const;

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
     * @param groups_y Y workgroup count (batched slots / heads).
     */
    void dispatch(Kernel k, std::span<const Buffer> buffers,
                  std::span<const std::uint32_t> push,
                  std::uint32_t groups_x, std::uint32_t groups_y = 1);

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
