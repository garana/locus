#pragma once

#include <cstdint>
#include <memory>

/**
 * A Q4_K x f32 prefill matmul on Metal's simdgroup_matrix (i#799, i#806),
 * run by the Vulkan backend on MoltenVK: the buffers are the Metal
 * buffers behind its VkDeviceMemory, and the command buffers go on the
 * VkQueue's own MTLCommandQueue (both from VK_EXT_metal_objects), so the
 * GPU runs them in submission order with the Vulkan work around them.
 * Only built on Apple (LOCUS_HAS_METAL); elsewhere create() is nullptr.
 */
namespace locus::backend::vk {

/**
 * One matmul: out[t * out_stride + out_off + r] = scale * (W[r] . x[t])
 * (or += with accumulate) for rows r < rows and tokens t < n_tok, where
 * x[t] is cols floats at t * x_stride + x_off. W is rows Q4_K rows of
 * cols / 256 blocks starting w_off bytes into its buffer. Buffers are
 * MTLBuffer ids; offsets other than w_off are in floats.
 */
struct MetalQ4kArgs {
    void* w = nullptr;
    std::uint64_t w_off = 0;
    void* x = nullptr;
    std::uint64_t x_off = 0;
    std::uint32_t x_stride = 0;
    void* out = nullptr;
    std::uint64_t out_off = 0;
    std::uint32_t out_stride = 0;
    std::uint32_t rows = 0, cols = 0, n_tok = 0;
    bool accumulate = false;
    float scale = 1.0f;
    /** The MTLSharedEvent (id) behind the Vulkan timeline semaphore:
     *  the matmul waits for wait_value (the work before it) and then
     *  signals signal_value (for the work after it); queue order alone
     *  does not order Metal and MoltenVK's command buffers. */
    void* event = nullptr;
    std::uint64_t wait_value = 0;
    std::uint64_t signal_value = 0;
};

class MetalMatmul {
public:
    /**
     * @param mtl_device The MTLDevice (id) behind the Vulkan device.
     * @param mtl_queue The MTLCommandQueue (id) behind its VkQueue.
     * @returns The kernel, or nullptr when it does not build (and on
     *     non-Apple builds).
     */
    static std::unique_ptr<MetalMatmul> create(void* mtl_device,
                                               void* mtl_queue);
    virtual ~MetalMatmul() = default;

    /**
     * Encodes one matmul into a new command buffer on the queue and
     * commits it: it runs after everything already submitted to that
     * queue and before anything submitted after this call.
     */
    virtual void commit_q4k(const MetalQ4kArgs& a) = 0;
};

}  // namespace locus::backend::vk
