#pragma once

#include <cstdint>
#include <memory>
#include <span>

namespace cppllm::backend::vk {

/**
 * Minimal synchronous Vulkan compute context: one device, one
 * compute queue, pipelines built from SPIR-V embedded at compile
 * time (glslangValidator --vn, per the ../pbw pattern).
 *
 * This is the M5 foundation: correctness-first, host-visible
 * buffers and a wait-idle per dispatch. Persistent device-local
 * weight buffers and batched command recording come with the full
 * GPU forward pass.
 *
 * Only available when the build found the Vulkan SDK and shader
 * compiler (CPPLLM_HAS_VULKAN_KERNELS); otherwise construction
 * throws std::runtime_error.
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
     * GPU f32 matvec: out[r] = dot(w row r, x).
     *
     * @param w Row-major rows*cols floats.
     * @throws std::runtime_error on device errors.
     */
    void matvec_f32(std::span<const float> w, std::uint32_t rows,
                    std::uint32_t cols, std::span<const float> x,
                    std::span<float> out);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cppllm::backend::vk
