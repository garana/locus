#include "locus/backend/vulkan/context.hpp"

#include "locus/backend/vulkan/mem_fit.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace locus::backend::vk {

bool VulkanContext::kernel_needs_f16(Kernel k) {
    switch (k) {
        case Kernel::kMatmulQ8_0H:
        case Kernel::kMatmulF32H:
        case Kernel::kMatmulQ8_0IdH:
        case Kernel::kMatmulPagedF32H:
        case Kernel::kMatmulPagedTF32H:
            return true;
        default:
            return false;
    }
}

std::uint32_t VulkanContext::matvec_groups(Kernel k,
                                           std::uint32_t rows) {
    switch (k) {
        case Kernel::kMatvecF32:
        case Kernel::kMatvecF16:
        case Kernel::kMatvecBF16:
        case Kernel::kMatvecQ8_0:
        case Kernel::kMatvecF32B:
        case Kernel::kMatvecF16B:
        case Kernel::kMatvecBF16B:
        case Kernel::kMatvecQ8_0B:
        case Kernel::kMatvecQ8_0Glu:
        case Kernel::kMatvecQ8_0GluB:
        case Kernel::kMatvecQ8_0Seg:
        case Kernel::kMatvecQ8_0ShB:
        case Kernel::kMatvecQ8_0GluShB:
        case Kernel::kMatvecQ8_0N:
        case Kernel::kMatvecQ8_0P:
        case Kernel::kMatvecQ8_0GluP:
        case Kernel::kMatvecF32N:
        case Kernel::kMatvecF16N:
        case Kernel::kMatvecBF16N:
        case Kernel::kMatvecF16N2:
        case Kernel::kMatvecBF16N2:
            return (rows + 3) / 4;
        case Kernel::kMatvecQ4_K:  // 8 rows of 16 lanes
        case Kernel::kMatvecQ4_KB:
        case Kernel::kMatvecQ4_KGluB:
        case Kernel::kMatvecQ6_K:
        case Kernel::kMatvecQ6_KB:
            return (rows + 7) / 8;
        default:
            return (rows + 63) / 64;
    }
}

bool VulkanContext::profiling() {
    static const bool on = std::getenv("LOCUS_VK_PROFILE") != nullptr;
    return on;
}

/**
 * @returns Kernel k's profile label (its kernel table row; a
 * placeholder in builds without Vulkan kernels, which never profile).
 */
const char* kernel_name(Kernel k);

void VulkanContext::report_profile() const {
    const Profile& p = profile();
    if (!profiling() || p.batches == 0) {
        return;
    }
    const double per = p.tokens > 0 ? static_cast<double>(p.tokens) : 1.0;
    std::fprintf(stderr,
                 "vk-profile: %llu tokens, %llu batches, %llu "
                 "dispatches; per token: %.2f batches, %.1f "
                 "dispatches, host %.2f ms, submit+wait %.2f ms, "
                 "gpu %.2f ms\n",
                 (unsigned long long)p.tokens,
                 (unsigned long long)p.batches,
                 (unsigned long long)p.dispatches,
                 static_cast<double>(p.batches) / per,
                 static_cast<double>(p.dispatches) / per,
                 p.host_ms / per, p.wait_ms / per, p.gpu_ms / per);
    constexpr int kKernels = static_cast<int>(Kernel::kCount_);
    int order[kKernels];
    for (int k = 0; k < kKernels; ++k) {
        order[k] = k;
    }
    std::sort(std::begin(order), std::end(order), [&](int a, int b) {
        return p.kernel_ms[a] > p.kernel_ms[b];
    });
    for (int k : order) {
        if (p.kernel_n[k] == 0) {
            continue;
        }
        std::fprintf(stderr,
                     "vk-profile:   %-14s %8.3f ms/token  %6.1f "
                     "dispatches/token  %7.1f us each\n",
                     kernel_name(static_cast<Kernel>(k)),
                     p.kernel_ms[k] / per,
                     static_cast<double>(p.kernel_n[k]) / per,
                     1e3 * p.kernel_ms[k] /
                         static_cast<double>(p.kernel_n[k]));
    }
}

}  // namespace locus::backend::vk

#if defined(LOCUS_HAS_VULKAN_KERNELS)

#include "locus/backend/vulkan/limits.hpp"

// The shaders were compiled with the limits CMake extracted from
// limits.hpp; the host must use the same values.
static_assert(locus::backend::vk::kAttnPartMax ==
                  LOCUS_VK_SHADER_ATTN_PART_MAX,
              "kAttnPartMax differs from the shaders' ATTN_PART_MAX");
static_assert(locus::backend::vk::kAttnDecodeQMax ==
                  LOCUS_VK_SHADER_ATTN_DECODE_Q_MAX,
              "kAttnDecodeQMax differs from the shaders' "
              "ATTN_DECODE_Q_MAX");
static_assert(locus::backend::vk::kAttnGrpDvMax ==
                  LOCUS_VK_SHADER_ATTN_GRP_DV_MAX,
              "kAttnGrpDvMax differs from the shaders' ATTN_GRP_DV_MAX");

#include <vulkan/vulkan.h>

#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "attn_paged_q_spv.h"
#include "attn_softmax_spv.h"
#include "kv_scatter_spv.h"
#include "kv_store_q_spv.h"
#include "matmul_f32_small_spv.h"
#include "matmul_f32_spv.h"
#include "matmul_paged_f32_small_spv.h"
#include "matmul_paged_f32_spv.h"
#include "matmul_paged_t_f32_small_spv.h"
#include "matmul_paged_t_f32_spv.h"
#include "matmul_q8_0_id_small_spv.h"
#include "matmul_q8_0_id_spv.h"
#include "matmul_q8_0_small_spv.h"
#include "matmul_q8_0_spv.h"
#include "matvec_b_spv.h"
#include "matvec_f16_b_spv.h"
#include "matvec_f16_spv.h"
#include "matvec_q4_0_spv.h"
#include "matvec_q5_0_spv.h"
#include "matvec_q4_k_spv.h"
#include "matvec_q5_k_spv.h"
#include "matvec_q6_k_spv.h"
#include "matvec_q2_k_spv.h"
#include "matvec_iq2_xxs_spv.h"
#include "matvec_iq3_xxs_spv.h"
#include "matvec_iq4_xs_spv.h"
#include "matvec_q8_0_b_spv.h"
#include "matvec_q8_0_spv.h"
#include "moe_combine_spv.h"
#include "moe_group_spv.h"
#include "moe_route_b_spv.h"
#include "moe_route_spv.h"
#include "matvec_spv.h"
#include "matvec_t_spv.h"
#include "rmsnorm_b_spv.h"
#include "rmsnorm_spv.h"
#include "rope_b_spv.h"
#include "rope_spv.h"
#include "silu_mul_spv.h"
#include "sum_splits_spv.h"
#include "matvec_t_q8_spv.h"
#include "matvec_q8_0_glu_spv.h"
#include "matvec_q8_0_glu_b_spv.h"
#include "attn_decode_spv.h"
#include "attn_combine_spv.h"
#include "matvec_q8_0_seg_spv.h"
#include "matvec_q8_0_sh_b_spv.h"
#include "matvec_q8_0_glu_sh_b_spv.h"
#include "attn_decode_grp_spv.h"
#include "matmul_q8_0_h_spv.h"
#include "matmul_f32_h_spv.h"
#include "matmul_q8_0_id_h_spv.h"
#include "matmul_paged_f32_h_spv.h"
#include "matmul_paged_t_f32_h_spv.h"
#include "matvec_q8_0_n_spv.h"
#include "matvec_q8_0_p_spv.h"
#include "matvec_q8_0_glu_p_spv.h"
#include "matvec_f32_n_spv.h"
#include "matmul_paged_f16_spv.h"
#include "matmul_paged_t_f16_spv.h"
#include "matmul_paged_f16_small_spv.h"
#include "matmul_paged_t_f16_small_spv.h"
#include "attn_decode_f16_spv.h"
#include "kv_scatter_f16_spv.h"
#include "attn_decode_grp_f16_spv.h"
#include "attn_decode_grp8_spv.h"
#include "attn_decode_grp8_f16_spv.h"
#include "matvec_bf16_b_spv.h"
#include "matvec_bf16_spv.h"
#include "matvec_f16_n_spv.h"
#include "matvec_bf16_n_spv.h"
#include "matvec_f16_n2_spv.h"
#include "matvec_bf16_n2_spv.h"
#include "copy_spv.h"
#include "rope_neox_spv.h"
#include "rope_neox_b_spv.h"
#include "matvec_q4_k_b_spv.h"
#include "matvec_q6_k_b_spv.h"
#include "matvec_q4_k_glu_b_spv.h"
#include "matmul_q4_k_spv.h"
#include "matmul_q6_k_spv.h"
#include "matmul_q4_k_id_spv.h"
#include "matmul_q6_k_id_spv.h"
#include "matmul_q4_k_small_spv.h"
#include "matmul_q6_k_small_spv.h"
#include "matmul_q4_k_id_small_spv.h"
#include "matmul_q6_k_id_small_spv.h"
#include "silu_mul_sg_spv.h"
#include "conv1d_spv.h"
#include "conv1d_transpose_spv.h"
#include "snake1d_spv.h"
#include "conv2d_spv.h"

namespace locus::backend::vk {

namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string("vulkan: ") + what);
    }
}

/**
 * Validation errors seen through the debug messenger (opt-in, see
 * LOCUS_VK_VALIDATION in Impl::init). Process-wide: the layer runs on
 * one instance at a time in the test that enables it.
 */
std::atomic<std::uint32_t> g_validation_errors{0};

/**
 * Debug-messenger callback: counts error-severity messages (a
 * GPU-assisted out-of-bounds buffer access is reported as an error)
 * and echoes every warning and error to stderr so a failing test
 * shows what the layer found.
 */
VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        g_validation_errors.fetch_add(1, std::memory_order_relaxed);
    }
    std::fprintf(stderr, "[vk-validation] %s\n",
                 data != nullptr && data->pMessage != nullptr
                     ? data->pMessage
                     : "(no message)");
    return VK_FALSE;
}

/**
 * Static description of each kernel's interface: one row per Kernel,
 * in enum order (kDescs; the order is checked at compile time).
 */
struct KernelDesc {
    Kernel kernel;
    /** Profile label (LOCUS_VK_PROFILE=2). */
    const char* name;
    const std::uint32_t* spv;
    std::size_t spv_bytes;
    std::uint32_t n_buffers;
    std::uint32_t push_bytes;
    /**
     * Smaller-shared-memory variant, used when the device's
     * maxComputeSharedMemorySize cannot fit `shared_bytes`. Null for
     * kernels with no variant; `shared_bytes` 0 then means the kernel
     * fits any device.
     */
    const std::uint32_t* alt_spv = nullptr;
    std::size_t alt_spv_bytes = 0;
    std::uint32_t shared_bytes = 0;
    std::uint32_t alt_shared_bytes = 0;
};

/**
 * Workgroup (shared) memory the tiled matmul (matmul.comp) declares
 * for a given BK: ws[BK][BM/4] + xs[BK][BN/4] as vec4 (16 B each) +
 * rows_x[BN] + rows_o[BN] (uint). Mirrors the shader's shared block;
 * the static_asserts pin the two tiles so a change here or in the
 * shader's BM/BN/BK forces a conscious update of both.
 */
constexpr std::uint32_t matmul_shared_bytes(std::uint32_t bk) {
    constexpr std::uint32_t bm = VulkanContext::kMatmulTileRows;
    constexpr std::uint32_t bn = VulkanContext::kMatmulTileToks;
    return bk * (bm / 4u) * 16u + bk * (bn / 4u) * 16u + bn * 4u * 2u;
}
static_assert(matmul_shared_bytes(VulkanContext::kMatmulBK) == 25088);
static_assert(matmul_shared_bytes(VulkanContext::kMatmulBKSmall) == 12800);

/**
 * Every kernel's interface, row i describing Kernel(i). One table
 * instead of parallel enum / name / descriptor lists, so a misordered
 * or missing row is a compile error (below), not a silent shift of
 * every later kernel's SPIR-V, buffer count and push size.
 */
constexpr KernelDesc kDescs[] = {
    {Kernel::kMatvecF32, "matvec_f32", locus_matvec_spv,
     sizeof(locus_matvec_spv), 3, 28},
    {Kernel::kMatvecF16, "matvec_f16", locus_matvec_f16_spv,
     sizeof(locus_matvec_f16_spv), 3, 28},
    {Kernel::kMatvecQ8_0, "matvec_q8_0", locus_matvec_q8_0_spv,
     sizeof(locus_matvec_q8_0_spv), 3, 28},
    {Kernel::kMatvecQ4_0, "matvec_q4_0", locus_matvec_q4_0_spv,
     sizeof(locus_matvec_q4_0_spv), 3, 28},
    {Kernel::kMatvecQ5_0, "matvec_q5_0", locus_matvec_q5_0_spv,
     sizeof(locus_matvec_q5_0_spv), 3, 28},
    {Kernel::kMatvecQ4_K, "matvec_q4_k", locus_matvec_q4_k_spv,
     sizeof(locus_matvec_q4_k_spv), 3, 28},
    {Kernel::kMatvecQ5_K, "matvec_q5_k", locus_matvec_q5_k_spv,
     sizeof(locus_matvec_q5_k_spv), 3, 28},
    {Kernel::kMatvecQ6_K, "matvec_q6_k", locus_matvec_q6_k_spv,
     sizeof(locus_matvec_q6_k_spv), 3, 28},
    {Kernel::kMatvecQ2_K, "matvec_q2_k", locus_matvec_q2_k_spv,
     sizeof(locus_matvec_q2_k_spv), 3, 28},
    {Kernel::kMatvecIQ2_XXS, "matvec_iq2_xxs", locus_matvec_iq2_xxs_spv,
     sizeof(locus_matvec_iq2_xxs_spv), 4, 28},
    {Kernel::kMatvecIQ3_XXS, "matvec_iq3_xxs", locus_matvec_iq3_xxs_spv,
     sizeof(locus_matvec_iq3_xxs_spv), 4, 28},
    {Kernel::kMatvecIQ4_XS, "matvec_iq4_xs", locus_matvec_iq4_xs_spv,
     sizeof(locus_matvec_iq4_xs_spv), 3, 28},
    {Kernel::kMatvecT, "matvec_t", locus_matvec_t_spv,
     sizeof(locus_matvec_t_spv), 3, 32},
    {Kernel::kRmsNorm, "rmsnorm", locus_rmsnorm_spv, sizeof(locus_rmsnorm_spv),
     3, 16},
    {Kernel::kRope, "rope", locus_rope_spv, sizeof(locus_rope_spv), 2, 36},
    {Kernel::kSiluMul, "silu_mul", locus_silu_mul_spv,
     sizeof(locus_silu_mul_spv), 3, 4},
    {Kernel::kAttnPagedQ, "attn_paged_q", locus_attn_paged_q_spv,
     sizeof(locus_attn_paged_q_spv), 5, 44},
    {Kernel::kKvStoreQ, "kv_store_q", locus_kv_store_q_spv,
     sizeof(locus_kv_store_q_spv), 2, 16},
    {Kernel::kMatvecF32B, "matvec_f32_b", locus_matvec_b_spv,
     sizeof(locus_matvec_b_spv), 4, 52},
    {Kernel::kMatvecF16B, "matvec_f16_b", locus_matvec_f16_b_spv,
     sizeof(locus_matvec_f16_b_spv), 4, 52},
    {Kernel::kMatvecQ8_0B, "matvec_q8_0_b", locus_matvec_q8_0_b_spv,
     sizeof(locus_matvec_q8_0_b_spv), 4, 52},
    {Kernel::kMoeRoute, "moe_route", locus_moe_route_spv,
     sizeof(locus_moe_route_spv), 3, 36},
    {Kernel::kMatmulQ8_0, "matmul_q8_0", locus_matmul_q8_0_spv,
     sizeof(locus_matmul_q8_0_spv), 3, 96, locus_matmul_q8_0_small_spv,
     sizeof(locus_matmul_q8_0_small_spv),
     matmul_shared_bytes(VulkanContext::kMatmulBK),
     matmul_shared_bytes(VulkanContext::kMatmulBKSmall)},
    {Kernel::kMatmulF32, "matmul_f32", locus_matmul_f32_spv,
     sizeof(locus_matmul_f32_spv), 3, 96, locus_matmul_f32_small_spv,
     sizeof(locus_matmul_f32_small_spv),
     matmul_shared_bytes(VulkanContext::kMatmulBK),
     matmul_shared_bytes(VulkanContext::kMatmulBKSmall)},
    {Kernel::kMatmulQ8_0Id, "matmul_q8_0_id", locus_matmul_q8_0_id_spv,
     sizeof(locus_matmul_q8_0_id_spv), 4, 96, locus_matmul_q8_0_id_small_spv,
     sizeof(locus_matmul_q8_0_id_small_spv),
     matmul_shared_bytes(VulkanContext::kMatmulBK),
     matmul_shared_bytes(VulkanContext::kMatmulBKSmall)},
    {Kernel::kKvScatter, "kv_scatter", locus_kv_scatter_spv,
     sizeof(locus_kv_scatter_spv), 3, 40},
    {Kernel::kMoeGroup, "moe_group", locus_moe_group_spv,
     sizeof(locus_moe_group_spv), 2, 16},
    {Kernel::kMoeCombine, "moe_combine", locus_moe_combine_spv,
     sizeof(locus_moe_combine_spv), 3, 8},
    {Kernel::kRmsNormB, "rmsnorm_b", locus_rmsnorm_b_spv,
     sizeof(locus_rmsnorm_b_spv), 3, 24},
    {Kernel::kRopeB, "rope_b", locus_rope_b_spv, sizeof(locus_rope_b_spv), 2,
     40},
    {Kernel::kMoeRouteB, "moe_route_b", locus_moe_route_b_spv,
     sizeof(locus_moe_route_b_spv), 3, 44},
    {Kernel::kMatmulPagedF32, "matmul_paged", locus_matmul_paged_f32_spv,
     sizeof(locus_matmul_paged_f32_spv), 4, 96,
     locus_matmul_paged_f32_small_spv, sizeof(locus_matmul_paged_f32_small_spv),
     matmul_shared_bytes(VulkanContext::kMatmulBK),
     matmul_shared_bytes(VulkanContext::kMatmulBKSmall)},
    {Kernel::kMatmulPagedTF32, "matmul_paged_t", locus_matmul_paged_t_f32_spv,
     sizeof(locus_matmul_paged_t_f32_spv), 4, 96,
     locus_matmul_paged_t_f32_small_spv,
     sizeof(locus_matmul_paged_t_f32_small_spv),
     matmul_shared_bytes(VulkanContext::kMatmulBK),
     matmul_shared_bytes(VulkanContext::kMatmulBKSmall)},
    {Kernel::kAttnSoftmax, "attn_softmax", locus_attn_softmax_spv,
     sizeof(locus_attn_softmax_spv), 1, 20},
    {Kernel::kSumSplits, "sum_splits", locus_sum_splits_spv,
     sizeof(locus_sum_splits_spv), 2, 16},
    {Kernel::kMatvecTQ8, "matvec_t_q8", locus_matvec_t_q8_spv,
     sizeof(locus_matvec_t_q8_spv), 3, 32},
    {Kernel::kMatvecQ8_0Glu, "matvec_q8_0_glu", locus_matvec_q8_0_glu_spv,
     sizeof(locus_matvec_q8_0_glu_spv), 4, 28},
    {Kernel::kMatvecQ8_0GluB, "matvec_q8_0_glu_b", locus_matvec_q8_0_glu_b_spv,
     sizeof(locus_matvec_q8_0_glu_b_spv), 5, 52},
    {Kernel::kAttnDecode, "attn_decode", locus_attn_decode_spv,
     sizeof(locus_attn_decode_spv), 6, 64},
    {Kernel::kAttnCombine, "attn_combine", locus_attn_combine_spv,
     sizeof(locus_attn_combine_spv), 2, 16},
    {Kernel::kMatvecQ8_0Seg, "matvec_q8_0_seg", locus_matvec_q8_0_seg_spv,
     sizeof(locus_matvec_q8_0_seg_spv), 5, 88},
    {Kernel::kMatvecQ8_0ShB, "matvec_q8_0_sh_b", locus_matvec_q8_0_sh_b_spv,
     sizeof(locus_matvec_q8_0_sh_b_spv), 5, 60},
    {Kernel::kMatvecQ8_0GluShB, "matvec_q8_0_glu_sh_b",
     locus_matvec_q8_0_glu_sh_b_spv, sizeof(locus_matvec_q8_0_glu_sh_b_spv), 7,
     60},
    {Kernel::kAttnDecodeGrp, "attn_decode_grp", locus_attn_decode_grp_spv,
     sizeof(locus_attn_decode_grp_spv), 6, 64},
    {Kernel::kMatmulQ8_0H, "matmul_q8_0_h", locus_matmul_q8_0_h_spv,
     sizeof(locus_matmul_q8_0_h_spv), 3, 96},
    {Kernel::kMatmulF32H, "matmul_f32_h", locus_matmul_f32_h_spv,
     sizeof(locus_matmul_f32_h_spv), 3, 96},
    {Kernel::kMatmulQ8_0IdH, "matmul_q8_0_id_h", locus_matmul_q8_0_id_h_spv,
     sizeof(locus_matmul_q8_0_id_h_spv), 4, 96},
    {Kernel::kMatmulPagedF32H, "matmul_paged_h", locus_matmul_paged_f32_h_spv,
     sizeof(locus_matmul_paged_f32_h_spv), 4, 96},
    {Kernel::kMatmulPagedTF32H, "matmul_paged_t_h",
     locus_matmul_paged_t_f32_h_spv, sizeof(locus_matmul_paged_t_f32_h_spv), 4,
     96},
    {Kernel::kMatvecQ8_0N, "matvec_q8_0_n", locus_matvec_q8_0_n_spv,
     sizeof(locus_matvec_q8_0_n_spv), 3, 52},
    {Kernel::kMatvecQ8_0P, "matvec_q8_0_p", locus_matvec_q8_0_p_spv,
     sizeof(locus_matvec_q8_0_p_spv), 4, 60},
    {Kernel::kMatvecQ8_0GluP, "matvec_q8_0_glu_p", locus_matvec_q8_0_glu_p_spv,
     sizeof(locus_matvec_q8_0_glu_p_spv), 5, 60},
    {Kernel::kMatvecF32N, "matvec_f32_n", locus_matvec_f32_n_spv,
     sizeof(locus_matvec_f32_n_spv), 3, 52},
    // F16-KV pool readers stage in f32 like the F32 ones, so they
    // step down to BK16 on tight devices the same way.
    {Kernel::kMatmulPagedF16, "matmul_paged_f16", locus_matmul_paged_f16_spv,
     sizeof(locus_matmul_paged_f16_spv), 4, 96,
     locus_matmul_paged_f16_small_spv, sizeof(locus_matmul_paged_f16_small_spv),
     matmul_shared_bytes(VulkanContext::kMatmulBK),
     matmul_shared_bytes(VulkanContext::kMatmulBKSmall)},
    {Kernel::kMatmulPagedTF16, "matmul_paged_t_f16",
     locus_matmul_paged_t_f16_spv, sizeof(locus_matmul_paged_t_f16_spv), 4, 96,
     locus_matmul_paged_t_f16_small_spv,
     sizeof(locus_matmul_paged_t_f16_small_spv),
     matmul_shared_bytes(VulkanContext::kMatmulBK),
     matmul_shared_bytes(VulkanContext::kMatmulBKSmall)},
    {Kernel::kAttnDecodeF16, "attn_decode_f16", locus_attn_decode_f16_spv,
     sizeof(locus_attn_decode_f16_spv), 6, 64},
    {Kernel::kKvScatterF16, "kv_scatter_f16", locus_kv_scatter_f16_spv,
     sizeof(locus_kv_scatter_f16_spv), 3, 40},
    {Kernel::kAttnDecodeGrpF16, "attn_decode_grp_f16",
     locus_attn_decode_grp_f16_spv, sizeof(locus_attn_decode_grp_f16_spv), 6,
     64},
    {Kernel::kRopeNeox, "rope_neox", locus_rope_neox_spv,
     sizeof(locus_rope_neox_spv), 2, 36},
    {Kernel::kRopeNeoxB, "rope_neox_b", locus_rope_neox_b_spv,
     sizeof(locus_rope_neox_b_spv), 2, 40},
    {Kernel::kMatvecQ4_KB, "matvec_q4_k_b", locus_matvec_q4_k_b_spv,
     sizeof(locus_matvec_q4_k_b_spv), 4, 52},
    {Kernel::kMatvecQ6_KB, "matvec_q6_k_b", locus_matvec_q6_k_b_spv,
     sizeof(locus_matvec_q6_k_b_spv), 4, 52},
    {Kernel::kMatmulQ4_K, "matmul_q4_k", locus_matmul_q4_k_spv,
     sizeof(locus_matmul_q4_k_spv), 3, 96, locus_matmul_q4_k_small_spv,
     sizeof(locus_matmul_q4_k_small_spv),
     matmul_shared_bytes(VulkanContext::kMatmulBK),
     matmul_shared_bytes(VulkanContext::kMatmulBKSmall)},
    {Kernel::kMatmulQ6_K, "matmul_q6_k", locus_matmul_q6_k_spv,
     sizeof(locus_matmul_q6_k_spv), 3, 96, locus_matmul_q6_k_small_spv,
     sizeof(locus_matmul_q6_k_small_spv),
     matmul_shared_bytes(VulkanContext::kMatmulBK),
     matmul_shared_bytes(VulkanContext::kMatmulBKSmall)},
    {Kernel::kMatmulQ4_KId, "matmul_q4_k_id", locus_matmul_q4_k_id_spv,
     sizeof(locus_matmul_q4_k_id_spv), 4, 96, locus_matmul_q4_k_id_small_spv,
     sizeof(locus_matmul_q4_k_id_small_spv),
     matmul_shared_bytes(VulkanContext::kMatmulBK),
     matmul_shared_bytes(VulkanContext::kMatmulBKSmall)},
    {Kernel::kMatmulQ6_KId, "matmul_q6_k_id", locus_matmul_q6_k_id_spv,
     sizeof(locus_matmul_q6_k_id_spv), 4, 96, locus_matmul_q6_k_id_small_spv,
     sizeof(locus_matmul_q6_k_id_small_spv),
     matmul_shared_bytes(VulkanContext::kMatmulBK),
     matmul_shared_bytes(VulkanContext::kMatmulBKSmall)},
    {Kernel::kSiluMulSg, "silu_mul_sg", locus_silu_mul_sg_spv,
     sizeof(locus_silu_mul_sg_spv), 4, 8},
    {Kernel::kConv1d, "conv1d", locus_conv1d_spv,
     sizeof(locus_conv1d_spv), 4, 40},
    {Kernel::kConv1dTranspose, "conv1d_transpose",
     locus_conv1d_transpose_spv, sizeof(locus_conv1d_transpose_spv), 4, 40},
    {Kernel::kSnake1d, "snake1d", locus_snake1d_spv,
     sizeof(locus_snake1d_spv), 2, 12},
    {Kernel::kMatvecQ4_KGluB, "matvec_q4_k_glu_b",
     locus_matvec_q4_k_glu_b_spv, sizeof(locus_matvec_q4_k_glu_b_spv), 5, 52},
    {Kernel::kConv2d, "conv2d", locus_conv2d_spv, sizeof(locus_conv2d_spv),
     4, 48},
    {Kernel::kAttnDecodeGrp8, "attn_decode_grp8",
     locus_attn_decode_grp8_spv, sizeof(locus_attn_decode_grp8_spv), 6, 64},
    {Kernel::kAttnDecodeGrp8F16, "attn_decode_grp8_f16",
     locus_attn_decode_grp8_f16_spv,
     sizeof(locus_attn_decode_grp8_f16_spv), 6, 64},
    {Kernel::kMatvecBF16, "matvec_bf16", locus_matvec_bf16_spv,
     sizeof(locus_matvec_bf16_spv), 3, 28},
    {Kernel::kMatvecBF16B, "matvec_bf16_b", locus_matvec_bf16_b_spv,
     sizeof(locus_matvec_bf16_b_spv), 4, 52},
    {Kernel::kMatvecF16N, "matvec_f16_n", locus_matvec_f16_n_spv,
     sizeof(locus_matvec_f16_n_spv), 3, 52},
    {Kernel::kMatvecBF16N, "matvec_bf16_n", locus_matvec_bf16_n_spv,
     sizeof(locus_matvec_bf16_n_spv), 3, 52},
    {Kernel::kMatvecF16N2, "matvec_f16_n2", locus_matvec_f16_n2_spv,
     sizeof(locus_matvec_f16_n2_spv), 3, 52},
    {Kernel::kMatvecBF16N2, "matvec_bf16_n2", locus_matvec_bf16_n2_spv,
     sizeof(locus_matvec_bf16_n2_spv), 3, 52},
    {Kernel::kCopy, "copy", locus_copy_spv, sizeof(locus_copy_spv), 2, 12},
};

/** @returns Whether row i of kDescs is Kernel(i), for every kernel. */
constexpr bool descs_in_enum_order() {
    if (std::size(kDescs) != static_cast<std::size_t>(Kernel::kCount_)) {
        return false;
    }
    for (std::size_t i = 0; i < std::size(kDescs); ++i) {
        if (kDescs[i].kernel != static_cast<Kernel>(i)) {
            return false;
        }
    }
    return true;
}
static_assert(descs_in_enum_order(),
              "kDescs rows must list every Kernel, in enum order");

const KernelDesc& kernel_desc(Kernel k) {
    return kDescs[static_cast<int>(k)];
}

}  // namespace

const char* kernel_name(Kernel k) { return kernel_desc(k).name; }

struct VulkanContext::Impl {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;

    /**
     * Descriptor sets cached by (kernel, buffers): a decode token
     * replays the same dispatch sequence, so after the first token
     * every set is a hit and recording skips the allocate + write.
     * Sets come from a growing list of pools. Destroying any buffer
     * drops the map at once (a recycled VkBuffer handle must never
     * hit a stale set) and frees the pools at the next end_batch,
     * when no recorded command can still reference them.
     */
    static constexpr std::size_t kMaxBufs = 8;
    struct SetKey {
        int kernel;
        std::array<VkBuffer, kMaxBufs> bufs;
        bool operator==(const SetKey&) const = default;
    };
    struct SetKeyHash {
        std::size_t operator()(const SetKey& k) const {
            std::size_t h = static_cast<std::size_t>(k.kernel);
            for (VkBuffer b : k.bufs) {
                h = h * 1000003u ^ std::hash<const void*>{}(b);
            }
            return h;
        }
    };
    std::unordered_map<SetKey, VkDescriptorSet, SetKeyHash> sets;
    std::vector<VkDescriptorPool> dpools;
    std::size_t sets_live = 0;
    bool sets_stale = false;

    VkDescriptorPool new_dpool() {
        VkDescriptorPoolSize dps{};
        dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        dps.descriptorCount = 8192;
        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 2048;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &dps;
        VkDescriptorPool p;
        check(vkCreateDescriptorPool(device, &dpi, nullptr, &p),
              "create descriptor pool");
        dpools.push_back(p);
        return p;
    }

    VkDescriptorSet alloc_set(VkDescriptorSetLayout dsl) {
        VkDescriptorSetAllocateInfo dsa{};
        dsa.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsa.descriptorSetCount = 1;
        dsa.pSetLayouts = &dsl;
        dsa.descriptorPool = dpools.back();
        VkDescriptorSet ds;
        if (vkAllocateDescriptorSets(device, &dsa, &ds) != VK_SUCCESS) {
            dsa.descriptorPool = new_dpool();
            check(vkAllocateDescriptorSets(device, &dsa, &ds),
                  "allocate descriptor set");
        }
        ++sets_live;
        return ds;
    }

    /** Frees every cached set; only when no batch is pending. */
    void reset_sets() {
        sets.clear();
        for (std::size_t i = 1; i < dpools.size(); ++i) {
            vkDestroyDescriptorPool(device, dpools[i], nullptr);
        }
        dpools.resize(1);
        check(vkResetDescriptorPool(device, dpools[0], 0),
              "reset descriptor pool");
        sets_live = 0;
        sets_stale = false;
    }

    struct Pipe {
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };
    Pipe pipes[static_cast<int>(Kernel::kCount_)];

    /** Open batch state (begin_batch .. end_batch). */
    VkCommandBuffer batch_cmd = VK_NULL_HANDLE;

    /**
     * LOCUS_VK_PROFILE state: a timestamp at batch begin, one after
     * each dispatch (its kernel in ts_kernel), one at batch end.
     */
    static constexpr std::uint32_t kMaxTs = 4096;
    Profile prof;
    std::vector<int> ts_kernel;
    bool per_kernel = false;  // LOCUS_VK_PROFILE=2
    VkQueryPool qpool = VK_NULL_HANDLE;
    double ts_period_ns = 0.0;

    /** Device maxComputeSharedMemorySize (bytes), read at init. */
    std::uint32_t max_shared_mem = 0;
    /** Whether the matmul smaller-tile step-down has been logged. */
    bool matmul_stepdown_logged = false;
    std::chrono::steady_clock::time_point batch_t0;

    struct HostBuffer {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        void* map = nullptr;
        VkDeviceSize bytes = 0;
    };

    /** VK_EXT_memory_budget present (and enabled). */
    bool has_budget = false;
    /**
     * The driver runs dispatches in submission order with writes
     * visible to later ones (MoltenVK), so dispatch() records no
     * compute-to-compute barrier.
     */
    bool serial_dispatch = false;
    /** shaderFloat16 supported and enabled (f16 kernels built). */
    bool has_f16 = false;
    /** Bytes currently allocated through make_buffer. */
    VkDeviceSize allocated = 0;

    /** @returns The heap backing host-visible coherent memory. */
    std::uint32_t host_heap() const {
        VkPhysicalDeviceMemoryProperties props;
        vkGetPhysicalDeviceMemoryProperties(phys, &props);
        const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
            if ((props.memoryTypes[i].propertyFlags & want) == want) {
                return props.memoryTypes[i].heapIndex;
            }
        }
        return 0;
    }

    /** Installs the debug messenger that feeds debug_callback. */
    void create_debug_messenger() {
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance,
                                  "vkCreateDebugUtilsMessengerEXT"));
        if (create == nullptr) {
            return;
        }
        VkDebugUtilsMessengerCreateInfoEXT ci{};
        ci.sType =
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        ci.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        ci.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        ci.pfnUserCallback = debug_callback;
        create(instance, &ci, nullptr, &messenger);
    }

    void init() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "locus";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        std::vector<const char*> ilayers;
        std::vector<const char*> iexts;
#if defined(__APPLE__)
        iexts.push_back("VK_KHR_portability_enumeration");
        ici.flags =
            VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
        // Opt-in validation (LOCUS_VK_VALIDATION): turn on the
        // Khronos validation layer with GPU-assisted validation, so
        // the layer bounds-checks shader buffer access on the device
        // and reports an out-of-range SSBO read (e.g. a whole-word
        // Q8_0 read past an under-sized weight buffer) through the
        // debug messenger. Off, and zero cost, by default.
        const bool validate =
            std::getenv("LOCUS_VK_VALIDATION") != nullptr;
        const VkValidationFeatureEnableEXT gpu_av[] = {
            VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT};
        VkValidationFeaturesEXT vf{};
        if (validate) {
            ilayers.push_back("VK_LAYER_KHRONOS_validation");
            iexts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            vf.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
            vf.enabledValidationFeatureCount = 1;
            vf.pEnabledValidationFeatures = gpu_av;
            vf.pNext = ici.pNext;
            ici.pNext = &vf;
        }
        ici.enabledLayerCount =
            static_cast<std::uint32_t>(ilayers.size());
        ici.ppEnabledLayerNames = ilayers.data();
        ici.enabledExtensionCount =
            static_cast<std::uint32_t>(iexts.size());
        ici.ppEnabledExtensionNames = iexts.data();
        check(vkCreateInstance(&ici, nullptr, &instance),
              "create instance");
        if (validate) {
            create_debug_messenger();
        }

        std::uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance, &n, nullptr);
        if (n == 0) {
            throw std::runtime_error("vulkan: no devices");
        }
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance, &n, devs.data());

        for (auto d : devs) {
            std::uint32_t qn = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qn,
                                                     nullptr);
            std::vector<VkQueueFamilyProperties> qs(qn);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qn,
                                                     qs.data());
            for (std::uint32_t i = 0; i < qn; ++i) {
                if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    phys = d;
                    queue_family = i;
                    break;
                }
            }
            if (phys != VK_NULL_HANDLE) {
                break;
            }
        }
        if (phys == VK_NULL_HANDLE) {
            throw std::runtime_error("vulkan: no compute queue");
        }

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        std::vector<const char*> dexts;
        // Enabled through dci.pNext when the device has it.
        VkPhysicalDeviceShaderFloat16Int8Features f16f{};
        f16f.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
#if defined(__APPLE__)
        dexts.push_back("VK_KHR_portability_subset");
#endif
        {
            std::uint32_t en = 0;
            vkEnumerateDeviceExtensionProperties(phys, nullptr, &en,
                                                 nullptr);
            std::vector<VkExtensionProperties> exts(en);
            vkEnumerateDeviceExtensionProperties(phys, nullptr, &en,
                                                 exts.data());
            bool has_driver_props = false;
            bool has_f16_ext = false;
            for (const auto& e : exts) {
                has_f16_ext =
                    has_f16_ext ||
                    std::strcmp(e.extensionName,
                                VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) ==
                        0;
                if (std::strcmp(e.extensionName,
                                VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) ==
                    0) {
                    has_budget = true;
                    dexts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
                }
                has_driver_props =
                    has_driver_props ||
                    std::strcmp(e.extensionName,
                                VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME) ==
                        0;
            }
            // LOCUS_VK_NO_F16=1 takes the no-shaderFloat16 path on any
            // device (to check the fallback).
            if (has_f16_ext && std::getenv("LOCUS_VK_NO_F16") == nullptr) {
                // The f16 kernels (--prefill-f16) need shaderFloat16;
                // without it they are not built and the flag is a
                // no-op.
                VkPhysicalDeviceFeatures2 f2{};
                f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                f2.pNext = &f16f;
                vkGetPhysicalDeviceFeatures2(phys, &f2);
                has_f16 = f16f.shaderFloat16 == VK_TRUE;
                if (has_f16) {
                    dexts.push_back(
                        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
                    f16f.shaderInt8 = VK_FALSE;
                    f16f.pNext = nullptr;
                    dci.pNext = &f16f;
                }
            }
            if (has_driver_props) {
                VkPhysicalDeviceDriverProperties dp{};
                dp.sType =
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
                VkPhysicalDeviceProperties2 p2{};
                p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                p2.pNext = &dp;
                vkGetPhysicalDeviceProperties2(phys, &p2);
                // MoltenVK records compute into serial Metal compute
                // encoders: each dispatch completes, writes visible,
                // before the next starts. A pipeline barrier there
                // only splits the encoder (~19 us each on an M4 Max),
                // so skip them; LOCUS_VK_BARRIERS=1 keeps them.
                serial_dispatch =
                    dp.driverID == VK_DRIVER_ID_MOLTENVK &&
                    std::getenv("LOCUS_VK_BARRIERS") == nullptr;
            }
        }
        dci.enabledExtensionCount =
            static_cast<std::uint32_t>(dexts.size());
        dci.ppEnabledExtensionNames = dexts.data();
        check(vkCreateDevice(phys, &dci, nullptr, &device),
              "create device");
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        VkCommandPoolCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.queueFamilyIndex = queue_family;
        check(vkCreateCommandPool(device, &cpi, nullptr, &pool),
              "create command pool");

        {
            VkPhysicalDeviceProperties pp;
            vkGetPhysicalDeviceProperties(phys, &pp);
            max_shared_mem = pp.limits.maxComputeSharedMemorySize;
        }
        if (!has_f16) {
            static bool told = false;
            if (!told) {
                told = true;
                std::fprintf(stderr,
                             "locus: Vulkan device has no shaderFloat16: "
                             "f16 kernels skipped (--prefill-f16 off)\n");
            }
        }
        for (int k = 0; k < static_cast<int>(Kernel::kCount_);
             ++k) {
            if (has_f16 ||
                !VulkanContext::kernel_needs_f16(static_cast<Kernel>(k))) {
                create_pipe(static_cast<Kernel>(k));
            }
        }

        new_dpool();

        if (VulkanContext::profiling()) {
            VkPhysicalDeviceProperties pp;
            vkGetPhysicalDeviceProperties(phys, &pp);
            ts_period_ns = pp.limits.timestampPeriod;
            VkQueryPoolCreateInfo qpi{};
            qpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpi.queryCount = kMaxTs;
            const char* level = std::getenv("LOCUS_VK_PROFILE");
            per_kernel = level != nullptr && std::atoi(level) >= 2;
            check(vkCreateQueryPool(device, &qpi, nullptr, &qpool),
                  "create query pool");
        }
    }

    void create_pipe(Kernel k) {
        const KernelDesc& kd = kernel_desc(k);
        Pipe& p = pipes[static_cast<int>(k)];

        std::vector<VkDescriptorSetLayoutBinding> binds(
            kd.n_buffers);
        for (std::uint32_t i = 0; i < kd.n_buffers; ++i) {
            binds[i].binding = i;
            binds[i].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dli{};
        dli.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dli.bindingCount = kd.n_buffers;
        dli.pBindings = binds.data();
        check(vkCreateDescriptorSetLayout(device, &dli, nullptr,
                                          &p.dsl),
              "create dsl");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = kd.push_bytes;
        VkPipelineLayoutCreateInfo pli{};
        pli.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &p.dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(device, &pli, nullptr,
                                     &p.layout),
              "create pipeline layout");

        // Pick the smaller-tile variant when the device's shared
        // memory cannot fit this kernel's default stage (LOCUS_VK_
        // MATMUL_SMALL forces it so a roomy device can test the
        // variant too). Keeps the GPU path on tight devices instead
        // of failing pipeline creation or vanishing to the CPU.
        const std::uint32_t* spv = kd.spv;
        std::size_t spv_bytes = kd.spv_bytes;
        if (kd.alt_spv != nullptr) {
            const bool over = kd.shared_bytes > max_shared_mem;
            const bool forced =
                std::getenv("LOCUS_VK_MATMUL_SMALL") != nullptr;
            if (over || forced) {
                spv = kd.alt_spv;
                spv_bytes = kd.alt_spv_bytes;
                if (!matmul_stepdown_logged) {
                    matmul_stepdown_logged = true;
                    if (over) {
                        std::fprintf(
                            stderr,
                            "locus: Vulkan matmul tile needs %u B "
                            "shared memory, over this device's "
                            "maxComputeSharedMemorySize %u B; using "
                            "the smaller-tile variant (%u B).\n",
                            kd.shared_bytes, max_shared_mem,
                            kd.alt_shared_bytes);
                    } else {
                        std::fprintf(
                            stderr,
                            "locus: LOCUS_VK_MATMUL_SMALL set; using "
                            "the smaller-tile matmul variant (%u B).\n",
                            kd.alt_shared_bytes);
                    }
                }
            }
        }

        VkShaderModuleCreateInfo smi{};
        smi.sType =
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = spv_bytes;
        smi.pCode = spv;
        VkShaderModule sm;
        check(vkCreateShaderModule(device, &smi, nullptr, &sm),
              "create shader module");

        VkComputePipelineCreateInfo cpci{};
        cpci.sType =
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = sm;
        cpci.stage.pName = "main";
        cpci.layout = p.layout;
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                       &cpci, nullptr,
                                       &p.pipeline),
              "create pipeline");
        vkDestroyShaderModule(device, sm, nullptr);
    }

    HostBuffer make_buffer(VkDeviceSize size) {
        HostBuffer b;
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device, &bci, nullptr, &b.buf),
              "create buffer");

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buf, &req);
        VkPhysicalDeviceMemoryProperties props;
        vkGetPhysicalDeviceMemoryProperties(phys, &props);
        std::uint32_t type = ~0u;
        const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t i = 0;
             i < props.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (props.memoryTypes[i].propertyFlags & want) ==
                    want) {
                type = i;
                break;
            }
        }
        if (type == ~0u) {
            throw std::runtime_error(
                "vulkan: no host-visible memory");
        }
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        check(vkAllocateMemory(device, &mai, nullptr, &b.mem),
              "allocate memory");
        b.bytes = req.size;
        allocated += req.size;
        check(vkBindBufferMemory(device, b.buf, b.mem, 0),
              "bind memory");
        check(vkMapMemory(device, b.mem, 0, size, 0, &b.map),
              "map memory");
        return b;
    }

    void destroy_buffer(HostBuffer& b) {
        if (b.mem != VK_NULL_HANDLE) {
            allocated -= b.bytes;
            vkUnmapMemory(device, b.mem);
            vkFreeMemory(device, b.mem, nullptr);
        }
        if (b.buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, b.buf, nullptr);
        }
        b = {};
    }

    ~Impl() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            if (qpool != VK_NULL_HANDLE) {
                vkDestroyQueryPool(device, qpool, nullptr);
            }
            for (VkDescriptorPool p : dpools) {
                vkDestroyDescriptorPool(device, p, nullptr);
            }
            for (const Pipe& p : pipes) {
                vkDestroyPipeline(device, p.pipeline, nullptr);
                vkDestroyPipelineLayout(device, p.layout,
                                        nullptr);
                vkDestroyDescriptorSetLayout(device, p.dsl,
                                             nullptr);
            }
            vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (messenger != VK_NULL_HANDLE) {
            auto destroy =
                reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(
                        instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy != nullptr) {
                destroy(instance, messenger, nullptr);
            }
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }
};

bool VulkanContext::available() {
    try {
        VulkanContext ctx;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::uint32_t VulkanContext::validation_errors() {
    return g_validation_errors.load(std::memory_order_relaxed);
}

VulkanContext::VulkanContext() : impl_(new Impl) { impl_->init(); }

VulkanContext::~VulkanContext() = default;

VulkanContext::Buffer VulkanContext::create_buffer(
    std::size_t bytes) {
    auto* hb = new Impl::HostBuffer(impl_->make_buffer(bytes));
    return Buffer{hb};
}

void VulkanContext::destroy_buffer(Buffer b) {
    auto* hb = static_cast<Impl::HostBuffer*>(b.impl);
    if (hb != nullptr) {
        impl_->sets.clear();
        impl_->sets_stale = true;
        impl_->destroy_buffer(*hb);
        delete hb;
    }
}

void VulkanContext::write_buffer(Buffer b,
                                 std::span<const std::byte> data) {
    auto* hb = static_cast<Impl::HostBuffer*>(b.impl);
    std::memcpy(hb->map, data.data(), data.size());
}

void VulkanContext::read_buffer(Buffer b,
                                std::span<std::byte> out) {
    auto* hb = static_cast<Impl::HostBuffer*>(b.impl);
    std::memcpy(out.data(), hb->map, out.size());
}

void* VulkanContext::mapped(Buffer b) {
    return static_cast<Impl::HostBuffer*>(b.impl)->map;
}

void VulkanContext::begin_batch() {
    Impl& im = *impl_;
    VkCommandBufferAllocateInfo cba{};
    cba.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = im.pool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(im.device, &cba,
                                   &im.batch_cmd),
          "allocate command buffer");
    VkCommandBufferBeginInfo cbb{};
    cbb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(im.batch_cmd, &cbb), "begin cmd");
    if (im.qpool != VK_NULL_HANDLE) {
        im.batch_t0 = std::chrono::steady_clock::now();
        im.ts_kernel.clear();
        vkCmdResetQueryPool(im.batch_cmd, im.qpool, 0, Impl::kMaxTs);
        vkCmdWriteTimestamp(im.batch_cmd,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            im.qpool, 0);
    }
}

void VulkanContext::dispatch(Kernel k,
                             std::span<const Buffer> buffers,
                             std::span<const std::uint32_t> push,
                             std::uint32_t groups_x,
                             std::uint32_t groups_y,
                             std::uint32_t groups_z) {
    Impl& im = *impl_;
    const Impl::Pipe& p = im.pipes[static_cast<int>(k)];
    const KernelDesc& kd = kernel_desc(k);
    if (buffers.size() != kd.n_buffers ||
        buffers.size() > Impl::kMaxBufs ||
        push.size() * 4 != kd.push_bytes) {
        throw std::runtime_error("vulkan: dispatch shape");
    }

    Impl::SetKey key{static_cast<int>(k), {}};
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        key.bufs[i] = static_cast<Impl::HostBuffer*>(buffers[i].impl)->buf;
    }
    VkDescriptorSet ds;
    if (auto it = im.sets.find(key); it != im.sets.end()) {
        ds = it->second;
    } else {
        ds = im.alloc_set(p.dsl);
        VkDescriptorBufferInfo infos[Impl::kMaxBufs];
        VkWriteDescriptorSet writes[Impl::kMaxBufs];
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            infos[i] = {key.bufs[i], 0, VK_WHOLE_SIZE};
            writes[i] = {};
            writes[i].sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = static_cast<std::uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(
            im.device, static_cast<std::uint32_t>(buffers.size()),
            writes, 0, nullptr);
        im.sets.emplace(key, ds);
    }

    vkCmdBindPipeline(im.batch_cmd,
                      VK_PIPELINE_BIND_POINT_COMPUTE,
                      p.pipeline);
    vkCmdBindDescriptorSets(
        im.batch_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout,
        0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(im.batch_cmd, p.layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       kd.push_bytes, push.data());
    vkCmdDispatch(im.batch_cmd, groups_x, groups_y, groups_z);
    ++im.prof.dispatches;
    // Slot 0 is the batch start; keep the last slot for its end.
    if (im.per_kernel && im.ts_kernel.size() + 2 < Impl::kMaxTs) {
        im.ts_kernel.push_back(static_cast<int>(k));
        vkCmdWriteTimestamp(
            im.batch_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            im.qpool,
            static_cast<std::uint32_t>(im.ts_kernel.size()));
    }

    // Serialize consecutive dispatches (writes feed reads), unless
    // the driver already does (serial_dispatch).
    if (im.serial_dispatch) {
        return;
    }
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(im.batch_cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, nullptr, 0, nullptr);
}

void VulkanContext::end_batch() {
    Impl& im = *impl_;
    using clock = std::chrono::steady_clock;
    const bool prof = im.qpool != VK_NULL_HANDLE;
    const auto last_ts =
        static_cast<std::uint32_t>(im.ts_kernel.size() + 1);
    if (prof) {
        vkCmdWriteTimestamp(im.batch_cmd,
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            im.qpool, last_ts);
    }
    check(vkEndCommandBuffer(im.batch_cmd), "end cmd");
    const clock::time_point t1 = clock::now();
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &im.batch_cmd;
    check(vkQueueSubmit(im.queue, 1, &si, VK_NULL_HANDLE),
          "submit");
    check(vkQueueWaitIdle(im.queue), "wait idle");
    ++im.prof.batches;
    if (prof) {
        const clock::time_point t2 = clock::now();
        using ms = std::chrono::duration<double, std::milli>;
        im.prof.host_ms += ms(t1 - im.batch_t0).count();
        im.prof.wait_ms += ms(t2 - t1).count();
        std::vector<std::uint64_t> ts(last_ts + 1);
        if (vkGetQueryPoolResults(
                im.device, im.qpool, 0, last_ts + 1,
                ts.size() * sizeof(ts[0]), ts.data(), sizeof(ts[0]),
                VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            const auto to_ms = [&](std::uint64_t a, std::uint64_t b) {
                return static_cast<double>(b - a) * im.ts_period_ns /
                       1e6;
            };
            im.prof.gpu_ms += to_ms(ts[0], ts[last_ts]);
            for (std::size_t i = 0; i < im.ts_kernel.size(); ++i) {
                im.prof.kernel_ms[im.ts_kernel[i]] +=
                    to_ms(ts[i], ts[i + 1]);
                ++im.prof.kernel_n[im.ts_kernel[i]];
            }
        }
    }
    vkFreeCommandBuffers(im.device, im.pool, 1, &im.batch_cmd);
    im.batch_cmd = VK_NULL_HANDLE;
    // Bound the cache when keys keep changing (e.g. a paging
    // budget cycling per-expert buffers).
    if (im.sets_stale || im.sets_live > 65536) {
        im.reset_sets();
    }
}

bool VulkanContext::serial_dispatch() const {
    return impl_->serial_dispatch;
}

bool VulkanContext::has_f16() const { return impl_->has_f16; }

std::uint64_t VulkanContext::memory_available() const {
    const Impl& im = *impl_;
    const std::uint32_t heap = im.host_heap();
    if (im.has_budget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        budget.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 props{};
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        props.pNext = &budget;
        vkGetPhysicalDeviceMemoryProperties2(im.phys, &props);
        return memory_available_from(true, budget.heapBudget[heap],
                                     budget.heapUsage[heap]);
    }
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(im.phys, &props);
    return memory_available_from(false, props.memoryHeaps[heap].size,
                                 im.allocated);
}

void VulkanContext::note_token() { ++impl_->prof.tokens; }

const VulkanContext::Profile& VulkanContext::profile() const {
    return impl_->prof;
}

void VulkanContext::copy_buffer(Buffer src, std::size_t src_off,
                                Buffer dst, std::size_t dst_off,
                                std::size_t bytes) {
    Impl& im = *impl_;
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(im.batch_cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                         &mb, 0, nullptr, 0, nullptr);
    VkBufferCopy region{src_off, dst_off, bytes};
    vkCmdCopyBuffer(im.batch_cmd,
                    static_cast<Impl::HostBuffer*>(src.impl)->buf,
                    static_cast<Impl::HostBuffer*>(dst.impl)->buf,
                    1, &region);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(im.batch_cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, nullptr, 0, nullptr);
}

void VulkanContext::matvec_f32(Buffer w, std::uint32_t rows,
                               std::uint32_t cols, Buffer x,
                               Buffer out) {
    begin_batch();
    const Buffer bufs[] = {w, x, out};
    const std::uint32_t push[] = {
        rows, cols, 0, 0, 0, 0, std::bit_cast<std::uint32_t>(1.0f)};
    dispatch(Kernel::kMatvecF32, bufs, push,
             matvec_groups(Kernel::kMatvecF32, rows));
    end_batch();
}

void VulkanContext::matvec_f32(std::span<const float> w,
                               std::uint32_t rows,
                               std::uint32_t cols,
                               std::span<const float> x,
                               std::span<float> out) {
    if (w.size() != static_cast<std::size_t>(rows) * cols ||
        x.size() != cols || out.size() != rows) {
        throw std::runtime_error("vulkan: matvec size mismatch");
    }
    Buffer wb = create_buffer(w.size_bytes());
    Buffer xb = create_buffer(x.size_bytes());
    Buffer ob = create_buffer(out.size_bytes());
    write_buffer(wb, std::as_bytes(w));
    write_buffer(xb, std::as_bytes(x));
    matvec_f32(wb, rows, cols, xb, ob);
    read_buffer(ob, std::as_writable_bytes(out));
    destroy_buffer(wb);
    destroy_buffer(xb);
    destroy_buffer(ob);
}

}  // namespace locus::backend::vk

#else  // !LOCUS_HAS_VULKAN_KERNELS

namespace locus::backend::vk {

struct VulkanContext::Impl {};

bool VulkanContext::available() { return false; }

std::uint32_t VulkanContext::validation_errors() { return 0; }

VulkanContext::VulkanContext() {
    throw std::runtime_error(
        "locus built without Vulkan kernels");
}

VulkanContext::~VulkanContext() = default;

const char* kernel_name(Kernel) { return "kernel"; }

namespace {

[[noreturn]] void no_kernels() {
    throw std::runtime_error(
        "locus built without Vulkan kernels");
}

}  // namespace

VulkanContext::Buffer VulkanContext::create_buffer(std::size_t) {
    no_kernels();
}
void VulkanContext::destroy_buffer(Buffer) { no_kernels(); }
void VulkanContext::write_buffer(Buffer,
                                 std::span<const std::byte>) {
    no_kernels();
}
void VulkanContext::read_buffer(Buffer, std::span<std::byte>) {
    no_kernels();
}
void* VulkanContext::mapped(Buffer) { no_kernels(); }
void VulkanContext::begin_batch() { no_kernels(); }
void VulkanContext::dispatch(Kernel, std::span<const Buffer>,
                             std::span<const std::uint32_t>,
                             std::uint32_t, std::uint32_t,
                             std::uint32_t) {
    no_kernels();
}
void VulkanContext::end_batch() { no_kernels(); }
void VulkanContext::note_token() { no_kernels(); }
std::uint64_t VulkanContext::memory_available() const { no_kernels(); }
bool VulkanContext::serial_dispatch() const { no_kernels(); }
bool VulkanContext::has_f16() const { no_kernels(); }
const VulkanContext::Profile& VulkanContext::profile() const {
    no_kernels();
}
void VulkanContext::copy_buffer(Buffer, std::size_t, Buffer,
                                std::size_t, std::size_t) {
    no_kernels();
}
void VulkanContext::matvec_f32(Buffer, std::uint32_t,
                               std::uint32_t, Buffer, Buffer) {
    no_kernels();
}
void VulkanContext::matvec_f32(std::span<const float>,
                               std::uint32_t, std::uint32_t,
                               std::span<const float>,
                               std::span<float>) {
    no_kernels();
}

}  // namespace locus::backend::vk

#endif
