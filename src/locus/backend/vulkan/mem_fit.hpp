#pragma once

#include <cstdint>

namespace locus::backend::vk {

/**
 * Device memory still allocatable, from what the driver reports.
 * With VK_EXT_memory_budget: the heap budget minus its usage.
 * Without it: 3/4 of the heap size (headroom for the OS and other
 * clients) minus what this context has allocated. Pure, so the
 * policy is testable without a GPU.
 *
 * @param has_budget Whether budget/usage come from the extension.
 * @param budget_or_heap Heap budget (extension) or heap size.
 * @param used Heap usage (extension) or this context's allocations.
 * @returns Available bytes, 0 when exhausted.
 */
constexpr std::uint64_t memory_available_from(bool has_budget,
                                              std::uint64_t budget_or_heap,
                                              std::uint64_t used) {
    const std::uint64_t cap =
        has_budget ? budget_or_heap : budget_or_heap / 4 * 3;
    return cap > used ? cap - used : 0;
}

/**
 * GPU MoE routing keeps every expert resident; allow it only when
 * the model's expert tensors fit in 80% of the available memory.
 *
 * @param expert_bytes All MoE layers' expert tensors, summed.
 * @param available From memory_available_from().
 * @returns true when the experts fit.
 */
constexpr bool moe_experts_fit(std::uint64_t expert_bytes,
                               std::uint64_t available) {
    return expert_bytes <= available / 5 * 4;
}

/**
 * On Apple (MoltenVK) experts imported from the model file in place
 * still count toward the GPU's working set (i#589), though the budget
 * does not show them used. With them, all experts must fit in 90% of
 * the available memory; the copied ones alone keep the 80% rule.
 *
 * @param copied Expert bytes the device holds copies of.
 * @param imported Expert bytes imported in place (0 off Apple).
 * @param available From memory_available_from().
 * @returns true when the experts fit.
 */
constexpr bool moe_experts_fit(std::uint64_t copied, std::uint64_t imported,
                               std::uint64_t available) {
    return moe_experts_fit(copied, available) &&
           copied + imported <= available - available / 10;
}

}  // namespace locus::backend::vk
