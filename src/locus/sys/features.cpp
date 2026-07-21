#include "locus/sys/features.hpp"

#include <cstdint>

#if defined(LOCUS_HAS_VULKAN)
#include <vulkan/vulkan.h>
#endif

namespace locus::sys {

Features detect() {
    Features f;
#if defined(__aarch64__)
    f.neon = true;
#elif defined(__x86_64__)
    f.avx2 = __builtin_cpu_supports("avx2") != 0;
    f.avx512f = __builtin_cpu_supports("avx512f") != 0;
#endif
#if defined(LOCUS_HAS_VULKAN)
    std::uint32_t version = 0;
    f.vulkan = vkEnumerateInstanceVersion(&version) == VK_SUCCESS &&
               version != 0;
#endif
    return f;
}

std::string to_string(const Features& f) {
    auto mark = [](bool on) { return on ? "+" : "-"; };
    return std::string("cpu[") + mark(f.neon) + "neon " + mark(f.avx2) +
           "avx2 " + mark(f.avx512f) + "avx512f] gpu[" + mark(f.vulkan) +
           "vulkan]";
}

}  // namespace locus::sys
