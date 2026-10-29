#include "locus/sys/features.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>

#if defined(LOCUS_HAS_VULKAN)
#include <vulkan/vulkan.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace locus::sys {

Features detect() {
    Features f;
#if defined(__aarch64__)
    f.neon = true;
#elif defined(__x86_64__)
    f.sse4 = __builtin_cpu_supports("sse4.1") != 0;
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

unsigned performance_cores() {
#if defined(__APPLE__)
    int n = 0;
    std::size_t len = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, nullptr,
                     0) == 0 &&
        n > 0) {
        return static_cast<unsigned>(n);
    }
#endif
    return 0;
}

namespace {

/** @returns s without leading and trailing spaces, tabs and NULs. */
std::string trim(std::string s) {
    const auto keep = [](char c) {
        return c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '\0';
    };
    std::size_t b = 0, e = s.size();
    while (b < e && !keep(s[b])) ++b;
    while (e > b && !keep(s[e - 1])) --e;
    return s.substr(b, e - b);
}

}  // namespace

std::string cpu_name() {
#if defined(__APPLE__)
    char buf[256] = {};
    std::size_t len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) ==
            0 &&
        len > 0) {
        return trim(std::string(buf, std::strlen(buf)));
    }
    return "";
#else
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line, implementer, part;
    while (std::getline(cpuinfo, line)) {
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string value = trim(line.substr(colon + 1));
        if (line.rfind("model name", 0) == 0) {
            return value;
        }
        if (line.rfind("CPU implementer", 0) == 0 && implementer.empty()) {
            implementer = value;
        } else if (line.rfind("CPU part", 0) == 0 && part.empty()) {
            part = value;
        }
    }
    std::ifstream board("/proc/device-tree/model");
    std::string model((std::istreambuf_iterator<char>(board)),
                      std::istreambuf_iterator<char>());
    model = trim(model);
    if (model.empty() && !implementer.empty() && !part.empty()) {
        model = "arm " + implementer + ":" + part;
    }
    return model;
#endif
}

std::string to_string(const Features& f) {
    auto mark = [](bool on) { return on ? "+" : "-"; };
    return std::string("cpu[") + mark(f.neon) + "neon " + mark(f.sse4) +
           "sse4 " + mark(f.avx2) + "avx2 " + mark(f.avx512f) +
           "avx512f] gpu[" + mark(f.vulkan) + "vulkan]";
}

}  // namespace locus::sys
