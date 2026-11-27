#include "locus/sys/meminfo.hpp"

#include <unistd.h>

#include <cstddef>
#include <cstdint>

#if defined(__APPLE__)
#include <libproc.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif
#if defined(__linux__)
#include <fstream>
#include <string>
#endif

namespace locus::sys {

std::uint64_t total_ram_bytes() {
#if defined(__linux__)
    const long pages = ::sysconf(_SC_PHYS_PAGES);
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(pages) *
           static_cast<std::uint64_t>(page_size);
#elif defined(__APPLE__)
    int mib[2] = {CTL_HW, HW_MEMSIZE};  // 64-bit physical memory size
    std::uint64_t mem = 0;
    std::size_t len = sizeof(mem);
    if (::sysctl(mib, 2, &mem, &len, nullptr, 0) != 0) {
        return 0;
    }
    return mem;
#else
    return 0;  // unknown platform: callers skip the check
#endif
}

std::uint64_t storage_read_bytes() {
#if defined(__linux__)
    std::ifstream f("/proc/self/io");
    std::string key;
    std::uint64_t v = 0;
    while (f >> key >> v) {
        if (key == "read_bytes:") return v;
    }
    return 0;
#elif defined(__APPLE__)
    rusage_info_v2 ri{};
    if (::proc_pid_rusage(::getpid(), RUSAGE_INFO_V2,
                          reinterpret_cast<rusage_info_t*>(&ri)) != 0) {
        return 0;
    }
    return ri.ri_diskio_bytesread;
#else
    return 0;
#endif
}

}  // namespace locus::sys
