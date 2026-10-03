#pragma once

#include <cstdint>

namespace locus::sys {

/**
 * Total physical RAM on this host, in bytes.
 *
 * Linux: sysconf(_SC_PHYS_PAGES) * page size. macOS/Darwin: sysctl
 * hw.memsize. On any platform where it cannot be determined this
 * returns 0, which callers treat as "unknown, skip the check" rather
 * than as zero memory -- a guard must not refuse to start just because
 * it could not measure RAM.
 *
 * @returns total physical RAM in bytes, or 0 if unknown.
 */
std::uint64_t total_ram_bytes();

}  // namespace locus::sys
