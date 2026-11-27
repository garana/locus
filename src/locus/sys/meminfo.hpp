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

/**
 * Bytes this process has read from storage so far (i#1219): Linux
 * /proc/self/io read_bytes, macOS proc_pid_rusage ri_diskio_bytesread.
 * Page-cache hits are not counted, so a streamed model's expert reads
 * from disk are.
 *
 * @returns the bytes read, or 0 when the platform does not report it.
 */
std::uint64_t storage_read_bytes();

}  // namespace locus::sys
