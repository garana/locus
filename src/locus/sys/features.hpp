#pragma once

#include <string>

namespace locus::sys {

/**
 * CPU/GPU capabilities visible to this process, combining what the
 * build enabled with what the running machine supports. Backends
 * use this to pick the best kernel variant at runtime (same scheme
 * as ../pbw: every variant is always compiled when the compiler
 * supports it; selection happens here, not at build time).
 */
struct Features {
    /** ARM NEON (baseline on all arm64 targets). */
    bool neon = false;
    /** x86-64 SSE4.1 detected on the running CPU. */
    bool sse4 = false;
    /** x86-64 AVX2 detected on the running CPU. */
    bool avx2 = false;
    /** x86-64 AVX-512F detected on the running CPU. */
    bool avx512f = false;
    /** Vulkan loader linked in and answering version queries. */
    bool vulkan = false;
};

/**
 * Probes the running machine.
 *
 * @returns The capability set; never throws, absent features are
 *     simply reported false.
 */
Features detect();

/**
 * Fastest-tier physical core count on a heterogeneous CPU (Apple
 * silicon's performance cores, hw.perflevel0.physicalcpu).
 *
 * @returns The count, or 0 where the platform does not report one
 *     (callers then treat every core as equal).
 */
unsigned performance_cores();

/**
 * The CPU's model name, for the metrics' hardware labels: macOS's
 * machdep.cpu.brand_string ("Apple M2 Pro"), Linux's first "model name"
 * in /proc/cpuinfo (x86), else the board in /proc/device-tree/model
 * (ARM boards such as "Raspberry Pi 5 Model B Rev 1.0").
 *
 * @returns The name with surrounding spaces trimmed, or "" when none of
 *     these is available.
 */
std::string cpu_name();

/**
 * @param f A capability set from detect().
 * @returns Human-readable one-line summary, e.g. for server logs.
 */
std::string to_string(const Features& f);

}  // namespace locus::sys
