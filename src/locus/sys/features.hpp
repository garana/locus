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
 * @param f A capability set from detect().
 * @returns Human-readable one-line summary, e.g. for server logs.
 */
std::string to_string(const Features& f);

}  // namespace locus::sys
