#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/model/arch.hpp"

namespace locus_tools {

/** Prints the supported model architectures. */
inline void print_archs() {
    std::printf("architectures:\n");
    for (const auto& a : locus::model::archs()) {
        std::printf("  %-10s %s\n",
                    std::string(a.name).c_str(),
                    std::string(a.description).c_str());
    }
}

/** Prints the backend registry, marking the auto-pick with '*'. */
inline void print_backends() {
    const auto& best = locus::backend::best_backend();
    std::printf("backends (best first):\n");
    for (const auto& b : locus::backend::backends()) {
        const char* note = !b.available      ? "  [unavailable]"
                           : !b.selectable   ? "  [not selectable]"
                           : &b == &best     ? "  [default]"
                                             : "";
        std::printf("%c %-8s %s%s\n", &b == &best ? '*' : ' ',
                    std::string(b.name).c_str(),
                    std::string(b.description).c_str(), note);
    }
}

/** Flag lines shared by every tool's --help output. */
inline constexpr const char* kCommonHelp =
    "options:\n"
    "  --backend NAME   select the math backend (see --backends)\n"
    "  --backends       list available backends and exit\n"
    "  --archs          list supported architectures and exit\n"
    "  --ctx N          cap the KV pool at N tokens (bounds the\n"
    "                   dirty memory a run can allocate; default\n"
    "                   min(model context, 4096))\n"
    "  --concurrent N   locus-run: submit the prompt N times and\n"
    "                   report aggregate tok/s (bench mode)\n"
    "  --metrics-path P locus-server: serve Prometheus metrics at\n"
    "                   path P (default /metrics)\n"
    "  --batch-decode   enable R10 cross-sequence batched decode\n"
    "  --no-batch-prefill  force per-token prompt ingestion\n"
    "  -h, --help       show this help and exit\n"
    "\n"
    "environment:\n"
    "  LOCUS_BACKEND           backend when --backend absent;\n"
    "                          same names as --backends\n"
    "  LOCUS_MOE_STATS=1       print expert-routing telemetry\n"
    "                          at exit (R8 working-set data)\n"
    "  LOCUS_BATCH_DEQUANT=1   in batched forward, dequant each\n"
    "                          weight row once then dot all N\n"
    "                          tokens (token-exact, not byte-\n"
    "                          identical to the fused matvec)\n"
    "  LOCUS_NO_READAHEAD=1    disable the default madvise\n"
    "                          readahead of upcoming weights\n"
    "                          (routed experts + next layer)\n"
    "  LOCUS_THREADS=N         cap the matvec thread fan-out\n"
    "                          (default: hardware cores; 1\n"
    "                          runs single-threaded)\n"
    "  LOCUS_PIN_STATIC=1      mlock non-expert weights at\n"
    "                          load (streaming models; R9)\n"
    "  LOCUS_WEIGHT_WINDOW=1   drop routed-expert pages right\n"
    "                          after use (madvise DONTNEED) so\n"
    "                          streamed models never build up\n"
    "                          memory pressure\n";

/** Backend flags peeled off argv; the rest stays positional. */
struct BackendArgs {
    std::vector<std::string> positional;
    /** From --backend NAME or --backend=NAME; empty if absent. */
    std::string choice;
    /** From --ctx N: KV pool cap in tokens (0 = default). */
    std::uint32_t ctx = 0;
    /** From --metrics-path P: locus-server metrics route (empty =
     * default /metrics). */
    std::string metrics_path;
    /** From --concurrent N: submit the prompt N times and report
     * aggregate throughput (bench mode; default 1 = normal run). */
    std::uint32_t concurrent = 1;
    /** --batch-decode: enable R10 cross-sequence batched decode. */
    bool batch_decode = false;
    /** --no-batch-prefill: force per-token prompt ingestion. */
    bool no_batch_prefill = false;
    /** --backends was given: list and exit. */
    bool list = false;
    /** --archs was given: list architectures and exit. */
    bool list_archs = false;
    /** --help / -h was given: print usage and exit. */
    bool help = false;
};

inline BackendArgs parse_backend_args(int argc, char** argv) {
    BackendArgs out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--backends") {
            out.list = true;
        } else if (a == "--archs") {
            out.list_archs = true;
        } else if (a == "--help" || a == "-h") {
            out.help = true;
        } else if (a == "--backend" && i + 1 < argc) {
            out.choice = argv[++i];
        } else if (a.rfind("--backend=", 0) == 0) {
            out.choice = std::string(a.substr(10));
        } else if (a == "--ctx" && i + 1 < argc) {
            out.ctx = static_cast<std::uint32_t>(
                std::atol(argv[++i]));
        } else if (a.rfind("--ctx=", 0) == 0) {
            out.ctx = static_cast<std::uint32_t>(
                std::atol(std::string(a.substr(6)).c_str()));
        } else if (a == "--metrics-path" && i + 1 < argc) {
            out.metrics_path = argv[++i];
        } else if (a.rfind("--metrics-path=", 0) == 0) {
            out.metrics_path = std::string(a.substr(15));
        } else if (a == "--concurrent" && i + 1 < argc) {
            out.concurrent = static_cast<std::uint32_t>(
                std::atol(argv[++i]));
        } else if (a.rfind("--concurrent=", 0) == 0) {
            out.concurrent = static_cast<std::uint32_t>(
                std::atol(std::string(a.substr(13)).c_str()));
        } else if (a == "--batch-decode") {
            out.batch_decode = true;
        } else if (a == "--no-batch-prefill") {
            out.no_batch_prefill = true;
        } else {
            out.positional.emplace_back(a);
        }
    }
    return out;
}

}  // namespace locus_tools
