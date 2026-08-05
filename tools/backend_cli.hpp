#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/arch.hpp"
#include "locus/sys/features.hpp"

namespace locus_tools {

/** Why a compiled backend is unavailable on THIS machine (empty when
 * available), derived from the detected CPU/GPU features. */
inline std::string backend_unavail_reason(std::string_view name) {
    const locus::sys::Features f = locus::sys::detect();
    if (name == "neon") {
        return f.neon ? "" : "not an ARM/NEON CPU";
    }
    if (name == "sse4") {
        return f.sse4 ? "" : "CPU lacks SSE4 (not on this arch)";
    }
    if (name == "avx2") {
        return f.avx2 ? "" : "CPU lacks AVX2 (not on this arch)";
    }
    if (name == "vulkan") {
        return f.vulkan ? "Vulkan device unusable"
                        : "no Vulkan loader on this machine";
    }
    if (name == "cuda") {
        return "no CUDA device or non-CUDA build";
    }
    return "unavailable on this machine";
}

/** Prints the supported model architectures. */
inline void print_archs() {
    std::printf("architectures:\n");
    for (const auto& a : locus::model::archs()) {
        std::printf("  %-10s %s\n",
                    std::string(a.name).c_str(),
                    std::string(a.description).c_str());
    }
}

/** Prints the backend registry, marking the auto-pick with '*'.
 * Compiled-but-unavailable backends are shown (not hidden) with the
 * reason they are inactive on this machine. */
inline void print_backends() {
    const auto& best = locus::backend::best_backend();
    std::printf("backends (best first):\n");
    for (const auto& b : locus::backend::backends()) {
        std::string note;
        if (!b.available) {
            note = "  [unavailable: " +
                   backend_unavail_reason(b.name) + "]";
        } else if (!b.selectable) {
            note = "  [not selectable]";
        } else if (&b == &best) {
            note = "  [default]";
        }
        std::printf("%c %-8s %s%s\n", &b == &best ? '*' : ' ',
                    std::string(b.name).c_str(),
                    std::string(b.description).c_str(), note.c_str());
    }
    // Variants compiled out for this architecture (e.g. the x86 SSE4/
    // AVX2 backends on an ARM build) are absent from the registry;
    // list them too so the operator sees the full matrix.
    for (const char* n : {"scalar", "neon", "sse4", "avx2", "vulkan",
                          "cuda"}) {
        if (locus::backend::find_backend(n) == nullptr) {
            std::printf(
                "  %-8s [unavailable: not built for this "
                "architecture]\n",
                n);
        }
    }
}

/** Prints the quantization / tensor types locus can load. */
inline void print_quants() {
    using T = locus::gguf::TensorType;
    struct Q {
        T type;
        const char* name;
        const char* group;
    };
    static constexpr Q kQuants[] = {
        {T::kF32, "F32", "float"},
        {T::kF16, "F16", "float"},
        {T::kQ4_0, "Q4_0", "legacy"},
        {T::kQ4_1, "Q4_1", "legacy"},
        {T::kQ5_0, "Q5_0", "legacy"},
        {T::kQ5_1, "Q5_1", "legacy"},
        {T::kQ8_0, "Q8_0", "legacy"},
        {T::kQ8_1, "Q8_1", "legacy"},
        {T::kQ2_K, "Q2_K", "k-quant"},
        {T::kQ3_K, "Q3_K", "k-quant"},
        {T::kQ4_K, "Q4_K", "k-quant"},
        {T::kQ5_K, "Q5_K", "k-quant"},
        {T::kQ6_K, "Q6_K", "k-quant"},
        {T::kQ8_K, "Q8_K", "k-quant"},
        {T::kIQ2_XXS, "IQ2_XXS", "iq"},
        {T::kIQ2_XS, "IQ2_XS", "iq"},
        {T::kIQ2_S, "IQ2_S", "iq"},
        {T::kIQ3_XXS, "IQ3_XXS", "iq"},
        {T::kIQ3_S, "IQ3_S", "iq"},
        {T::kIQ1_S, "IQ1_S", "iq"},
        {T::kIQ1_M, "IQ1_M", "iq"},
        {T::kIQ4_NL, "IQ4_NL", "iq"},
        {T::kIQ4_XS, "IQ4_XS", "iq"},
        {T::kTQ1_0, "TQ1_0", "ternary/BitNet"},
        {T::kTQ2_0, "TQ2_0", "ternary/BitNet"},
    };
    std::printf("quantization types (loadable weight formats):\n");
    for (const Q& q : kQuants) {
        std::printf("  %-9s %s\n", q.name, q.group);
    }
}

/** Prints the tokenizer families locus understands. */
inline void print_tokenizers() {
    std::printf("tokenizers (by tokenizer.ggml.model):\n");
    std::printf(
        "  spm       SentencePiece unigram (model=llama)\n");
    std::printf(
        "  bpe       byte-level BPE (model=gpt2); "
        "pretokenizers: gpt2, llama3\n");
}

/** Prints the full capability matrix: archs, backends, quants,
 * tokenizers. */
inline void print_capabilities() {
    print_archs();
    std::printf("\n");
    print_backends();
    std::printf("\n");
    print_quants();
    std::printf("\n");
    print_tokenizers();
}

/** Flag lines shared by every tool's --help output. */
inline constexpr const char* kCommonHelp =
    "options:\n"
    "  --backend NAME   select the math backend (see --backends)\n"
    "  --backends       list available backends and exit\n"
    "  --archs          list supported architectures and exit\n"
    "  --quants         list loadable quant types and exit\n"
    "  --tokenizers     list tokenizer families and exit\n"
    "  --capabilities   list archs+backends+quants+tokenizers, "
    "exit\n"
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
    /** --quants: list loadable quant types and exit. */
    bool list_quants = false;
    /** --tokenizers: list tokenizer families and exit. */
    bool list_tokenizers = false;
    /** --capabilities: dump archs+backends+quants+tokenizers. */
    bool list_capabilities = false;
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
        } else if (a == "--quants") {
            out.list_quants = true;
        } else if (a == "--tokenizers") {
            out.list_tokenizers = true;
        } else if (a == "--capabilities") {
            out.list_capabilities = true;
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
