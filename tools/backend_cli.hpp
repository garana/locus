#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/kv_quant.hpp"
#include "locus/model/arch.hpp"
#include "locus/sys/features.hpp"

#include "cli_spec.hpp"

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

/**
 * Public model id to advertise for a model file: the path's basename,
 * never the full on-disk path. /v1/models and /metrics echo this to
 * unauthenticated clients, so the absolute path (which leaks the
 * server's filesystem layout and username) must not become the model
 * id. Falls back to "locus" when the basename would be empty.
 * @param model_path Path passed on the command line.
 * @returns The basename, or "locus" when it would be empty.
 */
inline std::string public_model_id(const std::string& model_path) {
    const std::string name =
        std::filesystem::path(model_path).filename().string();
    return name.empty() ? "locus" : name;
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
        {T::kBF16, "BF16", "float"},
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
        "pretokenizers: gpt2, llama3, qwen2\n");
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
    "  --audio-decoder P locus-server: enable /v1/audio/speech with\n"
    "                   the neural-codec decoder GGUF at path P\n"
    "  --max-tokens N   locus-server: cap one request's output at N\n"
    "                   tokens (default 1024); also the length of a\n"
    "                   /v1/responses reply that sets no\n"
    "                   max_output_tokens (Codex sets none)\n"
    "  --audio-max-tokens N locus-server: cap one /v1/audio/speech\n"
    "                   request at N audio tokens (7 per 85 ms\n"
    "                   frame; default 1024)\n"
    "  --image-decoder P locus-server: enable /v1/images/generations\n"
    "                   (Janus-Pro models) with the VQ-16 decoder\n"
    "                   GGUF at path P\n"
    "  --image-model D  locus-server: enable /v1/images/generations\n"
    "                   with the image pipeline in directory D: FLUX.1\n"
    "                   (flux, t5encoder, clip_text and autoencoder_kl\n"
    "                   GGUFs) or PixArt-alpha (pixart, t5encoder and\n"
    "                   autoencoder_kl)\n"
    "  --prefill-f16    Vulkan: stage prefill matmul operands in\n"
    "                   f16 (faster prompt ingestion; opt-in, the\n"
    "                   f32 default is exact)\n"
    "  --speculative    prompt-lookup speculative decoding: draft\n"
    "                   tokens from n-grams already in the context,\n"
    "                   verified in one batched forward (greedy,\n"
    "                   same output; wins when output repeats input)\n"
    "  --kv-type T      KV cache precision: f32 (default, exact),\n"
    "                   f16 (half the cache and attention reads;\n"
    "                   GPU-accelerated for GQA models), q8, q4\n"
    "  --batch-decode   enable R10 cross-sequence batched decode\n"
    "  --no-batch-prefill  force per-token prompt ingestion\n"
    "  --timing         locus-run: print prefill and decode\n"
    "                   tok/s to stderr\n"
    "  --warmup         locus-run: run a throwaway 1-token\n"
    "                   request first (weight uploads, pipeline\n"
    "                   warm-up) so --timing excludes them\n"
    "  -h, --help       show this help and exit\n"
    "\n"
    "environment:\n"
    "  LOCUS_BACKEND           backend when --backend absent;\n"
    "                          same names as --backends\n"
    "  LOCUS_MOE_STATS=1       print expert-routing telemetry\n"
    "                          at exit (R8 working-set data)\n"
    "  LOCUS_VK_PROFILE=1      print per-token Vulkan batch,\n"
    "                          host, wait and GPU time at exit\n"
    "                          (=2 also splits GPU time per\n"
    "                          kernel)\n"
    "  LOCUS_VK_BARRIERS=1     Vulkan: keep the barrier between\n"
    "                          dispatches even on MoltenVK\n"
    "                          (whose serial encoders make it\n"
    "                          redundant; skipped by default)\n"
    "  LOCUS_VK_CPU_ROUTE=1    Vulkan: route MoE experts on the\n"
    "                          host (default routes on the GPU\n"
    "                          when all experts fit unbudgeted)\n"
    "  LOCUS_FLUX_GPU=0        FLUX on Vulkan: run the DiT per\n"
    "                          op instead of a resident GPU step\n"
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
    /** From --audio-decoder P: path to the neural-codec decoder GGUF
     * (arch "snac") enabling locus-server's /v1/audio/speech; empty =
     * the endpoint is disabled (503). */
    std::string audio_decoder;
    /** From --max-tokens N: per-request output-token cap (0 = the
     * server default). */
    std::uint32_t max_tokens = 0;
    /** From --audio-max-tokens N: per-request audio-token cap for
     * /v1/audio/speech (0 = the server default). */
    std::uint32_t audio_max_tokens = 0;
    /** From --image-decoder P: path to the VQ-16 decoder GGUF (arch
     * "vq16") enabling locus-server's /v1/images/generations; empty =
     * the endpoint is disabled (503). */
    std::string image_decoder;
    /** From --image-model D: directory of the PixArt-alpha pipeline's
     * GGUFs, enabling locus-server's /v1/images/generations for it;
     * empty = off. */
    std::string image_model;
    /** From --concurrent N: submit the prompt N times and report
     * aggregate throughput (bench mode; default 1 = normal run). */
    std::uint32_t concurrent = 1;
    /** --prefill-f16: f16 operand staging in Vulkan prefill. */
    bool prefill_f16 = false;
    /** --speculative: prompt-lookup speculative decoding. */
    bool speculative = false;
    /** From --kv-type T: KV cache precision (kv_type_ok false when
     * T names no type; kv_type_name keeps it for the error). */
    locus::kv::KvType kv_type = locus::kv::KvType::kF32;
    bool kv_type_ok = true;
    std::string kv_type_name;
    /** --batch-decode: enable R10 cross-sequence batched decode. */
    bool batch_decode = false;
    /** --no-batch-prefill: force per-token prompt ingestion. */
    bool no_batch_prefill = false;
    /** --timing: locus-run prints prefill/decode rates. */
    bool timing = false;
    /** --warmup: locus-run runs a throwaway request first. */
    bool warmup = false;
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
    /** The first numeric flag (--ctx, --concurrent, --audio-max-tokens)
     * whose value is not an integer in [0, INT_MAX], e.g. "--ctx"; empty
     * when all parsed. bad_number_value keeps the value for the error. */
    std::string bad_number_flag;
    std::string bad_number_value;
};

/**
 * Parses a --kv-type value.
 * @param name f32, f16, q8 or q4.
 * @param out Set to the type when name is valid.
 * @returns Whether name is valid.
 */
inline bool parse_kv_type(std::string_view name, locus::kv::KvType& out) {
    using locus::kv::KvType;
    if (name == "f32") {
        out = KvType::kF32;
    } else if (name == "f16") {
        out = KvType::kF16;
    } else if (name == "q8") {
        out = KvType::kQ8;
    } else if (name == "q4") {
        out = KvType::kQ4;
    } else {
        return false;
    }
    return true;
}

inline BackendArgs parse_backend_args(int argc, char** argv) {
    BackendArgs out;
    // Through parse_nonneg_int, so "-1" is an error instead of wrapping
    // to 4294967295 on the cast to uint32_t (i#176).
    const auto number = [&out](std::string_view flag,
                               const std::string& v, std::uint32_t& dst) {
        int n = 0;
        if (parse_nonneg_int(v, n)) {
            dst = static_cast<std::uint32_t>(n);
        } else if (out.bad_number_flag.empty()) {
            out.bad_number_flag = std::string(flag);
            out.bad_number_value = v;
        }
    };
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
            number("--ctx", argv[++i], out.ctx);
        } else if (a.rfind("--ctx=", 0) == 0) {
            number("--ctx", std::string(a.substr(6)), out.ctx);
        } else if (a == "--metrics-path" && i + 1 < argc) {
            out.metrics_path = argv[++i];
        } else if (a.rfind("--metrics-path=", 0) == 0) {
            out.metrics_path = std::string(a.substr(15));
        } else if (a == "--audio-decoder" && i + 1 < argc) {
            out.audio_decoder = argv[++i];
        } else if (a.rfind("--audio-decoder=", 0) == 0) {
            out.audio_decoder = std::string(a.substr(16));
        } else if (a == "--max-tokens" && i + 1 < argc) {
            number("--max-tokens", argv[++i], out.max_tokens);
        } else if (a.rfind("--max-tokens=", 0) == 0) {
            number("--max-tokens", std::string(a.substr(13)),
                   out.max_tokens);
        } else if (a == "--audio-max-tokens" && i + 1 < argc) {
            number("--audio-max-tokens", argv[++i], out.audio_max_tokens);
        } else if (a.rfind("--audio-max-tokens=", 0) == 0) {
            number("--audio-max-tokens", std::string(a.substr(19)),
                   out.audio_max_tokens);
        } else if (a == "--image-decoder" && i + 1 < argc) {
            out.image_decoder = argv[++i];
        } else if (a.rfind("--image-decoder=", 0) == 0) {
            out.image_decoder = std::string(a.substr(16));
        } else if (a == "--image-model" && i + 1 < argc) {
            out.image_model = argv[++i];
        } else if (a.rfind("--image-model=", 0) == 0) {
            out.image_model = std::string(a.substr(14));
        } else if (a == "--concurrent" && i + 1 < argc) {
            number("--concurrent", argv[++i], out.concurrent);
        } else if (a.rfind("--concurrent=", 0) == 0) {
            number("--concurrent", std::string(a.substr(13)),
                   out.concurrent);
        } else if (a == "--prefill-f16") {
            out.prefill_f16 = true;
        } else if (a == "--speculative") {
            out.speculative = true;
        } else if ((a == "--kv-type" && i + 1 < argc) ||
                   a.rfind("--kv-type=", 0) == 0) {
            out.kv_type_name =
                a == "--kv-type" ? argv[++i] : std::string(a.substr(10));
            out.kv_type_ok = parse_kv_type(out.kv_type_name, out.kv_type);
        } else if (a == "--batch-decode") {
            out.batch_decode = true;
        } else if (a == "--no-batch-prefill") {
            out.no_batch_prefill = true;
        } else if (a == "--timing") {
            out.timing = true;
        } else if (a == "--warmup") {
            out.warmup = true;
        } else {
            out.positional.emplace_back(a);
        }
    }
    return out;
}

}  // namespace locus_tools
