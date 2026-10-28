#include "locus/backend/registry.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "locus/backend/variants.hpp"
#include "locus/sys/features.hpp"

namespace locus::backend {

namespace {

const std::vector<Backend>& list() {
    static const std::vector<Backend> v = [] {
        const sys::Features f = sys::detect();
        std::vector<Backend> b;
        b.reserve(6);  // fits every arch's entries; also avoids a
                       // GCC-13 -Wstringop-overflow false positive
                       // on the vector-growth path.
// GCC 13 -Wextra flags designated initializers that omit trailing
// members (which are value-initialized to their defaults); clang does
// not. The Ops entries below deliberately set only the fields a
// backend provides, so silence that GCC-only false positive here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#if defined(__aarch64__)
        b.push_back({"neon",
                     "ARM NEON vectorized matvec (F32/BF16/Q8_0/"
                     "Q4_K/Q6_K/IQ1_S) and 8-bit activation dots "
                     "(Q4_K/Q6_K/Q8_0/Q4_0/IQ4_NL)",
                     f.neon, true,
                     {.matvec = &matvec_neon,
                      .matvec_q8k = &matvec_neon_q8k,
                      .dequant_row = &dequant_row,
                      .matvec_batch = &matvec_batch_neon,
                      .prefers_q8k = true}});
#endif
#if defined(__x86_64__)
        b.push_back({"avx2",
                     "x86-64 AVX2 vectorized matvec (F32/Q8_0)",
                     f.avx2, true,
                     {.matvec = &matvec_avx2,
                      .matvec_q8k = &matvec_q8k,  // scalar (no AVX2 Q8K)
                      .dequant_row = &dequant_row}});
        b.push_back({"sse4",
                     "x86-64 SSE4.1 vectorized matvec (F32/Q8_0/"
                     "Q4_K/Q6_K), 8-bit activation dots (K-quants), "
                     "Q4_K/Q6_K register-blocked batch",
                     f.sse4, true,
                     {.matvec = &matvec_sse4,
                      .matvec_q8k = &matvec_sse4_q8k,
                      .dequant_row = &dequant_row,
                      .matvec_batch = &matvec_batch_sse4,
                      .prefers_q8k = true}});
#endif
        // The scalar backend ships the R11 batched reference; neon
        // ships its register-blocked kernel (Q4_K/Q6_K + per-token
        // fallback). sse4/avx2 leave matvec_batch null for now and
        // use the per-token fallback until their kernels land
        // (owner: vx).
        b.push_back({"scalar", "portable reference (all types)",
                     true, true,
                     {.matvec = &matvec,
                      .matvec_q8k = &matvec_q8k,
                      .dequant_row = &dequant_row,
                      .matvec_batch = &matvec_batch_scalar,
                      .prefers_q8k = true}});
        const bool vk = f.vulkan && vulkan_backend_usable();
        b.push_back({"vulkan",
                     "GPU forward pass via Vulkan: paged/MLA "
                     "attention and MoE experts on GPU "
                     "(F32/F16/BF16/Q8_0/Q4_0/Q5_0/Q2_K/Q4_K/Q5_K/"
                     "Q6_K/IQ2_XXS/IQ3_XXS/IQ4_XS)",
                     vk, vk,
                     {.matvec = &matvec_vulkan,
                      .matvec_q8k = &matvec_q8k,  // scalar (pi: Vulkan)
                      .dequant_row = &dequant_row,
                      .alloc_kv = &vulkan_alloc_kv,
                      .mt_safe = false}});
        const bool cu = cuda_backend_usable();
        b.push_back({"cuda",
                     "NVIDIA CUDA matvec (F32/Q8_0/Q2_K/Q4_K/Q5_K/"
                     "Q6_K, IQ1_S/IQ1_M/IQ2_*/IQ3_XXS/IQ3_S/IQ4_NL/"
                     "IQ4_XS, TQ1_0/TQ2_0 on GPU, pooled weights; "
                     "other scalar)",
                     cu, cu,
                     {.matvec = &matvec_cuda,
                      .matvec_q8k = &matvec_cuda_q8k,
                      .dequant_row = &dequant_row,
                      .matvec_batch = &matvec_batch_cuda,
                      .self_parallel = true,
                      .prefetch = &cuda_prefetch}});
#pragma GCC diagnostic pop
        return b;
    }();
    return v;
}

}  // namespace

std::span<const Backend> backends() { return list(); }

const Backend* find_backend(std::string_view name) {
    for (const Backend& b : list()) {
        if (b.name == name) {
            return &b;
        }
    }
    return nullptr;
}

const Backend& best_backend() {
    for (const Backend& b : list()) {
        if (b.available && b.selectable) {
            return b;
        }
    }
    return *find_backend("scalar");  // always present
}

const Backend& resolve_backend(std::string_view choice) {
    std::string name(choice);
    if (name.empty()) {
        if (const char* env = std::getenv("LOCUS_BACKEND")) {
            name = env;
        }
    }
    if (name.empty()) {
        return best_backend();
    }
    const Backend* b = find_backend(name);
    if (b == nullptr) {
        throw std::invalid_argument("unknown backend: " + name);
    }
    if (!b->available) {
        throw std::invalid_argument(
            "backend not available on this machine: " + name);
    }
    if (!b->selectable) {
        throw std::invalid_argument(
            "backend not selectable for inference yet: " + name);
    }
    return *b;
}

bool backend_defaulted(std::string_view choice) {
    const char* env = std::getenv("LOCUS_BACKEND");
    return choice.empty() && (env == nullptr || *env == '\0');
}

std::string gpu_idle_warning(bool defaulted, const Backend& picked,
                             std::span<const Backend> list) {
    auto is_gpu = [](std::string_view name) {
        return name == "vulkan" || name == "cuda";
    };
    if (!defaulted || is_gpu(picked.name)) {
        return {};
    }
    for (const Backend& b : list) {
        if (is_gpu(b.name) && b.available && b.selectable) {
            const std::string gpu(b.name);
            return "warning: running on the CPU (" +
                   std::string(picked.name) +
                   ") by default; a GPU is usable: pass --backend " +
                   gpu + " (or set LOCUS_BACKEND=" + gpu + ")";
        }
    }
    return {};
}

}  // namespace locus::backend
