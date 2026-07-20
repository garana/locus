#include "cppllm/backend/registry.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "cppllm/backend/variants.hpp"
#include "cppllm/sys/features.hpp"

namespace cppllm::backend {

namespace {

const std::vector<Backend>& list() {
    static const std::vector<Backend> v = [] {
        const sys::Features f = sys::detect();
        std::vector<Backend> b;
#if defined(__aarch64__)
        b.push_back({"neon",
                     "ARM NEON vectorized matvec (F32/Q8_0)",
                     f.neon, true,
                     {&matvec_neon, &dequant_row, nullptr}});
#endif
        b.push_back({"scalar", "portable reference (all types)",
                     true, true,
                     {&matvec, &dequant_row, nullptr}});
        const bool vk = f.vulkan && vulkan_backend_usable();
        b.push_back({"vulkan",
                     "GPU forward pass via Vulkan: paged/MLA "
                     "attention and MoE experts on GPU "
                     "(F32/F16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K)",
                     vk, vk,
                     {&matvec_vulkan, &dequant_row,
                      &vulkan_alloc_kv}});
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
        if (const char* env = std::getenv("CPPLLM_BACKEND")) {
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

}  // namespace cppllm::backend
