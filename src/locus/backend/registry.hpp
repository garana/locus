#pragma once

#include <span>
#include <string_view>

#include "locus/backend/cpu_ops.hpp"

namespace locus::backend {

/** Op table a backend provides to the model runtime. Ops not in
 * the table (rmsnorm, rope, ...) are negligible on CPU backends;
 * the vulkan backend replaces the whole forward pass instead. */
struct Ops {
    void (*matvec)(const Mat&, std::span<const float>,
                   std::span<float>);
    void (*dequant_row)(const Mat&, std::uint32_t,
                        std::span<float>);
    /**
     * Optional KV-pool allocator (nullptr: heap). The vulkan
     * backend hands out a GPU-mapped pointer so the paged cache
     * lives in memory the attention kernel can read directly.
     */
    float* (*alloc_kv)(std::size_t n_floats);
    /**
     * Optional one-step-ahead weight-prefetch hint (nullptr on
     * backends without a pager). The model may call this at the R8
     * readahead points -- next layer's static tensors, and each
     * routed expert at MoE selection time -- with the exact same Mat
     * (host pointer) a later matvec() will use; a GPU backend begins
     * an async upload into its weight pool so the matvec finds it
     * resident. Fire-and-forget: it never blocks and may no-op.
     */
    void (*prefetch)(const Mat& w);
};

/**
 * One selectable math implementation. All variants are always
 * compiled in (pbw pattern); `available` reflects what the running
 * machine supports, `selectable` whether the backend can serve
 * inference today (the Vulkan entry is listed but not selectable
 * until M5 completes).
 */
struct Backend {
    std::string_view name;
    std::string_view description;
    bool available;
    bool selectable;
    Ops ops;
};

/** @returns All known backends, best-first for this build. */
std::span<const Backend> backends();

/** @returns The backend named `name`, or nullptr. */
const Backend* find_backend(std::string_view name);

/** @returns The best available, selectable backend. */
const Backend& best_backend();

/**
 * Resolves the backend to use: explicit choice > LOCUS_BACKEND
 * env var > best available.
 *
 * @param choice Backend name from the CLI, or empty.
 * @throws std::invalid_argument on unknown or unusable names.
 */
const Backend& resolve_backend(std::string_view choice);

}  // namespace locus::backend
