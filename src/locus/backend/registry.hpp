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
     * Optional cache-blocked batched matvec (R11): applies w to n
     * input vectors at once. Input x_batch is n * w.cols contiguous
     * with vector t at t*w.cols (token-major, as the model holds
     * activations). Output out_batch is w.rows * n contiguous in
     * ROW-MAJOR order -- result[row r, token t] at r*n + t. That
     * row-major output lets the caller row-slice the matrix across
     * threads (each slice writes a contiguous span) and matches the
     * register-blocked kernel's natural write pattern (n running
     * accumulators -> n contiguous writes per row). The model
     * transposes it back to token-major for downstream ops.
     *
     * A backend fills this with a kernel that reads/dequantizes each
     * weight block once and reuses it across all n columns, cutting
     * weight DRAM traffic ~n-fold on batched decode. Must be
     * byte-identical to n matvec() calls (same per-block
     * accumulation order). nullptr means "no batched kernel": the
     * model loops matvec() per token instead.
     */
    void (*matvec_batch)(const Mat&, std::span<const float>,
                         std::span<float>, std::uint32_t n) = nullptr;
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
    /**
     * True when matvec() may be called concurrently from several
     * threads (the R9 row-split, matvec_mt). The CPU kernels and
     * the CUDA pager are re-entrant; the Vulkan backend drives a
     * single-threaded VulkanContext singleton, so it sets this
     * false and matvec_mt runs it inline. Applies to the
     * per-op fallback path, which is the only time a GPU
     * backend's matvec() is reached through matvec_mt.
     */
    bool mt_safe = true;
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
