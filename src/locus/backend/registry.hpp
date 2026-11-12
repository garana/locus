#pragma once

#include <span>
#include <string>
#include <string_view>

#include "locus/backend/cpu_ops.hpp"

namespace locus::backend {

/**
 * GQA attention over a device KV mirror (Ops.kv_attend, i#466): n_q
 * queries of one sequence at positions base .. base + n_q - 1, query i
 * seeing positions 0 .. base + i. Offsets are in floats into the mirror,
 * which has the host paged pool's layout.
 */
struct KvAttend {
    const void* mirror = nullptr;
    /** The sequence's block table: position t is in blocks[t / bt]. */
    const std::uint32_t* blocks = nullptr;
    std::uint32_t n_blocks = 0;
    /** Floats per block, the layer's offset in a block, and V's offset
     *  past K inside the layer. */
    std::size_t block_stride = 0, layer_off = 0, v_off = 0;
    std::uint32_t block_tokens = 0;
    std::uint32_t n_heads = 0, n_kv_heads = 0, head_dim = 0;
    float kq_scale = 1.0f;
    std::uint32_t n_q = 0, base = 0;
    /**
     * Several sequences (batch decode): when set, query i is at position
     * q_pos[i] of the sequence whose block table starts at
     * blocks[q_table[i]], and base is ignored. Both hold n_q entries.
     */
    const std::uint32_t* q_pos = nullptr;
    const std::uint32_t* q_table = nullptr;
    /** n_q rows of n_heads * head_dim floats, q_stride floats apart. */
    const float* q = nullptr;
    /** Out: as q. */
    float* out = nullptr;
    std::size_t q_stride = 0;
    /** q and out are device pointers of a CUDA forward (i#517), not
     *  host ones: nothing is copied and the call does not wait. */
    bool device_io = false;
    /** With device_io: device copies of blocks (n_blocks entries) and
     *  of the query table (the n_q positions, then the n_q block-table
     *  offsets), uploaded once per forward; null: uploaded per call.
     *  The host blocks / q_pos / q_table stay set (bounds, chunk count). */
    const void* dev_blocks = nullptr;
    const void* dev_meta = nullptr;
};

/** Op table a backend provides to the model runtime. Ops not in
 * the table (rmsnorm, rope, ...) are negligible on CPU backends;
 * the vulkan backend replaces the whole forward pass instead. */
struct Ops {
    void (*matvec)(const Mat&, std::span<const float>,
                   std::span<float>);
    /**
     * Q8_K-activation matvec (ggml parity, DESIGN.md R16): quantizes
     * the activation to Q8_K and integer-dots it with the quantized
     * weight, bit-exact to llama.cpp on QK_K weight types (vs the
     * f32-activation dot in `matvec`). Each backend points this at its
     * own Q8_K entry (scalar matvec_q8k, matvec_cuda_q8k,
     * matvec_sse4_q8k), all of which fall back to the scalar Q8_K dot
     * for types they do not vectorize. Non-QK_K weight types delegate
     * to `matvec` internally, so this is always safe to call. The model
     * routes QK_K matvecs here once the ggml-parity flip lands; until
     * then it is an additive entry that leaves `matvec` the default.
     */
    void (*matvec_q8k)(const Mat&, std::span<const float>,
                       std::span<float>) = nullptr;
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
     *
     * A kernel that special-cases only some weight types MUST fall
     * back to n calls of the SAME backend's matvec() (scattered into
     * the row-major output) for the rest -- never to the scalar
     * reference. A SIMD backend does F32/Q8_0 in SIMD, so delegating
     * those to matvec_batch_scalar would diverge from this backend's
     * own per-token path and break batched==per-token.
     */
    void (*matvec_batch)(const Mat&, std::span<const float>,
                         std::span<float>, std::uint32_t n) = nullptr;
    /**
     * Optional batched matvec_q8k, in matvec_batch's layouts: each
     * token is quantized once and each weight row is dotted against
     * all n tokens while it is in cache. Must be byte-identical to n
     * matvec_q8k() calls (the same per-row, per-token dot). nullptr:
     * the model loops matvec_q8k per token.
     */
    void (*matvec_q8k_batch)(const Mat&, std::span<const float>,
                             std::span<float>, std::uint32_t n) = nullptr;
    /**
     * Optional batched matvec writing token-major output (out[t *
     * w.rows + r], the layout the model wants): a GPU kernel that
     * produces it natively skips matvec_batch's row-major round trip,
     * two transposes of every output on the host (i#671). When set,
     * the model's batched matmuls call it instead of matvec_batch.
     */
    void (*linear_tokens)(const Mat&, std::span<const float> x,
                          std::uint32_t n, std::span<float> out) = nullptr;
    /**
     * True when the backend parallelizes the weight rows itself and
     * wants the WHOLE matrix in each matvec, matvec_q8k and
     * matvec_batch call (a GPU backend: its kernel grids over all
     * rows in a single launch). The model then skips the CPU
     * row-slice fan-out (see splits_rows). Row slices would also
     * defeat a weight pool keyed by host pointer: slice 0 shares the
     * whole weight's pointer with a different size. Distinct from
     * mt_safe, which only says matvec() is re-entrant: CUDA is both.
     * CPU SIMD backends leave this false so the model still splits
     * rows across cores.
     */
    bool self_parallel = false;
    /**
     * Optional KV-pool allocator (nullptr: heap). The vulkan
     * backend hands out a GPU-mapped pointer so the paged cache
     * lives in memory the attention kernel can read directly.
     */
    float* (*alloc_kv)(std::size_t n_floats);
    /**
     * Releases a pool alloc_kv returned (nullptr: nothing to free).
     * The cache built on it calls this when it is destroyed.
     */
    void (*free_kv)(float* pool) = nullptr;
    /**
     * Optional device mirror of a host F32 KV pool (i#466; nullptr:
     * none). Allocates `n_floats` device floats, or returns nullptr when
     * the device cannot hold them (the cache then has no mirror).
     */
    void* (*kv_mirror_alloc)(std::size_t n_floats) = nullptr;
    /** Copies n host floats to mirror `dev` at float offset `off`;
     *  @returns false on a device error. */
    bool (*kv_mirror_store)(void* dev, std::size_t off, const float* src,
                            std::size_t n) = nullptr;
    /** Frees a kv_mirror_alloc allocation. */
    void (*kv_mirror_free)(void* dev) = nullptr;
    /**
     * Optional attention over a kv_mirror_alloc mirror (nullptr: the
     * host attends). Each (query, KV head) pair is computed the same way
     * whatever n_q is, so prefill and decode give the same bits.
     * @returns false on a device error (the caller then attends on the
     *     host).
     */
    bool (*kv_attend)(const KvAttend& a) = nullptr;
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
     * Optional: frees w's copy in the backend's weight pool now
     * (nullptr on backends without one). The MoE hybrid mode calls it
     * when an expert leaves the GPU hot set, so the pool's room goes
     * to the expert that replaced it. Never blocks on compute.
     */
    void (*evict)(const Mat& w);
    /** Optional: @returns the weight pool's byte budget. */
    std::size_t (*pool_budget)();
    /**
     * True when matvec() may be called concurrently from several
     * threads (the R9 row-split, matvec_mt). The CPU kernels and
     * the CUDA pager are re-entrant; the Vulkan backend drives a
     * single-threaded VulkanContext singleton, so it sets this
     * false and matvec_mt runs it inline.
     */
    bool mt_safe = true;
    /**
     * True when matvec_q8k is this backend's FAST path, so the R16
     * Q8-activation mode is its default (DESIGN.md R16): neon, sse4
     * and the scalar reference. avx2 (scalar Q8 dot vs its AVX2 f32
     * one), cuda (no block-32 Q8 kernel) and vulkan (scalar, inline
     * per-op fallback) keep the f32 dot unless the arch or the caller
     * asks for Q8.
     */
    bool prefers_q8k = false;
};

/**
 * @returns true when the model should split a matvec's weight rows
 *     across CPU threads for `op`: matvec is re-entrant (mt_safe) and
 *     the backend does not grid over all rows itself (self_parallel).
 */
inline bool splits_rows(const Ops& op) {
    return op.mt_safe && !op.self_parallel;
}

/**
 * One selectable math implementation. All variants the compiler
 * supports are compiled in; `available` reflects what the running
 * machine supports, `selectable` whether the backend can serve
 * inference (today every available backend can).
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

/**
 * True when resolve_backend(choice) picks by default: no CLI choice
 * and no LOCUS_BACKEND.
 *
 * @param choice Backend name from the CLI, or empty.
 */
bool backend_defaulted(std::string_view choice);

/**
 * Whether a Vulkan device of this kind is measured faster than the CPU:
 * a discrete GPU or an Apple integrated one. A CPU-type device
 * (llvmpipe) and other integrated GPUs are not: on a Raspberry Pi 5,
 * V3D decodes about 15 times slower than the CPU (i#607).
 *
 * @param type The device type, as VulkanDeviceInfo::type gives it.
 * @param name The device name.
 * @returns true when suggesting Vulkan over the CPU is right.
 */
bool vulkan_beats_cpu(std::string_view type, std::string_view name);

/** @returns vulkan_beats_cpu() for the device the calling thread's
 *  Vulkan backend runs on; false without a Vulkan device. Opens no
 *  device. */
bool vulkan_device_beats_cpu();

/**
 * The startup warning for a default pick that leaves a GPU idle: the
 * picked backend runs on the CPU while a GPU backend (vulkan, cuda) in
 * `list` is available and selectable. Vulkan counts only when its
 * device beats the CPU.
 *
 * @param defaulted Whether the backend was picked by default.
 * @param picked The backend in use.
 * @param list The backends to look for a usable GPU in.
 * @param vulkan_beats Whether the Vulkan device beats the CPU.
 * @returns One line naming the GPU backend to pass, or empty when
 *     there is nothing to warn about.
 */
std::string gpu_idle_warning(bool defaulted, const Backend& picked,
                             std::span<const Backend> list = backends(),
                             bool vulkan_beats = vulkan_device_beats_cpu());

}  // namespace locus::backend
