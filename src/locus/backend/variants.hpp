#pragma once

#include <span>
#include <string>

#include "locus/backend/cpu_ops.hpp"

namespace locus::backend {

struct KvAttend;  // registry.hpp

#if defined(__aarch64__)
/**
 * NEON-vectorized matvec: F32 and Q8_0 rows use NEON inner loops;
 * other weight types delegate to the scalar reference.
 */
void matvec_neon(const Mat& w, std::span<const float> x,
                 std::span<float> out);

/**
 * NEON 8-bit activation matvec (R16): quantizes x once (Q8_K for Q4_K
 * / Q6_K rows, Q8_0 for block-32 rows) and integer-dots it with the
 * weights. Q4_K / Q6_K use matvec_batch_neon_q8k's kernels with one
 * token (exact integers per block, one float step per block), so a
 * token's value is the same alone or in a batch, and within f32
 * rounding of scalar matvec_q8k. Block-32 rows are bit-identical to
 * the scalar dot. Other types delegate to scalar matvec_q8k.
 */
void matvec_neon_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out);

/**
 * NEON register-blocked batched matvec (R11, Ops::matvec_batch):
 * applies w to all n token vectors in x_batch (token-major, token t
 * at t*w.cols) into out_batch (row-major, out[r*n + t]). Q4_K and
 * Q6_K rows are weight-stationary -- each weight block is read and
 * dequantized ONCE, then FMA'd into all n token accumulators, so
 * weight DRAM traffic drops ~n-fold vs n matvec() calls. Other
 * types fall back to per-token matvec_neon. Byte-identical to n
 * matvec_neon() calls (same per-block accumulation order).
 * Single-threaded: the model row-slices and calls this per slice.
 */
void matvec_batch_neon(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch,
                       std::uint32_t n);

/**
 * matvec_neon_q8k on n tokens (Ops::matvec_q8k_batch): token-major
 * x_batch, row-major out_batch (out[r*n + t]). Each token is quantized
 * once, then each weight row is dotted against four tokens at a time
 * while it is in cache. Byte-identical to n matvec_neon_q8k() calls
 * (the same row dot on the same quants).
 */
void matvec_batch_neon_q8k(const Mat& w, std::span<const float> x_batch,
                           std::span<float> out_batch, std::uint32_t n);

/** quantize_activation_q8k with NEON: bit-identical to it (the same
 *  operations, the same rounding, no contraction). */
void quantize_activation_q8k_neon(std::span<const float> x, float* d,
                                  std::int8_t* qs, std::int16_t* bsums);
/** quantize_activation_q8_0 with NEON, bit-identical to it. */
void quantize_activation_q8_0_neon(std::span<const float> x, float* d,
                                   std::int8_t* qs);

#if defined(LOCUS_HAS_NEON_DOTPROD_VARIANT)
/**
 * The four NEON entry points above, compiled again with the 8-bit
 * dot-product extension (matvec_neon_dotprod.cpp) for a build whose
 * default target lacks it. Bit-identical to the baseline ones; the
 * registry uses them when sys::detect() reports dotprod.
 */
namespace neon_dotprod {
void matvec_neon(const Mat& w, std::span<const float> x,
                 std::span<float> out);
void matvec_neon_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out);
void matvec_batch_neon(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch, std::uint32_t n);
void matvec_batch_neon_q8k(const Mat& w, std::span<const float> x_batch,
                           std::span<float> out_batch, std::uint32_t n);
}  // namespace neon_dotprod
#endif

#if defined(LOCUS_HAS_NEON_I8MM_VARIANT)
/**
 * Q8_0 weights through the 8-bit matrix multiply (smmla,
 * matvec_neon_i8mm.cpp), in llama.cpp's float order (one scaled
 * integer dot per block): not bit-identical to the baseline kernels,
 * so the registry uses them only under cpu_fast() on a CPU with i8mm.
 * A token gets the same bits alone or batched. Other weight types go
 * to the baseline matvec_neon_q8k / matvec_batch_neon_q8k.
 */
namespace neon_i8mm {
void matvec_neon_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out);
void matvec_batch_neon_q8k(const Mat& w, std::span<const float> x_batch,
                           std::span<float> out_batch, std::uint32_t n);
}  // namespace neon_i8mm
#endif
#endif

#if defined(__x86_64__)
/**
 * SSE4-vectorized matvec: F32 and Q8_0 rows use 128-bit SSE4.1
 * inner loops; other weight types delegate to the scalar
 * reference. Compiled only on x86-64 hosts (per-source -msse4.1).
 * The baseline x86 vector path for CPUs without AVX2.
 */
void matvec_sse4(const Mat& w, std::span<const float> x,
                 std::span<float> out);

/**
 * SSE4 Q8_K-activation matvec (ggml parity, DESIGN.md R16): quantizes
 * x to Q8_K once and integer-dots it with the quantized weight using
 * SSE4.1 (per-lane integer sums, bit-identical to scalar matvec_q8k).
 * Q8_0 (against Q8_0 activations) and Q2_K..Q6_K are vectorized; other
 * types delegate to scalar matvec_q8k.
 */
void matvec_sse4_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out);

/**
 * matvec_sse4_q8k on n tokens (Ops::matvec_q8k_batch): each token's
 * activation quantized once, then each weight row dotted with a tile
 * of tokens while it is in cache, through the same row kernels, so the
 * result is byte-identical to n matvec_sse4_q8k calls. out_batch is
 * row-major (out[r*n + t]).
 */
void matvec_batch_sse4_q8k(const Mat& w, std::span<const float> x_batch,
                           std::span<float> out_batch, std::uint32_t n);

/**
 * SSE4 batched matvec (R11 Ops::matvec_batch): Q4_K/Q6_K use a
 * register-blocked kernel (weight block dequanted once, FMA'd into n
 * token accumulators -- byte-identical to n matvec_sse4() calls but
 * ~n-fold less weight traffic); other types delegate to
 * matvec_batch_scalar. out_batch is row-major (out[r*n + t]).
 */
void matvec_batch_sse4(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch, std::uint32_t n);

/**
 * AVX2-vectorized matvec: F32 and Q8_0 rows use AVX2 inner
 * loops; other weight types delegate to the scalar reference.
 * Compiled only on x86-64 hosts (per-source -mavx2).
 */
void matvec_avx2(const Mat& w, std::span<const float> x,
                 std::span<float> out);
#endif

/**
 * Vulkan matvec: weights with a kernel (F32, F16, BF16, Q8_0,
 * Q4_0, Q5_0, Q2_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ4_XS) run
 * on the GPU (uploaded once, resident, cached by host pointer);
 * other types delegate to the scalar reference.
 * Attention and norms stay on the CPU (hybrid backend).
 */
void matvec_vulkan(const Mat& w, std::span<const float> x,
                   std::span<float> out);

/**
 * matvec_batch on the GPU: w applied to n token-major inputs in one
 * batched matmul dispatch (rec_matmul), output row-major
 * (out[r * n + t]) as the Ops contract says. Weights are cached by
 * host pointer like matvec_vulkan's, so pass only stable (mmapped)
 * weights, never temporary host buffers.
 */
void matvec_batch_vulkan(const Mat& w, std::span<const float> x,
                         std::span<float> out, std::uint32_t n);

/**
 * Ops for the image models (T5, PixArt DiT, VAE) on Vulkan: matvec and
 * matvec_batch on the GPU, self_parallel, so dense::linear hands over
 * whole matrices and all tokens at once. Kept apart from the vulkan
 * backend entry, whose language-model paths do their own batching.
 */
/**
 * matvec_batch_vulkan's GPU work with the output token-major
 * (out[t * rows + r]), as the image models hold activations, so no
 * transpose either way. Same weight rule: stable (mmapped) weights only.
 */
void vulkan_linear_tokens(const Mat& w, std::span<const float> x,
                          std::uint32_t n, std::span<float> out);

/**
 * conv2d on the GPU (conv2d.comp: one invocation per output, summing
 * ci, ky, kx in conv2d's order), same arguments and checks
 * (conv2d_check_args). The weight and bias are cached by host pointer
 * like the matvec weights, so pass only stable (mmapped) ones.
 */
void conv2d_vulkan(std::span<const float> in, std::uint32_t in_ch,
                   std::uint32_t in_h, std::uint32_t in_w,
                   std::span<const float> weight, std::uint32_t out_ch,
                   std::uint32_t kh, std::uint32_t kw,
                   std::span<const float> bias, std::uint32_t stride,
                   std::uint32_t pad, std::uint32_t groups,
                   std::span<float> out);

struct Ops;
const Ops& vulkan_image_ops();

/**
 * Multi-head attention without a mask on the GPU, in one batch:
 * S_h = Q_h K_h^T / sqrt(hd), a softmax per row, O_h = P_h V_h. The
 * operands go to scratch buffers rewritten every call (never the weight
 * pool), so temporary host buffers are fine here.
 *
 * @param q nq rows of heads * hd floats (head h at h * hd).
 * @param kh Per-head keys: [heads][nk][hd].
 * @param vt Per-head values, transposed: [heads][hd][nk].
 * @param out nq rows of heads * hd floats.
 * @throws std::invalid_argument on short spans or a score buffer past
 *     2^32 floats.
 */
void vulkan_attention(std::span<const float> q, std::uint32_t nq,
                      std::span<const float> kh, std::span<const float> vt,
                      std::uint32_t nk, std::uint32_t heads, std::uint32_t hd,
                      std::span<float> out);

/**
 * @returns true when the Vulkan device can import host memory in
 * place (VK_EXT_external_memory_host, unified memory as on Apple):
 * mmapped F16, F32 and K-quant weights then cost it no device memory.
 * Q8_0 weights are still copied (repacked for the GPU), so a Q8_0 T5
 * would take about 5 GB of the pool. Without import the image models'
 * F16 / F32 weights (T5-XXL alone is 9.5 GB) would be copied into the
 * weight pool next to a language model's.
 */
bool vulkan_imports_host_memory();

/** @returns true when a usable Vulkan device + kernels exist. */
bool vulkan_backend_usable();

/**
 * @returns The Vulkan device's name, or "" when no device is usable.
 *     Opens the device on first use and takes the GPU lock, so call it
 *     once at startup, not on a hot path.
 */
std::string vulkan_device_name();

/** A Vulkan device with a compute queue (i#572). */
struct VulkanDeviceInfo {
    std::string name;
    /** "discrete", "integrated", "virtual", "cpu" or "other". */
    std::string type;
};

/** @returns The Vulkan devices with a compute queue, in the order
 *  vulkan_set_device counts them; empty without Vulkan. Opens no
 *  device. */
std::vector<VulkanDeviceInfo> vulkan_devices();

/**
 * Runs the Vulkan backend on device `index` of vulkan_devices() instead
 * of the first, for the whole process (as cuda_set_device). Call it
 * before the first Vulkan use (before the model loads); a call after
 * that names the same device or throws.
 * @throws std::out_of_range for an index past vulkan_devices();
 *     std::runtime_error when the backend already runs on another
 *     device, or in a build without Vulkan.
 */
void vulkan_set_device(int index);

/**
 * Runs the calling thread's Vulkan work on device `index` (i#573): each
 * device has its own backend state (context, kernels, weight uploads,
 * KV pools), so two threads bound to two devices run independently.
 * A thread never bound uses the process's device (vulkan_set_device).
 * A KV cache belongs to the device of the thread that made it; using
 * it from a thread bound to another throws. vulkan_set_prefill_f16,
 * vulkan_reset_state and vulkan_release_weights act on the calling
 * thread's device only.
 * @throws std::out_of_range for an index past vulkan_devices().
 */
void vulkan_bind_thread(int index);

/** @returns The device index the calling thread's Vulkan work uses. */
int vulkan_thread_device();

/**
 * @returns "moltenvk" when the Vulkan driver is MoltenVK, else
 *     "vulkan" (also when no device is usable). Asks the device once
 *     and caches the answer, so it is cheap on a hot path.
 */
const char* vulkan_api_name();

/**
 * Opt-in (--prefill-f16): the Vulkan prefill matmuls stage both
 * operands as f16 in shared memory (~+18% prefill on an M4 Max). Off
 * (the default) keeps prefill f32-exact.
 * @param on Whether to stage in f16 from now on.
 */
void vulkan_set_prefill_f16(bool on);

/** @returns Whether prefill stages operands in f16 (the effective
 *  vulkan_set_prefill_f16 setting; false without a usable device). */
bool vulkan_prefill_f16();

/**
 * @returns The query tokens per attention sub-block of the batched
 * Vulkan prefill for n queries of `heads` heads over n_pos positions:
 * as many as keep the heads x n_pos score rows within the scratch cap
 * (1 GiB of floats by default), at least 1 and at most n (i#313).
 */
std::uint32_t vulkan_attn_block(std::uint32_t n, std::uint32_t heads,
                                std::uint32_t n_pos);

/**
 * Sets the batched prefill's attention scratch cap to `floats` (0
 * restores the default). TEST USE ONLY: lets a small model run its
 * chunk's queries in several sub-blocks.
 */
void vulkan_set_attn_scratch_floats(std::size_t floats);

/**
 * Sets decode attention's partials cap to `blocks` per round (0
 * restores the default, 256). TEST USE ONLY: lets a short context take
 * the multi-round merge (i#116).
 */
void vulkan_set_attn_part_blocks(std::uint32_t blocks);

/** KV-pool allocator returning a GPU-mapped pointer (Ops hook). */
float* vulkan_alloc_kv(std::size_t n_floats);
/** Frees a pool from vulkan_alloc_kv (no-op for an unknown pointer, or
 *  once the Vulkan state is torn down at exit). */
void vulkan_free_kv(float* pool);
/** @returns The GPU-mapped KV pools alive now (for tests). */
std::size_t vulkan_kv_pool_count();

/**
 * Frees the process-lifetime Vulkan backend state: the weight pool
 * buffers and activation scratch; the GPU-mapped KV pools stay with
 * their caches, which free them (i#428). TEST
 * USE ONLY -- the singleton's pool is the intended resident working
 * set in production (one model per process); this exists so a test
 * process running many models across many cases does not accumulate
 * GPU allocations until the device runs out. Call between tests.
 */
void vulkan_reset_state();

/**
 * Batch-invariant Vulkan decode (#432): when on, a lone sequence also
 * decodes through vulkan_forward_multi's kernels, so a request's
 * logits are bit-identical alone or batched; off, it takes
 * vulkan_forward's single-sequence kernels. The default is per device:
 * on for Apple GPUs, where a lone sequence costs little (level on an
 * M2 Pro, -5% on an M4 Max), off elsewhere (-33% on V3D; others not
 * measured). Under the default, a lone sequence on a model with Q4_K
 * or Q6_K weights still takes vulkan_forward (their short-batch
 * kernels cost it 9 to 11% on an M2 Pro), so such a model is not
 * batch-invariant unless the mode is set. LOCUS_VK_BATCH_INVARIANT=1 /
 * 0 (or this call) sets it, for every weight type.
 */
void vulkan_set_batch_invariant(bool on);
/** Drops a vulkan_set_batch_invariant setting: back to the
 * LOCUS_VK_BATCH_INVARIANT or per-device default. */
void vulkan_reset_batch_invariant();
/** @returns Whether batch-invariant Vulkan decode is on. */
bool vulkan_batch_invariant();

/**
 * Drops the Vulkan weight-pool buffers (copies and in-place imports)
 * over host memory [p, p + n), as an unmap does: an image pipeline
 * whose parts do not all fit the GPU at once releases one part before
 * the next (FLUX: T5-XXL before the DiT). The next use uploads or
 * imports them again. No-op without Vulkan.
 * @throws std::logic_error inside an open batch.
 */
void vulkan_release_weights(const void* p, std::size_t n);

/**
 * CUDA matvec: F32, Q8_0, Q2_K, Q4_K, Q5_K, Q6_K and the
 * IQ1_S/IQ2_XXS/IQ3_XXS/IQ4_XS families run on an NVIDIA GPU
 * (pooled/paged); other types delegate to scalar.
 * When the build has no CUDA toolkit a stub delegates entirely to
 * scalar and reports the backend unusable.
 */
void matvec_cuda(const Mat& w, std::span<const float> x,
                 std::span<float> out);

/**
 * CUDA Q8_K-activation matvec (ggml parity, DESIGN.md R16): quantizes
 * x to Q8_K once on the host, uploads the shared quants, and integer-
 * dots them with the device-resident quantized weight -- bit-identical
 * to the scalar matvec_q8k. Ported types run on the GPU; the rest
 * delegate to scalar matvec_q8k. This is the device half of the eventual
 * ggml-parity flip; kept additive so the f32 matvec_cuda parity path
 * stays intact until the flip lands. Scalar-delegating stub on non-CUDA
 * builds.
 */
void matvec_cuda_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out);

/**
 * CUDA batched matvec (R11 Ops::matvec_batch, self_parallel):
 * one kernel launch over all rows x n tokens. Q4_K/Q6_K read each
 * weight block from VRAM once and FMA it into n token accumulators
 * (bit-identical to n matvec_cuda() calls, ~n-fold less VRAM weight
 * traffic, weight shared from the pool); other types fall back to n
 * matvec_cuda() calls. out_batch is row-major (out[r*n + t]); x_batch
 * token-major. Scalar-delegating stub on non-CUDA builds.
 */
void matvec_batch_cuda(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch, std::uint32_t n);

/** @returns true when a usable CUDA device is present. */
bool cuda_backend_usable();

/**
 * @returns The current CUDA device's name (cudaDeviceProp::name), or ""
 *     with no usable device or on a non-CUDA build.
 */
std::string cuda_device_name();

/**
 * Binds this process to CUDA device `ordinal` for all subsequent CUDA
 * work (i#24 inc 4, the device-select seam). Validates the ordinal
 * against the device count and fails LOUDLY -- throws std::out_of_range
 * for an out-of-range ordinal and std::runtime_error when no CUDA
 * device / non-CUDA build -- rather than silently falling back to
 * device 0. Records the ordinal so every CUDA matvec entry re-binds its
 * calling thread to it (CUDA's current device is per-thread, and the
 * stage executor runs forwards on its own worker thread). Must be the
 * first device-selecting CUDA call; cuda_backend_usable()'s
 * cudaGetDeviceCount is safe to call before it (it binds no pool).
 * @param ordinal Zero-based CUDA device index.
 */
void cuda_set_device(int ordinal);

/**
 * Frees every resident page in the CUDA weight pool (test seam). The
 * pool keys weights by host pointer, valid only while that buffer is
 * live; tests reset it between cases so transient weight buffers whose
 * addresses get recycled do not alias a stale device page. No-op on
 * non-CUDA builds. Inference does not need it while weights stay
 * mapped for the model's life; a model unload and reload must call it
 * (or cuda_evict per weight) before the new weights run.
 */
void cuda_pool_reset();

/**
 * Async weight prefetch (Ops.prefetch): begins a cudaMemcpyAsync of
 * w onto a dedicated stream into the CUDA weight pool, so a later
 * matvec_cuda(w) finds it resident. No-op for weight types the GPU
 * kernels do not handle, and a no-op stub on non-CUDA builds.
 */
void cuda_prefetch(const Mat& w);

/**
 * Frees w's page in the CUDA weight pool now (Ops.evict), unless a
 * matvec holds it. No-op when w is not resident, and on non-CUDA
 * builds. Call it before freeing w's host bytes when that address may
 * hold another weight later: the pool keys pages by host address.
 */
void cuda_evict(const Mat& w);

/**
 * @returns The CUDA weight pool's byte budget (Ops.pool_budget):
 *     LOCUS_GPU_POOL_MB, else about 80% of free VRAM; 0 on non-CUDA
 *     builds.
 */
std::size_t cuda_pool_budget();

/**
 * Device mirror of a host F32 KV pool (Ops.kv_mirror_*, i#466).
 * cuda_kv_mirror_alloc returns `n_floats` zeroed device floats, outside
 * the weight pool's budget, or nullptr when they do not fit (and always
 * on non-CUDA builds).
 */
void* cuda_kv_mirror_alloc(std::size_t n_floats);

/**
 * Copies n floats from host `src` to mirror `dev` at float offset
 * `off`; `src` may be reused on return.
 * @returns false on a CUDA error.
 */
bool cuda_kv_mirror_store(void* dev, std::size_t off, const float* src,
                          std::size_t n);

/**
 * Reads n floats of mirror `dev` at float offset `off` into `dst` (a
 * test seam). @returns false on a CUDA error.
 */
bool cuda_kv_mirror_load(const void* dev, std::size_t off, float* dst,
                         std::size_t n);

/** Frees a cuda_kv_mirror_alloc allocation. */
void cuda_kv_mirror_free(void* dev);

/**
 * GQA attention on the GPU over a cuda_kv_mirror_alloc mirror
 * (Ops.kv_attend): one block per (KV head, query), so a query's output
 * does not depend on how many queries share the launch. Sums in another
 * order than the host's gqa_head, so it matches it to rounding, not bits.
 * @returns false on a CUDA error or an unsupported shape (always on
 *     non-CUDA builds).
 */
bool cuda_kv_attend(const KvAttend& a);

// ---- Device primitives for a CUDA forward (i#517). Each takes and
// returns device buffers and does not wait: the work is queued in order
// and cuda_dev_finish waits for it and unpins every weight the calls
// pinned. All return false on a CUDA error or a weight the device cannot
// hold (and always on non-CUDA builds), and the caller then runs the
// host path. rmsnorm, RoPE and add round each step as the host ops do,
// so they equal them bit for bit; silu_mul uses the device expf.

/** @returns n device floats, or nullptr. */
float* cuda_dev_alloc(std::size_t n);
/** Frees a cuda_dev_alloc buffer. */
void cuda_dev_free(float* p);
/** Copies n floats to / from the device (these two wait). */
bool cuda_dev_upload(float* d, const float* h, std::size_t n);
bool cuda_dev_download(float* h, const float* d, std::size_t n);
/**
 * @returns A device copy of host vector h (a norm weight), kept in the
 *     weight pool by its host pointer and pinned until cuda_dev_finish;
 *     nullptr when it does not fit.
 */
const float* cuda_dev_const(const float* h, std::size_t n);
/** dout = w dx: matvec_cuda's kernel, so the same bits. */
bool cuda_dev_matvec(const Mat& w, const float* dx, float* dout);
/**
 * n token-major rows of x (n x cols) to n token-major rows of out
 * (n x rows): matvec_batch_cuda's kernels, so the same bits as n
 * cuda_dev_matvec calls. scratch holds rows * min(n, 32) floats.
 */
bool cuda_dev_matvec_batch(const Mat& w, const float* dx, float* dout,
                           std::uint32_t n, float* scratch);
/** rmsnorm of `tokens` rows of n floats with weight dw. */
bool cuda_dev_rmsnorm(const float* dx, const float* dw, float eps,
                      float* dout, std::uint32_t n, std::uint32_t tokens);
/**
 * rope_norm on `tokens` rows of n_heads * hd floats, stride floats apart,
 * from host-computed (cos, sin) pairs: hd / 2 per token in dcs.
 */
bool cuda_dev_rope(float* dx, const float* dcs, std::uint32_t n_heads,
                   std::uint32_t hd, std::size_t stride,
                   std::uint32_t tokens);
/** dg = silu(dg) * du over n floats. */
bool cuda_dev_silu_mul(float* dg, const float* du, std::size_t n);
/** dx += dy over n floats. */
bool cuda_dev_add(float* dx, const float* dy, std::size_t n);
/** Raw device bytes, for the forward's integer tables. */
void* cuda_dev_alloc_bytes(std::size_t bytes);
void cuda_dev_free_bytes(void* p);
/** Copies bytes to the device (waits). */
bool cuda_dev_upload_bytes(void* d, const void* h, std::size_t bytes);
/**
 * Writes token t's K and V rows (dk / dv, n x kv_dim) into the KV mirror
 * at float offset drows[t] + layer_off (V v_off further), and into the
 * staging buffer at t * stage_stride + stage_off (V kv_dim further), for
 * the host pool's copy. drows is a device array of n offsets, uploaded
 * once per forward (it does not depend on the layer).
 */
bool cuda_dev_kv_scatter(const float* dk, const float* dv,
                         std::uint32_t n, std::uint32_t kv_dim,
                         const std::uint64_t* drows, std::size_t layer_off,
                         std::size_t v_off, float* mirror, float* stage,
                         std::size_t stage_stride, std::size_t stage_off);
/** Waits for the queued work and unpins the weights. @returns false on
 *  a CUDA error. */
bool cuda_dev_finish();

/**
 * CUDA 1D convolution (F32), the device half of the conv1d scalar
 * reference (generative-media codec decoders). Same contract and
 * channels-major layout as backend::conv1d: `out_len` comes from
 * conv1d_out_len, weight is [out_ch][in_ch/groups][k], groups == in_ch
 * is depthwise. One thread per output element. Falls back to the scalar
 * reference on any CUDA error; a scalar-delegating stub on non-CUDA
 * builds.
 */
void conv1d_cuda(std::span<const float> in, std::uint32_t in_ch,
                 std::uint32_t in_len, std::span<const float> weight,
                 std::uint32_t out_ch, std::uint32_t k,
                 std::span<const float> bias, std::uint32_t stride,
                 std::uint32_t dilation, std::uint32_t pad,
                 std::uint32_t groups, std::span<float> out);

/**
 * CUDA transposed (upsampling) 1D convolution (F32); device half of
 * backend::conv1d_transpose. `out_len` from conv1d_transpose_out_len,
 * weight is input-channel-first [in_ch][out_ch/groups][k]. Each thread
 * gathers one output element (ix = (ox + pad - t*dilation) / stride
 * when it divides). Scalar fallback on error; stub on non-CUDA builds.
 */
void conv1d_transpose_cuda(std::span<const float> in,
                           std::uint32_t in_ch, std::uint32_t in_len,
                           std::span<const float> weight,
                           std::uint32_t out_ch, std::uint32_t k,
                           std::span<const float> bias,
                           std::uint32_t stride, std::uint32_t dilation,
                           std::uint32_t pad, std::uint32_t groups,
                           std::span<float> out);

/**
 * CUDA Snake activation in place (per-channel), device half of
 * backend::snake1d: x += (1 / (alpha + eps)) * sin(alpha * x)^2, one
 * thread per element. Scalar fallback on error; stub on non-CUDA builds.
 */
void snake1d_cuda(std::span<float> x, std::uint32_t ch,
                  std::uint32_t len, std::span<const float> alpha,
                  float eps = 1e-9f);

/**
 * CUDA 2D convolution (F32), the device half of backend::conv2d (image
 * VQ decoder, i#223). Same arguments and channels-major layout as
 * conv2d. One thread per output element, summed in conv2d's order
 * without fused multiply-add, so it is bit-identical to conv2d when the
 * host compiler does not contract either. A bad shape throws
 * std::invalid_argument (conv2d_check_args); a CUDA error falls back to
 * the scalar conv2d; the non-CUDA stub checks, then runs conv2d.
 */
void conv2d_cuda(std::span<const float> in, std::uint32_t in_ch,
                 std::uint32_t in_h, std::uint32_t in_w,
                 std::span<const float> weight, std::uint32_t out_ch,
                 std::uint32_t kh, std::uint32_t kw,
                 std::span<const float> bias, std::uint32_t stride,
                 std::uint32_t pad, std::uint32_t groups,
                 std::span<float> out);

}  // namespace locus::backend
