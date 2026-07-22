// CUDA matvec variant. Weights run through a device weight pool (the
// R8-GPU pager): each weight is uploaded once, keyed by host pointer,
// and reused across tokens under a byte budget with LRU eviction;
// weights too big for the budget fall back to an uncached per-call
// upload. Kernels dequantize F32/Q8_0/Q4_K/IQ1_S on-device (the pool
// stores the source quantized bytes); other types delegate to scalar.

#include "locus/backend/variants.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "iq_grids.h"  // iq1s_grid lookup table (global scope)

namespace locus::backend {

namespace {

/** Aborts to scalar on any CUDA error (caller falls back). */
bool cuda_ok(cudaError_t e) { return e == cudaSuccess; }

__global__ void matvec_f32_kernel(const float* w, const float* x,
                                  float* out, std::uint32_t rows,
                                  std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const float* row = w + static_cast<std::size_t>(r) * cols;
    float acc = 0.0f;
    for (std::uint32_t i = 0; i < cols; ++i) {
        acc += row[i] * x[i];
    }
    out[r] = acc;
}

/** One Q8_0 block is 34 bytes: f16 scale + 32 int8 quants. */
__global__ void matvec_q8_0_kernel(const std::uint8_t* w,
                                   const float* x, float* out,
                                   std::uint32_t rows,
                                   std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nblk = cols / 32;
    const std::size_t row_bytes =
        static_cast<std::size_t>(nblk) * 34;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nblk; ++b) {
        const std::uint8_t* blk = row + b * 34;
        std::uint16_t hbits =
            static_cast<std::uint16_t>(blk[0]) |
            (static_cast<std::uint16_t>(blk[1]) << 8);
        const float d = __half2float(__ushort_as_half(hbits));
        const std::int8_t* q =
            reinterpret_cast<const std::int8_t*>(blk + 2);
        const float* xb = x + static_cast<std::size_t>(b) * 32;
        float bacc = 0.0f;
        for (int g = 0; g < 32; ++g) {
            bacc += static_cast<float>(q[g]) * xb[g];
        }
        acc += d * bacc;
    }
    out[r] = acc;
}

/** Unpacks the 6-bit scale/min of K-quant sub-block j from the
 *  packed 12-byte scales array (ggml get_scale_min_k4). */
__device__ void scale_min_k4(int j, const std::uint8_t* q,
                             std::uint8_t& d, std::uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = static_cast<std::uint8_t>((q[j + 4] & 0x0f) |
                                      ((q[j - 4] >> 6) << 4));
        m = static_cast<std::uint8_t>((q[j + 4] >> 4) |
                                      ((q[j] >> 6) << 4));
    }
}

/** One Q4_K super-block is 144 bytes: d,dmin (f16) + scales[12] +
 *  qs[128], 256 weights. Dequant + accumulate in the exact element
 *  order of the scalar dequant_block_q4_k / dot_k_quant so the dot
 *  is token-exact with the CPU reference. */
__global__ void matvec_q4_k_kernel(const std::uint8_t* w,
                                   const float* x, float* out,
                                   std::uint32_t rows,
                                   std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 144;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            144;
        const std::uint16_t db =
            static_cast<std::uint16_t>(blk[0]) |
            (static_cast<std::uint16_t>(blk[1]) << 8);
        const std::uint16_t dmb =
            static_cast<std::uint16_t>(blk[2]) |
            (static_cast<std::uint16_t>(blk[3]) << 8);
        const float d = __half2float(__ushort_as_half(db));
        const float dmin = __half2float(__ushort_as_half(dmb));
        const std::uint8_t* scales = blk + 4;
        const std::uint8_t* q = blk + 16;
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        int is = 0, yi = 0;
        for (int j = 0; j < 256; j += 64) {
            std::uint8_t sc, mn;
            scale_min_k4(is + 0, scales, sc, mn);
            const float d1 = d * sc, m1 = dmin * mn;
            scale_min_k4(is + 1, scales, sc, mn);
            const float d2 = d * sc, m2 = dmin * mn;
            for (int l = 0; l < 32; ++l) {
                const float val =
                    d1 * static_cast<float>(q[l] & 0x0f) - m1;
                acc += val * xb[yi++];
            }
            for (int l = 0; l < 32; ++l) {
                const float val =
                    d2 * static_cast<float>(q[l] >> 4) - m2;
                acc += val * xb[yi++];
            }
            q += 32;
            is += 2;
        }
    }
    out[r] = acc;
}

/** One IQ1_S super-block is 50 bytes: d (f16) + qs[32] + qh[8]u16,
 *  256 weights. Each of 8 sub-blocks picks 4 grid entries (8 int8
 *  each) scaled by dl with a +/-0.125 delta. Dequant + accumulate in
 *  the exact order of the scalar dequant_block_iq1_s so the dot is
 *  token-exact. `grid` is the device copy of iq1s_grid. */
__global__ void matvec_iq1_s_kernel(const std::uint8_t* w,
                                    const std::uint64_t* grid,
                                    const float* x, float* out,
                                    std::uint32_t rows,
                                    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    constexpr float kIq1sDelta = 0.125f;
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 50;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            50;
        const std::uint16_t db =
            static_cast<std::uint16_t>(blk[0]) |
            (static_cast<std::uint16_t>(blk[1]) << 8);
        const float d = __half2float(__ushort_as_half(db));
        const std::uint8_t* qs = blk + 2;
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        int yi = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* qhp = blk + 34 + 2 * ib;
            const std::uint16_t qh =
                static_cast<std::uint16_t>(qhp[0]) |
                (static_cast<std::uint16_t>(qhp[1]) << 8);
            const float dl = d * (2 * ((qh >> 12) & 7) + 1);
            const float delta =
                (qh & 0x8000) ? -kIq1sDelta : kIq1sDelta;
            for (int l = 0; l < 4; ++l) {
                const std::uint32_t gi =
                    qs[l] | (((qh >> (3 * l)) & 7) << 8);
                const std::int8_t* g =
                    reinterpret_cast<const std::int8_t*>(&grid[gi]);
                for (int j = 0; j < 8; ++j) {
                    const float val =
                        dl * (static_cast<float>(g[j]) + delta);
                    acc += val * xb[yi++];
                }
            }
            qs += 4;
        }
    }
    out[r] = acc;
}

/** RAII-ish scoped device buffer; frees on scope exit. */
struct DevBuf {
    void* p = nullptr;
    bool alloc(std::size_t bytes) {
        return cuda_ok(cudaMalloc(&p, bytes));
    }
    ~DevBuf() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }
};

/**
 * Device weight pool (R8-GPU pager). Caches one device buffer per
 * weight, keyed by host pointer, so a weight is uploaded once and
 * reused across tokens under a byte budget with LRU eviction.
 * acquire() pins the page for the duration of a matvec so a
 * concurrent call cannot evict a weight in use; release() unpins.
 * A weight larger than the whole budget, or when every resident page
 * is pinned, returns nullptr so the caller uses an uncached per-call
 * upload -- preserving correctness on tiny-VRAM hosts. The pool holds
 * the source (quantized) bytes; the kernels dequantize on-device.
 *
 * prefetch() (phase 3) uploads a weight asynchronously on a dedicated
 * non-blocking stream so the copy overlaps the default-stream compute
 * of the current layer/expert; acquire() then waits on that copy's
 * event before handing the pointer to the kernel.
 */
class WeightPool {
 public:
    /** @returns device buffer for `host` (uploading on miss) with the
     *  page pinned, or nullptr if it cannot be cached. */
    const void* acquire(const void* host, std::size_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        if (budget_ == 0) {
            init_budget();
        }
        const std::uint64_t now = ++clock_;
        if (auto it = table_.find(host); it != table_.end()) {
            wait_ready(it->second);  // finish any in-flight prefetch
            it->second.last_use = now;
            it->second.pins++;
            hits_++;
            return it->second.dptr;
        }
        void* dptr = alloc(bytes);
        if (dptr == nullptr) {
            return nullptr;  // too big / all pinned: caller demand path
        }
        if (!cuda_ok(cudaMemcpy(dptr, host, bytes,
                                cudaMemcpyHostToDevice))) {
            cudaFree(dptr);
            used_ -= bytes;
            return nullptr;
        }
        upload_bytes_ += bytes;
        misses_++;
        table_.emplace(host,
                       Page{dptr, bytes, now, 1, nullptr, false});
        return dptr;
    }

    /** Fire-and-forget async upload into the pool (Ops.prefetch). */
    void prefetch(const void* host, std::size_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        if (budget_ == 0) {
            init_budget();
        }
        const std::uint64_t now = ++clock_;
        if (auto it = table_.find(host); it != table_.end()) {
            it->second.last_use = now;  // already resident/in flight
            return;
        }
        ensure_stream();
        void* dptr = alloc(bytes);
        if (dptr == nullptr) {
            return;  // no room now; matvec will demand-load it
        }
        cudaEvent_t ev = nullptr;
        if (stream_ != nullptr &&
            cuda_ok(cudaEventCreateWithFlags(
                &ev, cudaEventDisableTiming)) &&
            cuda_ok(cudaMemcpyAsync(dptr, host, bytes,
                                    cudaMemcpyHostToDevice,
                                    stream_)) &&
            cuda_ok(cudaEventRecord(ev, stream_))) {
            upload_bytes_ += bytes;
            prefetches_++;
            table_.emplace(host, Page{dptr, bytes, now, 0, ev, true});
            return;
        }
        // Async path unavailable: synchronous upload keeps the page
        // valid and reusable.
        if (ev != nullptr) {
            cudaEventDestroy(ev);
        }
        if (!cuda_ok(cudaMemcpy(dptr, host, bytes,
                                cudaMemcpyHostToDevice))) {
            cudaFree(dptr);
            used_ -= bytes;
            return;
        }
        upload_bytes_ += bytes;
        prefetches_++;
        table_.emplace(host,
                       Page{dptr, bytes, now, 0, nullptr, false});
    }

    void release(const void* host) {
        std::lock_guard<std::mutex> lk(mu_);
        if (auto it = table_.find(host);
            it != table_.end() && it->second.pins > 0) {
            it->second.pins--;
        }
    }

    ~WeightPool() {
        if (std::getenv("LOCUS_MOE_STATS") == nullptr) {
            return;
        }
        const std::uint64_t tot = hits_ + misses_;
        std::fprintf(
            stderr,
            "[cuda-pool] budget=%zuMB resident=%zuMB hits=%llu "
            "misses=%llu hit_rate=%.1f%% prefetches=%llu "
            "evictions=%llu uploaded=%lluMB\n",
            budget_ >> 20, used_ >> 20,
            static_cast<unsigned long long>(hits_),
            static_cast<unsigned long long>(misses_),
            tot ? 100.0 * static_cast<double>(hits_) /
                      static_cast<double>(tot)
                : 0.0,
            static_cast<unsigned long long>(prefetches_),
            static_cast<unsigned long long>(evictions_),
            static_cast<unsigned long long>(upload_bytes_ >> 20));
        // Process exit frees the device; skip cudaFree to avoid
        // ordering issues with CUDA context teardown.
    }

 private:
    struct Page {
        void* dptr;
        std::size_t bytes;
        std::uint64_t last_use;
        int pins;
        cudaEvent_t ev;   // set while an async prefetch is in flight
        bool inflight;
    };

    void ensure_stream() {
        if (stream_ == nullptr) {
            cudaStreamCreateWithFlags(&stream_,
                                      cudaStreamNonBlocking);
        }
    }

    /** Blocks until a page's in-flight prefetch copy is complete, so
     *  the default-stream kernel that follows reads valid data. */
    void wait_ready(Page& pg) {
        if (pg.inflight) {
            cudaEventSynchronize(pg.ev);
            cudaEventDestroy(pg.ev);
            pg.ev = nullptr;
            pg.inflight = false;
        }
    }

    void init_budget() {
        if (const char* mb = std::getenv("LOCUS_GPU_POOL_MB")) {
            budget_ = static_cast<std::size_t>(
                          std::strtoull(mb, nullptr, 10))
                      << 20;
        }
        if (budget_ == 0) {
            std::size_t free_b = 0, total_b = 0;
            budget_ = cuda_ok(cudaMemGetInfo(&free_b, &total_b))
                          ? free_b / 5 * 4  // ~80% of free VRAM
                          : (std::size_t{512} << 20);
        }
    }

    /** Reserves room (evicting LRU) then cudaMallocs `bytes`; nullptr
     *  if it cannot fit even after evicting every unpinned page.
     *  Updates used_ on success. */
    void* alloc(std::size_t bytes) {
        if (bytes > budget_) {
            return nullptr;
        }
        while (used_ + bytes > budget_) {
            if (!evict_one()) {
                return nullptr;
            }
        }
        void* p = nullptr;
        if (!cuda_ok(cudaMalloc(&p, bytes))) {
            while (evict_one()) {  // reclaim slack, then retry once
            }
            if (!cuda_ok(cudaMalloc(&p, bytes))) {
                return nullptr;
            }
        }
        used_ += bytes;
        return p;
    }

    bool evict_one() {
        auto victim = table_.end();
        std::uint64_t oldest = UINT64_MAX;
        for (auto it = table_.begin(); it != table_.end(); ++it) {
            if (it->second.pins == 0 &&
                it->second.last_use < oldest) {
                oldest = it->second.last_use;
                victim = it;
            }
        }
        if (victim == table_.end()) {
            return false;  // everything pinned
        }
        wait_ready(victim->second);  // don't free mid-copy
        cudaFree(victim->second.dptr);
        used_ -= victim->second.bytes;
        evictions_++;
        table_.erase(victim);
        return true;
    }

    std::unordered_map<const void*, Page> table_;
    cudaStream_t stream_ = nullptr;
    std::size_t budget_ = 0;
    std::size_t used_ = 0;
    std::uint64_t clock_ = 0;
    std::uint64_t hits_ = 0, misses_ = 0, evictions_ = 0;
    std::uint64_t prefetches_ = 0;
    std::uint64_t upload_bytes_ = 0;
    std::mutex mu_;
};

WeightPool& pool() {
    static WeightPool p;
    return p;
}

/** Lazily uploads iq1s_grid to the device once; nullptr on failure
 *  (caller falls back to scalar). */
const std::uint64_t* device_iq1s_grid() {
    static const std::uint64_t* d =
        []() -> const std::uint64_t* {
        void* p = nullptr;
        if (!cuda_ok(cudaMalloc(&p, sizeof(iq1s_grid)))) {
            return nullptr;
        }
        if (!cuda_ok(cudaMemcpy(p, iq1s_grid, sizeof(iq1s_grid),
                                cudaMemcpyHostToDevice))) {
            cudaFree(p);
            return nullptr;
        }
        return static_cast<const std::uint64_t*>(p);
    }();
    return d;
}

enum class Kind { kF32, kQ8_0, kQ4_K, kIQ1_S };

void launch_kernel(Kind kind, const void* dw,
                   const std::uint64_t* grid, const float* dxf,
                   float* doutf, std::uint32_t rows,
                   std::uint32_t cols) {
    const std::uint32_t threads = 256;
    const std::uint32_t blocks = (rows + threads - 1) / threads;
    const auto* dwb = static_cast<const std::uint8_t*>(dw);
    switch (kind) {
        case Kind::kF32:
            matvec_f32_kernel<<<blocks, threads>>>(
                static_cast<const float*>(dw), dxf, doutf, rows,
                cols);
            break;
        case Kind::kQ8_0:
            matvec_q8_0_kernel<<<blocks, threads>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kQ4_K:
            matvec_q4_k_kernel<<<blocks, threads>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ1_S:
            matvec_iq1_s_kernel<<<blocks, threads>>>(
                dwb, grid, dxf, doutf, rows, cols);
            break;
    }
}

bool run_matvec(const Mat& w, std::span<const float> x,
                std::span<float> out, std::size_t w_bytes,
                Kind kind, const std::uint64_t* grid = nullptr) {
    // Weight: resident pool, or an uncached per-call upload when it
    // cannot be cached (too big for the budget / all pages pinned).
    const void* dw = pool().acquire(w.data, w_bytes);
    const bool pooled = dw != nullptr;
    DevBuf dw_demand;
    if (!pooled) {
        if (!dw_demand.alloc(w_bytes) ||
            !cuda_ok(cudaMemcpy(dw_demand.p, w.data, w_bytes,
                                cudaMemcpyHostToDevice))) {
            return false;
        }
        dw = dw_demand.p;
    }

    bool ok = false;
    {
        DevBuf dx, dout;  // transient: change every call
        const std::size_t x_bytes = x.size() * sizeof(float);
        const std::size_t out_bytes =
            static_cast<std::size_t>(w.rows) * sizeof(float);
        if (dx.alloc(x_bytes) && dout.alloc(out_bytes) &&
            cuda_ok(cudaMemcpy(dx.p, x.data(), x_bytes,
                               cudaMemcpyHostToDevice))) {
            launch_kernel(kind, dw, grid,
                          static_cast<const float*>(dx.p),
                          static_cast<float*>(dout.p), w.rows,
                          w.cols);
            ok = cuda_ok(cudaGetLastError()) &&
                 cuda_ok(cudaDeviceSynchronize()) &&
                 cuda_ok(cudaMemcpy(out.data(), dout.p, out_bytes,
                                    cudaMemcpyDeviceToHost));
        }
    }

    if (pooled) {
        pool().release(w.data);
    }
    return ok;
}

/** Device upload size for a GPU-handled weight type; 0 otherwise. */
std::size_t device_weight_bytes(const Mat& w) {
    switch (w.type) {
        case gguf::TensorType::kF32:
            return static_cast<std::size_t>(w.rows) * w.cols *
                   sizeof(float);
        case gguf::TensorType::kQ8_0:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 32ull) * 34ull;
        case gguf::TensorType::kQ4_K:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 144ull;
        case gguf::TensorType::kIQ1_S:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 50ull;
        default:
            return 0;
    }
}

Kind kind_for(gguf::TensorType t) {
    switch (t) {
        case gguf::TensorType::kQ8_0:
            return Kind::kQ8_0;
        case gguf::TensorType::kQ4_K:
            return Kind::kQ4_K;
        case gguf::TensorType::kIQ1_S:
            return Kind::kIQ1_S;
        default:
            return Kind::kF32;
    }
}

}  // namespace

void matvec_cuda(const Mat& w, std::span<const float> x,
                 std::span<float> out) {
    const std::size_t bytes = device_weight_bytes(w);
    bool done = false;
    if (bytes > 0) {
        const std::uint64_t* grid =
            w.type == gguf::TensorType::kIQ1_S ? device_iq1s_grid()
                                               : nullptr;
        if (w.type != gguf::TensorType::kIQ1_S || grid != nullptr) {
            done = run_matvec(w, x, out, bytes, kind_for(w.type),
                              grid);
        }
    }
    if (!done) {
        matvec(w, x, out);  // fallback / other types: scalar
    }
}

void cuda_prefetch(const Mat& w) {
    const std::size_t bytes = device_weight_bytes(w);
    if (bytes > 0) {
        pool().prefetch(w.data, bytes);
    }
}

bool cuda_backend_usable() {
    int n = 0;
    return cuda_ok(cudaGetDeviceCount(&n)) && n > 0;
}

}  // namespace locus::backend
