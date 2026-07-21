// CUDA matvec variant. Correctness-first (streaming): each call
// uploads the weight rows it needs, launches a thread-per-row dot
// product, and copies the result back. Residency/prefetch policies
// (R8) build on top of this once it is proven token-exact. F32 and
// Q8_0 run on the GPU; other weight types delegate to the scalar
// reference on the host.

#include "locus/backend/variants.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

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

bool run_matvec(const Mat& w, std::span<const float> x,
                std::span<float> out, std::size_t w_bytes,
                bool q8_0) {
    DevBuf dw, dx, dout;
    const std::size_t x_bytes = x.size() * sizeof(float);
    const std::size_t out_bytes =
        static_cast<std::size_t>(w.rows) * sizeof(float);
    if (!dw.alloc(w_bytes) || !dx.alloc(x_bytes) ||
        !dout.alloc(out_bytes)) {
        return false;
    }
    if (!cuda_ok(cudaMemcpy(dw.p, w.data, w_bytes,
                            cudaMemcpyHostToDevice)) ||
        !cuda_ok(cudaMemcpy(dx.p, x.data(), x_bytes,
                            cudaMemcpyHostToDevice))) {
        return false;
    }
    const std::uint32_t threads = 256;
    const std::uint32_t blocks = (w.rows + threads - 1) / threads;
    if (q8_0) {
        matvec_q8_0_kernel<<<blocks, threads>>>(
            static_cast<const std::uint8_t*>(dw.p),
            static_cast<const float*>(dx.p),
            static_cast<float*>(dout.p), w.rows, w.cols);
    } else {
        matvec_f32_kernel<<<blocks, threads>>>(
            static_cast<const float*>(dw.p),
            static_cast<const float*>(dx.p),
            static_cast<float*>(dout.p), w.rows, w.cols);
    }
    if (!cuda_ok(cudaGetLastError()) ||
        !cuda_ok(cudaDeviceSynchronize())) {
        return false;
    }
    return cuda_ok(cudaMemcpy(out.data(), dout.p, out_bytes,
                              cudaMemcpyDeviceToHost));
}

}  // namespace

void matvec_cuda(const Mat& w, std::span<const float> x,
                 std::span<float> out) {
    bool done = false;
    if (w.type == gguf::TensorType::kF32) {
        const std::size_t bytes =
            static_cast<std::size_t>(w.rows) * w.cols *
            sizeof(float);
        done = run_matvec(w, x, out, bytes, /*q8_0=*/false);
    } else if (w.type == gguf::TensorType::kQ8_0) {
        const std::size_t bytes =
            static_cast<std::size_t>(w.rows) *
            (w.cols / 32ull) * 34ull;
        done = run_matvec(w, x, out, bytes, /*q8_0=*/true);
    }
    if (!done) {
        matvec(w, x, out);  // fallback / other types: scalar
    }
}

bool cuda_backend_usable() {
    int n = 0;
    return cuda_ok(cudaGetDeviceCount(&n)) && n > 0;
}

}  // namespace locus::backend
