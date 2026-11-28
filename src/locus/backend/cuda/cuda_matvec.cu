// CUDA matvec variant. Weights run through a device weight pool (the
// R8-GPU pager): each weight is uploaded once, keyed by host pointer,
// and reused across tokens under a byte budget with LRU eviction;
// weights too big for the budget fall back to an uncached per-call
// upload. Kernels dequantize F32/Q8_0/Q2_K/Q4_K/Q5_K/Q6_K and the
// IQ1_S/IQ2_XXS/IQ3_XXS/IQ4_XS families on-device (the pool stores the
// source quantized bytes); other types delegate to scalar.

#include "locus/backend/cuda/cuda_common.cuh"
#include "locus/backend/fallbacks.hpp"
#include "locus/backend/gpu_metrics.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/sys/mmap_file.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef LOCUS_HAS_CUBLAS
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "iq_grids.h"  // iq1s_grid lookup table (global scope)

namespace locus::backend {

// The helpers cuda_conv.cu and cuda_attend.cu share (cuda_common.cuh).
namespace cuda_detail {

bool cuda_ok(cudaError_t e) { return e == cudaSuccess; }

// i#24 inc 4: process-selected CUDA device ordinal (-1 = unset, i.e. the
// runtime default device 0). Set once by cuda_set_device(). Every CUDA
// matvec entry re-binds its calling thread to it via ensure_device(),
// because CUDA's current device is per-thread and the stage executor
// runs forwards on its own worker thread -- a cudaSetDevice on the main
// thread would not bind the worker that actually launches the kernels.
std::atomic<int> g_cuda_device{-1};

/**
 * Binds the calling thread to the process-selected CUDA device if one
 * was chosen. Cheap (a thread-local set) and idempotent; called at the
 * top of every CUDA matvec entry so whichever thread runs the forward
 * lands on the selected device, and so the device set precedes the
 * first weight-pool cudaMalloc by construction.
 */
void ensure_device() {
    // Acquire pairs with the release store in cuda_set_device so a worker
    // thread always observes the selected ordinal. In 4a the set precedes
    // worker creation (thread start is already a happens-before), so this
    // is belt-and-suspenders; it also keeps the seam correct if a device
    // is ever selected after a worker is live (the 4b direction).
    const int d = g_cuda_device.load(std::memory_order_acquire);
    if (d < 0) {
        return;  // no explicit device: the runtime default (device 0)
    }
    // One cudaSetDevice per thread per device change, not per call: the
    // op entries hit this thousands of times per token. Storing the
    // DEVICE (not a bool) keeps it correct under inc 4b, where different
    // executor threads bind different devices -- a bool would freeze the
    // first binding.
    thread_local int bound = -1;
    if (bound != d) {
        // Cache the binding only on success: if cudaSetDevice fails,
        // leave `bound` so the next call retries rather than believing a
        // failed bind took. The pool device-match guard is the net if a
        // kernel still runs misbound.
        if (cuda_ok(cudaSetDevice(d))) {
            bound = d;
        }
    }
}

}  // namespace cuda_detail

using namespace cuda_detail;

namespace {

/** Reads a little-endian IEEE half from two device bytes. */
__device__ inline float ld_f16(const std::uint8_t* p) {
    return __half2float(__ushort_as_half(
        static_cast<std::uint16_t>(p[0]) |
        (static_cast<std::uint16_t>(p[1]) << 8)));
}

/** Reads a little-endian u32 from four (possibly unaligned) bytes. */
__device__ inline std::uint32_t ld_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

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

/**
 * One row of a Q4_K or Q5_K x Q8_K integer dot on device; bit-exact port
 * of the scalar vec_dot_q4_k_q8_k / vec_dot_q5_k_q8_k: decode(blk, aux8)
 * unpacks a block's 256 quants, then come the 6-bit scale/min unpack,
 * the per-8 int16 products, the scale * int32 accumulate over 8 lanes
 * and the bsums min-correction, which the two types share.
 *
 * @param row The row's nsb blocks of kBlk bytes (d, dmin, scales[12] first).
 * @param d_y, qs_y, bsums_y The activation's Q8_K quants, flat.
 * @returns The row's dot.
 */
template <std::size_t kBlk, typename Decode>
__device__ float row_k_q8k(const std::uint8_t* row, std::uint32_t nsb,
                           const float* d_y, const std::int8_t* qs_y,
                           const std::int16_t* bsums_y, Decode decode) {
    const std::uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f,
                        kmask3 = 0x03030303;
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * kBlk;
        const float bd = __half2float(__ushort_as_half(
            static_cast<std::uint16_t>(blk[0] | (blk[1] << 8))));
        const float bdmin = __half2float(__ushort_as_half(
            static_cast<std::uint16_t>(blk[2] | (blk[3] << 8))));
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int8_t aux8[256];
        decode(blk, aux8);
        const std::int8_t* a = aux8;
        std::uint32_t utmp[4];
        const std::uint8_t* s = blk + 4;  // scales[12]
        for (int k = 0; k < 3; ++k) {
            utmp[k] = static_cast<std::uint32_t>(s[4 * k]) |
                      (static_cast<std::uint32_t>(s[4 * k + 1]) << 8) |
                      (static_cast<std::uint32_t>(s[4 * k + 2]) << 16) |
                      (static_cast<std::uint32_t>(s[4 * k + 3]) << 24);
        }
        utmp[3] = ((utmp[2] >> 4) & kmask2) |
                  (((utmp[1] >> 6) & kmask3) << 4);
        const std::uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const auto* scales = reinterpret_cast<const std::uint8_t*>(utmp);
        const auto* mins =
            reinterpret_cast<const std::uint8_t*>(utmp + 2);
        const std::int16_t* bs =
            bsums_y + static_cast<std::size_t>(b) * 16;
        int sumi = 0;
        for (int j = 0; j < 16; ++j) {
            sumi += bs[j] * mins[j / 2];
        }
        std::int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int is = 0;
        for (int j = 0; j < 8; ++j) {
            const std::int32_t scale = scales[is++];
            for (int k = 0; k < 4; ++k) {
                std::int16_t aux16[8];
                for (int l = 0; l < 8; ++l) {
                    aux16[l] = static_cast<std::int16_t>(q8[l] * a[l]);
                }
                for (int l = 0; l < 8; ++l) {
                    aux32[l] += scale * aux16[l];
                }
                q8 += 8;
                a += 8;
            }
        }
        const float d = bd * d_y[b];
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * static_cast<float>(aux32[l]);
        }
        sumf -= bdmin * d_y[b] * static_cast<float>(sumi);
    }
    for (int l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    return sumf;
}

/** Q4_K x Q8_K integer dot on device: row_k_q8k with the Q4_K nibble
 *  unpack (144-byte blocks). Activation quants arrive flat: d_y[nsb],
 *  qs_y[nsb*256], bsums_y[nsb*16]. One thread per row keeps the scalar
 *  accumulation order, so the float result is identical. */
__global__ void matvec_q4_k_q8k_kernel(
    const std::uint8_t* w, const float* d_y, const std::int8_t* qs_y,
    const std::int16_t* bsums_y, float* out, std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 144;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    out[r] = row_k_q8k<144>(
        row, nsb, d_y, qs_y, bsums_y,
        [](const std::uint8_t* blk, std::int8_t* a) {
            const std::uint8_t* q4 = blk + 16;
            for (int j = 0; j < 4; ++j) {
                for (int l = 0; l < 32; ++l) {
                    a[l] = static_cast<std::int8_t>(q4[l] & 0xF);
                }
                a += 32;
                for (int l = 0; l < 32; ++l) {
                    a[l] = static_cast<std::int8_t>(q4[l] >> 4);
                }
                a += 32;
                q4 += 32;
            }
        });
}

/** Q6_K x Q8_K integer dot; bit-exact port of scalar vec_dot_q6_k_q8_k
 *  (6-bit unpack ql+qh with the -32 offset, int8 scales, no min term). */
__global__ void matvec_q6_k_q8k_kernel(
    const std::uint8_t* w, const float* d_y, const std::int8_t* qs_y,
    float* out, std::uint32_t rows, std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 210;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 210;
        const std::uint8_t* ql = blk;
        const std::uint8_t* qh = blk + 128;
        const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
        const float bd = __half2float(__ushort_as_half(
            static_cast<std::uint16_t>(blk[208] | (blk[209] << 8))));
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int8_t aux8[256];
        std::int8_t* a = aux8;
        const std::uint8_t* q4 = ql;
        const std::uint8_t* qhp = qh;
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = static_cast<std::int8_t>(
                    ((q4[l] & 0xF) | (((qhp[l] >> 0) & 3) << 4)) - 32);
                a[l + 32] = static_cast<std::int8_t>(
                    ((q4[l + 32] & 0xF) | (((qhp[l] >> 2) & 3) << 4)) -
                    32);
                a[l + 64] = static_cast<std::int8_t>(
                    ((q4[l] >> 4) | (((qhp[l] >> 4) & 3) << 4)) - 32);
                a[l + 96] = static_cast<std::int8_t>(
                    ((q4[l + 32] >> 4) | (((qhp[l] >> 6) & 3) << 4)) -
                    32);
            }
            a += 128;
            q4 += 64;
            qhp += 32;
        }
        a = aux8;
        int is = 0;
        std::int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int j = 0; j < 16; ++j) {
            const std::int32_t scale = sc[is++];
            for (int k = 0; k < 2; ++k) {
                std::int16_t aux16[8];
                for (int l = 0; l < 8; ++l) {
                    aux16[l] = static_cast<std::int16_t>(q8[l] * a[l]);
                }
                for (int l = 0; l < 8; ++l) {
                    aux32[l] += scale * aux16[l];
                }
                q8 += 8;
                a += 8;
            }
        }
        const float d = bd * d_y[b];
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * static_cast<float>(aux32[l]);
        }
    }
    for (int l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    out[r] = sumf;
}

/** Q5_K x Q8_K integer dot; bit-exact port of scalar vec_dot_q5_k_q8_k:
 *  row_k_q8k with the nibble + hm 5th-bit unpack (176-byte blocks). */
__global__ void matvec_q5_k_q8k_kernel(
    const std::uint8_t* w, const float* d_y, const std::int8_t* qs_y,
    const std::int16_t* bsums_y, float* out, std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 176;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    out[r] = row_k_q8k<176>(
        row, nsb, d_y, qs_y, bsums_y,
        [](const std::uint8_t* blk, std::int8_t* a) {
            const std::uint8_t* hm = blk + 16;
            const std::uint8_t* q4 = blk + 48;
            std::uint8_t m = 1;
            for (int j = 0; j < 4; ++j) {
                for (int l = 0; l < 32; ++l) {
                    a[l] = static_cast<std::int8_t>(q4[l] & 0xF);
                }
                for (int l = 0; l < 32; ++l) {
                    a[l] = static_cast<std::int8_t>(
                        a[l] + ((hm[l] & m) ? 16 : 0));
                }
                a += 32;
                m <<= 1;
                for (int l = 0; l < 32; ++l) {
                    a[l] = static_cast<std::int8_t>(q4[l] >> 4);
                }
                for (int l = 0; l < 32; ++l) {
                    a[l] = static_cast<std::int8_t>(
                        a[l] + ((hm[l] & m) ? 16 : 0));
                }
                a += 32;
                m <<= 1;
                q4 += 32;
            }
        });
}

/** Q2_K x Q8_K integer dot; bit-exact port of scalar vec_dot_q2_k_q8_k
 *  (2-bit unpack per 16-group, 4-bit scale+min, bsums min-correction). */
__global__ void matvec_q2_k_q8k_kernel(
    const std::uint8_t* w, const float* d_y, const std::int8_t* qs_y,
    const std::int16_t* bsums_y, float* out, std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 84;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 84;
        const std::uint8_t* sc = blk;
        const std::uint8_t* q2 = blk + 16;
        const float bd = __half2float(__ushort_as_half(
            static_cast<std::uint16_t>(blk[80] | (blk[81] << 8))));
        const float bdmin = __half2float(__ushort_as_half(
            static_cast<std::uint16_t>(blk[82] | (blk[83] << 8))));
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        const std::int16_t* bs =
            bsums_y + static_cast<std::size_t>(b) * 16;
        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += bs[j] * (sc[j] >> 4);
        }
        const float dall = d_y[b] * bd;
        const float dmin = d_y[b] * bdmin;
        int isum = 0, is = 0;
        for (int k = 0; k < 2; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                int d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l = 0; l < 16; ++l) {
                    isuml += q8[l] * ((q2[l] >> shift) & 3);
                }
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) {
                    isuml += q8[l] * ((q2[l] >> shift) & 3);
                }
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * static_cast<float>(isum) -
                dmin * static_cast<float>(summs);
    }
    out[r] = sumf;
}

/** IQ2_XXS x Q8_K integer dot; bit-exact port of vec_dot_iq2_xxs_q8_k
 *  (grid+ksigns lookup, per-32 scale from aux1>>28, 0.125 factor). */
__global__ void matvec_iq2_xxs_q8k_kernel(
    const std::uint8_t* w, const std::uint64_t* grid,
    const std::uint8_t* ksigns, const float* d_y,
    const std::int8_t* qs_y, float* out, std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 66;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 66;
        const float d = ld_f16(blk) * d_y[b];
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint8_t* a = blk + 2 + 8 * ib32;
            const std::uint32_t aux1 = ld_u32(a + 4);
            const std::uint32_t ls = 2 * (aux1 >> 28) + 1;
            std::int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t* g =
                    reinterpret_cast<const std::uint8_t*>(grid + a[l]);
                const std::uint8_t signs =
                    ksigns[(aux1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += g[j] * q8[j] * ((signs & (1 << j)) ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * static_cast<std::int32_t>(ls);
        }
        sumf += d * static_cast<float>(bsum);
    }
    out[r] = 0.125f * sumf;
}

/** IQ2_XS x Q8_K integer dot; bit-exact port of vec_dot_iq2_xs_q8_k
 *  (u16 low-9 grid index + high-7 ksigns, two 4-bit sub-scales). */
__global__ void matvec_iq2_xs_q8k_kernel(
    const std::uint8_t* w, const std::uint64_t* grid,
    const std::uint8_t* ksigns, const float* d_y,
    const std::int8_t* qs_y, float* out, std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 74;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 74;
        const float d = ld_f16(blk) * d_y[b];
        const std::uint8_t* sc = blk + 66;
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint32_t ls1 = 2 * (sc[ib32] & 0xf) + 1;
            const std::uint32_t ls2 = 2 * (sc[ib32] >> 4) + 1;
            std::int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const std::uint8_t* p = blk + 2 + 2 * (4 * ib32 + l);
                const std::uint16_t q =
                    static_cast<std::uint16_t>(p[0] | (p[1] << 8));
                const std::uint8_t* g =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (q & 511));
                const std::uint8_t signs = ksigns[q >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += g[j] * q8[j] * ((signs & (1 << j)) ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * static_cast<std::int32_t>(ls1);
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const std::uint8_t* p = blk + 2 + 2 * (4 * ib32 + l);
                const std::uint16_t q =
                    static_cast<std::uint16_t>(p[0] | (p[1] << 8));
                const std::uint8_t* g =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (q & 511));
                const std::uint8_t signs = ksigns[q >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += g[j] * q8[j] * ((signs & (1 << j)) ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * static_cast<std::int32_t>(ls2);
        }
        sumf += d * static_cast<float>(bsum);
    }
    out[r] = 0.125f * sumf;
}

/** IQ2_S x Q8_K integer dot; bit-exact port of vec_dot_iq2_s_q8_k
 *  (qh high-bit grid extension, signs read inline -- no ksigns). */
__global__ void matvec_iq2_s_q8k_kernel(
    const std::uint8_t* w, const std::uint64_t* grid, const float* d_y,
    const std::int8_t* qs_y, float* out, std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 82;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 82;
        const float d = ld_f16(blk) * d_y[b];
        const std::uint8_t* qsp = blk + 2;
        const std::uint8_t* signs = blk + 34;
        const std::uint8_t* qh = blk + 66;
        const std::uint8_t* sc = blk + 74;
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const int ls1 = 1 + 2 * (sc[ib32] & 0xf);
            const int ls2 = 1 + 2 * (sc[ib32] >> 4);
            std::int32_t sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const std::uint8_t* g =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (qsp[l] |
                                ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * g[j] *
                             ((signs[l] & (1 << j)) ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const std::uint8_t* g =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (qsp[l] |
                                ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * g[j] *
                             ((signs[l] & (1 << j)) ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qsp += 4;
            signs += 4;
        }
        sumf += d * static_cast<float>(bsum);
    }
    out[r] = 0.125f * sumf;
}

/** TQ1_0 x Q8_K integer dot; bit-exact port of vec_dot_tq1_0_q8_k
 *  (base-3 unpack via pow3 + the (q*3)>>8 trit extraction, no table). */
__global__ void matvec_tq1_0_q8k_kernel(
    const std::uint8_t* w, const float* d_y, const std::int8_t* qs_y,
    float* out, std::uint32_t rows, std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 54;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 54;
        const std::uint8_t* qs = blk;
        const std::uint8_t* qh = blk + 48;
        const float bd = ld_f16(blk + 52);
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        int sum = 0;
        for (int j = 0; j < 32; j += 32) {
            for (int l = 0; l < 5; ++l) {
                for (int m = 0; m < 32; ++m) {
                    const std::uint8_t q =
                        static_cast<std::uint8_t>(qs[j + m] * pow3[l]);
                    const std::uint16_t xi =
                        (static_cast<std::uint16_t>(q) * 3) >> 8;
                    sum += (xi - 1) * q8[j * 5 + l * 32 + m];
                }
            }
        }
        for (int j = 32; j < 48; j += 16) {
            for (int l = 0; l < 5; ++l) {
                for (int m = 0; m < 16; ++m) {
                    const std::uint8_t q =
                        static_cast<std::uint8_t>(qs[j + m] * pow3[l]);
                    const std::uint16_t xi =
                        (static_cast<std::uint16_t>(q) * 3) >> 8;
                    sum += (xi - 1) * q8[j * 5 + l * 16 + m];
                }
            }
        }
        for (int l = 0; l < 4; ++l) {
            for (int j = 0; j < 4; ++j) {
                const std::uint8_t q =
                    static_cast<std::uint8_t>(qh[j] * pow3[l]);
                const std::uint16_t xi =
                    (static_cast<std::uint16_t>(q) * 3) >> 8;
                sum += (xi - 1) * q8[48 * 5 + l * 4 + j];
            }
        }
        sumf += static_cast<float>(sum) * (d_y[b] * bd);
    }
    out[r] = sumf;
}

/** TQ2_0 x Q8_K integer dot; bit-exact port of vec_dot_tq2_0_q8_k
 *  (2-bit trit unpack, value (bits&3)-1, no table). */
__global__ void matvec_tq2_0_q8k_kernel(
    const std::uint8_t* w, const float* d_y, const std::int8_t* qs_y,
    float* out, std::uint32_t rows, std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 66;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 66;
        const std::uint8_t* qs = blk;
        const float bd = ld_f16(blk + 64);
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int32_t sumi = 0;
        for (int j = 0; j < 64; j += 32) {
            for (int l = 0; l < 4; ++l) {
                for (int k = 0; k < 32; ++k) {
                    sumi += q8[j * 4 + l * 32 + k] *
                            (((qs[j + k] >> (l * 2)) & 3) - 1);
                }
            }
        }
        sumf += static_cast<float>(sumi) * (d_y[b] * bd);
    }
    out[r] = sumf;
}

/** IQ4_XS x Q8_K integer dot; bit-exact port of vec_dot_iq4_xs_q8_k
 *  (6-bit scales from scales_l+scales_h, iq4nl codebook `kvalues`). */
__global__ void matvec_iq4_xs_q8k_kernel(
    const std::uint8_t* w, const std::int8_t* kvalues, const float* d_y,
    const std::int8_t* qs_y, float* out, std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 136;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 136;
        const float d4d8 = ld_f16(blk) * d_y[b];
        std::uint16_t h =
            static_cast<std::uint16_t>(blk[2] | (blk[3] << 8));
        const std::uint8_t* scales_l = blk + 4;
        const std::uint8_t* qs = blk + 8;
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        for (int ib = 0; ib < 8; ib += 2) {
            const std::uint8_t ls1 =
                (scales_l[ib / 2] & 0xf) | ((h << 4) & 0x30);
            const std::uint8_t ls2 =
                (scales_l[ib / 2] >> 4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8 * (static_cast<int>(ls1) - 32);
            const float d2 = d4d8 * (static_cast<int>(ls2) - 32);
            std::int32_t sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * kvalues[qs[j] & 0xf];
                sumi2 += q8[j + 16] * kvalues[qs[j] >> 4];
            }
            sumf += d1 * static_cast<float>(sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * kvalues[qs[j] & 0xf];
                sumi2 += q8[j + 16] * kvalues[qs[j] >> 4];
            }
            sumf += d2 * static_cast<float>(sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    out[r] = sumf;
}

/** IQ3_XXS x Q8_K integer dot; bit-exact port of vec_dot_iq3_xxs_q8_k
 *  (iq3xxs_grid u32 entries + ksigns, per-32 scale, 0.25 factor). */
__global__ void matvec_iq3_xxs_q8k_kernel(
    const std::uint8_t* w, const std::uint32_t* grid,
    const std::uint8_t* ksigns, const float* d_y,
    const std::int8_t* qs_y, float* out, std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 98;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 98;
        const float d = ld_f16(blk) * d_y[b];
        const std::uint8_t* q3 = blk + 2;
        const std::uint8_t* gas = blk + 66;
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint32_t aux32 = ld_u32(gas);
            gas += 4;
            const std::uint32_t ls = 2 * (aux32 >> 28) + 1;
            std::int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t* g1 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + q3[2 * l + 0]);
                const std::uint8_t* g2 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + q3[2 * l + 1]);
                const std::uint8_t signs =
                    ksigns[(aux32 >> (7 * l)) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += g1[j] * q8[j + 0] *
                            ((signs & (1 << (j + 0))) ? -1 : 1);
                    sumi += g2[j] * q8[j + 4] *
                            ((signs & (1 << (j + 4))) ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * static_cast<std::int32_t>(ls);
        }
        sumf += d * static_cast<float>(bsum);
    }
    out[r] = 0.25f * sumf;
}

/** IQ3_S x Q8_K integer dot; bit-exact port of vec_dot_iq3_s_q8_k
 *  (iq3s_grid u32 + qh 9th-bit extension, inline signs). */
__global__ void matvec_iq3_s_q8k_kernel(
    const std::uint8_t* w, const std::uint32_t* grid, const float* d_y,
    const std::int8_t* qs_y, float* out, std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 110;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 110;
        const float d = ld_f16(blk) * d_y[b];
        const std::uint8_t* qh = blk + 66;
        const std::uint8_t* sc = blk + 106;
        const std::uint8_t* qsp = blk + 2;
        const std::uint8_t* signs = blk + 74;
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const std::uint32_t ls1 = 2 * (sc[ib32 / 2] & 0xf) + 1;
            const std::uint32_t ls2 = 2 * (sc[ib32 / 2] >> 4) + 1;
            std::int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t* g1 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (qsp[2 * l] |
                                ((qh[ib32] << (8 - 2 * l)) & 256)));
                const std::uint8_t* g2 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (qsp[2 * l + 1] |
                                ((qh[ib32] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += g1[j] * q8[j + 0] *
                            ((signs[l] & (1 << (j + 0))) ? -1 : 1);
                    sumi += g2[j] * q8[j + 4] *
                            ((signs[l] & (1 << (j + 4))) ? -1 : 1);
                }
                q8 += 8;
            }
            qsp += 8;
            signs += 4;
            bsum += sumi * static_cast<std::int32_t>(ls1);
            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t* g1 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (qsp[2 * l] |
                                ((qh[ib32 + 1] << (8 - 2 * l)) & 256)));
                const std::uint8_t* g2 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (qsp[2 * l + 1] |
                                ((qh[ib32 + 1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += g1[j] * q8[j + 0] *
                            ((signs[l] & (1 << (j + 0))) ? -1 : 1);
                    sumi += g2[j] * q8[j + 4] *
                            ((signs[l] & (1 << (j + 4))) ? -1 : 1);
                }
                q8 += 8;
            }
            qsp += 8;
            signs += 4;
            bsum += sumi * static_cast<std::int32_t>(ls2);
        }
        sumf += d * static_cast<float>(bsum);
    }
    out[r] = sumf;
}

/** IQ1_S x Q8_K integer dot; bit-exact port of vec_dot_iq1_s_q8_k
 *  (iq1s_grid u64, per-32 delta folded through the Q8_K bsums). */
__global__ void matvec_iq1_s_q8k_kernel(
    const std::uint8_t* w, const std::uint64_t* grid, const float* d_y,
    const std::int8_t* qs_y, const std::int16_t* bsums_y, float* out,
    std::uint32_t rows, std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 50;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 50;
        const float d = ld_f16(blk) * d_y[b];
        const std::uint8_t* qs = blk + 2;
        const std::uint8_t* qhb = blk + 34;
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        const std::int16_t* bs =
            bsums_y + static_cast<std::size_t>(b) * 16;
        std::int32_t sumi = 0, sumi1 = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint16_t qh = static_cast<std::uint16_t>(
                qhb[2 * ib] | (qhb[2 * ib + 1] << 8));
            const int ls = 2 * ((qh >> 12) & 7) + 1;
            const int delta = (qh & 0x8000) ? -1 : 1;
            std::int32_t lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const std::int8_t* g =
                    reinterpret_cast<const std::int8_t*>(
                        grid + (qs[l] | (((qh >> (3 * l)) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    lsum += q8[j] * g[j];
                }
                q8 += 8;
            }
            sumi += ls * lsum;
            sumi1 +=
                ls * delta * (bs[2 * ib + 0] + bs[2 * ib + 1]);
            qs += 4;
        }
        sumf += d * (static_cast<float>(sumi) +
                     0.125f * static_cast<float>(sumi1));
    }
    out[r] = sumf;
}

/** IQ1_M x Q8_K integer dot; bit-exact port of vec_dot_iq1_m_q8_k
 *  (iq1s_grid u64, scale from 4 nibbles, per-8 delta on q8 sums). */
__global__ void matvec_iq1_m_q8k_kernel(
    const std::uint8_t* w, const std::uint64_t* grid, const float* d_y,
    const std::int8_t* qs_y, float* out, std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 56;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 56;
        const std::uint8_t* qs = blk;
        const std::uint8_t* qh = blk + 32;
        const std::uint8_t* scb = blk + 48;
        const std::uint16_t scv[4] = {
            static_cast<std::uint16_t>(scb[0] | (scb[1] << 8)),
            static_cast<std::uint16_t>(scb[2] | (scb[3] << 8)),
            static_cast<std::uint16_t>(scb[4] | (scb[5] << 8)),
            static_cast<std::uint16_t>(scb[6] | (scb[7] << 8))};
        const std::uint16_t scale_u16 =
            (scv[0] >> 12) | ((scv[1] >> 8) & 0x00f0) |
            ((scv[2] >> 4) & 0x0f00) | (scv[3] & 0xf000);
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int32_t sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const int delta[4] = {(qh[0] & 0x08) ? -1 : 1,
                                  (qh[0] & 0x80) ? -1 : 1,
                                  (qh[1] & 0x08) ? -1 : 1,
                                  (qh[1] & 0x80) ? -1 : 1};
            std::int32_t sum1[2] = {0, 0}, sum2[2] = {0, 0};
            for (int l = 0; l < 4; ++l) {
                const std::int8_t* g =
                    reinterpret_cast<const std::int8_t*>(
                        grid +
                        (qs[l] |
                         ((static_cast<std::uint16_t>(qh[l / 2])
                           << (8 - 4 * (l % 2))) &
                          0x700)));
                std::int32_t lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * g[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l / 2] += lsum1;
                sum2[l / 2] += lsum2 * delta[l];
            }
            const int ls1 =
                2 * ((scv[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1;
            const int ls2 =
                2 * ((scv[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1;
            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }
        sumf += __half2float(__ushort_as_half(scale_u16)) * d_y[b] *
                (static_cast<float>(sumi1) +
                 0.125f * static_cast<float>(sumi2));
    }
    out[r] = sumf;
}

/** Q3_K x Q8_K integer dot; bit-exact port of vec_dot_q3_k_q8_k
 *  (2-bit + hmask 3rd-bit unpack with the -4 offset, 6-bit scales
 *  unpacked from the packed 12 bytes, scale-32 int32 accumulate). */
__global__ void matvec_q3_k_q8k_kernel(
    const std::uint8_t* w, const float* d_y, const std::int8_t* qs_y,
    float* out, std::uint32_t rows, std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 110;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * row_bytes;
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 110;
        const std::uint8_t* hm = blk;
        const std::uint8_t* q3 = blk + 32;
        const float bd = ld_f16(blk + 108);
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int8_t aux8[256];
        std::int8_t* a = aux8;
        std::uint8_t m = 1;
        const std::uint8_t* q3p = q3;
        for (int j = 0; j < 256; j += 128) {
            for (int sh = 0; sh < 8; sh += 2) {
                for (int l = 0; l < 32; ++l) {
                    a[l] = static_cast<std::int8_t>((q3p[l] >> sh) & 3);
                }
                for (int l = 0; l < 32; ++l) {
                    a[l] = static_cast<std::int8_t>(
                        a[l] - ((hm[l] & m) ? 0 : 4));
                }
                a += 32;
                m <<= 1;
            }
            q3p += 32;
        }
        std::uint32_t auxs[4];
        const std::uint8_t* s = blk + 96;
        for (int k = 0; k < 3; ++k) {
            auxs[k] = static_cast<std::uint32_t>(s[4 * k]) |
                      (static_cast<std::uint32_t>(s[4 * k + 1]) << 8) |
                      (static_cast<std::uint32_t>(s[4 * k + 2]) << 16) |
                      (static_cast<std::uint32_t>(s[4 * k + 3]) << 24);
        }
        const std::uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) |
                  (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) |
                  (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const auto* scales = reinterpret_cast<const std::int8_t*>(auxs);
        std::int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        a = aux8;
        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < 2; ++k) {
                std::int16_t aux16[8];
                for (int l = 0; l < 8; ++l) {
                    aux16[l] = static_cast<std::int16_t>(q8[l] * a[l]);
                }
                for (int l = 0; l < 8; ++l) {
                    aux32[l] += (scales[j] - 32) * aux16[l];
                }
                q8 += 8;
                a += 8;
            }
        }
        const float d = bd * d_y[b];
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * static_cast<float>(aux32[l]);
        }
    }
    for (int l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    out[r] = sumf;
}

/** Rows (one warp each) per block of the warp K-quant kernels. */
constexpr std::uint32_t kWarpRows = 4;

/** @returns the sum of v over the 32 lanes, in every lane. */
__device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

/** The scales of sub-blocks 2j and 2j + 1 of a Q4_K or Q5_K block (d
 *  and dmin f16 at bytes 0 and 2, scales[12] at 4): d1 = d * scale and
 *  m1 = dmin * min of sub-block 2j, d2 and m2 of 2j + 1. */
__device__ inline void k4_pair_scales(const std::uint8_t* blk,
                                      std::uint32_t j, float& d1,
                                      float& m1, float& d2, float& m2) {
    const float d = ld_f16(blk);
    const float dmin = ld_f16(blk + 2);
    std::uint8_t sc, mn;
    scale_min_k4(2 * j, blk + 4, sc, mn);
    d1 = d * sc;
    m1 = dmin * mn;
    scale_min_k4(2 * j + 1, blk + 4, sc, mn);
    d2 = d * sc;
    m2 = dmin * mn;
}

/** The 8 weights lane `lane` owns in Q4_K block blk, as d * q - m (the
 *  layout of matvec_q4_k_warp_kernel): vlo[k] is element 64j + c + k,
 *  vhi[k] element 64j + 32 + c + k (j = lane / 8, c = 4 (lane % 8)). */
__device__ inline void q4_k_lane_values(const std::uint8_t* blk,
                                        std::uint32_t lane, float vlo[4],
                                        float vhi[4]) {
    float d1, m1, d2, m2;
    k4_pair_scales(blk, lane / 8, d1, m1, d2, m2);
    const std::uint32_t q4 =
        reinterpret_cast<const std::uint32_t*>(blk + 16)[lane];
    for (int k = 0; k < 4; ++k) {
        const std::uint32_t q = (q4 >> (8 * k)) & 0xff;
        vlo[k] = d1 * static_cast<float>(q & 0x0f) - m1;
        vhi[k] = d2 * static_cast<float>(q >> 4) - m2;
    }
}

/** As q4_k_lane_values for a Q5_K block (the layout of
 *  matvec_q5_k_warp_kernel): the nibbles from ql, the 5th bits from
 *  qh. */
__device__ inline void q5_k_lane_values(const std::uint8_t* blk,
                                        std::uint32_t lane, float vlo[4],
                                        float vhi[4]) {
    const std::uint32_t j = lane / 8;
    const std::uint32_t c = 4 * (lane % 8);
    float d1, m1, d2, m2;
    k4_pair_scales(blk, j, d1, m1, d2, m2);
    const std::uint32_t ql4 =
        __ldg(reinterpret_cast<const std::uint32_t*>(blk + 48) + lane);
    const std::uint32_t qh4 =
        __ldg(reinterpret_cast<const std::uint32_t*>(blk + 16 + c));
    for (int k = 0; k < 4; ++k) {
        const std::uint32_t q = (ql4 >> (8 * k)) & 0xff;
        const std::uint32_t h = (qh4 >> (8 * k)) & 0xff;
        const std::uint32_t lo = (q & 0x0f) + (((h >> (2 * j)) & 1) << 4);
        const std::uint32_t hi =
            (q >> 4) + (((h >> (2 * j + 1)) & 1) << 4);
        vlo[k] = d1 * static_cast<float>(lo) - m1;
        vhi[k] = d2 * static_cast<float>(hi) - m2;
    }
}

/** @returns acc plus a lane's 8 products, in the order of the Q4_K,
 *  Q5_K and IQ4_XS kernels: vlo[k] * xb[k], then vhi[k] * xb[kHi + k],
 *  for k = 0..3 (xb 16-byte aligned; kHi 32 for the K-quants, 16 for
 *  IQ4_XS). */
template <std::uint32_t kHi = 32>
__device__ inline float lane_dot8(float acc, const float vlo[4],
                                  const float vhi[4], const float* xb) {
    const float4 xlo = __ldg(reinterpret_cast<const float4*>(xb));
    const float4 xhi = __ldg(reinterpret_cast<const float4*>(xb + kHi));
    const float xl[4] = {xlo.x, xlo.y, xlo.z, xlo.w};
    const float xh[4] = {xhi.x, xhi.y, xhi.z, xhi.w};
    for (int k = 0; k < 4; ++k) {
        acc += vlo[k] * xl[k];
        acc += vhi[k] * xh[k];
    }
    return acc;
}

/**
 * One Q4_K super-block is 144 bytes: d,dmin (f16) + scales[12] +
 * qs[128], 256 weights. One warp per row: lane `l` reads the 4 qs
 * bytes at 4l (one coalesced 128-byte read per warp and super-block),
 * which hold the low nibbles of elements 64j + c and the high nibbles
 * of 64j + 32 + c, with j = l / 8 and c = 4 (l % 8) .. +3; it
 * accumulates those 8 products over the row, then the warp sums the
 * lanes. The summation order is not the scalar one, so the result
 * matches the CPU only to rounding. Launch with 32 * kWarpRows
 * threads per block.
 */
__global__ void matvec_q4_k_warp_kernel(const std::uint8_t* w,
                                        const float* x, float* out,
                                        std::uint32_t rows,
                                        std::uint32_t cols) {
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    const std::uint32_t nsb = cols / 256;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * nsb * 144;
    const std::uint32_t j = lane / 8;
    const std::uint32_t c = 4 * (lane % 8);
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        float vlo[4], vhi[4];
        q4_k_lane_values(row + static_cast<std::size_t>(b) * 144, lane, vlo,
                         vhi);
        acc = lane_dot8(acc, vlo, vhi,
                        x + static_cast<std::size_t>(b) * 256 + 64 * j + c);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/**
 * One Q5_K super-block is 176 bytes: d,dmin (f16) + scales[12] +
 * qh[32] + ql[128], 256 weights. Element 64j + l (l = 0..31) is the
 * low nibble of ql[32j + l] plus 16 when bit 2j of qh[l] is set, and
 * element 64j + 32 + l the high nibble plus 16 when bit 2j + 1 is.
 * One warp per row, laid out as matvec_q4_k_warp_kernel: lane L reads
 * the 4 ql bytes at 4L (one coalesced 128-byte read per warp) and the
 * 4 qh bytes at 4 (L % 8), so j = L / 8 and l = 4 (L % 8) .. +3
 * (176-byte blocks keep 16-byte alignment, so both are 32-bit loads).
 * Each lane accumulates its 8 products over the row, then the warp
 * sums the lanes. The summation order is not the scalar one, so the
 * result matches the CPU only to rounding. Launch with 32 * kWarpRows
 * threads per block.
 */
__global__ void matvec_q5_k_warp_kernel(const std::uint8_t* w,
                                        const float* x, float* out,
                                        std::uint32_t rows,
                                        std::uint32_t cols) {
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    const std::uint32_t nsb = cols / 256;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * nsb * 176;
    const std::uint32_t j = lane / 8;
    const std::uint32_t c = 4 * (lane % 8);
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        float vlo[4], vhi[4];
        q5_k_lane_values(row + static_cast<std::size_t>(b) * 176, lane, vlo,
                         vhi);
        acc = lane_dot8(acc, vlo, vhi,
                        x + static_cast<std::size_t>(b) * 256 + 64 * j + c);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/**
 * Per-lane share of one Q6_K super-block for the warp kernels. A
 * Q6_K super-block is 210 bytes: ql[128] + qh[64] + sc[16] int8 + d
 * (f16), 256 weights. Element i = 128h + 32g + l (g = 0..3,
 * l = 0..31) takes the low nibble of ql[64h + 32(g & 1) + l] for
 * g < 2 and the high nibble for g >= 2, bits 2g..2g+1 of
 * qh[32h + l], and scale sc[8h + l / 16 + 2g]. Lane L owns the 4 ql
 * bytes at 4L (h = L / 16, g & 1 = (L % 16) / 8, l = 4 (L % 8) ..
 * +3), so a warp reads ql in one coalesced pass; each byte yields
 * element g (low nibble) and g + 2 (high nibble), 8 values in all.
 * The block is only 2-byte aligned (210-byte blocks), so it reads
 * 16-bit words. Writes the 8 dequantized values (y = d * sc * q, the
 * scalar's expression) to ylo[k] / yhi[k] and returns the offset of
 * ylo[0]'s element in the super-block (yhi[k] is 64 further).
 */
__device__ std::uint32_t q6_k_lane_values(const std::uint8_t* blk,
                                          std::uint32_t lane,
                                          float ylo[4], float yhi[4]) {
    const std::uint32_t h = lane / 16;
    const std::uint32_t g = (lane % 16) / 8;
    const std::uint32_t l0 = 4 * (lane % 8);
    const auto* ql16 =
        reinterpret_cast<const std::uint16_t*>(blk + 4 * lane);
    const auto* qh16 =
        reinterpret_cast<const std::uint16_t*>(blk + 128 + 32 * h + l0);
    const std::uint32_t ql =
        static_cast<std::uint32_t>(ql16[0]) |
        (static_cast<std::uint32_t>(ql16[1]) << 16);
    const std::uint32_t qh =
        static_cast<std::uint32_t>(qh16[0]) |
        (static_cast<std::uint32_t>(qh16[1]) << 16);
    const std::uint16_t db =
        static_cast<std::uint16_t>(blk[208]) |
        (static_cast<std::uint16_t>(blk[209]) << 8);
    const float d = __half2float(__ushort_as_half(db));
    const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
    const std::uint32_t is = 8 * h + l0 / 16 + 2 * g;
    const std::int8_t s_lo = sc[is], s_hi = sc[is + 4];
    for (int k = 0; k < 4; ++k) {
        const std::uint32_t lb = (ql >> (8 * k)) & 0xff;
        const std::uint32_t hb = (qh >> (8 * k)) & 0xff;
        const int q_lo = static_cast<int>((lb & 0x0f) |
                                          (((hb >> (2 * g)) & 3) << 4)) -
                         32;
        const int q_hi = static_cast<int>((lb >> 4) |
                                          (((hb >> (2 * g + 4)) & 3)
                                           << 4)) -
                         32;
        ylo[k] = d * s_lo * static_cast<float>(q_lo);
        yhi[k] = d * s_hi * static_cast<float>(q_hi);
    }
    return 128 * h + 32 * g + l0;
}

/** Q6_K matvec, one warp per row (q6_k_lane_values for the lane
 *  mapping): each lane accumulates its 8 products over the row, then
 *  the warp sums the lanes. The summation order is not the scalar
 *  one, so the result matches the CPU only to rounding. Launch with
 *  32 * kWarpRows threads per block. */
__global__ void matvec_q6_k_warp_kernel(const std::uint8_t* w,
                                        const float* x, float* out,
                                        std::uint32_t rows,
                                        std::uint32_t cols) {
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    const std::uint32_t nsb = cols / 256;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * nsb * 210;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        float ylo[4], yhi[4];
        const std::uint32_t e = q6_k_lane_values(
            row + static_cast<std::size_t>(b) * 210, lane, ylo, yhi);
        const float* xb = x + static_cast<std::size_t>(b) * 256 + e;
        const float4 xlo = __ldg(reinterpret_cast<const float4*>(xb));
        const float4 xhi =
            __ldg(reinterpret_cast<const float4*>(xb + 64));
        const float xl[4] = {xlo.x, xlo.y, xlo.z, xlo.w};
        const float xh[4] = {xhi.x, xhi.y, xhi.z, xhi.w};
        for (int k = 0; k < 4; ++k) {
            acc += ylo[k] * xl[k];
            acc += yhi[k] * xh[k];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/**
 * Copies one pass of a Q8_0 row (nb blocks of 34 bytes from src, 16-bit
 * words) into the warp's shared slice dst (16-byte aligned). A full
 * pass is 1088 bytes, a multiple of 16, so when the row starts 16-byte
 * aligned every pass does and the copy takes 68 16-byte loads; when it
 * starts 8-byte aligned (an 896-wide row is 952 bytes, so every other
 * one is), 136 8-byte loads (i#1254), not 544 16-bit ones. Each lane
 * issues all of its loads before any store (i#1197): a loop of
 * load-then-store kept one load in flight per warp.
 *
 * @param align The row start's alignment in bytes: 16, 8, or less.
 */
__device__ inline void q8_0_copy_pass(const std::uint16_t* src,
                                      std::uint16_t* dst, std::uint32_t nb,
                                      std::uint32_t align, std::uint32_t lane) {
    if (nb == 32 && align == 8) {
        const auto* s2 = reinterpret_cast<const uint2*>(src);
        auto* d2 = reinterpret_cast<uint2*>(dst);
        uint2 v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] = __ldg(s2 + 32 * k + lane);
        }
        uint2 e{};
        if (lane < 8) {
            e = __ldg(s2 + 128 + lane);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            d2[32 * k + lane] = v[k];
        }
        if (lane < 8) {
            d2[128 + lane] = e;
        }
        return;
    }
    if (nb == 32 && align == 16) {
        const auto* s4 = reinterpret_cast<const uint4*>(src);
        auto* d4 = reinterpret_cast<uint4*>(dst);
        const uint4 a = __ldg(s4 + lane);
        const uint4 b = __ldg(s4 + 32 + lane);
        uint4 c{};
        if (lane < 4) {
            c = __ldg(s4 + 64 + lane);
        }
        d4[lane] = a;
        d4[32 + lane] = b;
        if (lane < 4) {
            d4[64 + lane] = c;
        }
    } else if (nb == 32) {
        std::uint16_t v[17];
#pragma unroll
        for (int k = 0; k < 17; ++k) {
            v[k] = __ldg(src + lane + 32 * k);
        }
#pragma unroll
        for (int k = 0; k < 17; ++k) {
            dst[lane + 32 * k] = v[k];
        }
    } else {
        for (std::uint32_t i = lane; i < nb * 17; i += 32) {
            dst[i] = __ldg(src + i);
        }
    }
}

/** d times the dot of each of two Q8_0 blocks (rows; 17 16-bit words in
 *  shared memory each: d, then the 32 int8 quants) with the 32 floats
 *  at xb (16-byte aligned), in the order of the Q8_0 kernels: x is
 *  loaded once for both, and neither row's sum depends on the other. */
__device__ inline void q8_0_block_dot2(const std::uint16_t* blk0,
                                       const std::uint16_t* blk1,
                                       const float* xb, float& out0,
                                       float& out1) {
    const auto* x4 = reinterpret_cast<const float4*>(xb);
    float acc[2] = {0.0f, 0.0f};
    const std::uint16_t* blks[2] = {blk0, blk1};
    for (int k = 0; k < 8; ++k) {
        const float4 xv = __ldg(x4 + k);
        const float xs[4] = {xv.x, xv.y, xv.z, xv.w};
        for (int r = 0; r < 2; ++r) {
            const std::uint16_t lo = blks[r][1 + 2 * k];
            const std::uint16_t hi = blks[r][2 + 2 * k];
            const std::int8_t q[4] = {static_cast<std::int8_t>(lo & 0xff),
                                      static_cast<std::int8_t>(lo >> 8),
                                      static_cast<std::int8_t>(hi & 0xff),
                                      static_cast<std::int8_t>(hi >> 8)};
            for (int g = 0; g < 4; ++g) {
                acc[r] += static_cast<float>(q[g]) * xs[g];
            }
        }
    }
    out0 = __half2float(__ushort_as_half(blk0[0])) * acc[0];
    out1 = __half2float(__ushort_as_half(blk1[0])) * acc[1];
}

/**
 * One Q8_0 block is 34 bytes: f16 scale + 32 int8 quants. One warp
 * per two rows (i#1197: each x load serves both): lane L takes blocks
 * L, L + 32, ... of each. Each pass, the warp copies the rows' 32
 * consecutive blocks (1088 contiguous bytes each; blocks are only
 * 2-byte aligned) into shared memory with coalesced reads
 * (q8_0_copy_pass). Each block is summed as the scalar does (d times
 * the sum of q[g] * x[g], g in order), each lane accumulates its
 * blocks, and the warp sums the lanes: a row's arithmetic does not
 * depend on its partner row, so it is the bits of one warp per row.
 * The summation order across blocks is not the scalar one, so the
 * result matches the CPU only to rounding. Launch with 32 * kWarpRows
 * threads per block and ceil(rows / (2 * kWarpRows)) blocks.
 */
__global__ void matvec_q8_0_warp_kernel(const std::uint8_t* w,
                                        const float* x, float* out,
                                        std::uint32_t rows,
                                        std::uint32_t cols) {
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r0 = 2 * (blockIdx.x * kWarpRows + threadIdx.x / 32);
    if (r0 >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    // The second row, or the first again when rows is odd (its result
    // is not stored).
    const bool two = r0 + 1 < rows;
    const std::uint32_t r1 = two ? r0 + 1 : r0;
    __shared__ __align__(16) std::uint16_t sw[kWarpRows][2][32 * 17];
    std::uint16_t* mine0 = sw[threadIdx.x / 32][0];
    std::uint16_t* mine1 = sw[threadIdx.x / 32][1];
    const std::uint32_t nblk = cols / 32;
    const auto* row0 = reinterpret_cast<const std::uint16_t*>(
        w + static_cast<std::size_t>(r0) * nblk * 34);
    const auto* row1 = reinterpret_cast<const std::uint16_t*>(
        w + static_cast<std::size_t>(r1) * nblk * 34);
    // Each row's start alignment: 16, 8, or less (16-bit loads).
    const auto align_of = [](const void* p) -> std::uint32_t {
        const auto a = reinterpret_cast<std::uintptr_t>(p);
        return (a & 15) == 0 ? 16 : (a & 7) == 0 ? 8 : 2;
    };
    const std::uint32_t al0 = align_of(row0);
    const std::uint32_t al1 = align_of(row1);
    float acc0 = 0.0f, acc1 = 0.0f;
    for (std::uint32_t b0 = 0; b0 < nblk; b0 += 32) {
        const std::uint32_t nb = min(32u, nblk - b0);
        const std::size_t off = static_cast<std::size_t>(b0) * 17;
        q8_0_copy_pass(row0 + off, mine0, nb, al0, lane);
        q8_0_copy_pass(row1 + off, mine1, nb, al1, lane);
        __syncwarp();
        if (lane < nb) {
            const std::uint32_t b = b0 + lane;
            float d0, d1;
            q8_0_block_dot2(mine0 + lane * 17, mine1 + lane * 17,
                            x + static_cast<std::size_t>(b) * 32, d0, d1);
            acc0 += d0;
            acc1 += d1;
        }
        __syncwarp();  // the next pass overwrites this warp's slices
    }
    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if (lane == 0) {
        out[r0] = acc0;
        if (two) {
            out[r1] = acc1;
        }
    }
}

/**
 * Quantizes x to Q8_0 activations, 32 floats a block (i#1254), by the
 * CPU's rule (quantize_block32_act): d = amax / 127, stored rounded to
 * f16; each quant is x * (1 / d) of the unrounded d, rounded to
 * nearest even, its low byte kept (a NaN or an infinite product, as
 * when 1 / d overflows, gives 0). One warp a block, lane j element j;
 * the max is exact in any order. Launch with 256 threads a block and
 * ceil(nblk / 8) blocks.
 */
__global__ void quantize_q8_0_act_kernel(const float* __restrict__ x,
                                         float* __restrict__ d,
                                         std::int8_t* __restrict__ q,
                                         std::uint32_t nblk) {
    const std::uint32_t b = blockIdx.x * 8 + threadIdx.x / 32;
    if (b >= nblk) {
        return;  // a whole warp leaves together
    }
    const std::uint32_t lane = threadIdx.x % 32;
    const float v = x[static_cast<std::size_t>(b) * 32 + lane];
    // Seeded at 0 as the CPU's max: fmaxf drops a NaN, so an all-NaN
    // block gives amax 0 (d 0), not NaN.
    float amax = fmaxf(0.0f, fabsf(v));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    const float db = __fdiv_rn(amax, 127.0f);
    const float id = db != 0.0f ? __fdiv_rn(1.0f, db) : 0.0f;
    const float p = __fmul_rn(v, id);
    q[static_cast<std::size_t>(b) * 32 + lane] =
        isfinite(p) ? static_cast<std::int8_t>(__float2int_rn(p) & 0xff)
                    : std::int8_t{0};
    if (lane == 0) {
        d[b] = __half2float(__float2half_rn(db));
    }
}

/** The integer dot of 4 int8 pairs packed in a and b, added to c. The
 *  build targets sm_50 (CMakeLists.txt), so the byte multiplies run;
 *  __dp4a needs an sm_61 target (a newer card JITs compute_50). */
__device__ inline int dot4_i8(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    // sm_5x has no dp4a: the bytes, sign-extended, multiplied.
    return c + (a << 24 >> 24) * (b << 24 >> 24) +
           (a << 16 >> 24) * (b << 16 >> 24) +
           (a << 8 >> 24) * (b << 8 >> 24) + (a >> 24) * (b >> 24);
#endif
}

/**
 * One Q8_0 block's term against Q8_0 activations, the CPU's
 * vec_dot_block32_q8_0 step: float(p) * (d_w * d_x), p the exact
 * integer dot. Rounded multiplies (no fma), so the matvec and the
 * prefill tile give a block the same bits.
 */
__device__ inline float q8_0_q8x_term(int p, float dw, float dx) {
    return __fmul_rn(__int2float_rn(p), __fmul_rn(dw, dx));
}

/**
 * Q8_0 matvec on Q8_0 activations (quantize_q8_0_act_kernel; i#1254):
 * one warp a row, llama.cpp's mul_mat_vec_q layout: a pass takes 8
 * consecutive blocks, lane l bytes 8 (l % 4) .. + 7 of block l / 4,
 * read straight from global memory (weights by 16-bit words, x by 8
 * bytes), the 4 lanes of a block add their integer dots exactly, then
 * one float term a block (q8_0_q8x_term). Each term goes to lane kb %
 * 32 and adds there in block order, then the warp's xor tree sums the
 * lanes: the prefill tile's steps (matvec_tile_q8_0_q8x_kernel), so a
 * token's bits are the same in a prefill or a decode. Launch with 32 *
 * kWarpRows threads a block and ceil(rows / kWarpRows) blocks.
 */
__global__ void matvec_q8_0_q8x_kernel(const std::uint8_t* __restrict__ w,
                                       const float* __restrict__ xd,
                                       const std::int8_t* __restrict__ xq,
                                       float* __restrict__ out,
                                       std::uint32_t rows,
                                       std::uint32_t cols) {
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    const std::uint32_t nblk = cols / 32;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) * nblk * 34;
    const std::uint32_t g = lane / 4, sub = lane % 4;
    float acc = 0.0f;
    for (std::uint32_t it = 0; it * 8 < nblk; ++it) {
        const std::uint32_t kb = it * 8 + g;
        int p = 0;
        float dw = 0.0f, dx = 0.0f;
        if (kb < nblk) {
            const auto* blk = reinterpret_cast<const std::uint16_t*>(
                row + static_cast<std::size_t>(kb) * 34);
            const std::uint32_t w0 =
                __ldg(blk + 1 + 4 * sub) |
                static_cast<std::uint32_t>(__ldg(blk + 2 + 4 * sub)) << 16;
            const std::uint32_t w1 =
                __ldg(blk + 3 + 4 * sub) |
                static_cast<std::uint32_t>(__ldg(blk + 4 + 4 * sub)) << 16;
            const int2 xv = __ldg(reinterpret_cast<const int2*>(
                xq + static_cast<std::size_t>(kb) * 32 + 8 * sub));
            // Integer multiplies: an exact dot in float multiply-adds
            // (as the prefill tile) ran Qwen2.5 decode at 60.5 tok/s
            // against 70.3 here.
            p = dot4_i8(static_cast<int>(w0), xv.x, 0);
            p = dot4_i8(static_cast<int>(w1), xv.y, p);
            dw = __half2float(__ushort_as_half(__ldg(blk)));
            dx = __ldg(xd + kb);
        }
        p += __shfl_xor_sync(0xffffffffu, p, 1);
        p += __shfl_xor_sync(0xffffffffu, p, 2);
        const float term = q8_0_q8x_term(p, dw, dx);
        // Lane L owns block 8 it + L % 8 when L / 8 == it % 4: blocks
        // L, L + 32, ... in order, as the tile's lane L.
        const float t = __shfl_sync(0xffffffffu, term, 4 * (lane % 8));
        if (lane / 8 == it % 4 && it * 8 + lane % 8 < nblk) {
            acc = __fadd_rn(acc, t);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/**
 * The prefill tile for Q8_0 on Q8_0 activations (i#1254): R rows by T
 * tokens a warp, the weights staged as matvec_tile_q8_0_kernel stages
 * them; lane l takes block b0 + l of every row, and for each token the
 * block's exact integer dot and its term (q8_0_q8x_term), added in
 * block order, then the warp's xor tree: matvec_q8_0_q8x_kernel's bits.
 * xd / xq hold n tokens' activations (token t at t * nblk and t *
 * cols). Token-major output.
 */
template <std::uint32_t R, std::uint32_t T>
__global__ void matvec_tile_q8_0_q8x_kernel(
    const std::uint8_t* __restrict__ w, const float* __restrict__ xd,
    const std::int8_t* __restrict__ xq, float* __restrict__ out,
    std::uint32_t rows, std::uint32_t cols, std::uint32_t n) {
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r0 = (blockIdx.x * kWarpRows + threadIdx.x / 32) * R;
    if (r0 >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    __shared__ __align__(16) std::uint16_t sw[kWarpRows][R][32 * 17];
    auto& mine = sw[threadIdx.x / 32];
    const std::uint32_t t0 = blockIdx.y * T;
    const std::uint32_t nblk = cols / 32;
    const std::uint16_t* row16[R];
#pragma unroll
    for (std::uint32_t i = 0; i < R; ++i) {
        // A row past the end reads the last one and is not written.
        row16[i] = reinterpret_cast<const std::uint16_t*>(
            w + static_cast<std::size_t>(min(r0 + i, rows - 1)) * nblk * 34);
    }
    float acc[R][T];
#pragma unroll
    for (std::uint32_t i = 0; i < R; ++i) {
#pragma unroll
        for (std::uint32_t t = 0; t < T; ++t) {
            acc[i][t] = 0.0f;
        }
    }
    for (std::uint32_t b0 = 0; b0 < nblk; b0 += 32) {
        const std::uint32_t nb = min(32u, nblk - b0);
#pragma unroll
        for (std::uint32_t i = 0; i < R; ++i) {
            for (std::uint32_t k = lane; k < nb * 17; k += 32) {
                mine[i][k] =
                    __ldg(row16[i] + static_cast<std::size_t>(b0) * 17 + k);
            }
        }
        __syncwarp();
        if (lane < nb) {
            const std::uint32_t b = b0 + lane;
            // The integer dots in float: every product and partial sum
            // is an integer under 2^24 (|p| <= 32 * 127 * 128), so each
            // step is exact and pf is p in any order; sm_5x runs float
            // multiply-adds far faster than byte ones. Each weight word
            // is converted once a block, for all T tokens.
            float pf[R][T];
#pragma unroll
            for (std::uint32_t i = 0; i < R; ++i) {
#pragma unroll
                for (std::uint32_t t = 0; t < T; ++t) {
                    pf[i][t] = 0.0f;
                }
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float wf[R][4];
#pragma unroll
                for (std::uint32_t i = 0; i < R; ++i) {
                    const std::uint16_t* blk16 = mine[i] + lane * 17;
                    const int wq = static_cast<int>(
                        blk16[1 + 2 * k] |
                        static_cast<std::uint32_t>(blk16[2 + 2 * k]) << 16);
#pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        wf[i][g] = static_cast<float>(
                            static_cast<std::int8_t>(wq >> (8 * g)));
                    }
                }
#pragma unroll
                for (std::uint32_t t = 0; t < T; ++t) {
                    // A token past n reads token n - 1 (not stored).
                    const std::size_t tok = min(t0 + t, n - 1);
                    const int xv = __ldg(reinterpret_cast<const int*>(
                        xq + tok * cols + static_cast<std::size_t>(b) * 32 +
                        4 * k));
#pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        const float xf = static_cast<float>(
                            static_cast<std::int8_t>(xv >> (8 * g)));
#pragma unroll
                        for (std::uint32_t i = 0; i < R; ++i) {
                            pf[i][t] = fmaf(wf[i][g], xf, pf[i][t]);
                        }
                    }
                }
            }
#pragma unroll
            for (std::uint32_t i = 0; i < R; ++i) {
                const float dw = __half2float(
                    __ushort_as_half(mine[i][lane * 17]));
#pragma unroll
                for (std::uint32_t t = 0; t < T; ++t) {
                    const std::size_t tok = min(t0 + t, n - 1);
                    const float dx = __ldg(xd + tok * nblk + b);
                    acc[i][t] = __fadd_rn(
                        acc[i][t], __fmul_rn(pf[i][t], __fmul_rn(dw, dx)));
                }
            }
        }
        __syncwarp();  // the next pass overwrites this warp's slices
    }
#pragma unroll
    for (std::uint32_t i = 0; i < R; ++i) {
#pragma unroll
        for (std::uint32_t t = 0; t < T; ++t) {
            const float s = warp_sum(acc[i][t]);
            if (lane == 0 && r0 + i < rows && t0 + t < n) {
                out[static_cast<std::size_t>(t0 + t) * rows + r0 + i] = s;
            }
        }
    }
}

/** Copies the 16-entry IQ4 codebook (device `kvalues`) into shared
 *  memory for the block and syncs; every thread must call it. */
__device__ void load_iq4_codebook(const std::int8_t* kvalues,
                                  std::int8_t* skv) {
    if (threadIdx.x < 16) {
        skv[threadIdx.x] = kvalues[threadIdx.x];
    }
    __syncthreads();
}

/** The 32 codebook values of an IQ4_NL block (9 16-bit words in shared
 *  memory: d, then qs[16]): vlo[i] is element i, vhi[i] element
 *  16 + i. */
__device__ inline void iq4_nl_block_values(const std::uint16_t* blk16,
                                           const std::int8_t* skv,
                                           float vlo[16], float vhi[16]) {
    for (int i = 0; i < 16; ++i) {
        const std::uint32_t q = (blk16[1 + i / 2] >> (8 * (i % 2))) & 0xff;
        vlo[i] = static_cast<float>(skv[q & 0x0f]);
        vhi[i] = static_cast<float>(skv[q >> 4]);
    }
}

/** @returns d times the dot of an IQ4_NL block's values with the 32
 *  floats at xb (16-byte aligned), in the order of the IQ4_NL kernels:
 *  element 4k + m with its partner 16 past it, k and m 0..3. */
__device__ inline float iq4_nl_block_dot(float d, const float vlo[16],
                                         const float vhi[16],
                                         const float* xb) {
    const auto* x4 = reinterpret_cast<const float4*>(xb);
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) {  // elements 4k .. 4k + 3
        const float4 lo4 = __ldg(x4 + k);
        const float4 hi4 = __ldg(x4 + 4 + k);
        const float xl[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
        const float xh[4] = {hi4.x, hi4.y, hi4.z, hi4.w};
        for (int m = 0; m < 4; ++m) {
            const int i = 4 * k + m;
            s += vlo[i] * xl[m] + vhi[i] * xh[m];
        }
    }
    return d * s;
}

/**
 * One IQ4_NL block is 18 bytes (d f16 + qs[16]), 32 weights: element i
 * is the low nibble of qs[i] and element 16 + i the high nibble, each
 * through the kvalues codebook. One warp per row: lane L takes blocks
 * L, L + 32, ...; each pass the warp copies its row's 32 consecutive
 * blocks (576 bytes, as 16-bit words: blocks are only 2-byte aligned)
 * into shared memory, consecutive lanes reading consecutive words so
 * the reads coalesce. Each lane sums its block as the scalar does
 * (d times the sum over i of kv[lo] * x[i] + kv[hi] * x[i + 16]), and
 * the warp sums the lanes, so the result matches the CPU only to
 * rounding. Launch with 32 * kWarpRows threads per block.
 */
__global__ void matvec_iq4_nl_warp_kernel(const std::uint8_t* w,
                                          const std::int8_t* kvalues,
                                          const float* x, float* out,
                                          std::uint32_t rows,
                                          std::uint32_t cols) {
    __shared__ std::int8_t skv[16];
    __shared__ std::uint16_t sw[kWarpRows][32 * 9];
    load_iq4_codebook(kvalues, skv);  // before any warp returns
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    std::uint16_t* mine = sw[threadIdx.x / 32];
    const std::uint32_t nblk = cols / 32;
    const auto* row16 = reinterpret_cast<const std::uint16_t*>(
        w + static_cast<std::size_t>(r) * nblk * 18);
    float acc = 0.0f;
    for (std::uint32_t b0 = 0; b0 < nblk; b0 += 32) {
        const std::uint32_t nb = min(32u, nblk - b0);
        for (std::uint32_t i = lane; i < nb * 9; i += 32) {
            mine[i] = __ldg(row16 + static_cast<std::size_t>(b0) * 9 + i);
        }
        __syncwarp();
        if (lane < nb) {
            const std::uint32_t b = b0 + lane;
            const std::uint16_t* blk16 = mine + lane * 9;
            const float d = __half2float(__ushort_as_half(blk16[0]));
            float vlo[16], vhi[16];
            iq4_nl_block_values(blk16, skv, vlo, vhi);
            acc += iq4_nl_block_dot(d, vlo, vhi,
                                    x + static_cast<std::size_t>(b) * 32);
        }
        __syncwarp();  // the next pass overwrites this warp's slice
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/** The 8 weights lane `lane` owns in IQ4_XS block blk, as dl times the
 *  codebook value (the layout of matvec_iq4_xs_warp_kernel): vlo[k] is
 *  element 32 ib + c + k of the block, vhi[k] the one 16 past it
 *  (ib = lane / 4, c = 4 (lane % 4)). */
__device__ inline void iq4_xs_lane_values(const std::uint8_t* blk,
                                          std::uint32_t lane,
                                          const std::int8_t* skv,
                                          float vlo[4], float vhi[4]) {
    const std::uint32_t ib = lane / 4;
    const float d = ld_f16(blk);
    const std::uint32_t scales_h = static_cast<std::uint32_t>(blk[2]) |
                                   (static_cast<std::uint32_t>(blk[3]) << 8);
    const int ls =
        static_cast<int>(((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0x0f) |
                         (((scales_h >> (2 * ib)) & 3) << 4));
    const float dl = d * (ls - 32);
    const std::uint32_t q4 =
        __ldg(reinterpret_cast<const std::uint32_t*>(blk + 8) + lane);
    for (int k = 0; k < 4; ++k) {
        const std::uint32_t q = (q4 >> (8 * k)) & 0xff;
        vlo[k] = dl * skv[q & 0x0f];
        vhi[k] = dl * skv[q >> 4];
    }
}

/**
 * One IQ4_XS super-block is 136 bytes: d (f16) + scales_h (u16) +
 * scales_l[4] + qs[128], 256 weights in 8 sub-blocks of 32. In
 * sub-block ib, element j is the low nibble of qs[16 ib + j] and
 * element 16 + j the high nibble, each through the kvalues codebook,
 * times dl = d * (ls - 32) with ls the sub-block's 6-bit scale. One
 * warp per row: lane L reads the 4 qs bytes at 4L (one coalesced
 * 128-byte read per warp; 136-byte blocks keep 8-byte alignment), so
 * ib = L / 4 and j = 4 (L % 4) .. +3. Each lane accumulates its 8
 * products over the row, then the warp sums the lanes, so the result
 * matches the CPU only to rounding. Launch with 32 * kWarpRows threads
 * per block.
 */
__global__ void matvec_iq4_xs_warp_kernel(const std::uint8_t* w,
                                          const std::int8_t* kvalues,
                                          const float* x, float* out,
                                          std::uint32_t rows,
                                          std::uint32_t cols) {
    __shared__ std::int8_t skv[16];
    load_iq4_codebook(kvalues, skv);  // before any warp returns
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    const std::uint32_t nsb = cols / 256;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * nsb * 136;
    const std::uint32_t ib = lane / 4;
    const std::uint32_t c = 4 * (lane % 4);
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        float vlo[4], vhi[4];
        iq4_xs_lane_values(row + static_cast<std::size_t>(b) * 136, lane,
                           skv, vlo, vhi);
        acc = lane_dot8<16>(
            acc, vlo, vhi,
            x + static_cast<std::size_t>(b) * 256 + 32 * ib + c);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/** One Q2_K super-block is 84 bytes: scales[16] + qs[64] + d,dmin
 *  (f16), 256 weights. The scalar fills y sequentially; we still
 *  materialize tmp[256] then dot i=0..255 so the accumulation order
 *  is bit-for-bit dot_k_quant. Grid-free. */
__global__ void matvec_q2_k_kernel(const std::uint8_t* w,
                                   const float* x, float* out,
                                   std::uint32_t rows,
                                   std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 84;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            84;
        const std::uint8_t* scales = blk;
        const std::uint8_t* qp = blk + 16;
        const float d = ld_f16(blk + 80);
        const float mn = ld_f16(blk + 82);
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        float tmp[256];
        float* y = tmp;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                std::uint8_t sc = scales[is++];
                float dl = d * (sc & 0x0f), ml = mn * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((qp[l] >> shift) & 3) - ml;
                }
                sc = scales[is++];
                dl = d * (sc & 0x0f);
                ml = mn * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((qp[l + 16] >> shift) & 3) - ml;
                }
                shift += 2;
            }
            qp += 32;
        }
        for (int i = 0; i < 256; ++i) {
            acc += tmp[i] * xb[i];
        }
    }
    out[r] = acc;
}


/**
 * Copies `n16` 16-bit words of one super-block from global `src` into
 * the warp's shared slice `dst`, consecutive lanes reading consecutive
 * words (coalesced), then syncs the warp. The grid-quant super-blocks
 * are only 2-byte aligned, so they are read as 16-bit words and then
 * as bytes from shared memory at any offset.
 */
__device__ inline void warp_stage_u16(const std::uint16_t* src,
                                      std::uint32_t n16,
                                      std::uint16_t* dst,
                                      std::uint32_t lane) {
    for (std::uint32_t i = lane; i < n16; i += 32) {
        dst[i] = __ldg(src + i);
    }
    __syncwarp();
}

/** Loads the 8 activations a grid-quant lane owns (two float4). */
__device__ inline void lane_x8(const float* xb, float xs[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(xb));
    const float4 c = __ldg(reinterpret_cast<const float4*>(xb + 4));
    xs[0] = a.x;
    xs[1] = a.y;
    xs[2] = a.z;
    xs[3] = a.w;
    xs[4] = c.x;
    xs[5] = c.y;
    xs[6] = c.z;
    xs[7] = c.w;
}

/** @returns acc plus a lane's 8 IQ3 weights times its activations xs:
 *  the 4 u8 magnitudes of grid entry g1 then of g2, each times db and
 *  negated where `signs` has bit j (g1) or j + 4 (g2) set, in the
 *  order of the IQ3 warp kernels. */
__device__ inline float iq3_grid8_dot(float acc, std::uint32_t g1,
                                      std::uint32_t g2, float db,
                                      std::uint32_t signs,
                                      const float xs[8]) {
    for (int j = 0; j < 4; ++j) {
        const float y1 = db * static_cast<float>((g1 >> (8 * j)) & 0xff) *
                         ((signs & (1u << j)) ? -1.0f : 1.0f);
        const float y2 = db * static_cast<float>((g2 >> (8 * j)) & 0xff) *
                         ((signs & (1u << (j + 4))) ? -1.0f : 1.0f);
        acc += y1 * xs[j];
        acc += y2 * xs[j + 4];
    }
    return acc;
}

/**
 * One IQ3_XXS super-block is 98 bytes: d (f16) + qs[64] +
 * scales_and_signs[32], 256 weights in 8 groups of 32, each group 4
 * units of 8 (two 4-byte grid entries through qs, signs through
 * ksigns). One warp per row: lane L owns group g = L / 4, unit
 * l = L % 4, so the 32 lanes cover the 32 units of a super-block; each
 * super-block is staged in shared memory first (warp_stage_u16).
 * Each lane dequantizes its 8 weights as dequant_block_iq3_xxs does
 * and accumulates them over the row, then the warp sums the lanes,
 * so the result matches the CPU only to rounding. `grid` = device
 * iq3xxs_grid (u32), `ksigns` = device ksigns_iq2xs. Launch with
 * 32 * kWarpRows threads per block.
 */
__global__ void matvec_iq3_xxs_warp_kernel(const std::uint8_t* w,
                                           const std::uint32_t* grid,
                                           const std::uint8_t* ksigns,
                                           const float* x, float* out,
                                           std::uint32_t rows,
                                           std::uint32_t cols) {
    __shared__ std::uint16_t sw[kWarpRows][49];  // 98 bytes
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    std::uint16_t* mine = sw[threadIdx.x / 32];
    const auto* sb = reinterpret_cast<const std::uint8_t*>(mine);
    const std::uint32_t nsb = cols / 256;
    const auto* row16 = reinterpret_cast<const std::uint16_t*>(
        w + static_cast<std::size_t>(r) * nsb * 98);
    const std::uint32_t g = lane / 4;
    const std::uint32_t l = lane % 4;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        warp_stage_u16(row16 + static_cast<std::size_t>(b) * 49, 49, mine,
                       lane);
        const float d = __half2float(__ushort_as_half(mine[0]));
        const std::uint8_t* qs = sb + 2 + 8 * g;
        // scales_and_signs word of group g (bytes 66 + 4g): words 33 + 2g.
        const std::uint32_t aux =
            static_cast<std::uint32_t>(mine[33 + 2 * g]) |
            (static_cast<std::uint32_t>(mine[34 + 2 * g]) << 16);
        const float db = d * (0.5f + (aux >> 28)) * 0.5f;
        const std::uint32_t signs = __ldg(ksigns + ((aux >> (7 * l)) & 127));
        const std::uint32_t g1 = __ldg(grid + qs[2 * l]);
        const std::uint32_t g2 = __ldg(grid + qs[2 * l + 1]);
        float xs[8];
        lane_x8(x + static_cast<std::size_t>(b) * 256 + 32 * g + 8 * l, xs);
        acc = iq3_grid8_dot(acc, g1, g2, db, signs, xs);
        __syncwarp();  // the next stage overwrites this warp's slice
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/**
 * One IQ3_S super-block is 110 bytes: d | qs[64] | qh[8] | signs[32] |
 * scales[4], 256 weights in 8 groups of 32, each 4 units of 8. Unit
 * l of group g takes grid entries qs[8g + 2l] and qs[8g + 2l + 1],
 * each with a ninth index bit from qh[g] (bits 2l and 2l + 1), the
 * sign byte signs[4g + l], and the scale nibble scales[g / 2] (low
 * for even g). One warp per row, laid out as
 * matvec_iq3_xxs_warp_kernel (lane L owns g = L / 4, l = L % 4; each
 * super-block staged in shared memory). The result matches the CPU
 * only to rounding. `grid` = device iq3s_grid (u32). Launch with
 * 32 * kWarpRows threads per block.
 */
__global__ void matvec_iq3_s_warp_kernel(const std::uint8_t* w,
                                         const std::uint32_t* grid,
                                         const float* x, float* out,
                                         std::uint32_t rows,
                                         std::uint32_t cols) {
    __shared__ std::uint16_t sw[kWarpRows][55];  // 110 bytes
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    std::uint16_t* mine = sw[threadIdx.x / 32];
    const auto* sb = reinterpret_cast<const std::uint8_t*>(mine);
    const std::uint32_t nsb = cols / 256;
    const auto* row16 = reinterpret_cast<const std::uint16_t*>(
        w + static_cast<std::size_t>(r) * nsb * 110);
    const std::uint32_t g = lane / 4;
    const std::uint32_t l = lane % 4;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        warp_stage_u16(row16 + static_cast<std::size_t>(b) * 55, 55, mine,
                       lane);
        const float d = __half2float(__ushort_as_half(mine[0]));
        const std::uint8_t* qs = sb + 2 + 8 * g;
        const std::uint32_t qh = sb[66 + g];
        const std::uint32_t signs = sb[74 + 4 * g + l];
        const std::uint32_t sc = (sb[106 + g / 2] >> (4 * (g % 2))) & 0xf;
        const float db = d * (1 + 2 * sc);
        const std::uint32_t g1 =
            __ldg(grid + (qs[2 * l] | ((qh << (8 - 2 * l)) & 256)));
        const std::uint32_t g2 =
            __ldg(grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
        float xs[8];
        lane_x8(x + static_cast<std::size_t>(b) * 256 + 32 * g + 8 * l, xs);
        acc = iq3_grid8_dot(acc, g1, g2, db, signs, xs);
        __syncwarp();  // the next stage overwrites this warp's slice
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/** Adds the 8 weights of one IQ2 grid entry `gv` (8 u8 magnitudes),
 *  each db * magnitude, negated where `signs` has the bit set, times
 *  the lane's 8 activations, to `acc` (the scalar's y[j] = db * g[j] *
 *  sign, in j order). */
__device__ inline float iq2_grid8_dot(std::uint64_t gv, float db,
                                      std::uint32_t signs,
                                      const float xs[8], float acc) {
    for (int j = 0; j < 8; ++j) {
        const float y = db * static_cast<float>((gv >> (8 * j)) & 0xff) *
                        ((signs & (1u << j)) ? -1.0f : 1.0f);
        acc += y * xs[j];
    }
    return acc;
}

/**
 * One IQ2_XXS super-block is 66 bytes: d (f16) + 32 u16, 256 weights
 * in 8 groups of 32, each 4 units of 8. Group g's 8 bytes at 2 + 8g
 * hold 4 grid indices (one per unit l) and a 32-bit word with the scale
 * (top 4 bits) and 4 sign indices (7 bits each, through ksigns). One
 * warp per row: lane L owns g = L / 4, l = L % 4; each super-block is
 * staged in shared memory (warp_stage_u16). The result matches the CPU
 * only to rounding. `grid` = device iq2xxs_grid (u64), `ksigns` =
 * device ksigns_iq2xs. Launch with 32 * kWarpRows threads per block.
 */
__global__ void matvec_iq2_xxs_warp_kernel(const std::uint8_t* w,
                                           const std::uint64_t* grid,
                                           const std::uint8_t* ksigns,
                                           const float* x, float* out,
                                           std::uint32_t rows,
                                           std::uint32_t cols) {
    __shared__ std::uint16_t sw[kWarpRows][33];  // 66 bytes
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    std::uint16_t* mine = sw[threadIdx.x / 32];
    const auto* sb = reinterpret_cast<const std::uint8_t*>(mine);
    const std::uint32_t nsb = cols / 256;
    const auto* row16 = reinterpret_cast<const std::uint16_t*>(
        w + static_cast<std::size_t>(r) * nsb * 66);
    const std::uint32_t g = lane / 4;
    const std::uint32_t l = lane % 4;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        warp_stage_u16(row16 + static_cast<std::size_t>(b) * 33, 33, mine,
                       lane);
        const float d = __half2float(__ushort_as_half(mine[0]));
        // bytes 6 + 8g .. 9 + 8g: the group's scale / signs word.
        const std::uint32_t aux =
            static_cast<std::uint32_t>(mine[3 + 4 * g]) |
            (static_cast<std::uint32_t>(mine[4 + 4 * g]) << 16);
        const float db = d * (0.5f + (aux >> 28)) * 0.25f;
        const std::uint32_t signs = __ldg(ksigns + ((aux >> (7 * l)) & 127));
        const std::uint64_t gv = __ldg(grid + sb[2 + 8 * g + l]);
        float xs[8];
        lane_x8(x + static_cast<std::size_t>(b) * 256 + 32 * g + 8 * l, xs);
        acc = iq2_grid8_dot(gv, db, signs, xs, acc);
        __syncwarp();  // the next stage overwrites this warp's slice
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/**
 * One IQ2_XS super-block is 74 bytes: d (f16) + qs[32] u16 + scales[8],
 * 256 weights in 8 groups of 32, each 4 units of 8. Unit l of group g
 * is qs[4g + l]: its low 9 bits index iq2xs_grid, its high 7 bits
 * index ksigns; its scale is nibble l / 2 of scales[g]. One warp per
 * row: lane L owns g = L / 4, l = L % 4; each super-block is staged in
 * shared memory. The result matches the CPU only to rounding. `grid` =
 * device iq2xs_grid (u64). Launch with 32 * kWarpRows threads per
 * block.
 */
__global__ void matvec_iq2_xs_warp_kernel(const std::uint8_t* w,
                                          const std::uint64_t* grid,
                                          const std::uint8_t* ksigns,
                                          const float* x, float* out,
                                          std::uint32_t rows,
                                          std::uint32_t cols) {
    __shared__ std::uint16_t sw[kWarpRows][37];  // 74 bytes
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    std::uint16_t* mine = sw[threadIdx.x / 32];
    const auto* sb = reinterpret_cast<const std::uint8_t*>(mine);
    const std::uint32_t nsb = cols / 256;
    const auto* row16 = reinterpret_cast<const std::uint16_t*>(
        w + static_cast<std::size_t>(r) * nsb * 74);
    const std::uint32_t g = lane / 4;
    const std::uint32_t l = lane % 4;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        warp_stage_u16(row16 + static_cast<std::size_t>(b) * 37, 37, mine,
                       lane);
        const float d = __half2float(__ushort_as_half(mine[0]));
        const std::uint32_t q = mine[1 + 4 * g + l];
        const std::uint32_t sc = (sb[66 + g] >> (4 * (l / 2))) & 0xf;
        const float db = d * (0.5f + sc) * 0.25f;
        const std::uint32_t signs = __ldg(ksigns + (q >> 9));
        const std::uint64_t gv = __ldg(grid + (q & 511));
        float xs[8];
        lane_x8(x + static_cast<std::size_t>(b) * 256 + 32 * g + 8 * l, xs);
        acc = iq2_grid8_dot(gv, db, signs, xs, acc);
        __syncwarp();  // the next stage overwrites this warp's slice
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/**
 * One IQ2_S super-block is 82 bytes: d + qs[64] + qh[8] + scales[8],
 * 256 weights in 8 groups of 32, each 4 units of 8. Unit l of group g
 * takes grid index byte qs[4g + l] plus bits 2l, 2l + 1 of qh[g] as
 * index bits 8, 9 (10-bit iq2s_grid index), the raw sign byte
 * qs[32 + 4g + l], and nibble l / 2 of scales[g]. One warp per row:
 * lane L owns g = L / 4, l = L % 4; each super-block is staged in
 * shared memory. The result matches the CPU only to rounding. `grid` =
 * device iq2s_grid (u64). Launch with 32 * kWarpRows threads per block.
 */
__global__ void matvec_iq2_s_warp_kernel(const std::uint8_t* w,
                                         const std::uint64_t* grid,
                                         const float* x, float* out,
                                         std::uint32_t rows,
                                         std::uint32_t cols) {
    __shared__ std::uint16_t sw[kWarpRows][41];  // 82 bytes
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    std::uint16_t* mine = sw[threadIdx.x / 32];
    const auto* sb = reinterpret_cast<const std::uint8_t*>(mine);
    const std::uint32_t nsb = cols / 256;
    const auto* row16 = reinterpret_cast<const std::uint16_t*>(
        w + static_cast<std::size_t>(r) * nsb * 82);
    const std::uint32_t g = lane / 4;
    const std::uint32_t l = lane % 4;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        warp_stage_u16(row16 + static_cast<std::size_t>(b) * 41, 41, mine,
                       lane);
        const float d = __half2float(__ushort_as_half(mine[0]));
        const std::uint32_t qh = sb[66 + g];
        const std::uint32_t idx =
            sb[2 + 4 * g + l] | ((qh << (8 - 2 * l)) & 0x300);
        const std::uint32_t signs = sb[34 + 4 * g + l];
        const std::uint32_t sc = (sb[74 + g] >> (4 * (l / 2))) & 0xf;
        const float db = d * (0.5f + sc) * 0.25f;
        const std::uint64_t gv = __ldg(grid + idx);
        float xs[8];
        lane_x8(x + static_cast<std::size_t>(b) * 256 + 32 * g + 8 * l, xs);
        acc = iq2_grid8_dot(gv, db, signs, xs, acc);
        __syncwarp();  // the next stage overwrites this warp's slice
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/** Adds the 8 weights of one IQ1 grid entry `gv` (8 int8 values),
 *  each dl * (value + delta), times the lane's 8 activations, to `acc`
 *  (the scalar's expression, in j order). */
__device__ inline float iq1_grid8_dot(std::uint64_t gv, float dl,
                                      float delta, const float xs[8],
                                      float acc) {
    for (int j = 0; j < 8; ++j) {
        const auto v = static_cast<std::int8_t>((gv >> (8 * j)) & 0xff);
        acc += dl * (static_cast<float>(v) + delta) * xs[j];
    }
    return acc;
}

/**
 * One IQ1_S super-block is 50 bytes: d (f16) + qs[32] + qh[8] u16, 256
 * weights in 8 groups of 32, each 4 units of 8. Group g's qh word
 * (bytes 34 + 2g) holds the 3-bit scale (bits 12..14), the delta sign
 * (bit 15) and 3 high grid-index bits per unit (bits 3l..3l + 2); unit
 * l's grid index is qs[4g + l] plus those bits as bits 8..10. Each
 * weight is dl * (grid value + delta). One warp per row: lane L owns
 * g = L / 4, l = L % 4; each super-block is staged in shared memory.
 * The result matches the CPU only to rounding. `grid` = device
 * iq1s_grid. Launch with 32 * kWarpRows threads per block.
 */
__global__ void matvec_iq1_s_warp_kernel(const std::uint8_t* w,
                                         const std::uint64_t* grid,
                                         const float* x, float* out,
                                         std::uint32_t rows,
                                         std::uint32_t cols) {
    __shared__ std::uint16_t sw[kWarpRows][25];  // 50 bytes
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    std::uint16_t* mine = sw[threadIdx.x / 32];
    const auto* sb = reinterpret_cast<const std::uint8_t*>(mine);
    const std::uint32_t nsb = cols / 256;
    const auto* row16 = reinterpret_cast<const std::uint16_t*>(
        w + static_cast<std::size_t>(r) * nsb * 50);
    const std::uint32_t g = lane / 4;
    const std::uint32_t l = lane % 4;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        warp_stage_u16(row16 + static_cast<std::size_t>(b) * 25, 25, mine,
                       lane);
        const float d = __half2float(__ushort_as_half(mine[0]));
        const std::uint32_t qh = mine[17 + g];  // bytes 34 + 2g
        const float dl = d * (2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
        const std::uint32_t gi =
            sb[2 + 4 * g + l] | (((qh >> (3 * l)) & 7) << 8);
        const std::uint64_t gv = __ldg(grid + gi);
        float xs[8];
        lane_x8(x + static_cast<std::size_t>(b) * 256 + 32 * g + 8 * l, xs);
        acc = iq1_grid8_dot(gv, dl, delta, xs, acc);
        __syncwarp();  // the next stage overwrites this warp's slice
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/**
 * One IQ1_M super-block is 56 bytes: qs[32] | qh[16] | scales[8] (four
 * u16), 256 weights in 8 groups of 32, each 4 units of 8. The f16
 * scale d is the top nibbles of the four scale words; group g's two
 * 3-bit sub-scales are bits 6 (g % 2) and 6 (g % 2) + 3 of scale word
 * g / 2 (units 0, 1 and 2, 3). Unit l's grid index is qs[4g + l] plus
 * three bits of qh byte 32 + 2g + l / 2 (the low nibble for even l,
 * the high for odd) as bits 8..10, and that nibble's top bit picks the
 * delta sign. Each weight is dl * (grid value + delta). One warp per
 * row: lane L owns g = L / 4, l = L % 4; each super-block is staged in
 * shared memory. The result matches the CPU only to rounding. `grid` =
 * device iq1s_grid. Launch with 32 * kWarpRows threads per block.
 */
__global__ void matvec_iq1_m_warp_kernel(const std::uint8_t* w,
                                         const std::uint64_t* grid,
                                         const float* x, float* out,
                                         std::uint32_t rows,
                                         std::uint32_t cols) {
    __shared__ std::uint16_t sw[kWarpRows][28];  // 56 bytes
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    std::uint16_t* mine = sw[threadIdx.x / 32];
    const auto* sb = reinterpret_cast<const std::uint8_t*>(mine);
    const std::uint32_t nsb = cols / 256;
    const auto* row16 = reinterpret_cast<const std::uint16_t*>(
        w + static_cast<std::size_t>(r) * nsb * 56);
    const std::uint32_t g = lane / 4;
    const std::uint32_t l = lane % 4;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        warp_stage_u16(row16 + static_cast<std::size_t>(b) * 28, 28, mine,
                       lane);
        const std::uint32_t sc0 = mine[24], sc1 = mine[25];
        const std::uint32_t sc2 = mine[26], sc3 = mine[27];
        const auto su16 = static_cast<unsigned short>(
            (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) |
            (sc3 & 0xf000));
        const float d = __half2float(__ushort_as_half(su16));
        const std::uint32_t scw = mine[24 + g / 2];
        const std::uint32_t sub = (scw >> (6 * (g % 2) + 3 * (l / 2))) & 0x7;
        const float dl = d * (2 * sub + 1);
        const std::uint32_t qh = sb[32 + 2 * g + l / 2];
        const std::uint32_t gi =
            sb[4 * g + l] | ((qh << ((l % 2) ? 4 : 8)) & 0x700);
        const float delta =
            (qh & ((l % 2) ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
        const std::uint64_t gv = __ldg(grid + gi);
        float xs[8];
        lane_x8(x + static_cast<std::size_t>(b) * 256 + 32 * g + 8 * l, xs);
        acc = iq1_grid8_dot(gv, dl, delta, xs, acc);
        __syncwarp();  // the next stage overwrites this warp's slice
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        out[r] = acc;
    }
}

/** TQ1_0 (54B): qs[48] | qh[4] | d. Ternary base-3 packing (5 digits
 *  per qs byte, 4 per qh byte); grid-free. Materialize then dot to
 *  match dot_k_quant / dequant_block_tq1_0 exactly (uint8 wrap kept). */
__global__ void matvec_tq1_0_kernel(const std::uint8_t* w,
                                    const float* x, float* out,
                                    std::uint32_t rows,
                                    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 54;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            54;
        const std::uint8_t* qs = blk;
        const std::uint8_t* qh = blk + 48;
        const float d = ld_f16(blk + 52);
        float tmp[256];
        float* y = tmp;
        for (int n = 0; n < 5; ++n) {
            for (int m = 0; m < 32; ++m) {
                std::uint8_t q =
                    static_cast<std::uint8_t>(qs[m] * pow3[n]);
                std::int16_t xi =
                    (static_cast<std::uint16_t>(q) * 3) >> 8;
                *y++ = static_cast<float>(xi - 1) * d;
            }
        }
        for (int n = 0; n < 5; ++n) {
            for (int m = 0; m < 16; ++m) {
                std::uint8_t q =
                    static_cast<std::uint8_t>(qs[32 + m] * pow3[n]);
                std::int16_t xi =
                    (static_cast<std::uint16_t>(q) * 3) >> 8;
                *y++ = static_cast<float>(xi - 1) * d;
            }
        }
        for (int n = 0; n < 4; ++n) {
            for (int j = 0; j < 4; ++j) {
                std::uint8_t q =
                    static_cast<std::uint8_t>(qh[j] * pow3[n]);
                std::int16_t xi =
                    (static_cast<std::uint16_t>(q) * 3) >> 8;
                *y++ = static_cast<float>(xi - 1) * d;
            }
        }
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        for (int i = 0; i < 256; ++i) {
            acc += tmp[i] * xb[i];
        }
    }
    out[r] = acc;
}

/** TQ2_0 (66B): qs[64] | d. Ternary at 2 bits/element; grid-free.
 *  Matches dequant_block_tq2_0. */
__global__ void matvec_tq2_0_kernel(const std::uint8_t* w,
                                    const float* x, float* out,
                                    std::uint32_t rows,
                                    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 66;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            66;
        const std::uint8_t* qs = blk;
        const float d = ld_f16(blk + 64);
        float tmp[256];
        float* y = tmp;
        for (int j = 0; j < 64; j += 32) {
            for (int l = 0; l < 4; ++l) {
                for (int m = 0; m < 32; ++m) {
                    const int q = (qs[j + m] >> (l * 2)) & 3;
                    *y++ = static_cast<float>(q - 1) * d;
                }
            }
        }
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        for (int i = 0; i < 256; ++i) {
            acc += tmp[i] * xb[i];
        }
    }
    out[r] = acc;
}

// R11 batched kernels: one launch, one warp per row, n token
// accumulators per lane. Each weight
// block is read from VRAM once and reused across all n tokens (weight
// traffic ~n-fold down). Each token t's accumulator runs the identical
// dequant+FMA order as the per-token kernel, so the result is
// byte-identical to n matvec_cuda() calls.
// out is row-major: out[r*n + t]. x is token-major: token t at t*cols.
// Q4_K, Q6_K and Q8_0 use the prefill tiles below instead (i#1195).
constexpr std::uint32_t kMaxBatch = 32;

/** lane_dot8 on activations already loaded (xl = xb[0..3], xh =
 *  xb[kHi..kHi+3]): the same products in the same order. */
__device__ inline float lane_dot8_x(float acc, const float vlo[4],
                                    const float vhi[4], const float xl[4],
                                    const float xh[4]) {
    for (int k = 0; k < 4; ++k) {
        acc += vlo[k] * xl[k];
        acc += vhi[k] * xh[k];
    }
    return acc;
}

/** The Q4_K super-block for matvec_tile_smem_kernel: the per-token kernel's
 *  lane values, at x offset 64 j + c and its high half 32 floats on. */
struct TileQ4K {
    static constexpr std::uint32_t kBytes = 144;
    static constexpr std::uint32_t kHi = 32;
    __device__ static std::uint32_t values(const std::uint8_t* blk,
                                           std::uint32_t lane, float lo[4],
                                           float hi[4]) {
        q4_k_lane_values(blk, lane, lo, hi);
        return 64 * (lane / 8) + 4 * (lane % 8);
    }
};

/** The Q6_K super-block for matvec_tile_smem_kernel: the per-token kernel's
 *  lane values, at the x offset it returns and 64 floats on. */
struct TileQ6K {
    static constexpr std::uint32_t kBytes = 210;
    static constexpr std::uint32_t kHi = 64;
    __device__ static std::uint32_t values(const std::uint8_t* blk,
                                           std::uint32_t lane, float lo[4],
                                           float hi[4]) {
        return q6_k_lane_values(blk, lane, lo, hi);
    }
};

/**
 * Prefill tile for a K-quant Q (TileQ4K, TileQ6K; i#1195, i#1200):
 * each warp computes R rows by T tokens, so one load of a token's
 * activations serves R rows and one decode of a row's weights serves T
 * tokens. Per super-block the block's threads stage the T tokens' 256
 * floats in shared memory once, and every warp reads them from there
 * (each warp loading them from global memory bound the speed). Per
 * (row, token) the lane runs the per-token kernel's products in its
 * order and the warp sums the lanes the same way, so the result is
 * byte-identical to the per-token kernel. One launch covers all n
 * tokens (grid.y tiles them) and writes the token-major output
 * (out[t * rows + r]) directly. A warp past the last row stays to load
 * and to sync, and stores nothing. Launch with 32 * W threads per
 * block, W warps of R rows each.
 */
template <std::uint32_t R, std::uint32_t T, std::uint32_t W, class Q>
__global__ void matvec_tile_smem_kernel(const std::uint8_t* w,
                                        const float* x, float* out,
                                        std::uint32_t rows,
                                        std::uint32_t cols, std::uint32_t n) {
    __shared__ float4 xs[T][64];  // T tokens by 256 floats
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r0 = (blockIdx.x * W + threadIdx.x / 32) * R;
    const bool active = r0 < rows;
    const std::uint32_t t0 = blockIdx.y * T;
    const std::uint32_t nsb = cols / 256;
    const std::uint8_t* row[R];
#pragma unroll
    for (std::uint32_t i = 0; i < R; ++i) {
        // A row past the end reads the last one and is not written.
        row[i] = w + static_cast<std::size_t>(min(r0 + i, rows - 1)) * nsb *
                         Q::kBytes;
    }
    float acc[R][T];
#pragma unroll
    for (std::uint32_t i = 0; i < R; ++i) {
#pragma unroll
        for (std::uint32_t t = 0; t < T; ++t) {
            acc[i][t] = 0.0f;
        }
    }
    for (std::uint32_t b = 0; b < nsb; ++b) {
        // Stage: T x 64 float4, a token past n as zeros (never read).
        for (std::uint32_t k = threadIdx.x; k < T * 64; k += 32 * W) {
            const std::uint32_t t = k / 64, c = k % 64;
            xs[t][c] = t0 + t < n
                           ? __ldg(reinterpret_cast<const float4*>(
                                       x + static_cast<std::size_t>(t0 + t) *
                                               cols +
                                       static_cast<std::size_t>(b) * 256) +
                                   c)
                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();
        if (active) {
            float vlo[R][4], vhi[R][4];
            std::uint32_t e = 0;  // the lane's x offset, as in every row
#pragma unroll
            for (std::uint32_t i = 0; i < R; ++i) {
                e = Q::values(row[i] + static_cast<std::size_t>(b) * Q::kBytes,
                              lane, vlo[i], vhi[i]);
            }
#pragma unroll
            for (std::uint32_t t = 0; t < T; ++t) {
                if (t0 + t >= n) {
                    continue;
                }
                const float4 lo = xs[t][e / 4];
                const float4 hi = xs[t][(e + Q::kHi) / 4];
                const float xl[4] = {lo.x, lo.y, lo.z, lo.w};
                const float xh[4] = {hi.x, hi.y, hi.z, hi.w};
#pragma unroll
                for (std::uint32_t i = 0; i < R; ++i) {
                    acc[i][t] = lane_dot8_x(acc[i][t], vlo[i], vhi[i], xl, xh);
                }
            }
        }
        __syncthreads();  // the next super-block overwrites xs
    }
    if (!active) {
        return;  // after the last barrier: no sync is skipped
    }
#pragma unroll
    for (std::uint32_t i = 0; i < R; ++i) {
#pragma unroll
        for (std::uint32_t t = 0; t < T; ++t) {
            const float s = warp_sum(acc[i][t]);
            if (lane == 0 && r0 + i < rows && t0 + t < n) {
                out[static_cast<std::size_t>(t0 + t) * rows + r0 + i] = s;
            }
        }
    }
}

/**
 * The prefill tile for Q8_0 (i#1195): R rows by T tokens a warp. A
 * pass stages 32 blocks of each of the R rows in shared memory, as the
 * per-token kernel stages its rows'; lane l then takes block b0 + l
 * of every row, loads each token's 32 activations of that block once
 * and adds them into all R rows' block sums. Each block sum runs the
 * products in q8_0_block_dot2's order and is added as d * sum, and the
 * warp sums the lanes the same way, so the result is byte-identical
 * to the per-token kernel. Token-major output, all n tokens a launch.
 */
template <std::uint32_t R, std::uint32_t T>
__global__ void matvec_tile_q8_0_kernel(const std::uint8_t* w,
                                        const float* x, float* out,
                                        std::uint32_t rows,
                                        std::uint32_t cols,
                                        std::uint32_t n) {
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r0 = (blockIdx.x * kWarpRows + threadIdx.x / 32) * R;
    if (r0 >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    __shared__ std::uint16_t sw[kWarpRows][R][32 * 17];
    auto& mine = sw[threadIdx.x / 32];
    const std::uint32_t t0 = blockIdx.y * T;
    const std::uint32_t nblk = cols / 32;
    const std::uint16_t* row16[R];
#pragma unroll
    for (std::uint32_t i = 0; i < R; ++i) {
        // A row past the end reads the last one and is not written.
        row16[i] = reinterpret_cast<const std::uint16_t*>(
            w + static_cast<std::size_t>(min(r0 + i, rows - 1)) * nblk * 34);
    }
    float acc[R][T];
#pragma unroll
    for (std::uint32_t i = 0; i < R; ++i) {
#pragma unroll
        for (std::uint32_t t = 0; t < T; ++t) {
            acc[i][t] = 0.0f;
        }
    }
    for (std::uint32_t b0 = 0; b0 < nblk; b0 += 32) {
        const std::uint32_t nb = min(32u, nblk - b0);
#pragma unroll
        for (std::uint32_t i = 0; i < R; ++i) {
            for (std::uint32_t k = lane; k < nb * 17; k += 32) {
                mine[i][k] =
                    __ldg(row16[i] + static_cast<std::size_t>(b0) * 17 + k);
            }
        }
        __syncwarp();
        if (lane < nb) {
            const std::uint32_t b = b0 + lane;
            float d[R];
#pragma unroll
            for (std::uint32_t i = 0; i < R; ++i) {
                d[i] = __half2float(__ushort_as_half(mine[i][lane * 17]));
            }
#pragma unroll
            for (std::uint32_t t = 0; t < T; ++t) {
                if (t0 + t >= n) {
                    continue;
                }
                const auto* x4 = reinterpret_cast<const float4*>(
                    x + static_cast<std::size_t>(t0 + t) * cols +
                    static_cast<std::size_t>(b) * 32);
                float bacc[R];
#pragma unroll
                for (std::uint32_t i = 0; i < R; ++i) {
                    bacc[i] = 0.0f;
                }
                for (int k = 0; k < 8; ++k) {
                    const float4 xv = __ldg(x4 + k);
                    const float xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                    for (std::uint32_t i = 0; i < R; ++i) {
                        const std::uint16_t* blk16 = mine[i] + lane * 17;
                        const std::uint16_t lo = blk16[1 + 2 * k];
                        const std::uint16_t hi = blk16[2 + 2 * k];
                        const std::int8_t q[4] = {
                            static_cast<std::int8_t>(lo & 0xff),
                            static_cast<std::int8_t>(lo >> 8),
                            static_cast<std::int8_t>(hi & 0xff),
                            static_cast<std::int8_t>(hi >> 8)};
                        for (int g = 0; g < 4; ++g) {
                            bacc[i] += static_cast<float>(q[g]) * xs[g];
                        }
                    }
                }
#pragma unroll
                for (std::uint32_t i = 0; i < R; ++i) {
                    acc[i][t] += d[i] * bacc[i];
                }
            }
        }
        __syncwarp();  // the next pass overwrites this warp's slices
    }
#pragma unroll
    for (std::uint32_t i = 0; i < R; ++i) {
#pragma unroll
        for (std::uint32_t t = 0; t < T; ++t) {
            const float s = warp_sum(acc[i][t]);
            if (lane == 0 && r0 + i < rows && t0 + t < n) {
                out[static_cast<std::size_t>(t0 + t) * rows + r0 + i] = s;
            }
        }
    }
}

/** Batched matvec_q5_k_warp_kernel: one warp per row, each lane with n
 *  token accumulators. A super-block's 8 weights per lane are
 *  dequantized once, as d * q - m (the per-token kernel's factor), and
 *  applied to every token in the per-token kernel's order, with the
 *  same warp sum per token, so the result is byte-identical to n
 *  per-token calls. Launch with 32 * kWarpRows threads per block. */
__global__ void matvec_batch_q5_k_kernel(const std::uint8_t* w,
                                         const float* x, float* out,
                                         std::uint32_t rows,
                                         std::uint32_t cols,
                                         std::uint32_t n) {
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    const std::uint32_t nsb = cols / 256;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * nsb * 176;
    const std::uint32_t j = lane / 8;
    const std::uint32_t c = 4 * (lane % 8);
    float acc[kMaxBatch];
    for (std::uint32_t t = 0; t < n; ++t) {
        acc[t] = 0.0f;
    }
    for (std::uint32_t b = 0; b < nsb; ++b) {
        float vlo[4], vhi[4];
        q5_k_lane_values(row + static_cast<std::size_t>(b) * 176, lane, vlo,
                         vhi);
        for (std::uint32_t t = 0; t < n; ++t) {
            acc[t] = lane_dot8(acc[t], vlo, vhi,
                               x + static_cast<std::size_t>(t) * cols +
                                   static_cast<std::size_t>(b) * 256 +
                                   64 * j + c);
        }
    }
    for (std::uint32_t t = 0; t < n; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) {
            out[static_cast<std::size_t>(r) * n + t] = s;
        }
    }
}

/** Batched matvec_iq4_nl_warp_kernel: the same shared-memory staging of
 *  32 blocks per pass, read once for all n tokens; each lane looks its
 *  block's 32 codebook values up once, then per token sums them in the
 *  per-token kernel's order, with the same warp sum, so the result is
 *  byte-identical to n per-token calls. Launch with 32 * kWarpRows
 *  threads per block. */
__global__ void matvec_batch_iq4_nl_kernel(const std::uint8_t* w,
                                           const std::int8_t* kvalues,
                                           const float* x, float* out,
                                           std::uint32_t rows,
                                           std::uint32_t cols,
                                           std::uint32_t n) {
    __shared__ std::int8_t skv[16];
    __shared__ std::uint16_t sw[kWarpRows][32 * 9];
    load_iq4_codebook(kvalues, skv);  // before any warp returns
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    std::uint16_t* mine = sw[threadIdx.x / 32];
    const std::uint32_t nblk = cols / 32;
    const auto* row16 = reinterpret_cast<const std::uint16_t*>(
        w + static_cast<std::size_t>(r) * nblk * 18);
    float acc[kMaxBatch];
    for (std::uint32_t t = 0; t < n; ++t) {
        acc[t] = 0.0f;
    }
    for (std::uint32_t b0 = 0; b0 < nblk; b0 += 32) {
        const std::uint32_t nb = min(32u, nblk - b0);
        for (std::uint32_t i = lane; i < nb * 9; i += 32) {
            mine[i] = __ldg(row16 + static_cast<std::size_t>(b0) * 9 + i);
        }
        __syncwarp();
        if (lane < nb) {
            const std::uint32_t b = b0 + lane;
            const std::uint16_t* blk16 = mine + lane * 9;
            const float d = __half2float(__ushort_as_half(blk16[0]));
            float vlo[16], vhi[16];
            iq4_nl_block_values(blk16, skv, vlo, vhi);
            for (std::uint32_t t = 0; t < n; ++t) {
                acc[t] += iq4_nl_block_dot(
                    d, vlo, vhi,
                    x + static_cast<std::size_t>(t) * cols +
                        static_cast<std::size_t>(b) * 32);
            }
        }
        __syncwarp();  // the next pass overwrites this warp's slice
    }
    for (std::uint32_t t = 0; t < n; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) {
            out[static_cast<std::size_t>(r) * n + t] = s;
        }
    }
}

/** Batched matvec_iq4_xs_warp_kernel: one warp per row, each lane with
 *  n token accumulators. A super-block's 8 weights per lane are looked
 *  up once as dl * kv (the per-token kernel's first product), then
 *  applied to every token in the per-token kernel's order, with the
 *  same warp sum, so the result is byte-identical to n per-token calls.
 *  Launch with 32 * kWarpRows threads per block. */
__global__ void matvec_batch_iq4_xs_kernel(const std::uint8_t* w,
                                           const std::int8_t* kvalues,
                                           const float* x, float* out,
                                           std::uint32_t rows,
                                           std::uint32_t cols,
                                           std::uint32_t n) {
    __shared__ std::int8_t skv[16];
    load_iq4_codebook(kvalues, skv);  // before any warp returns
    const std::uint32_t lane = threadIdx.x % 32;
    const std::uint32_t r = blockIdx.x * kWarpRows + threadIdx.x / 32;
    if (r >= rows) {
        return;  // a whole warp leaves together: no partial shuffles
    }
    const std::uint32_t nsb = cols / 256;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * nsb * 136;
    const std::uint32_t ib = lane / 4;
    const std::uint32_t c = 4 * (lane % 4);
    float acc[kMaxBatch];
    for (std::uint32_t t = 0; t < n; ++t) {
        acc[t] = 0.0f;
    }
    for (std::uint32_t b = 0; b < nsb; ++b) {
        float vlo[4], vhi[4];
        iq4_xs_lane_values(row + static_cast<std::size_t>(b) * 136, lane,
                           skv, vlo, vhi);
        for (std::uint32_t t = 0; t < n; ++t) {
            acc[t] = lane_dot8<16>(acc[t], vlo, vhi,
                                   x + static_cast<std::size_t>(t) * cols +
                                       static_cast<std::size_t>(b) * 256 +
                                       32 * ib + c);
        }
    }
    for (std::uint32_t t = 0; t < n; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) {
            out[static_cast<std::size_t>(r) * n + t] = s;
        }
    }
}

}  // namespace

bool cuda_detail::sync_counted() {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = cuda_ok(cudaDeviceSynchronize());
    const auto dt = std::chrono::steady_clock::now() - t0;
    GpuMetrics& gm = gpu_metrics(GpuApi::kCuda);
    gm.submits.fetch_add(1, std::memory_order_relaxed);
    gm.wait_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(dt)
                .count()),
        std::memory_order_relaxed);
    return ok;
}

namespace {

/**
 * One set of reusable device buffers for a matvec call's activations
 * and outputs, on one device. Each slot only grows, so after the first
 * calls a matvec needs no cudaMalloc / cudaFree. The slots are not
 * counted against the weight pool's budget; they come out of the VRAM
 * the budget leaves free.
 */
struct ScratchSet {
    static constexpr int kSlots = 6;  // 4 and 5: Q8_0 activations
    int device = 0;
    void* p[kSlots] = {};
    std::size_t cap[kSlots] = {};
};

std::mutex g_scratch_mu;
std::vector<ScratchSet*> g_scratch_free;

/**
 * Leases a ScratchSet for the calling thread's current device: one
 * from the free list, or a new one, returned to the list when the
 * lease ends. Concurrent calls each get their own set, so the number of
 * sets is the peak number of concurrent CUDA matvecs. Sets are never
 * freed; process exit releases the device memory, as for the weight
 * pool (no cudaFree racing context teardown).
 */
class ScratchLease {
  public:
    ScratchLease() {
        int dev = 0;
        cudaGetDevice(&dev);
        std::lock_guard<std::mutex> lk(g_scratch_mu);
        for (auto it = g_scratch_free.begin(); it != g_scratch_free.end();
             ++it) {
            if ((*it)->device == dev) {
                set_ = *it;
                g_scratch_free.erase(it);
                return;
            }
        }
        set_ = new ScratchSet;
        set_->device = dev;
    }
    ~ScratchLease() {
        std::lock_guard<std::mutex> lk(g_scratch_mu);
        g_scratch_free.push_back(set_);
    }
    ScratchLease(const ScratchLease&) = delete;
    ScratchLease& operator=(const ScratchLease&) = delete;

    /** @returns slot i's buffer, grown to at least `bytes` (at least 1),
     *  or nullptr if the device allocation failed. */
    void* get(int i, std::size_t bytes) {
        bytes = std::max<std::size_t>(bytes, 1);
        if (set_->cap[i] >= bytes) {
            return set_->p[i];
        }
        GpuMetrics& gm = gpu_metrics(GpuApi::kCuda);
        if (set_->p[i] != nullptr) {
            cudaFree(set_->p[i]);
            gm.add_bytes(MemKind::kScratch,
                         -static_cast<std::int64_t>(set_->cap[i]));
            set_->p[i] = nullptr;
            set_->cap[i] = 0;
        }
        void* p = nullptr;
        if (!cuda_ok(cudaMalloc(&p, bytes))) {
            return nullptr;
        }
        gm.add_bytes(MemKind::kScratch, static_cast<std::int64_t>(bytes));
        set_->p[i] = p;
        set_->cap[i] = bytes;
        return p;
    }

  private:
    ScratchSet* set_ = nullptr;
};

/** False once the pool is destroyed at exit, so a mapping unmapped
 *  after it (static teardown order) does not touch it (i#1164). */
std::atomic<bool> g_pool_alive{false};

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
 *
 * Lifetime: a page is keyed by (host address, size) and never compared
 * with the host bytes again, so it trusts that an address keeps its
 * bytes. A GGUF mapping's pages are dropped when it is unmapped (the
 * unmap hook, release_range(), i#1164), so a model unload and reload
 * needs nothing more. Weights in buffers of their own (not a mapped
 * file) must still drop their pages before the buffer is freed and
 * its address reused, with cuda_evict() per weight or
 * cuda_pool_reset(); else a matvec reads the old weights' device copy.
 */
class WeightPool {
 public:
    /** @returns device buffer for `host` (uploading on miss) with the
     *  page pinned, or nullptr if it cannot be cached. */
    const void* acquire(const void* host, std::size_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        bind_check_locked();
        if (budget_ == 0) {
            init_budget();
        }
        const std::uint64_t now = ++clock_;
        if (auto it = table_.find(host); it != table_.end()) {
            if (it->second.bytes == bytes) {
                wait_ready(it->second);  // finish in-flight prefetch
                it->second.last_use = now;
                it->second.pins++;
                hits_++;
                return it->second.dptr;
            }
            // Same host pointer, different size: the address was
            // recycled for another weight (weights are keyed by host
            // pointer, valid only while that buffer is live). Drop the
            // stale page and re-upload. A live pin would mean the same
            // weight is in use, which cannot have a different size.
            wait_ready(it->second);
            cudaFree(it->second.dptr);
            used_ -= it->second.bytes;
            note_used();
            table_.erase(it);
        }
        void* dptr = alloc(bytes);
        if (dptr == nullptr) {
            return nullptr;  // too big / all pinned: caller demand path
        }
        if (!cuda_ok(cudaMemcpy(dptr, host, bytes,
                                cudaMemcpyHostToDevice))) {
            cudaFree(dptr);
            used_ -= bytes;
            note_used();
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
        bind_check_locked();
        if (budget_ == 0) {
            init_budget();
        }
        const std::uint64_t now = ++clock_;
        if (auto it = table_.find(host); it != table_.end()) {
            if (it->second.bytes == bytes) {
                it->second.last_use = now;  // resident / in flight
                return;
            }
            if (it->second.pins != 0) {
                return;  // in use; let a later acquire refresh it
            }
            wait_ready(it->second);  // recycled address: drop stale
            cudaFree(it->second.dptr);
            used_ -= it->second.bytes;
            note_used();
            table_.erase(it);
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
            note_used();
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

    /** Frees `host`'s page now (Ops.evict). A pinned page stays; LRU
     *  frees it later. */
    void drop(const void* host) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = table_.find(host);
        if (it == table_.end() || it->second.pins != 0) {
            return;
        }
        wait_ready(it->second);  // don't free mid-copy
        cudaFree(it->second.dptr);
        used_ -= it->second.bytes;
        note_used();
        drops_++;
        table_.erase(it);
    }

    /**
     * Frees every page whose host address lies in [p, p + n): a mapped
     * model file is going away (sys::add_unmap_hook, i#1164). Its pages
     * would otherwise hold device memory for the rest of the process,
     * and a later mapping at a reused address would be served the old
     * model's copies (pages are keyed by host address). A page still
     * pinned means a kernel may read it, so the device is synchronized
     * first; unmapping a model while it runs is the caller's bug.
     * The hook runs on whatever thread unmaps the file, whose CUDA
     * device may never have been set (cuda_set_device is per thread):
     * with one GPU that is device 0 anyway, but once several GPUs share
     * the process (i#924) cudaFree and cudaDeviceSynchronize here need
     * the pool's device_ set first.
     */
    void release_range(const void* p, std::size_t n) {
        std::lock_guard<std::mutex> lk(mu_);
        if (table_.empty()) {
            return;
        }
        const auto lo = reinterpret_cast<std::uintptr_t>(p);
        const std::uintptr_t hi = lo + n;
        bool synced = false;
        for (auto it = table_.begin(); it != table_.end();) {
            const auto h = reinterpret_cast<std::uintptr_t>(it->first);
            if (h < lo || h >= hi) {
                ++it;
                continue;
            }
            if (it->second.pins != 0 && !synced) {
                cudaDeviceSynchronize();
                synced = true;
            }
            wait_ready(it->second);  // don't free mid-copy
            cudaFree(it->second.dptr);
            used_ -= it->second.bytes;
            drops_++;
            it = table_.erase(it);
        }
        note_used();
    }

    /** @returns The pages resident and their bytes (tests). */
    std::pair<std::size_t, std::size_t> resident() {
        std::lock_guard<std::mutex> lk(mu_);
        return {table_.size(), used_};
    }

    /** @returns The byte budget (Ops.pool_budget). */
    std::size_t budget() {
        std::lock_guard<std::mutex> lk(mu_);
        bind_check_locked();
        if (budget_ == 0) {
            init_budget();
        }
        return budget_;
    }

    /** Frees every resident page and empties the table. For tests:
     *  the pool keys weights by host pointer (valid only while that
     *  buffer is live), so transient test weights whose addresses get
     *  recycled across unrelated cases would otherwise alias. Not used
     *  in inference, where weight pointers are stable for the run. */
    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& [host, pg] : table_) {
            wait_ready(pg);
            cudaFree(pg.dptr);
        }
        table_.clear();
        used_ = 0;
        note_used();
        device_ = -1;  // next access re-binds (i#24): a reset may precede
                       // a rebind to a different device; a stale device_
                       // would make bind_check_locked throw spuriously.
    }

    ~WeightPool() {
        g_pool_alive.store(false);
        if (std::getenv("LOCUS_MOE_STATS") == nullptr) {
            return;
        }
        const std::uint64_t tot = hits_ + misses_;
        std::fprintf(
            stderr,
            "[cuda-pool] budget=%zuMB resident=%zuMB hits=%llu "
            "misses=%llu hit_rate=%.1f%% prefetches=%llu "
            "evictions=%llu drops=%llu uploaded=%lluMB\n",
            budget_ >> 20, used_ >> 20,
            static_cast<unsigned long long>(hits_),
            static_cast<unsigned long long>(misses_),
            tot ? 100.0 * static_cast<double>(hits_) /
                      static_cast<double>(tot)
                : 0.0,
            static_cast<unsigned long long>(prefetches_),
            static_cast<unsigned long long>(evictions_),
            static_cast<unsigned long long>(drops_),
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
        note_used();
        return p;
    }

    /**
     * Records the device the pool first bound to (its first access) and
     * throws if a later access runs on a DIFFERENT current device -- a
     * device misbinding, e.g. cudaSetDevice ran too late or on the wrong
     * thread. It THROWS rather than assert()s on purpose: a misbinding is
     * a Release-only configuration failure, and Release is -DNDEBUG so an
     * assert would compile out exactly where this matters (i#24 inc 4).
     * ensure_device() at each matvec entry keeps the current device
     * consistent, so in a correctly-started process this never fires.
     * Caller holds mu_.
     */
    void bind_check_locked() {
        int cur = -1;
        if (!cuda_ok(cudaGetDevice(&cur))) {
            return;  // cannot query the device; nothing to check
        }
        if (device_ < 0) {
            device_ = cur;
        } else if (cur != device_) {
            throw std::runtime_error(
                "CUDA weight pool bound to device " +
                std::to_string(device_) +
                " but a kernel is running on device " +
                std::to_string(cur) +
                "; cudaSetDevice must precede the first weight upload "
                "(i#24 device-select)");
        }
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
        note_used();
        evictions_++;
        table_.erase(victim);
        return true;
    }

    /** Mirrors used_ into the weights gauge (the only weight pool). */
    void note_used() {
        gpu_metrics(GpuApi::kCuda)
            .bytes[static_cast<int>(MemKind::kWeights)]
            .store(static_cast<std::int64_t>(used_),
                   std::memory_order_relaxed);
    }

    std::unordered_map<const void*, Page> table_;
    cudaStream_t stream_ = nullptr;
    int device_ = -1;  // device the pool bound to (i#24, bind_check_locked)
    std::size_t budget_ = 0;
    std::size_t used_ = 0;
    std::uint64_t clock_ = 0;
    std::uint64_t hits_ = 0, misses_ = 0, evictions_ = 0, drops_ = 0;
    std::uint64_t prefetches_ = 0;
    std::uint64_t upload_bytes_ = 0;
    std::mutex mu_;
};

/** sys unmap hook: a model file going away drops its device pages. */
void on_unmap(const void* p, std::size_t n);

WeightPool& pool() {
    static WeightPool p;
    static const bool hooked = [] {
        g_pool_alive.store(true);
        sys::add_unmap_hook(&on_unmap);
        return true;
    }();
    (void)hooked;
    return p;
}

void on_unmap(const void* p, std::size_t n) {
    if (g_pool_alive.load()) {
        pool().release_range(p, n);
    }
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
        gpu_metrics(GpuApi::kCuda)
            .add_bytes(MemKind::kTable, sizeof(iq1s_grid));
        return static_cast<const std::uint64_t*>(p);
    }();
    return d;
}

/** Uploads a host lookup table to the device once; nullptr on
 *  failure (callers fall back to scalar). */
const void* upload_table(const void* host, std::size_t bytes) {
    void* p = nullptr;
    if (!cuda_ok(cudaMalloc(&p, bytes))) {
        return nullptr;
    }
    if (!cuda_ok(cudaMemcpy(p, host, bytes,
                            cudaMemcpyHostToDevice))) {
        cudaFree(p);
        return nullptr;
    }
    gpu_metrics(GpuApi::kCuda)
        .add_bytes(MemKind::kTable, static_cast<std::int64_t>(bytes));
    return p;
}

const std::uint64_t* device_iq2xxs_grid() {
    static const auto* d = static_cast<const std::uint64_t*>(
        upload_table(iq2xxs_grid, sizeof(iq2xxs_grid)));
    return d;
}

const std::uint32_t* device_iq3xxs_grid() {
    static const auto* d = static_cast<const std::uint32_t*>(
        upload_table(iq3xxs_grid, sizeof(iq3xxs_grid)));
    return d;
}

const std::uint64_t* device_iq2xs_grid() {
    static const auto* d = static_cast<const std::uint64_t*>(
        upload_table(iq2xs_grid, sizeof(iq2xs_grid)));
    return d;
}

const std::uint64_t* device_iq2s_grid() {
    static const auto* d = static_cast<const std::uint64_t*>(
        upload_table(iq2s_grid, sizeof(iq2s_grid)));
    return d;
}

const std::uint32_t* device_iq3s_grid() {
    static const auto* d = static_cast<const std::uint32_t*>(
        upload_table(iq3s_grid, sizeof(iq3s_grid)));
    return d;
}

/** Sign lookup shared by IQ2_XXS and IQ3_XXS. */
const std::uint8_t* device_ksigns() {
    static const auto* d = static_cast<const std::uint8_t*>(
        upload_table(ksigns_iq2xs, sizeof(ksigns_iq2xs)));
    return d;
}

/** IQ4_NL 16-entry codebook used by IQ4_XS. */
const std::int8_t* device_kvalues_iq4nl() {
    static const auto* d = static_cast<const std::int8_t*>(
        upload_table(kvalues_iq4nl, sizeof(kvalues_iq4nl)));
    return d;
}

enum class Kind {
    kF32,
    kQ8_0,
    kQ4_K,
    kQ5_K,
    kQ6_K,
    kIQ1_S,
    kQ2_K,
    kIQ2_XXS,
    kIQ2_XS,
    kIQ2_S,
    kIQ3_XXS,
    kIQ3_S,
    kIQ4_XS,
    kIQ4_NL,
    kIQ1_M,
    kTQ1_0,
    kTQ2_0
};

/** Device lookup tables threaded to the grid/codebook kernels; unused
 *  fields stay null for types that need none. */
struct CudaTables {
    const std::uint64_t* g64 = nullptr;    // iq1s / iq2xxs grid
    const std::uint32_t* g32 = nullptr;    // iq3xxs grid
    const std::uint8_t* ksigns = nullptr;  // iq2xxs / iq3xxs signs
    const std::int8_t* kvalues = nullptr;  // iq4_xs codebook
};

/** @returns Whether Q8_0 matvecs and prefill tiles run on Q8_0
 *  activations (i#1254): LOCUS_CUDA_Q8_INT8 not "0". A numeric mode:
 *  the disk prefix key holds it. */
bool q8_int8_on() {
    static const bool on = [] {
        const char* e = std::getenv("LOCUS_CUDA_Q8_INT8");
        return e == nullptr || std::string_view(e) != "0";
    }();
    return on;
}

/**
 * Queues the Q8_0 activation quantizer on count floats at dx (count a
 * multiple of 32) into slots 4 (scales) and 5 (quants) of s.
 * @returns false, queuing nothing, when a buffer cannot be allocated.
 */
bool quantize_act(ScratchLease& s, const float* dx, std::size_t count,
                  const float*& xd, const std::int8_t*& xq) {
    const std::size_t nblk = count / 32;
    auto* d = static_cast<float*>(s.get(4, nblk * sizeof(float)));
    auto* q = static_cast<std::int8_t*>(s.get(5, count));
    if (d == nullptr || q == nullptr) {
        return false;
    }
    quantize_q8_0_act_kernel<<<(nblk + 7) / 8, 256>>>(
        dx, d, q, static_cast<std::uint32_t>(nblk));
    xd = d;
    xq = q;
    return true;
}

void launch_kernel(Kind kind, const void* dw,
                   const CudaTables& t, const float* dxf,
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
        case Kind::kQ8_0: {
            const float* xd = nullptr;
            const std::int8_t* xq = nullptr;
            ScratchLease s;  // the default stream orders its reuse
            if (q8_int8_on() && quantize_act(s, dxf, cols, xd, xq)) {
                matvec_q8_0_q8x_kernel<<<(rows + kWarpRows - 1) /
                                             kWarpRows,
                                         32 * kWarpRows>>>(
                    dwb, xd, xq, doutf, rows, cols);
                break;
            }
            matvec_q8_0_warp_kernel<<<
                (rows + 2 * kWarpRows - 1) / (2 * kWarpRows),
                32 * kWarpRows>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        }
        case Kind::kQ4_K:
            matvec_q4_k_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kQ5_K:
            matvec_q5_k_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kQ6_K:
            matvec_q6_k_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ1_S:
            matvec_iq1_s_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, t.g64, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ1_M:
            matvec_iq1_m_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, t.g64, dxf, doutf, rows, cols);
            break;
        case Kind::kQ2_K:
            matvec_q2_k_kernel<<<blocks, threads>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ2_XXS:
            matvec_iq2_xxs_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, t.g64, t.ksigns, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ2_XS:
            matvec_iq2_xs_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, t.g64, t.ksigns, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ2_S:
            matvec_iq2_s_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, t.g64, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ3_S:
            matvec_iq3_s_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, t.g32, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ3_XXS:
            matvec_iq3_xxs_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, t.g32, t.ksigns, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ4_XS:
            matvec_iq4_xs_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, t.kvalues, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ4_NL:
            matvec_iq4_nl_warp_kernel<<<
                (rows + kWarpRows - 1) / kWarpRows, 32 * kWarpRows>>>(
                dwb, t.kvalues, dxf, doutf, rows, cols);
            break;
        case Kind::kTQ1_0:
            matvec_tq1_0_kernel<<<blocks, threads>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kTQ2_0:
            matvec_tq2_0_kernel<<<blocks, threads>>>(
                dwb, dxf, doutf, rows, cols);
            break;
    }
}

/**
 * A weight's device copy for one call: the resident pool's, pinned
 * until the lease ends, or an uncached per-call upload when the pool
 * cannot hold it (too big for the budget, or every page pinned).
 */
class WeightLease {
  public:
    WeightLease(const void* host, std::size_t bytes) : host_(host) {
        dev_ = pool().acquire(host, bytes);
        pooled_ = dev_ != nullptr;
        if (!pooled_ && demand_.alloc(bytes) &&
            cuda_ok(cudaMemcpy(demand_.p, host, bytes,
                               cudaMemcpyHostToDevice))) {
            dev_ = demand_.p;
        }
    }
    ~WeightLease() {
        if (pooled_) {
            pool().release(host_);
        }
    }
    WeightLease(const WeightLease&) = delete;
    WeightLease& operator=(const WeightLease&) = delete;

    /** @returns The device copy, or nullptr when the upload failed. */
    const void* get() const { return dev_; }

  private:
    const void* host_;
    const void* dev_ = nullptr;
    bool pooled_ = false;
    DevBuf demand_;
};

bool run_matvec(const Mat& w, std::span<const float> x,
                std::span<float> out, std::size_t w_bytes,
                Kind kind, const CudaTables& t = {}) {
    const WeightLease weight(w.data, w_bytes);
    const void* dw = weight.get();
    if (dw == nullptr) {
        return false;
    }

    bool ok = false;
    {
        ScratchLease scratch;  // x / out buffers, reused across calls
        const std::size_t x_bytes = x.size() * sizeof(float);
        const std::size_t out_bytes =
            static_cast<std::size_t>(w.rows) * sizeof(float);
        void* dx = scratch.get(0, x_bytes);
        void* dout = scratch.get(1, out_bytes);
        if (dx != nullptr && dout != nullptr &&
            cuda_ok(cudaMemcpy(dx, x.data(), x_bytes,
                               cudaMemcpyHostToDevice))) {
            launch_kernel(kind, dw, t, static_cast<const float*>(dx),
                          static_cast<float*>(dout), w.rows, w.cols);
            ok = cuda_ok(cudaGetLastError()) &&
                 sync_counted() &&
                 cuda_ok(cudaMemcpy(out.data(), dout, out_bytes,
                                    cudaMemcpyDeviceToHost));
        }
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
        case gguf::TensorType::kQ5_K:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 176ull;
        case gguf::TensorType::kQ6_K:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 210ull;
        case gguf::TensorType::kIQ1_S:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 50ull;
        case gguf::TensorType::kQ2_K:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 84ull;
        case gguf::TensorType::kIQ2_XXS:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 66ull;
        case gguf::TensorType::kIQ2_XS:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 74ull;
        case gguf::TensorType::kIQ2_S:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 82ull;
        case gguf::TensorType::kIQ3_S:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 110ull;
        case gguf::TensorType::kIQ3_XXS:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 98ull;
        case gguf::TensorType::kIQ4_XS:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 136ull;
        case gguf::TensorType::kIQ4_NL:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 32ull) * 18ull;
        case gguf::TensorType::kIQ1_M:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 56ull;
        case gguf::TensorType::kTQ1_0:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 54ull;
        case gguf::TensorType::kTQ2_0:
            return static_cast<std::size_t>(w.rows) *
                   (w.cols / 256ull) * 66ull;
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
        case gguf::TensorType::kQ5_K:
            return Kind::kQ5_K;
        case gguf::TensorType::kQ6_K:
            return Kind::kQ6_K;
        case gguf::TensorType::kIQ1_S:
            return Kind::kIQ1_S;
        case gguf::TensorType::kQ2_K:
            return Kind::kQ2_K;
        case gguf::TensorType::kIQ2_XXS:
            return Kind::kIQ2_XXS;
        case gguf::TensorType::kIQ2_XS:
            return Kind::kIQ2_XS;
        case gguf::TensorType::kIQ2_S:
            return Kind::kIQ2_S;
        case gguf::TensorType::kIQ3_S:
            return Kind::kIQ3_S;
        case gguf::TensorType::kIQ3_XXS:
            return Kind::kIQ3_XXS;
        case gguf::TensorType::kIQ4_XS:
            return Kind::kIQ4_XS;
        case gguf::TensorType::kIQ4_NL:
            return Kind::kIQ4_NL;
        case gguf::TensorType::kIQ1_M:
            return Kind::kIQ1_M;
        case gguf::TensorType::kTQ1_0:
            return Kind::kTQ1_0;
        case gguf::TensorType::kTQ2_0:
            return Kind::kTQ2_0;
        default:
            return Kind::kF32;
    }
}

/** @returns Whether `type` has a batched kernel (launch_tile or
 *  launch_batch). */
bool has_batch_kernel(gguf::TensorType type) {
    switch (type) {
        case gguf::TensorType::kQ4_K:
        case gguf::TensorType::kQ5_K:
        case gguf::TensorType::kQ6_K:
        case gguf::TensorType::kQ8_0:
        case gguf::TensorType::kIQ4_NL:
        case gguf::TensorType::kIQ4_XS:
            return true;
        default:
            return false;
    }
}

/**
 * Launches the batched kernel of `type` (has_batch_kernel, but not
 * has_tile_kernel: those run launch_tile) on the default stream.
 * @param kvalues The device IQ4 codebook; read by IQ4_NL and IQ4_XS.
 * @param x n <= kMaxBatch tokens, token-major (token t at t * cols).
 * @param out rows * n floats, row-major (out[r * n + t]).
 * @returns false, launching nothing, for a type with no batched kernel.
 */
bool launch_batch(gguf::TensorType type, const std::uint8_t* w,
                  const std::int8_t* kvalues, const float* x, float* out,
                  std::uint32_t rows, std::uint32_t cols, std::uint32_t n) {
    const std::uint32_t blocks = (rows + kWarpRows - 1) / kWarpRows;
    switch (type) {
        case gguf::TensorType::kQ5_K:
            matvec_batch_q5_k_kernel<<<blocks, 32 * kWarpRows>>>(
                w, x, out, rows, cols, n);
            return true;
        case gguf::TensorType::kIQ4_NL:
            matvec_batch_iq4_nl_kernel<<<blocks, 32 * kWarpRows>>>(
                w, kvalues, x, out, rows, cols, n);
            return true;
        case gguf::TensorType::kIQ4_XS:
            matvec_batch_iq4_xs_kernel<<<blocks, 32 * kWarpRows>>>(
                w, kvalues, x, out, rows, cols, n);
            return true;
        default:
            return false;  // has_batch_kernel and this switch disagree
    }
}

/** @returns true if `type` has a prefill tile kernel (i#1195). */
bool has_tile_kernel(gguf::TensorType type) {
    return type == gguf::TensorType::kQ4_K ||
           type == gguf::TensorType::kQ6_K || type == gguf::TensorType::kQ8_0;
}

/** A K-quant weight (TileQ4K, TileQ6K) dequantized to f32, row-major:
 *  one warp per (row, super-block), each lane writing the 8 values the
 *  per-token kernel decodes for it. Launch with 128 threads a block. */
template <class Q>
__global__ void dequant_ksb_kernel(const std::uint8_t* w, float* dst,
                                   std::uint32_t rows, std::uint32_t nsb) {
    const std::uint32_t lane = threadIdx.x % 32;
    const std::size_t idx =
        std::size_t{blockIdx.x} * (blockDim.x / 32) + threadIdx.x / 32;
    if (idx >= std::size_t{rows} * nsb) {
        return;
    }
    float lo[4], hi[4];
    const std::uint32_t e = Q::values(w + idx * Q::kBytes, lane, lo, hi);
    float* d = dst + idx * 256;  // row r's super-block b: (r * nsb + b)
    for (int k = 0; k < 4; ++k) {
        d[e + k] = lo[k];
        d[e + Q::kHi + k] = hi[k];
    }
}

/** A Q8_0 weight dequantized to f32 (d * q), one thread per value. */
__global__ void dequant_q8_0_kernel(const std::uint8_t* w, float* dst,
                                    std::size_t n) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const std::uint8_t* blk = w + (i / 32) * 34;
    std::uint16_t dh;
    memcpy(&dh, blk, 2);
    const float d = __half2float(__ushort_as_half(dh));
    dst[i] = d * static_cast<float>(static_cast<std::int8_t>(blk[2 + i % 32]));
}

}  // namespace

bool cuda_prefill_gemm_on() {
#ifdef LOCUS_HAS_CUBLAS
    static const bool on = [] {
        const char* e = std::getenv("LOCUS_CUDA_GEMM");
        return e == nullptr || std::string(e) != "0";
    }();
    return on;
#else
    return false;
#endif
}

bool cuda_q8_int8_on() { return q8_int8_on(); }

bool cuda_quantize_q8_0_act(std::span<const float> x, float* d,
                            std::int8_t* q) {
    if (!cuda_backend_usable()) {
        return false;
    }
    ensure_device();
    ScratchLease s;
    const std::size_t bytes = x.size() * sizeof(float);
    void* dx = s.get(0, bytes);
    const float* xd = nullptr;
    const std::int8_t* xq = nullptr;
    return dx != nullptr &&
           cuda_ok(cudaMemcpy(dx, x.data(), bytes,
                              cudaMemcpyHostToDevice)) &&
           quantize_act(s, static_cast<const float*>(dx), x.size(), xd,
                        xq) &&
           cuda_ok(cudaGetLastError()) &&
           cuda_ok(cudaMemcpy(d, xd, x.size() / 32 * sizeof(float),
                              cudaMemcpyDeviceToHost)) &&
           cuda_ok(cudaMemcpy(q, xq, x.size(), cudaMemcpyDeviceToHost));
}

namespace {

/** The fewest tokens a prefill matmul runs as a GEMM (i#1200), as
 *  Metal's kPrefillMin: below it the tile keeps the per-token bits. */
constexpr std::uint32_t kGemmMinTokens = 32;
/** The largest dequantized weight the GEMM stages (bytes): a bigger one
 *  (a 128k-vocabulary head is about 1 GB) keeps the tile. */
constexpr std::size_t kGemmMaxBytes = std::size_t{256} << 20;

/**
 * The prefill matmul of n >= kGemmMinTokens tokens as dequantize plus
 * cuBLAS sgemm (i#1200): llama.cpp's path on this GPU, far faster than
 * the tile, with its own sum order, so not the per-token kernel's bits.
 * LOCUS_CUDA_GEMM=0 turns it off (a numeric mode: the disk prefix key
 * holds it). Same layouts as launch_tile.
 * @returns false when the GEMM does not apply (launching nothing), or
 *     when cublasSgemm fails after the dequantize ran; either way the
 *     caller's tile kernel then writes all of out.
 */
bool launch_gemm(gguf::TensorType type, const std::uint8_t* w,
                 const float* x, float* out, std::uint32_t rows,
                 std::uint32_t cols, std::uint32_t n) {
#ifdef LOCUS_HAS_CUBLAS
    const std::size_t bytes = std::size_t{rows} * cols * sizeof(float);
    if (!cuda_prefill_gemm_on() || n < kGemmMinTokens ||
        bytes > kGemmMaxBytes ||
        !has_tile_kernel(type)) {
        return false;
    }
    // One cuBLAS handle per device (cuBLAS takes one handle from several
    // threads, not across devices), made once under a lock.
    int dev = 0;
    cudaGetDevice(&dev);
    cublasHandle_t handle = nullptr;
    {
        static std::mutex mu;
        static std::unordered_map<int, cublasHandle_t> handles;
        std::lock_guard<std::mutex> lk(mu);
        auto [it, fresh] = handles.try_emplace(dev, nullptr);
        if (fresh && cublasCreate(&it->second) != CUBLAS_STATUS_SUCCESS) {
            it->second = nullptr;
        }
        handle = it->second;
    }
    if (handle == nullptr) {
        return false;
    }
    // The dequantized weight in this call's own scratch set (i#1200): a
    // concurrent call, or one on another device, leases another, so no
    // thread dequantizes into the buffer another's GEMM is about to
    // read. Held across both launches; a later user of the set queues
    // behind them on the device's default stream.
    ScratchLease lease;
    auto* deq = static_cast<float*>(lease.get(0, bytes));
    if (deq == nullptr) {
        return false;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t sbs = std::size_t{rows} * nsb;
    switch (type) {
        case gguf::TensorType::kQ4_K:
            dequant_ksb_kernel<TileQ4K>
                <<<static_cast<unsigned>((sbs + 3) / 4), 128>>>(w, deq, rows,
                                                                nsb);
            break;
        case gguf::TensorType::kQ6_K:
            dequant_ksb_kernel<TileQ6K>
                <<<static_cast<unsigned>((sbs + 3) / 4), 128>>>(w, deq, rows,
                                                                nsb);
            break;
        default: {  // kQ8_0
            const std::size_t vals = std::size_t{rows} * cols;
            dequant_q8_0_kernel<<<static_cast<unsigned>((vals + 255) / 256),
                                  256>>>(w, deq, vals);
            break;
        }
    }
    // Row-major W (rows x cols) is column-major W^T, and token-major x
    // (n x cols) column-major x^T: out^T = W x^T, rows x n column-major,
    // is the token-major out.
    const float one = 1.0f, zero = 0.0f;
    return cuda_ok(cudaGetLastError()) &&
           cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                       static_cast<int>(rows), static_cast<int>(n),
                       static_cast<int>(cols), &one, deq,
                       static_cast<int>(cols), x, static_cast<int>(cols),
                       &zero, out, static_cast<int>(rows)) ==
               CUBLAS_STATUS_SUCCESS;
#else
    (void)type, (void)w, (void)x, (void)out, (void)rows, (void)cols, (void)n;
    return false;
#endif
}

/**
 * Launches the prefill tile kernel of `type` (has_tile_kernel) on the
 * default stream: all n tokens in one launch. A chunk of at least
 * kGemmMinTokens tokens goes to launch_gemm instead, where it applies.
 * @param x n tokens, token-major (token t at t * cols).
 * @param out n * rows floats, token-major (out[t * rows + r]).
 * @returns false, launching nothing, for a type with no tile kernel.
 */
bool launch_tile(gguf::TensorType type, const std::uint8_t* w,
                 const float* x, float* out, std::uint32_t rows,
                 std::uint32_t cols, std::uint32_t n) {
    if (launch_gemm(type, w, x, out, rows, cols, n)) {
        return true;
    }
    // 4 rows by 8 tokens a warp was the fastest of 2 to 8 rows by 4 to
    // 16 tokens on a GTX 750 Ti (Llama 3.2 1B Q4_K_M, 511 tokens: 116.6
    // tok/s; 4 x 16 94.7, 8 x 8 104.2).
    constexpr std::uint32_t R = 4, T = 8;
    const dim3 grid((rows + kWarpRows * R - 1) / (kWarpRows * R),
                    (n + T - 1) / T);
    // Q4_K and Q6_K stage each super-block's activations in shared
    // memory, 4 warps a block (i#1200): 125.8 to 159.1 tok/s on the same
    // prefill; 8 warps a block gave 148.2.
    switch (type) {
        case gguf::TensorType::kQ4_K:
            matvec_tile_smem_kernel<R, T, kWarpRows, TileQ4K>
                <<<grid, 32 * kWarpRows>>>(w, x, out, rows, cols, n);
            return true;
        case gguf::TensorType::kQ6_K:
            matvec_tile_smem_kernel<R, T, kWarpRows, TileQ6K>
                <<<grid, 32 * kWarpRows>>>(w, x, out, rows, cols, n);
            return true;
        case gguf::TensorType::kQ8_0: {
            const float* xd = nullptr;
            const std::int8_t* xq = nullptr;
            ScratchLease s;  // the default stream orders its reuse
            if (q8_int8_on() &&
                quantize_act(s, x, std::size_t{n} * cols, xd, xq)) {
                // 2 rows by 8 tokens a warp: of 1 to 8 rows by 4 to 16
                // tokens on the GTX 750 Ti, the fastest (Llama 3.2 1B
                // Q8_0, 15 tokens: 113.1 tok/s; 4 x 8 81.2, 4 x 4 85.3,
                // 2 x 4 86.2).
                constexpr std::uint32_t Rq = 2, Tq = 8;
                const dim3 gq((rows + kWarpRows * Rq - 1) / (kWarpRows * Rq),
                              (n + Tq - 1) / Tq);
                matvec_tile_q8_0_q8x_kernel<Rq, Tq>
                    <<<gq, 32 * kWarpRows>>>(w, xd, xq, out, rows, cols, n);
                return true;
            }
            matvec_tile_q8_0_kernel<R, T><<<grid, 32 * kWarpRows>>>(
                w, x, out, rows, cols, n);
            return true;
        }
        default:
            return false;
    }
}

/** @returns The device codebook `type`'s batched kernel reads, nullptr
 *  for a type that reads none; sets ok false when its upload failed. */
const std::int8_t* batch_tables(gguf::TensorType type, bool& ok) {
    ok = true;
    if (type != gguf::TensorType::kIQ4_NL &&
        type != gguf::TensorType::kIQ4_XS) {
        return nullptr;
    }
    const std::int8_t* kv = device_kvalues_iq4nl();
    ok = kv != nullptr;
    return kv;
}

/** One-launch batched matvec (has_batch_kernel types): weight pulled
 *  from the pool (shared across the n tokens), x_batch + out_batch
 *  staged once. */
bool run_batch(const Mat& w, std::span<const float> x_batch,
               std::span<float> out_batch, std::size_t w_bytes,
               std::uint32_t n) {
    bool tables_ok = false;
    const std::int8_t* kv = batch_tables(w.type, tables_ok);
    if (!tables_ok) {
        return false;
    }
    const WeightLease weight(w.data, w_bytes);
    const void* dw = weight.get();
    if (dw == nullptr) {
        return false;
    }
    bool ok = false;
    {
        ScratchLease scratch;
        const std::size_t x_bytes = x_batch.size() * sizeof(float);
        const std::size_t out_bytes = out_batch.size() * sizeof(float);
        void* dx = scratch.get(0, x_bytes);
        void* dout = scratch.get(1, out_bytes);
        if (dx != nullptr && dout != nullptr &&
            cuda_ok(cudaMemcpy(dx, x_batch.data(), x_bytes,
                               cudaMemcpyHostToDevice))) {
            ok = launch_batch(w.type, static_cast<const std::uint8_t*>(dw),
                              kv, static_cast<const float*>(dx),
                              static_cast<float*>(dout), w.rows, w.cols,
                              n) &&
                 cuda_ok(cudaGetLastError()) &&
                 sync_counted() &&
                 cuda_ok(cudaMemcpy(out_batch.data(), dout, out_bytes,
                                    cudaMemcpyDeviceToHost));
        }
    }
    return ok;
}

/**
 * The prefill tile (has_tile_kernel types, i#1195) for any n, from the
 * host: one launch for all n tokens, its token-major result transposed
 * into out_batch's rows. The same bits as n per-token calls.
 * @param out_batch rows * n floats, row-major (out[r * n + t]).
 */
bool run_tile(const Mat& w, std::span<const float> x_batch,
              std::span<float> out_batch, std::size_t w_bytes,
              std::uint32_t n) {
    const WeightLease weight(w.data, w_bytes);
    const void* dw = weight.get();
    if (dw == nullptr) {
        return false;
    }
    std::vector<float> tok_major(out_batch.size());
    bool ok = false;
    {
        ScratchLease scratch;
        const std::size_t x_bytes = x_batch.size() * sizeof(float);
        const std::size_t out_bytes = out_batch.size() * sizeof(float);
        void* dx = scratch.get(0, x_bytes);
        void* dout = scratch.get(1, out_bytes);
        ok = dx != nullptr && dout != nullptr &&
             cuda_ok(cudaMemcpy(dx, x_batch.data(), x_bytes,
                                cudaMemcpyHostToDevice)) &&
             launch_tile(w.type, static_cast<const std::uint8_t*>(dw),
                         static_cast<const float*>(dx),
                         static_cast<float*>(dout), w.rows, w.cols, n) &&
             cuda_ok(cudaGetLastError()) && sync_counted() &&
             cuda_ok(cudaMemcpy(tok_major.data(), dout, out_bytes,
                                cudaMemcpyDeviceToHost));
    }
    if (!ok) {
        return false;
    }
    for (std::uint32_t t = 0; t < n; ++t) {
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out_batch[static_cast<std::size_t>(r) * n + t] =
                tok_major[static_cast<std::size_t>(t) * w.rows + r];
        }
    }
    return true;
}

/**
 * run_batch for any n: slices of at most kMaxBatch tokens, each slice's
 * row-major result copied into its columns of out_batch. Every slice
 * runs the same kernel, so the result stays byte-identical to n
 * per-token calls.
 * @param out_batch rows * n floats, row-major (out[r * n + t]).
 * @returns false on the first slice that fails (out_batch then holds a
 *          partial result).
 */
bool run_batch_sliced(const Mat& w, std::span<const float> x_batch,
                      std::span<float> out_batch, std::size_t w_bytes,
                      std::uint32_t n) {
    if (has_tile_kernel(w.type)) {
        return run_tile(w, x_batch, out_batch, w_bytes, n);
    }
    if (n <= kMaxBatch) {
        return run_batch(w, x_batch, out_batch, w_bytes, n);
    }
    const std::size_t cols = w.cols;
    std::vector<float> part(static_cast<std::size_t>(w.rows) * kMaxBatch);
    for (std::uint32_t t0 = 0; t0 < n; t0 += kMaxBatch) {
        const std::uint32_t m = std::min(kMaxBatch, n - t0);
        const std::span<float> ps(part.data(),
                                  static_cast<std::size_t>(w.rows) * m);
        if (!run_batch(w, x_batch.subspan(t0 * cols, m * cols), ps, w_bytes,
                       m)) {
            return false;
        }
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            std::copy_n(part.data() + static_cast<std::size_t>(r) * m, m,
                        out_batch.data() + static_cast<std::size_t>(r) * n +
                            t0);
        }
    }
    return true;
}

/** @returns true if a Q8_K-activation CUDA kernel exists for `type`. */
bool cuda_has_q8k_kernel(gguf::TensorType type) {
    switch (type) {
        case gguf::TensorType::kQ2_K:
        case gguf::TensorType::kQ3_K:
        case gguf::TensorType::kQ4_K:
        case gguf::TensorType::kQ5_K:
        case gguf::TensorType::kQ6_K:
        case gguf::TensorType::kIQ2_XXS:
        case gguf::TensorType::kIQ2_XS:
        case gguf::TensorType::kIQ2_S:
        case gguf::TensorType::kTQ1_0:
        case gguf::TensorType::kTQ2_0:
        case gguf::TensorType::kIQ4_XS:
        case gguf::TensorType::kIQ3_XXS:
        case gguf::TensorType::kIQ3_S:
        case gguf::TensorType::kIQ1_S:
        case gguf::TensorType::kIQ1_M:
            return true;
        default:
            return false;
    }
}

/** Q8_K-activation device matvec: host-quantizes x (the same quant
 *  matvec_q8k uses), uploads the flat quants and the pooled/uploaded
 *  weight, and launches the per-type integer-dot kernel. Returns false
 *  on any CUDA failure so the caller can fall back to scalar. Only call
 *  for a type cuda_has_q8k_kernel() accepts. */
bool run_matvec_q8k(const Mat& w, std::span<const float> x,
                    std::span<float> out, std::size_t w_bytes,
                    const CudaTables& t) {
    const WeightLease weight(w.data, w_bytes);
    const void* dw = weight.get();
    if (dw == nullptr) {
        return false;
    }
    const std::size_t nsb = x.size() / 256;
    std::vector<float> hd(nsb);
    std::vector<std::int8_t> hqs(x.size());
    std::vector<std::int16_t> hbs(nsb * 16);
    quantize_activation_q8k(x, hd.data(), hqs.data(), hbs.data());

    bool ok = false;
    {
        ScratchLease scratch;
        const std::size_t out_bytes =
            static_cast<std::size_t>(w.rows) * sizeof(float);
        void* dd = scratch.get(0, nsb * sizeof(float));
        void* dqs = scratch.get(1, x.size() * sizeof(std::int8_t));
        void* dbs = scratch.get(2, nsb * 16 * sizeof(std::int16_t));
        void* dout = scratch.get(3, out_bytes);
        if (dd != nullptr && dqs != nullptr && dbs != nullptr &&
            dout != nullptr &&
            cuda_ok(cudaMemcpy(dd, hd.data(), nsb * sizeof(float),
                               cudaMemcpyHostToDevice)) &&
            cuda_ok(cudaMemcpy(dqs, hqs.data(), x.size(),
                               cudaMemcpyHostToDevice)) &&
            cuda_ok(cudaMemcpy(dbs, hbs.data(),
                               nsb * 16 * sizeof(std::int16_t),
                               cudaMemcpyHostToDevice))) {
            const std::uint32_t threads = 256;
            const std::uint32_t blocks =
                (w.rows + threads - 1) / threads;
            const auto* dwb = static_cast<const std::uint8_t*>(dw);
            const auto* ddf = static_cast<const float*>(dd);
            const auto* dqsb = static_cast<const std::int8_t*>(dqs);
            const auto* dbsb = static_cast<const std::int16_t*>(dbs);
            auto* doutf = static_cast<float*>(dout);
            switch (w.type) {
                case gguf::TensorType::kQ4_K:
                    matvec_q4_k_q8k_kernel<<<blocks, threads>>>(
                        dwb, ddf, dqsb, dbsb, doutf, w.rows, w.cols);
                    break;
                case gguf::TensorType::kQ5_K:
                    matvec_q5_k_q8k_kernel<<<blocks, threads>>>(
                        dwb, ddf, dqsb, dbsb, doutf, w.rows, w.cols);
                    break;
                case gguf::TensorType::kQ6_K:
                    matvec_q6_k_q8k_kernel<<<blocks, threads>>>(
                        dwb, ddf, dqsb, doutf, w.rows, w.cols);
                    break;
                case gguf::TensorType::kQ2_K:
                    matvec_q2_k_q8k_kernel<<<blocks, threads>>>(
                        dwb, ddf, dqsb, dbsb, doutf, w.rows, w.cols);
                    break;
                case gguf::TensorType::kQ3_K:
                    matvec_q3_k_q8k_kernel<<<blocks, threads>>>(
                        dwb, ddf, dqsb, doutf, w.rows, w.cols);
                    break;
                case gguf::TensorType::kIQ2_XXS:
                    matvec_iq2_xxs_q8k_kernel<<<blocks, threads>>>(
                        dwb, t.g64, t.ksigns, ddf, dqsb, doutf, w.rows,
                        w.cols);
                    break;
                case gguf::TensorType::kIQ2_XS:
                    matvec_iq2_xs_q8k_kernel<<<blocks, threads>>>(
                        dwb, t.g64, t.ksigns, ddf, dqsb, doutf, w.rows,
                        w.cols);
                    break;
                case gguf::TensorType::kIQ2_S:
                    matvec_iq2_s_q8k_kernel<<<blocks, threads>>>(
                        dwb, t.g64, ddf, dqsb, doutf, w.rows, w.cols);
                    break;
                case gguf::TensorType::kTQ1_0:
                    matvec_tq1_0_q8k_kernel<<<blocks, threads>>>(
                        dwb, ddf, dqsb, doutf, w.rows, w.cols);
                    break;
                case gguf::TensorType::kTQ2_0:
                    matvec_tq2_0_q8k_kernel<<<blocks, threads>>>(
                        dwb, ddf, dqsb, doutf, w.rows, w.cols);
                    break;
                case gguf::TensorType::kIQ4_XS:
                    matvec_iq4_xs_q8k_kernel<<<blocks, threads>>>(
                        dwb, t.kvalues, ddf, dqsb, doutf, w.rows,
                        w.cols);
                    break;
                case gguf::TensorType::kIQ3_XXS:
                    matvec_iq3_xxs_q8k_kernel<<<blocks, threads>>>(
                        dwb, t.g32, t.ksigns, ddf, dqsb, doutf, w.rows,
                        w.cols);
                    break;
                case gguf::TensorType::kIQ3_S:
                    matvec_iq3_s_q8k_kernel<<<blocks, threads>>>(
                        dwb, t.g32, ddf, dqsb, doutf, w.rows, w.cols);
                    break;
                case gguf::TensorType::kIQ1_S:
                    matvec_iq1_s_q8k_kernel<<<blocks, threads>>>(
                        dwb, t.g64, ddf, dqsb, dbsb, doutf, w.rows,
                        w.cols);
                    break;
                case gguf::TensorType::kIQ1_M:
                    matvec_iq1_m_q8k_kernel<<<blocks, threads>>>(
                        dwb, t.g64, ddf, dqsb, doutf, w.rows, w.cols);
                    break;
                default:
                    break;
            }
            ok = cuda_ok(cudaGetLastError()) &&
                 sync_counted() &&
                 cuda_ok(cudaMemcpy(out.data(), dout, out_bytes,
                                    cudaMemcpyDeviceToHost));
        }
    }
    return ok;
}

}  // namespace

void cuda_pool_reset() { pool().reset(); }

std::pair<std::size_t, std::size_t> cuda_pool_resident() {
    return pool().resident();
}

void matvec_cuda_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out) {
    ensure_device();
    if (cuda_has_q8k_kernel(w.type) && w.cols % 256 == 0) {
        const std::size_t bytes = device_weight_bytes(w);
        // IQ kernels need their grid/sign tables uploaded; if an upload
        // fails, fall through to scalar for this call.
        CudaTables t;
        bool tables_ok = true;
        switch (w.type) {
            case gguf::TensorType::kIQ2_XXS:
                t.g64 = device_iq2xxs_grid();
                t.ksigns = device_ksigns();
                tables_ok = t.g64 != nullptr && t.ksigns != nullptr;
                break;
            case gguf::TensorType::kIQ2_XS:
                t.g64 = device_iq2xs_grid();
                t.ksigns = device_ksigns();
                tables_ok = t.g64 != nullptr && t.ksigns != nullptr;
                break;
            case gguf::TensorType::kIQ2_S:
                t.g64 = device_iq2s_grid();
                tables_ok = t.g64 != nullptr;
                break;
            case gguf::TensorType::kIQ4_XS:
                t.kvalues = device_kvalues_iq4nl();
                tables_ok = t.kvalues != nullptr;
                break;
            case gguf::TensorType::kIQ3_XXS:
                t.g32 = device_iq3xxs_grid();
                t.ksigns = device_ksigns();
                tables_ok = t.g32 != nullptr && t.ksigns != nullptr;
                break;
            case gguf::TensorType::kIQ3_S:
                t.g32 = device_iq3s_grid();
                tables_ok = t.g32 != nullptr;
                break;
            case gguf::TensorType::kIQ1_S:
            case gguf::TensorType::kIQ1_M:
                t.g64 = device_iq1s_grid();
                tables_ok = t.g64 != nullptr;
                break;
            default:
                break;  // k-quants + ternary need no tables
        }
        if (bytes > 0 && tables_ok &&
            run_matvec_q8k(w, x, out, bytes, t)) {
            return;
        }
    }
    matvec_q8k(w, x, out);  // other types / failure: scalar Q8_K
}

namespace {

/** Fills t with the device tables w's kernel reads.
 *  @returns false when an upload failed. */
bool tables_for(gguf::TensorType type, CudaTables& t) {
    switch (type) {
        case gguf::TensorType::kIQ1_S:
        case gguf::TensorType::kIQ1_M:
            t.g64 = device_iq1s_grid();
            return t.g64 != nullptr;
        case gguf::TensorType::kIQ2_XXS:
            t.g64 = device_iq2xxs_grid();
            t.ksigns = device_ksigns();
            return t.g64 != nullptr && t.ksigns != nullptr;
        case gguf::TensorType::kIQ2_XS:
            t.g64 = device_iq2xs_grid();
            t.ksigns = device_ksigns();
            return t.g64 != nullptr && t.ksigns != nullptr;
        case gguf::TensorType::kIQ2_S:
            t.g64 = device_iq2s_grid();
            return t.g64 != nullptr;
        case gguf::TensorType::kIQ3_S:
            t.g32 = device_iq3s_grid();
            return t.g32 != nullptr;
        case gguf::TensorType::kIQ3_XXS:
            t.g32 = device_iq3xxs_grid();
            t.ksigns = device_ksigns();
            return t.g32 != nullptr && t.ksigns != nullptr;
        case gguf::TensorType::kIQ4_XS:
        case gguf::TensorType::kIQ4_NL:
            t.kvalues = device_kvalues_iq4nl();
            return t.kvalues != nullptr;
        default:
            return true;  // F32/Q8_0/Q2_K/Q4_K/Q5_K/Q6_K need none
    }
}

}  // namespace

void matvec_cuda(const Mat& w, std::span<const float> x,
                 std::span<float> out) {
    ensure_device();
    const std::size_t bytes = device_weight_bytes(w);
    bool done = false;
    if (bytes > 0) {
        // Grid/codebook kernels need their device tables uploaded;
        // if any upload failed we fall back to scalar for that call.
        CudaTables t;
        if (tables_for(w.type, t)) {
            done = run_matvec(w, x, out, bytes, kind_for(w.type), t);
            if (!done) {
                static Fallback& f = fallback(
                    "cuda", "cpu", "device error or out of memory");
                f.add();
            }
        } else {
            static Fallback& f =
                fallback("cuda", "cpu", "device table upload failed");
            f.add();
        }
    } else {
        static Fallback& f =
            fallback("cuda", "cpu", "no matvec kernel for the weight type");
        f.add();
    }
    if (!done) {
        matvec(w, x, out);  // fallback / other types: scalar
    }
}

void cuda_prefetch(const Mat& w) {
    ensure_device();
    const std::size_t bytes = device_weight_bytes(w);
    if (bytes > 0) {
        pool().prefetch(w.data, bytes);
    }
}

void cuda_evict(const Mat& w) {
    ensure_device();
    pool().drop(w.data);
}

std::size_t cuda_pool_budget() {
    ensure_device();
    return pool().budget();
}

void matvec_batch_cuda(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch, std::uint32_t n) {
    ensure_device();
    // Q4_K/Q6_K/Q8_0 get the prefill tile, one launch for all n tokens;
    // Q5_K/IQ4_NL/IQ4_XS the register-blocked kernel, one launch per
    // kMaxBatch tokens; other types (incl F32)
    // fall back to n matvec_cuda() calls scattered to row-major: the
    // same-backend fallback rule, so batched stays byte-identical to
    // per-token.
    const bool batch_type = has_batch_kernel(w.type);
    if (n == 1) {
        // One token: the per-token kernel is the same result and has no
        // per-token accumulator array to keep in registers.
        matvec_cuda(w, x_batch, out_batch);
        return;
    }
    if (batch_type) {
        const std::size_t bytes = device_weight_bytes(w);
        if (run_batch_sliced(w, x_batch, out_batch, bytes, n)) {
            return;
        }
        static Fallback& f = fallback("cuda", "per_token",
                                      "device error or out of memory");
        f.add();
    } else {
        static Fallback& f = fallback("cuda", "per_token",
                                      "no batch kernel for the weight type");
        f.add();
    }
    std::vector<float> col(w.rows);
    const std::size_t xc = w.cols;
    for (std::uint32_t t = 0; t < n; ++t) {
        matvec_cuda(
            w,
            x_batch.subspan(static_cast<std::size_t>(t) * xc, xc),
            col);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out_batch[static_cast<std::size_t>(r) * n + t] = col[r];
        }
    }
}

bool cuda_backend_usable() {
    int n = 0;
    return cuda_ok(cudaGetDeviceCount(&n)) && n > 0;
}

std::string cuda_device_name() {
    if (!cuda_backend_usable()) return "";
    int dev = 0;
    cudaDeviceProp p{};
    if (!cuda_ok(cudaGetDevice(&dev)) ||
        !cuda_ok(cudaGetDeviceProperties(&p, dev))) {
        return "";
    }
    return p.name;
}

void cuda_set_device(int ordinal) {
    if (ordinal < 0) {
        throw std::out_of_range(
            "CUDA device ordinal must be >= 0, got " +
            std::to_string(ordinal));
    }
    int count = 0;
    if (!cuda_ok(cudaGetDeviceCount(&count)) || count <= 0) {
        throw std::runtime_error(
            "cannot select CUDA device " + std::to_string(ordinal) +
            ": no CUDA device present on this host");
    }
    if (ordinal >= count) {
        throw std::out_of_range(
            "CUDA device " + std::to_string(ordinal) + " out of range; " +
            std::to_string(count) + " device(s) present");
    }
    if (!cuda_ok(cudaSetDevice(ordinal))) {
        throw std::runtime_error("cudaSetDevice(" +
                                 std::to_string(ordinal) + ") failed");
    }
    g_cuda_device.store(ordinal, std::memory_order_release);
}

// ---- Device primitives for a CUDA forward (i#517). Every op reads and
// writes device buffers and returns without waiting; cuda_dev_finish
// waits once and unpins the weights the calls pinned. The norm, RoPE
// and add kernels round each step as the host op does (no fused
// multiply-add), so their results equal the host's bit for bit.

namespace {

/** Weights pinned by cuda_dev_matvec since the last cuda_dev_finish. */
std::vector<const void*>& dev_pins() {
    thread_local std::vector<const void*> pins;
    return pins;
}

/** rmsnorm per token: one block of 256 per token; each thread sums the
 *  squares of its stride, the warps reduce by a fixed xor tree, then
 *  warp 0 the 8 warp sums (i#1254: a sum by thread 0 alone took 18 us
 *  for 896 floats). A fixed order, the same for a token in a prefill or
 *  a decode, close to the host's index order (DESIGN 16). */
__global__ void dev_rmsnorm_kernel(const float* __restrict__ x,
                                   const float* __restrict__ w, float eps,
                                   float* __restrict__ out, std::uint32_t n) {
    __shared__ float part[32];
    __shared__ float scale;
    const float* xt = x + std::size_t{blockIdx.x} * n;
    float* ot = out + std::size_t{blockIdx.x} * n;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float ss = 0.0f;
    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        ss = __fadd_rn(ss, __fmul_rn(xt[i], xt[i]));
    }
    for (int o = 16; o > 0; o >>= 1) {
        ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    }
    if (lane == 0) {
        part[warp] = ss;
    }
    __syncthreads();
    if (warp == 0) {
        const unsigned nw = blockDim.x / 32;
        float s = lane < nw ? part[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
        }
        if (lane == 0) {
            scale = __fdiv_rn(
                1.0f,
                __fsqrt_rn(__fadd_rn(__fdiv_rn(s, static_cast<float>(n)),
                                     eps)));
        }
    }
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        ot[i] = __fmul_rn(__fmul_rn(xt[i], scale), w[i]);
    }
}

/** RoPE (rope_norm's pairs) per token from host-computed cos / sin:
 *  cs holds, per token, head_dim / 2 (cos, sin) pairs. */
__global__ void dev_rope_kernel(float* __restrict__ x,
                                const float* __restrict__ cs,
                                std::uint32_t n_heads, std::uint32_t hd,
                                std::size_t stride, bool neox) {
    const std::uint32_t t = blockIdx.y;
    const std::uint32_t pairs = hd / 2;
    const std::uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_heads * pairs) {
        return;
    }
    const std::uint32_t h = k / pairs, p = k % pairs;
    // rope_norm pairs (2p, 2p + 1); NEOX pairs (p, p + hd/2) (i#1196).
    float* head = x + t * stride + std::size_t{h} * hd;
    float* v0 = neox ? head + p : head + 2 * p;
    float* v1 = neox ? head + p + pairs : head + 2 * p + 1;
    const float c = cs[(std::size_t{t} * pairs + p) * 2];
    const float s = cs[(std::size_t{t} * pairs + p) * 2 + 1];
    const float x0 = *v0, x1 = *v1;
    *v0 = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
    *v1 = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
}

/** x[t * width + i] += b[i] for every row t (grid.y). */
__global__ void dev_add_rows_kernel(float* __restrict__ x,
                                    const float* __restrict__ b,
                                    std::size_t width) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < width) {
        float* row = x + std::size_t{blockIdx.y} * width;
        row[i] = __fadd_rn(row[i], b[i]);
    }
}

/** gate = gate / (1 + exp(-gate)) * up (silu_mul). */
__global__ void dev_silu_mul_kernel(float* __restrict__ g,
                                    const float* __restrict__ u,
                                    std::size_t n) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) {
        const float gi = g[i];
        g[i] = __fmul_rn(__fdiv_rn(gi, __fadd_rn(1.0f, expf(-gi))), u[i]);
    }
}

/** x += y. */
__global__ void dev_add_kernel(float* __restrict__ x,
                               const float* __restrict__ y, std::size_t n) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = __fadd_rn(x[i], y[i]);
    }
}

/** out[t * rows + r] = in[r * n + t]: a batch kernel's row-major result
 *  to the token-major layout of the rest of the forward. */
__global__ void dev_transpose_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     std::uint32_t rows, std::uint32_t n) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < std::size_t{rows} * n) {
        const std::size_t r = i / n, t = i % n;
        out[t * rows + r] = in[i];
    }
}

unsigned blocks_for(std::size_t n, unsigned threads = 256) {
    return static_cast<unsigned>((n + threads - 1) / threads);
}

/** Token t's K and V rows (kv_dim floats each) to the mirror at
 *  rows[t] + layer_off (V v_off further) and to the host staging copy
 *  at t * stage_stride + stage_off (V kv_dim further). */
__global__ void dev_kv_scatter_kernel(
    const float* __restrict__ k, const float* __restrict__ v,
    std::uint32_t kv_dim, const std::uint64_t* __restrict__ rows,
    std::size_t layer_off, std::size_t v_off, float* __restrict__ mirror,
    float* __restrict__ stage, std::size_t stage_stride,
    std::size_t stage_off) {
    const std::uint32_t t = blockIdx.y;
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) {
        return;
    }
    const float kk = k[std::size_t{t} * kv_dim + i];
    const float vv = v[std::size_t{t} * kv_dim + i];
    float* m = mirror + rows[t] + layer_off + i;
    m[0] = kk;
    m[v_off] = vv;
    float* s = stage + t * stage_stride + stage_off + i;
    s[0] = kk;
    s[kv_dim] = vv;
}

}  // namespace

float* cuda_dev_alloc(std::size_t n) {
    ensure_device();
    void* p = nullptr;
    return cuda_ok(cudaMalloc(&p, n * sizeof(float))) ? static_cast<float*>(p)
                                                      : nullptr;
}

void cuda_dev_free(float* p) {
    ensure_device();
    cudaFree(p);
}

bool cuda_dev_upload(float* d, const float* h, std::size_t n) {
    return cuda_ok(
        cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
}

bool cuda_dev_download(float* h, const float* d, std::size_t n) {
    return cuda_ok(
        cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
}

const float* cuda_dev_const(const float* h, std::size_t n) {
    ensure_device();
    const void* p = pool().acquire(h, n * sizeof(float));
    if (p != nullptr) {
        dev_pins().push_back(h);
    }
    return static_cast<const float*>(p);
}

bool cuda_dev_matvec(const Mat& w, const float* dx, float* dout) {
    ensure_device();
    const std::size_t bytes = device_weight_bytes(w);
    CudaTables t;
    if (bytes == 0 || !tables_for(w.type, t)) {
        return false;
    }
    const void* dw = pool().acquire(w.data, bytes);
    if (dw == nullptr) {
        return false;  // too big for the pool: the caller uses the host
    }
    dev_pins().push_back(w.data);
    launch_kernel(kind_for(w.type), dw, t, dx, dout, w.rows, w.cols);
    return cuda_ok(cudaGetLastError());
}

bool cuda_dev_matvec_batch(const Mat& w, const float* dx, float* dout,
                           std::uint32_t n, float* scratch) {
    bool tables_ok = false;
    const std::int8_t* kv = batch_tables(w.type, tables_ok);
    if (n == 1 || !has_batch_kernel(w.type) || !tables_ok) {
        // The per-token kernel, which the batch kernels equal bit for bit.
        for (std::uint32_t t = 0; t < n; ++t) {
            if (!cuda_dev_matvec(w, dx + std::size_t{t} * w.cols,
                                 dout + std::size_t{t} * w.rows)) {
                return false;
            }
        }
        return true;
    }
    ensure_device();
    const std::size_t bytes = device_weight_bytes(w);
    const void* dw = pool().acquire(w.data, bytes);
    if (dw == nullptr) {
        return false;
    }
    dev_pins().push_back(w.data);
    const auto* dwb = static_cast<const std::uint8_t*>(dw);
    // The prefill tiles (i#1195): all n tokens in one launch, the output
    // token-major as dout wants it.
    if (launch_tile(w.type, dwb, dx, dout, w.rows, w.cols, n)) {
        return cuda_ok(cudaGetLastError());
    }
    // At most kMaxBatch tokens per launch, as run_batch_sliced does.
    for (std::uint32_t t0 = 0; t0 < n; t0 += kMaxBatch) {
        const std::uint32_t m = std::min(kMaxBatch, n - t0);
        const float* xs = dx + std::size_t{t0} * w.cols;
        if (!launch_batch(w.type, dwb, kv, xs, scratch, w.rows, w.cols,
                          m)) {
            return false;
        }
        dev_transpose_kernel<<<blocks_for(std::size_t{w.rows} * m), 256>>>(
            scratch, dout + std::size_t{t0} * w.rows, w.rows, m);
    }
    return cuda_ok(cudaGetLastError());
}

bool cuda_dev_rmsnorm(const float* dx, const float* dw, float eps,
                      float* dout, std::uint32_t n, std::uint32_t tokens) {
    dev_rmsnorm_kernel<<<tokens, 256>>>(dx, dw, eps, dout, n);
    return cuda_ok(cudaGetLastError());
}

bool cuda_dev_rope(float* dx, const float* dcs, std::uint32_t n_heads,
                   std::uint32_t hd, std::size_t stride,
                   std::uint32_t tokens, bool neox) {
    const std::size_t work = std::size_t{n_heads} * (hd / 2);
    dev_rope_kernel<<<dim3(blocks_for(work), tokens), 256>>>(
        dx, dcs, n_heads, hd, stride, neox);
    return cuda_ok(cudaGetLastError());
}

bool cuda_dev_add_rows(float* dx, const float* db, std::size_t width,
                       std::uint32_t tokens) {
    dev_add_rows_kernel<<<dim3(blocks_for(width), tokens), 256>>>(dx, db,
                                                                 width);
    return cuda_ok(cudaGetLastError());
}

bool cuda_dev_silu_mul(float* dg, const float* du, std::size_t n) {
    dev_silu_mul_kernel<<<blocks_for(n), 256>>>(dg, du, n);
    return cuda_ok(cudaGetLastError());
}

bool cuda_dev_add(float* dx, const float* dy, std::size_t n) {
    dev_add_kernel<<<blocks_for(n), 256>>>(dx, dy, n);
    return cuda_ok(cudaGetLastError());
}

void* cuda_dev_alloc_bytes(std::size_t bytes) {
    ensure_device();
    void* p = nullptr;
    return cuda_ok(cudaMalloc(&p, bytes)) ? p : nullptr;
}

void cuda_dev_free_bytes(void* p) {
    ensure_device();
    cudaFree(p);
}

bool cuda_dev_upload_bytes(void* d, const void* h, std::size_t bytes) {
    return cuda_ok(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
}

bool cuda_dev_kv_scatter(const float* dk, const float* dv,
                         std::uint32_t n, std::uint32_t kv_dim,
                         const std::uint64_t* drows, std::size_t layer_off,
                         std::size_t v_off, float* mirror, float* stage,
                         std::size_t stage_stride, std::size_t stage_off) {
    dev_kv_scatter_kernel<<<dim3(blocks_for(kv_dim), n), 256>>>(
        dk, dv, kv_dim, drows, layer_off, v_off, mirror, stage,
        stage_stride, stage_off);
    return cuda_ok(cudaGetLastError());
}

bool cuda_dev_finish() {
    const bool ok = sync_counted();
    for (const void* p : dev_pins()) {
        pool().release(p);
    }
    dev_pins().clear();
    return ok;
}

}  // namespace locus::backend
