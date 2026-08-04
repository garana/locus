// CUDA matvec variant. Weights run through a device weight pool (the
// R8-GPU pager): each weight is uploaded once, keyed by host pointer,
// and reused across tokens under a byte budget with LRU eviction;
// weights too big for the budget fall back to an uncached per-call
// upload. Kernels dequantize F32/Q8_0/Q2_K/Q4_K/Q5_K/Q6_K and the
// IQ1_S/IQ2_XXS/IQ3_XXS/IQ4_XS families on-device (the pool stores the
// source quantized bytes); other types delegate to scalar.

#include "locus/backend/variants.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "iq_grids.h"  // iq1s_grid lookup table (global scope)

namespace locus::backend {

namespace {

/** Aborts to scalar on any CUDA error (caller falls back). */
bool cuda_ok(cudaError_t e) { return e == cudaSuccess; }

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

/** IQ4_NL: 18-byte / 32-element block (d f16 + qs[16]). Low nibble ->
 *  first 16 lanes, high nibble -> next 16, mapped through the
 *  kvalues_iq4nl codebook. Same per-block accumulation as dot_iq4_nl. */
__global__ void matvec_iq4_nl_kernel(const std::uint8_t* w,
                                     const std::int8_t* kvalues,
                                     const float* x, float* out,
                                     std::uint32_t rows,
                                     std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nblk = cols / 32;
    const std::size_t row_bytes =
        static_cast<std::size_t>(nblk) * 18;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nblk; ++b) {
        const std::uint8_t* blk = row + b * 18;
        const float d = ld_f16(blk);
        const std::uint8_t* q = blk + 2;
        const float* xb = x + static_cast<std::size_t>(b) * 32;
        float s = 0.0f;
        for (int i = 0; i < 16; ++i) {
            s += static_cast<float>(kvalues[q[i] & 0x0f]) * xb[i] +
                 static_cast<float>(kvalues[q[i] >> 4]) * xb[i + 16];
        }
        acc += d * s;
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

/** Q4_K x Q8_K integer dot on device; bit-exact port of the scalar
 *  vec_dot_q4_k_q8_k (aux8 nibble unpack, per-8 int16 products, scale*
 *  int32 accumulate over 8 lanes, bsums min-correction). Activation
 *  quants arrive flat: d_y[nsb], qs_y[nsb*256], bsums_y[nsb*16]. One
 *  thread per row keeps the scalar accumulation order, so the float
 *  result is identical. */
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
    const std::uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f,
                        kmask3 = 0x03030303;
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 144;
        const float bd = __half2float(__ushort_as_half(
            static_cast<std::uint16_t>(blk[0] | (blk[1] << 8))));
        const float bdmin = __half2float(__ushort_as_half(
            static_cast<std::uint16_t>(blk[2] | (blk[3] << 8))));
        const std::uint8_t* q4 = blk + 16;
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int8_t aux8[256];
        std::int8_t* a = aux8;
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
        std::uint32_t utmp[4];
        const std::uint8_t* s = blk + 4;  // Q4_K scales[12]
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
        a = aux8;
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
    out[r] = sumf;
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

/** Q5_K x Q8_K integer dot; bit-exact port of scalar vec_dot_q5_k_q8_k
 *  (nibble + hm 5th-bit unpack, 6-bit scales, bsums min-correction). */
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
    const std::uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f,
                        kmask3 = 0x03030303;
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float sumf = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk =
            row + static_cast<std::size_t>(b) * 176;
        const float bd = __half2float(__ushort_as_half(
            static_cast<std::uint16_t>(blk[0] | (blk[1] << 8))));
        const float bdmin = __half2float(__ushort_as_half(
            static_cast<std::uint16_t>(blk[2] | (blk[3] << 8))));
        const std::uint8_t* hm = blk + 16;
        const std::uint8_t* q4 = blk + 48;
        const std::int8_t* q8 = qs_y + static_cast<std::size_t>(b) * 256;
        std::int8_t aux8[256];
        std::int8_t* a = aux8;
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
        std::uint32_t utmp[4];
        const std::uint8_t* s = blk + 4;
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
        a = aux8;
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
    out[r] = sumf;
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

/** One Q5_K super-block is 176 bytes: d,dmin (f16) + scales[12] +
 *  qh[32] + ql[128], 256 weights. The 5th bit comes from qh, its bit
 *  position advancing per 64-group (u1/u2). Fill order is sequential,
 *  so the dot accumulates on the fly matching dequant_block_q5_k. */
__global__ void matvec_q5_k_kernel(const std::uint8_t* w,
                                   const float* x, float* out,
                                   std::uint32_t rows,
                                   std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 176;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            176;
        const std::uint16_t db =
            static_cast<std::uint16_t>(blk[0]) |
            (static_cast<std::uint16_t>(blk[1]) << 8);
        const std::uint16_t dmb =
            static_cast<std::uint16_t>(blk[2]) |
            (static_cast<std::uint16_t>(blk[3]) << 8);
        const float d = __half2float(__ushort_as_half(db));
        const float dmin = __half2float(__ushort_as_half(dmb));
        const std::uint8_t* scales = blk + 4;
        const std::uint8_t* qh = blk + 16;
        const std::uint8_t* ql = blk + 48;
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        int is = 0, yi = 0;
        std::uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            std::uint8_t sc, mn;
            scale_min_k4(is + 0, scales, sc, mn);
            const float d1 = d * sc, m1 = dmin * mn;
            scale_min_k4(is + 1, scales, sc, mn);
            const float d2 = d * sc, m2 = dmin * mn;
            for (int l = 0; l < 32; ++l) {
                const float val =
                    d1 * static_cast<float>(
                             (ql[l] & 0x0f) +
                             ((qh[l] & u1) ? 16 : 0)) -
                    m1;
                acc += val * xb[yi++];
            }
            for (int l = 0; l < 32; ++l) {
                const float val =
                    d2 * static_cast<float>(
                             (ql[l] >> 4) +
                             ((qh[l] & u2) ? 16 : 0)) -
                    m2;
                acc += val * xb[yi++];
            }
            ql += 32;
            is += 2;
            u1 = static_cast<std::uint8_t>(u1 << 2);
            u2 = static_cast<std::uint8_t>(u2 << 2);
        }
    }
    out[r] = acc;
}

/** One Q6_K super-block is 210 bytes: ql[128] + qh[64] + sc[16]int8
 *  + d (f16), 256 weights. The scalar fills y in a scattered order
 *  (y[l], y[l+32], y[l+64], y[l+96]); to keep the dot bit-identical
 *  to dot_k_quant we materialize the block then accumulate i=0..255. */
__global__ void matvec_q6_k_kernel(const std::uint8_t* w,
                                   const float* x, float* out,
                                   std::uint32_t rows,
                                   std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 210;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            210;
        const std::uint8_t* qlp = blk;
        const std::uint8_t* qhp = blk + 128;
        const std::int8_t* scp =
            reinterpret_cast<const std::int8_t*>(blk + 192);
        const std::uint16_t db =
            static_cast<std::uint16_t>(blk[208]) |
            (static_cast<std::uint16_t>(blk[209]) << 8);
        const float d = __half2float(__ushort_as_half(db));
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        float tmp[256];
        float* y = tmp;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = static_cast<int>(
                                   (qlp[l] & 0x0f) |
                                   (((qhp[l] >> 0) & 3) << 4)) -
                               32;
                const int q2 = static_cast<int>(
                                   (qlp[l + 32] & 0x0f) |
                                   (((qhp[l] >> 2) & 3) << 4)) -
                               32;
                const int q3 = static_cast<int>(
                                   (qlp[l] >> 4) |
                                   (((qhp[l] >> 4) & 3) << 4)) -
                               32;
                const int q4 = static_cast<int>(
                                   (qlp[l + 32] >> 4) |
                                   (((qhp[l] >> 6) & 3) << 4)) -
                               32;
                y[l] = d * scp[is] * static_cast<float>(q1);
                y[l + 32] = d * scp[is + 2] * static_cast<float>(q2);
                y[l + 64] = d * scp[is + 4] * static_cast<float>(q3);
                y[l + 96] = d * scp[is + 6] * static_cast<float>(q4);
            }
            y += 128;
            qlp += 64;
            qhp += 32;
            scp += 8;
        }
        for (int i = 0; i < 256; ++i) {
            acc += tmp[i] * xb[i];
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

/** One IQ1_M super-block is 56 bytes: qs[32] | qh[16] | scales[8]. No
 *  d field -- the f16 scale is assembled from the top nibble of each
 *  of 4 scale u16s; two 3-bit sub-scales per 32-group; grid index is a
 *  qs byte + 3 high bits from qh; value dl*(grid[j] +/- 0.125).
 *  `grid` = device iq1s_grid. Matches dequant_block_iq1_m. */
__global__ void matvec_iq1_m_kernel(const std::uint8_t* w,
                                    const std::uint64_t* grid,
                                    const float* x, float* out,
                                    std::uint32_t rows,
                                    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    constexpr float kDelta = 0.125f;
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 56;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            56;
        std::uint16_t sc[4];
        for (int k = 0; k < 4; ++k) {
            const std::uint8_t* p = blk + 48 + 2 * k;
            sc[k] = static_cast<std::uint16_t>(p[0]) |
                    (static_cast<std::uint16_t>(p[1]) << 8);
        }
        const std::uint16_t su16 =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
            ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = __half2float(__ushort_as_half(su16));
        const std::uint8_t* qsp = blk;
        const std::uint8_t* qhp = blk + 32;
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        float tmp[256];
        float* y = tmp;
        for (int ib = 0; ib < 8; ++ib) {
            const float dl1 =
                d *
                (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
            const float dl2 =
                d *
                (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
            std::uint16_t idx[4];
            idx[0] = qsp[0] | ((qhp[0] << 8) & 0x700);
            idx[1] = qsp[1] | ((qhp[0] << 4) & 0x700);
            idx[2] = qsp[2] | ((qhp[1] << 8) & 0x700);
            idx[3] = qsp[3] | ((qhp[1] << 4) & 0x700);
            float delta[4];
            delta[0] = (qhp[0] & 0x08) ? -kDelta : kDelta;
            delta[1] = (qhp[0] & 0x80) ? -kDelta : kDelta;
            delta[2] = (qhp[1] & 0x08) ? -kDelta : kDelta;
            delta[3] = (qhp[1] & 0x80) ? -kDelta : kDelta;
            for (int l = 0; l < 2; ++l) {
                const std::int8_t* g =
                    reinterpret_cast<const std::int8_t*>(grid +
                                                         idx[l]);
                for (int j = 0; j < 8; ++j) {
                    *y++ = dl1 * (g[j] + delta[l]);
                }
            }
            for (int l = 2; l < 4; ++l) {
                const std::int8_t* g =
                    reinterpret_cast<const std::int8_t*>(grid +
                                                         idx[l]);
                for (int j = 0; j < 8; ++j) {
                    *y++ = dl2 * (g[j] + delta[l]);
                }
            }
            qsp += 4;
            qhp += 2;
        }
        for (int i = 0; i < 256; ++i) {
            acc += tmp[i] * xb[i];
        }
    }
    out[r] = acc;
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

/** One IQ2_XXS super-block is 66 bytes: d (f16) + 32 u16, 256
 *  weights. Each 32-group picks 4 grid entries (8 u8 each) and a sign
 *  byte from ksigns; `grid` = device iq2xxs_grid (u64), `ksigns` =
 *  device ksigns_iq2xs. Materialize then dot for dot_k_quant order. */
__global__ void matvec_iq2_xxs_kernel(const std::uint8_t* w,
                                      const std::uint64_t* grid,
                                      const std::uint8_t* ksigns,
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
        const float d = ld_f16(blk);
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        float tmp[256];
        float* y = tmp;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint8_t* a = blk + 2 + 8 * ib32;
            const std::uint32_t aux1 = ld_u32(a + 4);
            const float db = d * (0.5f + (aux1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t* g =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + a[l]);
                const std::uint8_t signs =
                    ksigns[(aux1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * g[j] *
                           ((signs & (1 << j)) ? -1.0f : 1.0f);
                }
                y += 8;
            }
        }
        for (int i = 0; i < 256; ++i) {
            acc += tmp[i] * xb[i];
        }
    }
    out[r] = acc;
}

/** One IQ2_XS super-block is 74 bytes: d (f16) + qs[32] u16 +
 *  scales[8]. Each 32-group has two 4-bit sub-scales; each qs low 9
 *  bits index iq2xs_grid (8 u8), high 7 bits index ksigns_iq2xs.
 *  `grid` = device iq2xs_grid (u64). Matches dequant_block_iq2_xs. */
__global__ void matvec_iq2_xs_kernel(const std::uint8_t* w,
                                     const std::uint64_t* grid,
                                     const std::uint8_t* ksigns,
                                     const float* x, float* out,
                                     std::uint32_t rows,
                                     std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 74;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            74;
        const float d = ld_f16(blk);
        const std::uint8_t* scales = blk + 66;
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        float tmp[256];
        float* y = tmp;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const float db0 =
                d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
            const float db1 =
                d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t* p = blk + 2 + 2 * (4 * ib32 + l);
                const std::uint16_t q =
                    static_cast<std::uint16_t>(p[0]) |
                    (static_cast<std::uint16_t>(p[1]) << 8);
                const std::uint8_t* g =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (q & 511));
                const std::uint8_t signs = ksigns[q >> 9];
                const float db = (l / 2 == 0) ? db0 : db1;
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * g[j] *
                           ((signs & (1 << j)) ? -1.0f : 1.0f);
                }
                y += 8;
            }
        }
        for (int i = 0; i < 256; ++i) {
            acc += tmp[i] * xb[i];
        }
    }
    out[r] = acc;
}

/** One IQ2_S super-block is 82 bytes: d + qs[64] + qh[8] + scales[8].
 *  qs = 32 grid-index bytes then 32 sign bytes; qh adds 2 high index
 *  bits per element (10-bit iq2s_grid index); signs are raw bytes.
 *  `grid` = device iq2s_grid (u64). Matches dequant_block_iq2_s. */
__global__ void matvec_iq2_s_kernel(const std::uint8_t* w,
                                    const std::uint64_t* grid,
                                    const float* x, float* out,
                                    std::uint32_t rows,
                                    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 82;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            82;
        const float d = ld_f16(blk);
        const std::uint8_t* qh = blk + 66;
        const std::uint8_t* scales = blk + 74;
        const std::uint8_t* qsp = blk + 2;
        const std::uint8_t* signs = blk + 2 + 32;
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        float tmp[256];
        float* y = tmp;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const float db0 =
                d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
            const float db1 =
                d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = (l / 2 == 0) ? db0 : db1;
                const std::uint32_t idx =
                    qsp[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                const std::uint8_t* g =
                    reinterpret_cast<const std::uint8_t*>(grid + idx);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * g[j] *
                           ((signs[l] & (1 << j)) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qsp += 4;
            signs += 4;
        }
        for (int i = 0; i < 256; ++i) {
            acc += tmp[i] * xb[i];
        }
    }
    out[r] = acc;
}

/** One IQ3_S super-block is 110 bytes: d | qs[64] | qh[8] | signs[32]
 *  | scales[4]. Two 32-groups per outer step; each element's 9-bit
 *  iq3s_grid index is a qs byte + one high bit from qh; grid entry is
 *  4 u8; signs raw. `grid` = device iq3s_grid (u32). Matches
 *  dequant_block_iq3_s. */
__global__ void matvec_iq3_s_kernel(const std::uint8_t* w,
                                    const std::uint32_t* grid,
                                    const float* x, float* out,
                                    std::uint32_t rows,
                                    std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 110;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            110;
        const float d = ld_f16(blk);
        const std::uint8_t* qs = blk + 2;
        const std::uint8_t* qh = blk + 66;
        const std::uint8_t* signs = blk + 74;
        const std::uint8_t* scales = blk + 106;
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        float tmp[256];
        float* y = tmp;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t* g1 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (qs[2 * l] |
                                ((qh[0] << (8 - 2 * l)) & 256)));
                const std::uint8_t* g2 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (qs[2 * l + 1] |
                                ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j] = db1 * g1[j] *
                           ((signs[l] & (1 << j)) ? -1.0f : 1.0f);
                    y[j + 4] =
                        db1 * g2[j] *
                        ((signs[l] & (1 << (j + 4))) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t* g1 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (qs[2 * l] |
                                ((qh[1] << (8 - 2 * l)) & 256)));
                const std::uint8_t* g2 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + (qs[2 * l + 1] |
                                ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j] = db2 * g1[j] *
                           ((signs[l] & (1 << j)) ? -1.0f : 1.0f);
                    y[j + 4] =
                        db2 * g2[j] *
                        ((signs[l] & (1 << (j + 4))) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
        for (int i = 0; i < 256; ++i) {
            acc += tmp[i] * xb[i];
        }
    }
    out[r] = acc;
}

/** One IQ3_XXS super-block is 98 bytes: d (f16) + qs[64] +
 *  scales_and_signs[32], 256 weights. Each 32-group picks 8 grid
 *  entries (4 u8 each) via qs and a sign byte; `grid` = device
 *  iq3xxs_grid (u32), `ksigns` = device ksigns_iq2xs. */
__global__ void matvec_iq3_xxs_kernel(const std::uint8_t* w,
                                      const std::uint32_t* grid,
                                      const std::uint8_t* ksigns,
                                      const float* x, float* out,
                                      std::uint32_t rows,
                                      std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 98;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            98;
        const float d = ld_f16(blk);
        const std::uint8_t* qs = blk + 2;
        const std::uint8_t* sas = blk + 2 + 64;
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        float tmp[256];
        float* y = tmp;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint32_t aux = ld_u32(sas + 4 * ib32);
            const float db = d * (0.5f + (aux >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t signs =
                    ksigns[(aux >> (7 * l)) & 127];
                const std::uint8_t* g1 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + qs[2 * l]);
                const std::uint8_t* g2 =
                    reinterpret_cast<const std::uint8_t*>(
                        grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j] = db * g1[j] *
                           ((signs & (1 << j)) ? -1.0f : 1.0f);
                    y[j + 4] = db * g2[j] *
                               ((signs & (1 << (j + 4))) ? -1.0f
                                                         : 1.0f);
                }
                y += 8;
            }
            qs += 8;
        }
        for (int i = 0; i < 256; ++i) {
            acc += tmp[i] * xb[i];
        }
    }
    out[r] = acc;
}

/** One IQ4_XS super-block is 136 bytes: d (f16) + scales_h (u16) +
 *  scales_l[4] + qs[128], 256 weights. Grid-free: each nibble indexes
 *  the 16-entry kvalues_iq4nl codebook (device `kvalues`). Scalar fill
 *  interleaves y[j]/y[j+16], so materialize then dot i=0..255. */
__global__ void matvec_iq4_xs_kernel(const std::uint8_t* w,
                                     const std::int8_t* kvalues,
                                     const float* x, float* out,
                                     std::uint32_t rows,
                                     std::uint32_t cols) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 136;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc = 0.0f;
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            136;
        const float d = ld_f16(blk);
        const std::uint16_t scales_h =
            static_cast<std::uint16_t>(blk[2]) |
            (static_cast<std::uint16_t>(blk[3]) << 8);
        const std::uint8_t* scales_l = blk + 4;
        const std::uint8_t* qs = blk + 8;
        const float* xb = x + static_cast<std::size_t>(b) * 256;
        float tmp[256];
        float* y = tmp;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls =
                ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f) |
                (((scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j] = dl * kvalues[qs[j] & 0x0f];
                y[j + 16] = dl * kvalues[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
        for (int i = 0; i < 256; ++i) {
            acc += tmp[i] * xb[i];
        }
    }
    out[r] = acc;
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

// R11 batched kernels: one launch, one thread per row, n token
// accumulators. Each weight block is read from VRAM once and reused
// across all n tokens (weight traffic ~n-fold down). Each token t's
// accumulator runs the identical dequant+FMA order as the per-token
// kernel, so the result is byte-identical to n matvec_cuda() calls.
// out is row-major: out[r*n + t]. x is token-major: token t at t*cols.
constexpr std::uint32_t kMaxBatch = 32;

__global__ void matvec_batch_q4_k_kernel(const std::uint8_t* w,
                                         const float* x, float* out,
                                         std::uint32_t rows,
                                         std::uint32_t cols,
                                         std::uint32_t n) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 144;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc[kMaxBatch];
    for (std::uint32_t t = 0; t < n; ++t) {
        acc[t] = 0.0f;
    }
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
        const std::size_t xoff = static_cast<std::size_t>(b) * 256;
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
                for (std::uint32_t t = 0; t < n; ++t) {
                    acc[t] += val *
                              x[static_cast<std::size_t>(t) * cols +
                                xoff + yi];
                }
                ++yi;
            }
            for (int l = 0; l < 32; ++l) {
                const float val =
                    d2 * static_cast<float>(q[l] >> 4) - m2;
                for (std::uint32_t t = 0; t < n; ++t) {
                    acc[t] += val *
                              x[static_cast<std::size_t>(t) * cols +
                                xoff + yi];
                }
                ++yi;
            }
            q += 32;
            is += 2;
        }
    }
    for (std::uint32_t t = 0; t < n; ++t) {
        out[static_cast<std::size_t>(r) * n + t] = acc[t];
    }
}

__global__ void matvec_batch_q6_k_kernel(const std::uint8_t* w,
                                         const float* x, float* out,
                                         std::uint32_t rows,
                                         std::uint32_t cols,
                                         std::uint32_t n) {
    const std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const std::uint32_t nsb = cols / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(nsb) * 210;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) *
                                      row_bytes;
    float acc[kMaxBatch];
    for (std::uint32_t t = 0; t < n; ++t) {
        acc[t] = 0.0f;
    }
    for (std::uint32_t b = 0; b < nsb; ++b) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(b) *
                                            210;
        const std::uint8_t* qlp = blk;
        const std::uint8_t* qhp = blk + 128;
        const std::int8_t* scp =
            reinterpret_cast<const std::int8_t*>(blk + 192);
        const std::uint16_t db =
            static_cast<std::uint16_t>(blk[208]) |
            (static_cast<std::uint16_t>(blk[209]) << 8);
        const float d = __half2float(__ushort_as_half(db));
        float tmp[256];
        float* y = tmp;
        for (int nn = 0; nn < 256; nn += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = static_cast<int>(
                                   (qlp[l] & 0x0f) |
                                   (((qhp[l] >> 0) & 3) << 4)) -
                               32;
                const int q2 = static_cast<int>(
                                   (qlp[l + 32] & 0x0f) |
                                   (((qhp[l] >> 2) & 3) << 4)) -
                               32;
                const int q3 = static_cast<int>(
                                   (qlp[l] >> 4) |
                                   (((qhp[l] >> 4) & 3) << 4)) -
                               32;
                const int q4 = static_cast<int>(
                                   (qlp[l + 32] >> 4) |
                                   (((qhp[l] >> 6) & 3) << 4)) -
                               32;
                y[l] = d * scp[is] * static_cast<float>(q1);
                y[l + 32] = d * scp[is + 2] * static_cast<float>(q2);
                y[l + 64] = d * scp[is + 4] * static_cast<float>(q3);
                y[l + 96] = d * scp[is + 6] * static_cast<float>(q4);
            }
            y += 128;
            qlp += 64;
            qhp += 32;
            scp += 8;
        }
        const std::size_t xoff = static_cast<std::size_t>(b) * 256;
        for (int i = 0; i < 256; ++i) {
            for (std::uint32_t t = 0; t < n; ++t) {
                acc[t] += tmp[i] *
                          x[static_cast<std::size_t>(t) * cols +
                            xoff + i];
            }
        }
    }
    for (std::uint32_t t = 0; t < n; ++t) {
        out[static_cast<std::size_t>(r) * n + t] = acc[t];
    }
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
        case Kind::kQ8_0:
            matvec_q8_0_kernel<<<blocks, threads>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kQ4_K:
            matvec_q4_k_kernel<<<blocks, threads>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kQ5_K:
            matvec_q5_k_kernel<<<blocks, threads>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kQ6_K:
            matvec_q6_k_kernel<<<blocks, threads>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ1_S:
            matvec_iq1_s_kernel<<<blocks, threads>>>(
                dwb, t.g64, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ1_M:
            matvec_iq1_m_kernel<<<blocks, threads>>>(
                dwb, t.g64, dxf, doutf, rows, cols);
            break;
        case Kind::kQ2_K:
            matvec_q2_k_kernel<<<blocks, threads>>>(
                dwb, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ2_XXS:
            matvec_iq2_xxs_kernel<<<blocks, threads>>>(
                dwb, t.g64, t.ksigns, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ2_XS:
            matvec_iq2_xs_kernel<<<blocks, threads>>>(
                dwb, t.g64, t.ksigns, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ2_S:
            matvec_iq2_s_kernel<<<blocks, threads>>>(
                dwb, t.g64, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ3_S:
            matvec_iq3_s_kernel<<<blocks, threads>>>(
                dwb, t.g32, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ3_XXS:
            matvec_iq3_xxs_kernel<<<blocks, threads>>>(
                dwb, t.g32, t.ksigns, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ4_XS:
            matvec_iq4_xs_kernel<<<blocks, threads>>>(
                dwb, t.kvalues, dxf, doutf, rows, cols);
            break;
        case Kind::kIQ4_NL:
            matvec_iq4_nl_kernel<<<blocks, threads>>>(
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

bool run_matvec(const Mat& w, std::span<const float> x,
                std::span<float> out, std::size_t w_bytes,
                Kind kind, const CudaTables& t = {}) {
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
            launch_kernel(kind, dw, t,
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

/** One-launch batched Q4_K/Q6_K matvec: weight pulled from the pool
 *  (shared across the n tokens), x_batch + out_batch staged once. */
bool run_batch(const Mat& w, std::span<const float> x_batch,
               std::span<float> out_batch, std::size_t w_bytes,
               std::uint32_t n, bool q6) {
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
        DevBuf dx, dout;
        const std::size_t x_bytes = x_batch.size() * sizeof(float);
        const std::size_t out_bytes = out_batch.size() * sizeof(float);
        if (dx.alloc(x_bytes) && dout.alloc(out_bytes) &&
            cuda_ok(cudaMemcpy(dx.p, x_batch.data(), x_bytes,
                               cudaMemcpyHostToDevice))) {
            const std::uint32_t threads = 256;
            const std::uint32_t blocks =
                (w.rows + threads - 1) / threads;
            const auto* dwb = static_cast<const std::uint8_t*>(dw);
            const auto* dxf = static_cast<const float*>(dx.p);
            auto* doutf = static_cast<float*>(dout.p);
            if (q6) {
                matvec_batch_q6_k_kernel<<<blocks, threads>>>(
                    dwb, dxf, doutf, w.rows, w.cols, n);
            } else {
                matvec_batch_q4_k_kernel<<<blocks, threads>>>(
                    dwb, dxf, doutf, w.rows, w.cols, n);
            }
            ok = cuda_ok(cudaGetLastError()) &&
                 cuda_ok(cudaDeviceSynchronize()) &&
                 cuda_ok(cudaMemcpy(out_batch.data(), dout.p,
                                    out_bytes,
                                    cudaMemcpyDeviceToHost));
        }
    }
    if (pooled) {
        pool().release(w.data);
    }
    return ok;
}

/** @returns true if a Q8_K-activation CUDA kernel exists for `type`. */
bool cuda_has_q8k_kernel(gguf::TensorType type) {
    switch (type) {
        case gguf::TensorType::kQ2_K:
        case gguf::TensorType::kQ4_K:
        case gguf::TensorType::kQ5_K:
        case gguf::TensorType::kQ6_K:
        case gguf::TensorType::kIQ2_XXS:
        case gguf::TensorType::kIQ2_XS:
        case gguf::TensorType::kIQ2_S:
        case gguf::TensorType::kTQ1_0:
        case gguf::TensorType::kTQ2_0:
        case gguf::TensorType::kIQ4_XS:
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
    const std::size_t nsb = x.size() / 256;
    std::vector<float> hd(nsb);
    std::vector<std::int8_t> hqs(x.size());
    std::vector<std::int16_t> hbs(nsb * 16);
    quantize_activation_q8k(x, hd.data(), hqs.data(), hbs.data());

    bool ok = false;
    {
        DevBuf dd, dqs, dbs, dout;
        const std::size_t out_bytes =
            static_cast<std::size_t>(w.rows) * sizeof(float);
        if (dd.alloc(nsb * sizeof(float)) &&
            dqs.alloc(x.size() * sizeof(std::int8_t)) &&
            dbs.alloc(nsb * 16 * sizeof(std::int16_t)) &&
            dout.alloc(out_bytes) &&
            cuda_ok(cudaMemcpy(dd.p, hd.data(), nsb * sizeof(float),
                               cudaMemcpyHostToDevice)) &&
            cuda_ok(cudaMemcpy(dqs.p, hqs.data(), x.size(),
                               cudaMemcpyHostToDevice)) &&
            cuda_ok(cudaMemcpy(dbs.p, hbs.data(),
                               nsb * 16 * sizeof(std::int16_t),
                               cudaMemcpyHostToDevice))) {
            const std::uint32_t threads = 256;
            const std::uint32_t blocks =
                (w.rows + threads - 1) / threads;
            const auto* dwb = static_cast<const std::uint8_t*>(dw);
            const auto* ddf = static_cast<const float*>(dd.p);
            const auto* dqsb = static_cast<const std::int8_t*>(dqs.p);
            const auto* dbsb = static_cast<const std::int16_t*>(dbs.p);
            auto* doutf = static_cast<float*>(dout.p);
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
                default:
                    break;
            }
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

}  // namespace

void cuda_pool_reset() { pool().reset(); }

void matvec_cuda_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out) {
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

void matvec_cuda(const Mat& w, std::span<const float> x,
                 std::span<float> out) {
    const std::size_t bytes = device_weight_bytes(w);
    bool done = false;
    if (bytes > 0) {
        // Grid/codebook kernels need their device tables uploaded;
        // if any upload failed we fall back to scalar for that call.
        CudaTables t;
        bool tables_ok = true;
        switch (w.type) {
            case gguf::TensorType::kIQ1_S:
            case gguf::TensorType::kIQ1_M:
                t.g64 = device_iq1s_grid();
                tables_ok = t.g64 != nullptr;
                break;
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
            case gguf::TensorType::kIQ3_S:
                t.g32 = device_iq3s_grid();
                tables_ok = t.g32 != nullptr;
                break;
            case gguf::TensorType::kIQ3_XXS:
                t.g32 = device_iq3xxs_grid();
                t.ksigns = device_ksigns();
                tables_ok = t.g32 != nullptr && t.ksigns != nullptr;
                break;
            case gguf::TensorType::kIQ4_XS:
            case gguf::TensorType::kIQ4_NL:
                t.kvalues = device_kvalues_iq4nl();
                tables_ok = t.kvalues != nullptr;
                break;
            default:
                break;  // F32/Q8_0/Q2_K/Q4_K/Q5_K/Q6_K need none
        }
        if (tables_ok) {
            done = run_matvec(w, x, out, bytes, kind_for(w.type), t);
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

void matvec_batch_cuda(const Mat& w, std::span<const float> x_batch,
                       std::span<float> out_batch, std::uint32_t n) {
    // Q4_K/Q6_K get the one-launch register-blocked kernel; other
    // types (incl F32/Q8_0) fall back to n matvec_cuda() calls
    // scattered to row-major -- the same-backend fallback rule, so
    // batched stays byte-identical to per-token.
    if (n <= kMaxBatch && (w.type == gguf::TensorType::kQ4_K ||
                           w.type == gguf::TensorType::kQ6_K)) {
        const std::size_t bytes = device_weight_bytes(w);
        if (run_batch(w, x_batch, out_batch, bytes, n,
                      w.type == gguf::TensorType::kQ6_K)) {
            return;
        }
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

}  // namespace locus::backend
