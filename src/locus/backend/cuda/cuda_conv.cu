// CUDA convolutions for the generative-media decoders (F32): conv1d and
// its transpose and snake1d (codec decoders), conv2d (the image VQ
// decoder, i#223). Each call uploads its inputs, runs one kernel and
// copies the output back; any CUDA error or unsupported shape falls back
// to the scalar reference in cpu_ops.cpp.

#include "locus/backend/cuda/cuda_common.cuh"
#include "locus/backend/fallbacks.hpp"
#include "locus/backend/variants.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace locus::backend {

using namespace cuda_detail;

namespace {

/** A convolution's device buffers: its input, weight and optional
 *  bias uploaded, and room for its output. */
struct ConvBufs {
    DevBuf in, w, b, out;
    const float* bias = nullptr;  ///< nullptr when there is no bias

    /** @returns false on a failed allocation or upload. */
    bool upload(std::span<const float> x, std::span<const float> weight,
                std::span<const float> bias_h, std::size_t out_n) {
        if (!in.alloc(x.size() * sizeof(float)) ||
            !w.alloc(weight.size() * sizeof(float)) ||
            !out.alloc(out_n * sizeof(float))) {
            return false;
        }
        if (!bias_h.empty()) {
            if (!b.alloc(bias_h.size() * sizeof(float))) {
                return false;
            }
            bias = static_cast<const float*>(b.p);
        }
        if (!cuda_ok(cudaMemcpy(in.p, x.data(), x.size() * sizeof(float),
                                cudaMemcpyHostToDevice)) ||
            !cuda_ok(cudaMemcpy(w.p, weight.data(),
                                weight.size() * sizeof(float),
                                cudaMemcpyHostToDevice))) {
            return false;
        }
        return bias == nullptr ||
               cuda_ok(cudaMemcpy(b.p, bias_h.data(),
                                  bias_h.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
    }
};

// --- conv1d family (generative-media codec decoders, F32) ----------
// One thread per output element; the per-element math mirrors the
// scalar reference in cpu_ops.cpp so the two agree within fp32 rounding.
// Device buffers use the DevBuf RAII helper (cuda_common.cuh).

__global__ void conv1d_kernel(const float* in, const float* weight,
                              const float* bias, float* out,
                              std::uint32_t in_len, std::uint32_t out_ch,
                              std::uint32_t out_len, std::uint32_t k,
                              std::uint32_t stride, std::uint32_t dilation,
                              std::uint32_t pad, std::uint32_t icg,
                              std::uint32_t ocg) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_ch * out_len) {
        return;
    }
    const std::uint32_t oc = idx / out_len;
    const std::uint32_t ox = idx % out_len;
    const std::uint32_t ic0 = (oc / ocg) * icg;  // group's first in-ch
    float acc = bias != nullptr ? bias[oc] : 0.0f;
    const long long base =
        static_cast<long long>(ox) * stride - static_cast<long long>(pad);
    for (std::uint32_t ci = 0; ci < icg; ++ci) {
        const float* inrow = in + static_cast<std::size_t>(ic0 + ci) * in_len;
        const float* wrow =
            weight + (static_cast<std::size_t>(oc) * icg + ci) * k;
        for (std::uint32_t t = 0; t < k; ++t) {
            const long long ix = base + static_cast<long long>(t) * dilation;
            if (ix >= 0 && ix < static_cast<long long>(in_len)) {
                acc += wrow[t] * inrow[ix];
            }
        }
    }
    out[static_cast<std::size_t>(oc) * out_len + ox] = acc;
}

__global__ void conv1d_transpose_kernel(
    const float* in, const float* weight, const float* bias, float* out,
    std::uint32_t in_len, std::uint32_t out_ch, std::uint32_t out_len,
    std::uint32_t k, std::uint32_t stride, std::uint32_t dilation,
    std::uint32_t pad, std::uint32_t icg, std::uint32_t ocg) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_ch * out_len) {
        return;
    }
    const std::uint32_t oc = idx / out_len;
    const std::uint32_t ox = idx % out_len;
    const std::uint32_t g = oc / ocg;    // this oc's group
    const std::uint32_t ocl = oc % ocg;  // its index in the group
    const std::uint32_t ic0 = g * icg;   // group's first in-ch
    float acc = bias != nullptr ? bias[oc] : 0.0f;
    for (std::uint32_t t = 0; t < k; ++t) {
        // ox = ix*stride - pad + t*dilation => ix = (ox + pad -
        // t*dilation) / stride, when that divides evenly.
        const long long num = static_cast<long long>(ox) + pad -
                              static_cast<long long>(t) * dilation;
        if (num < 0 || num % static_cast<long long>(stride) != 0) {
            continue;
        }
        const long long ix = num / static_cast<long long>(stride);
        if (ix < 0 || ix >= static_cast<long long>(in_len)) {
            continue;
        }
        for (std::uint32_t ci = 0; ci < icg; ++ci) {
            const float inv = in[static_cast<std::size_t>(ic0 + ci) * in_len +
                                 static_cast<std::size_t>(ix)];
            const float wv =
                weight[((static_cast<std::size_t>(ic0 + ci) * ocg) + ocl) * k +
                       t];
            acc += inv * wv;
        }
    }
    out[static_cast<std::size_t>(oc) * out_len + ox] = acc;
}

__global__ void snake1d_kernel(float* x, const float* alpha,
                               std::uint32_t ch, std::uint32_t len,
                               float eps) {
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ch * len) {
        return;
    }
    const float a = alpha[idx / len];
    const float s = sinf(a * x[idx]);
    x[idx] += (1.0f / (a + eps)) * s * s;
}

/** Uploads in/weight/(bias), runs one conv kernel, copies out back;
 *  false on any CUDA error (the caller then runs the scalar path). */
bool run_conv1d_dev(bool transpose, std::span<const float> in,
                    std::uint32_t in_ch, std::uint32_t in_len,
                    std::span<const float> weight, std::uint32_t out_ch,
                    std::uint32_t k, std::span<const float> bias,
                    std::uint32_t stride, std::uint32_t dilation,
                    std::uint32_t pad, std::uint32_t groups,
                    std::span<float> out) {
    if (groups == 0 || in_ch % groups != 0 || out_ch % groups != 0 ||
        stride == 0) {
        return false;  // stride 0 would divide by zero below / on device
    }
    const std::uint32_t icg = in_ch / groups;   // in-channels per group
    const std::uint32_t ocg = out_ch / groups;  // out-channels per group
    // Spans must match the layout the kernel indexes; a short weight or
    // bias would make the device read past the buffer (and can wedge the
    // context), so reject here rather than launch.
    const std::size_t want_w = transpose
                                   ? std::size_t{in_ch} * ocg * k
                                   : std::size_t{out_ch} * icg * k;
    if (in.size() != std::size_t{in_ch} * in_len ||
        weight.size() != want_w ||
        (!bias.empty() && bias.size() != out_ch)) {
        return false;
    }
    const std::uint32_t out_len =
        transpose ? conv1d_transpose_out_len(in_len, k, stride, dilation, pad)
                  : conv1d_out_len(in_len, k, stride, dilation, pad);
    if (out.size() != static_cast<std::size_t>(out_ch) * out_len) {
        return false;
    }
    if (out.empty()) {
        return true;  // out_ch == 0 or out_len == 0: nothing to compute
    }
    ConvBufs d;
    if (!d.upload(in, weight, bias, out.size())) {
        return false;
    }
    const std::uint32_t threads = 256;
    const std::uint32_t total = out_ch * out_len;
    const std::uint32_t blocks = (total + threads - 1) / threads;
    if (transpose) {
        conv1d_transpose_kernel<<<blocks, threads>>>(
            static_cast<const float*>(d.in.p),
            static_cast<const float*>(d.w.p), d.bias,
            static_cast<float*>(d.out.p), in_len, out_ch, out_len, k,
            stride, dilation, pad, icg, ocg);
    } else {
        conv1d_kernel<<<blocks, threads>>>(
            static_cast<const float*>(d.in.p),
            static_cast<const float*>(d.w.p), d.bias,
            static_cast<float*>(d.out.p), in_len, out_ch, out_len, k,
            stride, dilation, pad, icg, ocg);
    }
    if (!cuda_ok(cudaGetLastError()) || !sync_counted()) {
        return false;
    }
    return cuda_ok(cudaMemcpy(out.data(), d.out.p,
                              out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
}

bool run_snake1d_dev(std::span<float> x, std::uint32_t ch,
                     std::uint32_t len, std::span<const float> alpha,
                     float eps) {
    if (x.size() != static_cast<std::size_t>(ch) * len ||
        alpha.size() != ch) {
        return false;
    }
    if (x.empty()) {
        return true;  // nothing to do
    }
    DevBuf dx, da;
    if (!dx.alloc(x.size() * sizeof(float)) ||
        !da.alloc(alpha.size() * sizeof(float))) {
        return false;
    }
    if (!cuda_ok(cudaMemcpy(dx.p, x.data(), x.size() * sizeof(float),
                            cudaMemcpyHostToDevice)) ||
        !cuda_ok(cudaMemcpy(da.p, alpha.data(), alpha.size() * sizeof(float),
                            cudaMemcpyHostToDevice))) {
        return false;
    }
    const std::uint32_t threads = 256;
    const std::uint32_t total = ch * len;
    const std::uint32_t blocks = (total + threads - 1) / threads;
    snake1d_kernel<<<blocks, threads>>>(static_cast<float*>(dx.p),
                                        static_cast<const float*>(da.p), ch,
                                        len, eps);
    if (!cuda_ok(cudaGetLastError()) || !sync_counted()) {
        return false;
    }
    return cuda_ok(cudaMemcpy(x.data(), dx.p, x.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
}

// --- conv2d (image VQ decoder, F32, i#223) ---------------------------
// One thread per output element, summing in the scalar reference's
// order (bias, then ci, ky, kx, skipping padded taps). __fmul_rn /
// __fadd_rn stop nvcc from fusing the multiply-add, so the result is
// bit-identical to backend::conv2d on a host whose scalar conv2d cannot
// fuse either (the x86 baseline has no FMA; aarch64 and FMA x86 builds
// fuse the scalar reference), not just within fp32 rounding.

__global__ void conv2d_kernel(const float* in, const float* weight,
                              const float* bias, float* out,
                              std::uint32_t in_h, std::uint32_t in_w,
                              std::uint32_t out_ch, std::uint32_t out_h,
                              std::uint32_t out_w, std::uint32_t kh,
                              std::uint32_t kw, std::uint32_t stride,
                              std::uint32_t pad, std::uint32_t icg,
                              std::uint32_t ocg) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t oplane = static_cast<std::size_t>(out_h) * out_w;
    if (idx >= out_ch * oplane) {
        return;
    }
    const auto oc = static_cast<std::uint32_t>(idx / oplane);
    const std::size_t rem = idx % oplane;
    const auto oy = static_cast<std::uint32_t>(rem / out_w);
    const auto ox = static_cast<std::uint32_t>(rem % out_w);
    const std::uint32_t ic0 = (oc / ocg) * icg;  // group's first in-ch
    const std::size_t iplane = static_cast<std::size_t>(in_h) * in_w;
    const long long by =
        static_cast<long long>(oy) * stride - static_cast<long long>(pad);
    const long long bx =
        static_cast<long long>(ox) * stride - static_cast<long long>(pad);
    float acc = bias != nullptr ? bias[oc] : 0.0f;
    for (std::uint32_t ci = 0; ci < icg; ++ci) {
        const float* ip = in + (ic0 + ci) * iplane;
        const float* wp =
            weight + (static_cast<std::size_t>(oc) * icg + ci) * kh * kw;
        for (std::uint32_t ky = 0; ky < kh; ++ky) {
            const long long iy = by + ky;
            if (iy < 0 || iy >= static_cast<long long>(in_h)) {
                continue;
            }
            for (std::uint32_t kx = 0; kx < kw; ++kx) {
                const long long ix = bx + kx;
                if (ix >= 0 && ix < static_cast<long long>(in_w)) {
                    acc = __fadd_rn(acc, __fmul_rn(wp[ky * kw + kx],
                                                   ip[iy * in_w + ix]));
                }
            }
        }
    }
    out[idx] = acc;
}

/** Uploads in/weight/(bias), runs conv2d_kernel, copies out back. The
 *  caller has run conv2d_check_args, so false means only a device
 *  problem (a grid past the limit, or any CUDA error); the caller then
 *  runs the scalar path. */
bool run_conv2d_dev(std::span<const float> in, std::uint32_t in_ch,
                    std::uint32_t in_h, std::uint32_t in_w,
                    std::span<const float> weight, std::uint32_t out_ch,
                    std::uint32_t kh, std::uint32_t kw,
                    std::span<const float> bias, std::uint32_t stride,
                    std::uint32_t pad, std::uint32_t groups,
                    std::span<float> out) {
    const std::uint32_t icg = in_ch / groups;   // in-channels per group
    const std::uint32_t ocg = out_ch / groups;  // out-channels per group
    const std::uint32_t out_h = conv1d_out_len(in_h, kh, stride, 1, pad);
    const std::uint32_t out_w = conv1d_out_len(in_w, kw, stride, 1, pad);
    if (out.empty()) {
        return true;  // out_ch, out_h or out_w is 0: nothing to compute
    }
    const std::size_t threads = 256;
    const std::size_t blocks = (out.size() + threads - 1) / threads;
    if (blocks > 0x7fffffffu) {
        return false;  // past sm_50's gridDim.x limit (2^31 - 1)
    }
    ConvBufs d;
    if (!d.upload(in, weight, bias, out.size())) {
        return false;
    }
    conv2d_kernel<<<static_cast<unsigned>(blocks),
                    static_cast<unsigned>(threads)>>>(
        static_cast<const float*>(d.in.p), static_cast<const float*>(d.w.p),
        d.bias, static_cast<float*>(d.out.p), in_h, in_w, out_ch, out_h,
        out_w, kh, kw, stride, pad, icg, ocg);
    if (!cuda_ok(cudaGetLastError()) || !sync_counted()) {
        return false;
    }
    return cuda_ok(cudaMemcpy(out.data(), d.out.p,
                              out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
}

}  // namespace

void conv1d_cuda(std::span<const float> in, std::uint32_t in_ch,
                 std::uint32_t in_len, std::span<const float> weight,
                 std::uint32_t out_ch, std::uint32_t k,
                 std::span<const float> bias, std::uint32_t stride,
                 std::uint32_t dilation, std::uint32_t pad,
                 std::uint32_t groups, std::span<float> out) {
    ensure_device();
    if (!run_conv1d_dev(false, in, in_ch, in_len, weight, out_ch, k, bias,
                        stride, dilation, pad, groups, out)) {
        static Fallback& f = fallback(
            "cuda", "cpu", "conv1d device error or unsupported shape");
        f.add();
        conv1d(in, in_ch, in_len, weight, out_ch, k, bias, stride, dilation,
               pad, groups, out);
    }
}

void conv1d_transpose_cuda(std::span<const float> in,
                           std::uint32_t in_ch, std::uint32_t in_len,
                           std::span<const float> weight,
                           std::uint32_t out_ch, std::uint32_t k,
                           std::span<const float> bias,
                           std::uint32_t stride, std::uint32_t dilation,
                           std::uint32_t pad, std::uint32_t groups,
                           std::span<float> out) {
    ensure_device();
    if (!run_conv1d_dev(true, in, in_ch, in_len, weight, out_ch, k, bias,
                        stride, dilation, pad, groups, out)) {
        static Fallback& f = fallback(
            "cuda", "cpu",
            "conv1d_transpose device error or unsupported shape");
        f.add();
        conv1d_transpose(in, in_ch, in_len, weight, out_ch, k, bias, stride,
                         dilation, pad, groups, out);
    }
}

void snake1d_cuda(std::span<float> x, std::uint32_t ch,
                  std::uint32_t len, std::span<const float> alpha,
                  float eps) {
    ensure_device();
    if (!run_snake1d_dev(x, ch, len, alpha, eps)) {
        static Fallback& f = fallback(
            "cuda", "cpu", "snake1d device error or unsupported shape");
        f.add();
        snake1d(x, ch, len, alpha, eps);
    }
}

void conv2d_cuda(std::span<const float> in, std::uint32_t in_ch,
                 std::uint32_t in_h, std::uint32_t in_w,
                 std::span<const float> weight, std::uint32_t out_ch,
                 std::uint32_t kh, std::uint32_t kw,
                 std::span<const float> bias, std::uint32_t stride,
                 std::uint32_t pad, std::uint32_t groups,
                 std::span<float> out) {
    // A bad shape throws here; the scalar fallback below is only for a
    // device failure, never for arguments conv2d would misread.
    conv2d_check_args(in, in_ch, in_h, in_w, weight, out_ch, kh, kw, bias,
                      stride, pad, groups, out);
    ensure_device();
    if (!run_conv2d_dev(in, in_ch, in_h, in_w, weight, out_ch, kh, kw, bias,
                        stride, pad, groups, out)) {
        static Fallback& f = fallback(
            "cuda", "cpu", "conv2d device error or unsupported shape");
        f.add();
        conv2d(in, in_ch, in_h, in_w, weight, out_ch, kh, kw, bias, stride,
               pad, groups, out);
    }
}

}  // namespace locus::backend
