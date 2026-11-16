#include "locus/backend/metal/conv.hpp"

#include <cstring>
#include <mutex>
#include <stdexcept>

#include "locus/backend/metal/context.hpp"
#include "locus/backend/metal/matvec.hpp"

namespace locus::backend::metal {

namespace {

/** Conv2dParams in kernels.metal. */
struct Conv2dParams {
    std::uint32_t in_ch, in_h, in_w, out_ch, oh, ow, kh, kw, stride, pad;
    std::uint32_t has_bias, K, P;
};

/** The conv's shared buffers, grown as needed (device_mutex). */
struct ConvScratch {
    Context::Buffer in, w, b, out, tab;
};

void grow(Context& ctx, Context::Buffer& b, std::size_t bytes) {
    if (b.handle == nullptr || b.bytes < bytes) {
        ctx.free(b);
        b = ctx.alloc(bytes);
    }
}

void put(Context::Buffer& b, std::span<const float> v) {
    std::memcpy(b.data, v.data(), v.size_bytes());
}

}  // namespace

bool conv2d(std::span<const float> in, std::uint32_t in_ch,
            std::uint32_t in_h, std::uint32_t in_w,
            std::span<const float> weight, std::uint32_t out_ch,
            std::uint32_t kh, std::uint32_t kw, std::span<const float> bias,
            std::uint32_t stride, std::uint32_t pad, std::uint32_t groups,
            std::span<float> out) {
    if (groups != 1 || !Context::usable()) {
        return false;
    }
    if (stride == 0 || kh == 0 || kw == 0 || kh > 0xffff || kw > 0xffff ||
        in_h + 2 * pad < kh || in_w + 2 * pad < kw) {
        throw std::invalid_argument("metal conv2d: bad kernel or stride");
    }
    const std::uint32_t oh = (in_h + 2 * pad - kh) / stride + 1;
    const std::uint32_t ow = (in_w + 2 * pad - kw) / stride + 1;
    const std::uint32_t K = in_ch * kh * kw, P = oh * ow;
    if (in.size() != std::size_t{in_ch} * in_h * in_w ||
        weight.size() != std::size_t{out_ch} * K ||
        out.size() != std::size_t{out_ch} * P ||
        (!bias.empty() && bias.size() != out_ch)) {
        throw std::invalid_argument("metal conv2d: span sizes");
    }
    if (out_ch == 0 || P == 0) {
        return true;
    }
    std::lock_guard lk(device_mutex());
    Context& ctx = Context::get();
    static ConvScratch s;
    grow(ctx, s.in, in.size_bytes());
    grow(ctx, s.w, weight.size_bytes());
    grow(ctx, s.b, bias.empty() ? 4 : bias.size_bytes());
    grow(ctx, s.out, out.size_bytes());
    // k's input offset and (ky, kx), so the kernel divides nothing.
    grow(ctx, s.tab, std::size_t{K} * 8);
    auto* tab = static_cast<std::uint32_t*>(s.tab.data);
    for (std::uint32_t k = 0; k < K; ++k) {
        const std::uint32_t ci = k / (kh * kw), r = k % (kh * kw);
        const std::uint32_t ky = r / kw, kx = r % kw;
        tab[2 * k] = (ci * in_h + ky) * in_w + kx;
        tab[2 * k + 1] = ky << 16 | kx;
    }
    put(s.in, in);
    put(s.w, weight);
    put(s.b, bias);
    const Conv2dParams prm{in_ch, in_h,   in_w, out_ch, oh,
                           ow,    kh,     kw,   stride, pad,
                           bias.empty() ? 0u : 1u, K, P};
    const std::uint32_t tiles =
        ((P + 31) / 32) * ((out_ch + 63) / 64);
    const Context::Arg a[] = {
        {&s.in, 0}, {&s.w, 0}, {&s.b, 0}, {&s.out, 0}, {&s.tab, 0}};
    ctx.dispatch("conv2d_gemm", a,
                 std::as_bytes(std::span<const Conv2dParams>(&prm, 1)),
                 tiles * 128, 128);
    std::memcpy(out.data(), s.out.data, out.size_bytes());
    return true;
}

}  // namespace locus::backend::metal
