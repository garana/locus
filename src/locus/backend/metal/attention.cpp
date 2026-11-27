#include "locus/backend/metal/attention.hpp"

#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>

#include "locus/backend/metal/context.hpp"
#include "locus/backend/metal/matvec.hpp"

namespace locus::backend::metal {

namespace {

/** SeqAttnParams in kernels.metal. */
struct SeqAttnParams {
    std::uint32_t n_q, n_kv, heads, head_dim;
    std::uint32_t q_off, q_stride, k_off, k_stride;
    std::uint32_t v_off, v_stride, o_off, o_stride;
    float scale;
};

/** The attention's shared buffers, grown as needed (device_mutex). */
struct AttnScratch {
    Context::Buffer q, k, v, out;
};

void grow(Context& ctx, Context::Buffer& b, std::size_t bytes) {
    if (b.handle == nullptr || b.bytes < bytes) {
        ctx.free(b);
        b = ctx.alloc(bytes);
    }
}

}  // namespace

bool attention(std::span<const float> q, std::uint32_t nq,
               std::span<const float> k, std::span<const float> v,
               std::uint32_t nk, std::uint32_t heads, std::uint32_t hd,
               std::span<float> out) {
    if (!Context::usable() || hd == 0 || hd % 8 != 0 || hd > 128) {
        return false;
    }
    const std::size_t w = std::size_t{heads} * hd;
    if (q.size() != std::size_t{nq} * w || k.size() != std::size_t{nk} * w ||
        v.size() != k.size() || out.size() != q.size()) {
        throw std::invalid_argument("metal attention: span sizes");
    }
    if (nq == 0 || heads == 0) {
        return true;
    }
    if (nk == 0) {
        throw std::invalid_argument("metal attention: no keys");
    }
    std::lock_guard lk(device_mutex());
    Context& ctx = Context::get();
    static AttnScratch s;
    grow(ctx, s.q, q.size_bytes());
    grow(ctx, s.k, k.size_bytes());
    grow(ctx, s.v, v.size_bytes());
    grow(ctx, s.out, out.size_bytes());
    std::memcpy(s.q.data, q.data(), q.size_bytes());
    std::memcpy(s.k.data, k.data(), k.size_bytes());
    std::memcpy(s.v.data, v.data(), v.size_bytes());
    const auto W = static_cast<std::uint32_t>(w);
    const SeqAttnParams p{nq, nk, heads, hd, 0, W, 0, W, 0, W, 0, W,
                          1.0f / std::sqrt(static_cast<float>(hd))};
    const Context::Arg a[] = {
        {&s.q, 0}, {&s.k, 0}, {&s.v, 0}, {&s.out, 0}};
    // 64 queries of a head a threadgroup of 256 (kSeqAttnSg).
    ctx.dispatch("seq_attn_mma", a,
                 std::as_bytes(std::span<const SeqAttnParams>(&p, 1)),
                 ((nq + 63) / 64) * heads * 256, 256);
    std::memcpy(out.data(), s.out.data, out.size_bytes());
    return true;
}

}  // namespace locus::backend::metal
