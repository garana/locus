#include "locus/backend/metal/seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "locus/backend/metal/context.hpp"
#include "locus/backend/metal/matvec.hpp"

namespace locus::backend::metal {

namespace {

// The parameter blocks of kernels.metal.
struct BiasParams {
    std::uint64_t off;
    std::uint32_t n, n_tok, stride;
};
struct SeqLnParams {
    std::uint32_t d, n, x_off, x_stride, o_off, o_stride, shift_off,
        scale_off;
    float eps;
    std::uint32_t affine;
};
struct SeqNormRopeParams {
    std::uint32_t n, heads, hd, x_off, x_stride, rope_row0;
    float eps;
};
struct SeqEltParams {
    std::uint32_t n, d, x_off, x_stride, y_off, y_stride, g_off, op;
};
struct SeqAttnParams {
    std::uint32_t n_q, n_kv, heads, head_dim;
    std::uint32_t q_off, q_stride, k_off, k_stride;
    std::uint32_t v_off, v_stride, o_off, o_stride;
    float scale;
};

template <typename P>
std::span<const std::byte> bytes_of(const P& p) {
    return std::as_bytes(std::span(&p, 1));
}

constexpr std::uint32_t kRowGroup = 256;  // seq_ln's threadgroup

}  // namespace

struct Seq::Impl {
    std::unique_lock<std::mutex> lk;
    Context& ctx;
    Context::Buffer slots[kSlots];
    std::size_t n[kSlots] = {};
    // This batch's constants (biases, norm weights): a grow-only buffer
    // and its floats in use; buffers outgrown mid-batch are kept until
    // the batch ends (its ops still read them).
    Context::Buffer consts;
    std::size_t consts_used = 0;
    std::vector<Context::Buffer> retired;

    Impl() : lk(device_mutex()), ctx(Context::get()) {}

    ~Impl() {
        for (auto& b : slots) {
            ctx.free(b);
        }
        ctx.free(consts);
        for (auto& b : retired) {
            ctx.free(b);
        }
    }

    const Context::Buffer& slot(Slot i) const {
        if (i >= kSlots || n[i] == 0) {
            throw std::invalid_argument("metal::Seq: slot not reserved");
        }
        return slots[i];
    }

    /** Copies v into this batch's constants; @returns the argument. */
    Context::Arg put(std::span<const float> v) {
        const std::size_t cap = consts.bytes / 4;
        if (consts.handle == nullptr || consts_used + v.size() > cap) {
            if (consts.handle != nullptr) {
                retired.push_back(consts);
            }
            consts = ctx.alloc(
                std::max(v.size(), 2 * cap + (std::size_t{1} << 18)) * 4);
            consts_used = 0;
        }
        const std::size_t at = consts_used;
        std::memcpy(static_cast<float*>(consts.data) + at, v.data(),
                    v.size_bytes());
        consts_used += (v.size() + 63) & ~std::size_t{63};
        return {&consts, at * 4};
    }

    void end() {
        ctx.end_batch();
        consts_used = 0;
        for (auto& b : retired) {
            ctx.free(b);
        }
        retired.clear();
    }

    void elt(Rows x, Rows y, std::uint32_t nr, std::uint32_t d,
             const Context::Arg& gate, std::uint32_t g_off,
             std::uint32_t op) {
        const SeqEltParams p{nr,    d,        x.off, x.stride,
                             y.off, y.stride, g_off, op};
        const Context::Arg a[] = {{&slot(x.slot), 0}, {&slot(y.slot), 0}, gate};
        ctx.dispatch("seq_elt", a, bytes_of(p), nr * d, 256);
    }
};

bool Seq::available() { return Context::usable(); }

Seq::Seq() {
    if (!Context::usable()) {
        throw std::runtime_error("metal::Seq: Metal is not usable");
    }
    im_ = std::make_unique<Impl>();
    im_->ctx.begin_batch();
}

Seq::~Seq() {
    if (im_->ctx.batch_open()) {
        im_->end();
    }
}

void Seq::reserve(Slot i, std::size_t n) {
    if (i >= kSlots || n == 0) {
        throw std::invalid_argument("metal::Seq: bad slot or size");
    }
    if (im_->n[i] >= n) {
        return;
    }
    // A bigger buffer: ops recorded on the old one still run, so submit
    // them first, then keep the contents.
    flush();
    Context::Buffer b = im_->ctx.alloc(n * 4);
    if (im_->n[i] > 0) {
        std::memcpy(b.data, im_->slots[i].data, im_->n[i] * 4);
    }
    im_->ctx.free(im_->slots[i]);
    im_->slots[i] = b;
    im_->n[i] = n;
}

void Seq::write(Slot i, std::size_t off, std::span<const float> v) {
    const auto& b = im_->slot(i);
    if (off + v.size() > im_->n[i]) {
        throw std::invalid_argument("metal::Seq: write past the slot");
    }
    flush();
    std::memcpy(static_cast<float*>(b.data) + off, v.data(), v.size_bytes());
}

void Seq::read(Slot i, std::size_t off, std::span<float> out) {
    const auto& b = im_->slot(i);
    if (off + out.size() > im_->n[i]) {
        throw std::invalid_argument("metal::Seq: read past the slot");
    }
    flush();
    std::memcpy(out.data(), static_cast<const float*>(b.data) + off,
                out.size_bytes());
}

void Seq::flush() {
    im_->end();
    im_->ctx.begin_batch();
}

void Seq::linear(const Mat& w, std::span<const float> b, Rows x, Rows out,
                 std::uint32_t n) {
    if (!b.empty() && b.size() != w.rows) {
        throw std::invalid_argument("metal::Seq::linear: bias size");
    }
    Context& ctx = im_->ctx;
    const auto& wb = cached_weight(ctx, w);
    const auto& xb = im_->slot(x.slot);
    const auto& ob = im_->slot(out.slot);
    if (!record_matmul(ctx, w.type, w.rows, w.cols, n, wb, 0, xb, x.off,
                       x.stride, ob, out.off, out.stride, false)) {
        for (std::uint32_t t = 0; t < n; ++t) {
            if (t > 0) {
                ctx.overlap_next();
            }
            record_matvec(ctx, w.type, w.rows, w.cols, wb, 0, xb,
                          x.off + t * x.stride, ob,
                          out.off + std::uint64_t{t} * out.stride, false);
        }
    }
    if (!b.empty()) {
        const Context::Arg bias = im_->put(b);
        const BiasParams p{out.off, w.rows, n, out.stride};
        const Context::Arg a[] = {{&ob, 0}, bias};
        ctx.dispatch("add_bias", a, bytes_of(p), n * w.rows, 256);
    }
}

void Seq::linear_add(const Mat& w, Rows x, Rows out, std::uint32_t n,
                     float scale) {
    Context& ctx = im_->ctx;
    if (!record_matmul(ctx, w.type, w.rows, w.cols, n, cached_weight(ctx, w),
                       0, im_->slot(x.slot), x.off, x.stride,
                       im_->slot(out.slot), out.off, out.stride, true,
                       scale)) {
        throw std::invalid_argument("metal::Seq::linear_add: no matmul");
    }
}

void Seq::ln_mod(Rows x, Rows out, std::uint32_t n, std::uint32_t d,
                 Slot mod, std::uint32_t shift_off, std::uint32_t scale_off,
                 float eps) {
    const SeqLnParams p{d,          n,          x.off,     x.stride,
                        out.off,    out.stride, shift_off, scale_off,
                        eps,        0};
    const Context::Arg a[] = {{&im_->slot(x.slot), 0},
                              {&im_->slot(mod), 0},
                              {&im_->slot(out.slot), 0}};
    im_->ctx.dispatch("seq_ln", a, bytes_of(p), n * kRowGroup, kRowGroup);
}

void Seq::layer_norm(Rows x, Rows out, std::uint32_t n, std::uint32_t d,
                     std::span<const float> w, std::span<const float> b,
                     float eps) {
    if (w.size() != d || b.size() != d) {
        throw std::invalid_argument("metal::Seq::layer_norm: w / b size");
    }
    std::vector<float> wb(std::size_t{2} * d);
    std::copy(w.begin(), w.end(), wb.begin());
    std::copy(b.begin(), b.end(), wb.begin() + d);
    const Context::Arg m = im_->put(wb);
    const SeqLnParams p{d,       n,          x.off, x.stride,
                        out.off, out.stride, d,     0,
                        eps,     1};
    const Context::Arg a[] = {{&im_->slot(x.slot), 0}, m,
                              {&im_->slot(out.slot), 0}};
    im_->ctx.dispatch("seq_ln", a, bytes_of(p), n * kRowGroup, kRowGroup);
}

void Seq::head_norm_rope(Rows x, std::uint32_t n, std::uint32_t heads,
                         std::uint32_t hd, std::span<const float> norm,
                         Slot cos_s, Slot sin_s, std::uint32_t rope_row0,
                         float eps) {
    if (norm.size() != hd) {
        throw std::invalid_argument("metal::Seq::head_norm_rope: norm size");
    }
    const Context::Arg w = im_->put(norm);
    const SeqNormRopeParams p{n, heads, hd, x.off, x.stride, rope_row0, eps};
    const Context::Arg a[] = {{&im_->slot(x.slot), 0},
                              w,
                              {&im_->slot(cos_s), 0},
                              {&im_->slot(sin_s), 0}};
    im_->ctx.dispatch("seq_head_norm_rope", a, bytes_of(p), n * heads * 32,
                      32);
}

void Seq::attention(Rows q, Rows k, Rows v, Rows out, std::uint32_t n,
                    std::uint32_t heads, std::uint32_t hd, Slot kh, Slot vt,
                    Slot scores) {
    cross_attention(q, n, k, v, n, out, heads, hd, kh, vt, scores);
}

void Seq::cross_attention(Rows q, std::uint32_t n_q, Rows k, Rows v,
                          std::uint32_t n_kv, Rows out, std::uint32_t heads,
                          std::uint32_t hd, Slot, Slot, Slot) {
    if (n_q == 0 || n_kv == 0 || hd == 0 || hd % 8 != 0 || hd > 128) {
        throw std::invalid_argument("metal::Seq::attention: bad sizes");
    }
    const SeqAttnParams p{n_q,      n_kv,     heads,   hd,
                          q.off,    q.stride, k.off,   k.stride,
                          v.off,    v.stride, out.off, out.stride,
                          1.0f / std::sqrt(static_cast<float>(hd))};
    const Context::Arg a[] = {{&im_->slot(q.slot), 0},
                              {&im_->slot(k.slot), 0},
                              {&im_->slot(v.slot), 0},
                              {&im_->slot(out.slot), 0}};
    // 64 queries of a head a threadgroup of 256 (kSeqAttnSg in
    // kernels.metal).
    im_->ctx.dispatch("seq_attn_mma", a, bytes_of(p),
                      ((n_q + 63) / 64) * heads * 256, 256);
}

void Seq::gelu_tanh(Rows x, std::uint32_t n, std::uint32_t d) {
    const auto& xb = im_->slot(x.slot);
    im_->elt(x, x, n, d, {&xb, 0}, 0, 2);
}

void Seq::gelu_erf(Rows x, std::uint32_t n, std::uint32_t d) {
    const auto& xb = im_->slot(x.slot);
    im_->elt(x, x, n, d, {&xb, 0}, 0, 3);
}

void Seq::add(Rows x, Rows y, std::uint32_t n, std::uint32_t d) {
    const auto& xb = im_->slot(x.slot);
    im_->elt(x, y, n, d, {&xb, 0}, 0, 1);
}

void Seq::gated_add(Rows x, Rows y, std::uint32_t n, std::uint32_t d,
                    Slot gate_s, std::uint32_t gate_off) {
    im_->elt(x, y, n, d, {&im_->slot(gate_s), 0}, gate_off, 0);
}

}  // namespace locus::backend::metal
