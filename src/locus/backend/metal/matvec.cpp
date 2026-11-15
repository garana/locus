// The Metal matvec (i#866); see matvec.hpp.
#include "locus/backend/metal/matvec.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "locus/backend/metal/context.hpp"
#include "locus/backend/registry.hpp"

namespace locus::backend::metal {

namespace {

using gguf::TensorType;

/** A matvec kernel, its weight block (elements and bytes) and its
 *  threadgroup (threads and the rows it takes). */
struct Kind {
    const char* kernel = nullptr;
    std::uint32_t block = 0;
    std::uint32_t bytes = 0;
    std::uint32_t group = 128;  // 4 simdgroups
    std::uint32_t rows = 8;
};

Kind kind_of(TensorType t) {
    switch (t) {
        case TensorType::kF32:
            return {"mv_f32", 4, 16};
        case TensorType::kF16:
            return {"mv_f16", 4, 8};
        case TensorType::kBF16:
            return {"mv_bf16", 4, 8};
        case TensorType::kQ8_0:
            return {"mv_q8_0", 32, 34};
        case TensorType::kQ4_0:
            return {"mv_q4_0", 32, 18};
        case TensorType::kQ5_0:
            return {"mv_q5_0", 32, 22};
        case TensorType::kQ5_K:
            return {"mv_q5_k", 256, 176};
        case TensorType::kQ4_K:
            return {"mv_q4_k", 256, 144, 64, 4};
        case TensorType::kQ6_K:
            return {"mv_q6_k", 256, 210};
        default:
            return {};
    }
}

/** MvParams in kernels.metal. */
struct MvParams {
    std::uint64_t w_off, out_off;
    std::uint32_t rows, cols, accumulate, x_off;
    float scale;
};


/** A cached weight: its buffer and the matrix it was made from. */
struct Cached {
    Context::Buffer buf;
    gguf::TensorType type;
    std::uint32_t rows, cols;
    std::uint64_t print;  // fingerprint() of the bytes when uploaded
};

/**
 * FNV-1a over three 64-byte windows of a weight's bytes (its start,
 * middle and end): another matrix of the same type and shape that
 * turns up at a freed one's address (a model unloaded and another
 * mapped there, a temporary) almost surely differs in one of them, so
 * the cache uploads it afresh instead of serving the old bytes.
 */
std::uint64_t fingerprint(const std::byte* p, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto mix = [&](std::size_t at) {
        const std::size_t end = std::min(n, at + 64);
        for (std::size_t i = at; i < end; ++i) {
            h = (h ^ static_cast<std::uint8_t>(p[i])) * 1099511628211ULL;
        }
    };
    mix(0);
    mix(n / 2 > 32 ? n / 2 - 32 : 0);
    mix(n > 64 ? n - 64 : 0);
    return h;
}

/** Weights by host pointer, and the x / out scratch buffers. */
struct State {
    std::unordered_map<const std::byte*, Cached> weights;
    Context::Buffer x, out;
};

State& state() {
    static State s;
    return s;
}

/** Makes b hold at least n bytes (contents not kept). */
void grow(Context& ctx, Context::Buffer& b, std::size_t n) {
    if (b.handle == nullptr || b.bytes < n) {
        ctx.free(b);
        b = ctx.alloc(n);
    }
}

std::size_t weight_bytes(const Mat& w, const Kind& k) {
    return static_cast<std::size_t>(w.rows) * (w.cols / k.block) * k.bytes;
}

const Context::Buffer& weight(Context& ctx, const Mat& w, const Kind& k) {
    auto& ws = state().weights;
    const std::size_t bytes = weight_bytes(w, k);
    const std::uint64_t print = fingerprint(w.data, bytes);
    const auto it = ws.find(w.data);
    if (it != ws.end()) {
        Cached& c = it->second;
        if (c.type == w.type && c.rows == w.rows && c.cols == w.cols &&
            c.print == print) {
            return c.buf;
        }
        // Another matrix at a freed one's address: upload it afresh.
        ctx.free(c.buf);
        ws.erase(it);
    }
    Cached c{ctx.upload({w.data, bytes}), w.type, w.rows, w.cols, print};
    return ws.emplace(w.data, c).first->second.buf;
}

/** Checks the shape and stages x; @returns the kernel's kind. */
Kind prepare(Context& ctx, const Mat& w, std::span<const float> x,
             std::span<float> out) {
    if (x.size() != w.cols || out.size() != w.rows) {
        throw std::invalid_argument("metal matvec: span sizes");
    }
    const Kind k = kind_of(w.type);
    if (k.kernel == nullptr || w.cols % k.block != 0) {
        throw std::invalid_argument("metal matvec: no kernel");
    }
    State& s = state();
    grow(ctx, s.x, x.size_bytes());
    grow(ctx, s.out, out.size_bytes());
    std::memcpy(s.x.data, x.data(), x.size_bytes());
    return k;
}

void record(Context& ctx, const Mat& w, const Kind& k) {
    State& s = state();
    record_matvec(ctx, w.type, w.rows, w.cols, weight(ctx, w, k), 0, s.x,
                  0, s.out, 0, false);
}

}  // namespace

bool matvec_supported(gguf::TensorType t) {
    return kind_of(t).kernel != nullptr;
}

bool matvec_shape_ok(gguf::TensorType t, std::uint32_t cols) {
    const Kind k = kind_of(t);
    return k.kernel != nullptr && cols % k.block == 0;
}

void record_matvec(Context& ctx, gguf::TensorType t, std::uint32_t rows,
                   std::uint32_t cols, const Context::Buffer& w,
                   std::uint64_t w_off, const Context::Buffer& x,
                   std::uint32_t x_off, const Context::Buffer& out,
                   std::uint64_t out_off, bool accumulate) {
    const Kind k = kind_of(t);
    if (k.kernel == nullptr || cols % k.block != 0 || x_off % 4 != 0) {
        throw std::invalid_argument("metal record_matvec: no kernel");
    }
    const MvParams p{w_off, out_off, rows, cols, accumulate ? 1u : 0u,
                     x_off, 1.0f};
    const Context::Arg args[] = {{&w, 0}, {&x, 0}, {&out, 0}};
    const std::uint32_t groups = (rows + k.rows - 1) / k.rows;
    ctx.dispatch(k.kernel, args, std::as_bytes(std::span(&p, 1)),
                 groups * k.group, k.group);
}

namespace {

/** MmParams in kernels.metal. */
struct MmParams {
    std::uint64_t w_off, x_off, out_off;
    std::uint32_t rows, cols, n_tok, x_stride, out_stride, accumulate;
    float scale;
    std::uint32_t row_tiles;
};

}  // namespace

bool record_matmul(Context& ctx, gguf::TensorType t, std::uint32_t rows,
                   std::uint32_t cols, std::uint32_t n_tok,
                   const Context::Buffer& w, std::uint64_t w_off,
                   const Context::Buffer& x, std::uint64_t x_off,
                   std::uint32_t x_stride, const Context::Buffer& out,
                   std::uint64_t out_off, std::uint32_t out_stride,
                   bool accumulate) {
    using TT = gguf::TensorType;
    const char* kernel = t == TT::kQ4_K   ? "mm_q4_k"
                         : t == TT::kQ6_K ? "mm_q6_k"
                         : t == TT::kQ8_0 ? "mm_q8_0"
                         : t == TT::kF32  ? "mm_f32"
                         : t == TT::kF16  ? "mm_f16"
                         : t == TT::kBF16 ? "mm_bf16"
                                          : nullptr;
    // A k stage is 32 columns: the K-quants need whole 256-blocks.
    const std::uint32_t block =
        t == TT::kQ4_K || t == TT::kQ6_K ? 256 : 32;
    if (kernel == nullptr || cols % block != 0 || n_tok == 0) {
        return false;
    }
    const std::uint32_t row_tiles = (rows + 63) / 64;
    const std::uint32_t tok_tiles = (n_tok + 31) / 32;
    const MmParams p{w_off,    x_off,      out_off,
                     rows,     cols,       n_tok,
                     x_stride, out_stride, accumulate ? 1u : 0u,
                     1.0f,     row_tiles};
    const Context::Arg args[] = {{&w, 0}, {&x, 0}, {&out, 0}};
    ctx.dispatch(kernel, args, std::as_bytes(std::span(&p, 1)),
                 row_tiles * tok_tiles * 128, 128);
    return true;
}

std::size_t matvec_weight_bytes(gguf::TensorType t, std::uint32_t rows,
                                std::uint32_t cols) {
    const Kind k = kind_of(t);
    return k.kernel == nullptr
               ? 0
               : std::size_t{rows} * (cols / k.block) * k.bytes;
}

void matvec(const Mat& w, std::span<const float> x, std::span<float> out) {
    const Kind kd = kind_of(w.type);
    if (kd.kernel == nullptr || w.cols % kd.block != 0) {
        backend::matvec(w, x, out);  // no kernel for this type
        return;
    }
    std::lock_guard lk(device_mutex());
    Context& ctx = Context::get();
    const Kind k = prepare(ctx, w, x, out);
    record(ctx, w, k);
    std::memcpy(out.data(), state().out.data, out.size_bytes());
}

double matvec_repeat(const Mat& w, std::span<const float> x,
                     std::span<float> out, std::uint32_t reps) {
    if (reps == 0) {
        throw std::invalid_argument("metal matvec_repeat: reps 0");
    }
    std::lock_guard lk(device_mutex());
    Context& ctx = Context::get();
    const Kind k = prepare(ctx, w, x, out);
    weight(ctx, w, k);  // uploaded before the clock starts
    ctx.begin_batch();
    try {
        for (std::uint32_t i = 0; i < reps; ++i) {
            record(ctx, w, k);
        }
    } catch (...) {
        ctx.end_batch();
        throw;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ctx.end_batch();
    const auto t1 = std::chrono::steady_clock::now();
    std::memcpy(out.data(), state().out.data, out.size_bytes());
    return std::chrono::duration<double, std::milli>(t1 - t0).count() /
           reps;
}

void matvec_batch(const Mat& w, std::span<const float> x,
                  std::span<float> out, std::uint32_t n) {
    if (n == 0) {
        return;
    }
    if (x.size() < std::size_t{n} * w.cols ||
        out.size() < std::size_t{n} * w.rows) {
        throw std::invalid_argument("metal matvec_batch: span sizes");
    }
    const Kind k = kind_of(w.type);
    if (k.kernel == nullptr || w.cols % k.block != 0 || w.cols % 4 != 0) {
        matvec_batch_scalar(w, x, out, n);  // no kernel for this type
        return;
    }
    std::lock_guard lk(device_mutex());
    Context& ctx = Context::get();
    State& s = state();
    grow(ctx, s.x, std::size_t{n} * w.cols * 4);
    grow(ctx, s.out, std::size_t{n} * w.rows * 4);
    std::memcpy(s.x.data, x.data(), std::size_t{n} * w.cols * 4);
    const Context::Buffer& wb = weight(ctx, w, k);
    ctx.begin_batch();
    try {
        // The matmul where the type has one, else a matvec a token.
        if (!record_matmul(ctx, w.type, w.rows, w.cols, n, wb, 0, s.x, 0,
                           w.cols, s.out, 0, w.rows, false)) {
            for (std::uint32_t t = 0; t < n; ++t) {
                if (t > 0) {
                    ctx.overlap_next();
                }
                record_matvec(ctx, w.type, w.rows, w.cols, wb, 0, s.x,
                              t * w.cols, s.out, std::uint64_t{t} * w.rows,
                              false);
            }
        }
    } catch (...) {
        ctx.end_batch();
        throw;
    }
    ctx.end_batch();
    // Token-major on the GPU; the Ops contract is row-major.
    const auto* tok = static_cast<const float*>(s.out.data);
    for (std::uint32_t t = 0; t < n; ++t) {
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out[std::size_t{r} * n + t] = tok[std::size_t{t} * w.rows + r];
        }
    }
}

const Ops& image_ops() {
    static const Ops ops{.matvec = &matvec,
                         .dequant_row = &dequant_row,
                         .matvec_batch = &matvec_batch,
                         .self_parallel = true,
                         .alloc_kv = nullptr,
                         .free_kv = nullptr,
                         .prefetch = nullptr,
                         .evict = nullptr,
                         .pool_budget = nullptr,
                         .mt_safe = false};
    return ops;
}

void drop_weights() {
    if (!Context::usable()) {
        return;
    }
    std::lock_guard lk(device_mutex());
    Context& ctx = Context::get();
    for (auto& [ptr, c] : state().weights) {
        ctx.free(c.buf);
    }
    state().weights.clear();
}

}  // namespace locus::backend::metal
