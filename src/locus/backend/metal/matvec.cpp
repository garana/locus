// The Metal matvec (i#866); see matvec.hpp.
#include "locus/backend/metal/matvec.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <mutex>
#include <unordered_map>

#include "locus/backend/metal/context.hpp"

namespace locus::backend::metal {

namespace {

using gguf::TensorType;

/** A matvec kernel and its weight block: elements and bytes. */
struct Kind {
    const char* kernel = nullptr;
    std::uint32_t block = 0;
    std::uint32_t bytes = 0;
};

Kind kind_of(TensorType t) {
    switch (t) {
        case TensorType::kF32:
            return {"mv_f32", 4, 16};
        case TensorType::kQ8_0:
            return {"mv_q8_0", 32, 34};
        case TensorType::kQ4_K:
            return {"mv_q4_k", 256, 144};
        case TensorType::kQ6_K:
            return {"mv_q6_k", 256, 210};
        default:
            return {};
    }
}

/** MvParams in kernels.metal. */
struct MvParams {
    std::uint32_t rows, cols, out_off, accumulate, x_off;
    float scale;
};

constexpr std::uint32_t kGroup = 128;  // 4 simdgroups
constexpr std::uint32_t kRowsPerGroup = 8;

/** A cached weight: its buffer and the matrix it was made from. */
struct Cached {
    Context::Buffer buf;
    gguf::TensorType type;
    std::uint32_t rows, cols;
};

/** Weights by host pointer, and the x / out scratch buffers. */
struct State {
    std::unordered_map<const std::byte*, Cached> weights;
    Context::Buffer x, out;
};

// Guards state() and the context: one matvec at a time, from any
// thread (the forward will own its own buffers, i#866).
std::mutex& state_mu() {
    static std::mutex mu;
    return mu;
}

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
    const auto it = ws.find(w.data);
    if (it != ws.end()) {
        Cached& c = it->second;
        if (c.type == w.type && c.rows == w.rows && c.cols == w.cols) {
            return c.buf;
        }
        // Another matrix at a freed one's address: upload it afresh.
        ctx.free(c.buf);
        ws.erase(it);
    }
    Cached c{ctx.upload({w.data, weight_bytes(w, k)}), w.type, w.rows,
             w.cols};
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
    const Context::Buffer& wb = weight(ctx, w, k);
    const MvParams p{w.rows, w.cols, 0, 0, 0, 1.0f};
    const Context::Arg args[] = {{&wb, 0}, {&s.x, 0}, {&s.out, 0}};
    const std::uint32_t groups = (w.rows + kRowsPerGroup - 1) / kRowsPerGroup;
    ctx.dispatch(k.kernel, args, std::as_bytes(std::span(&p, 1)),
                 groups * kGroup, kGroup);
}

}  // namespace

bool matvec_supported(gguf::TensorType t) {
    return kind_of(t).kernel != nullptr;
}

void matvec(const Mat& w, std::span<const float> x, std::span<float> out) {
    const Kind kd = kind_of(w.type);
    if (kd.kernel == nullptr || w.cols % kd.block != 0) {
        backend::matvec(w, x, out);  // no kernel for this type
        return;
    }
    std::lock_guard lk(state_mu());
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
    std::lock_guard lk(state_mu());
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

void drop_weights() {
    if (!Context::usable()) {
        return;
    }
    std::lock_guard lk(state_mu());
    Context& ctx = Context::get();
    for (auto& [ptr, c] : state().weights) {
        ctx.free(c.buf);
    }
    state().weights.clear();
}

}  // namespace locus::backend::metal
