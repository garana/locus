#include "locus/image/t5_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

#include "locus/image/dense.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::image {

namespace {

/** transformers' NewGELUActivation (gelu_new, the tanh form). */
void gelu_tanh(std::span<float> x) {
    constexpr float kC = 0.79788456080286535588f;  // sqrt(2 / pi)
    for (float& v : x) {
        v = 0.5f * v * (1.0f + std::tanh(kC * (v + 0.044715f * v * v * v)));
    }
}

/** Runs fn(i) for i in [0, n) on the thread pool. */
void parallel(std::size_t n, const std::function<void(std::size_t)>& fn) {
    sys::ThreadPool::instance().parallel_for(n, fn);
}

/**
 * out[t][r] = w[r] . x[t] for n tokens, token-major in and out. The
 * rows are split across threads; each thread runs its slice over every
 * token, so the slice stays in cache while the tokens stream past it.
 * A backend that does not split rows (see backend::splits_rows) runs
 * as one slice.
 */
void linear(const backend::Ops& op, const backend::Mat& w,
            std::span<const float> x, std::uint32_t n,
            std::span<float> out) {
    if (op.self_parallel && op.matvec_batch != nullptr) {
        // A GPU: the whole matrix over every token in one call.
        dense::linear(op, w, {}, x, n, out);
        return;
    }
    const std::uint32_t rows = w.rows, cols = w.cols;
    const std::uint32_t threads =
        std::max(1u, std::thread::hardware_concurrency());
    const std::uint32_t slices =
        backend::splits_rows(op)
            ? std::max(1u, std::min(rows / 64, 2 * threads))
            : 1;
    const std::uint32_t per = (rows + slices - 1) / slices;
    parallel(slices, [&](std::size_t s) {
        const std::uint32_t r0 = static_cast<std::uint32_t>(s) * per;
        if (r0 >= rows) return;
        const std::uint32_t nr = std::min(per, rows - r0);
        const backend::Mat ws = backend::mat_rows(w, r0, nr);
        for (std::uint32_t t = 0; t < n; ++t) {
            op.matvec(ws, x.subspan(std::size_t{t} * cols, cols),
                      out.subspan(std::size_t{t} * rows + r0, nr));
        }
    });
}

/** T5's rms norm (no mean, no bias) over each of n rows of width d. */
void rms_norm(std::span<const float> x, std::span<const float> w, float eps,
              std::uint32_t n, std::uint32_t d, std::span<float> out) {
    parallel(n, [&](std::size_t t) {
        backend::rmsnorm(x.subspan(t * d, d), w, eps, out.subspan(t * d, d));
    });
}

}  // namespace

std::uint32_t T5Encoder::relative_bucket(std::int32_t d,
                                         std::uint32_t buckets,
                                         std::uint32_t max_distance) {
    const std::uint32_t half = buckets / 2;
    const std::uint32_t exact = half / 2;
    const std::uint32_t base = d > 0 ? half : 0;
    const std::uint32_t n = static_cast<std::uint32_t>(d < 0 ? -d : d);
    if (n < exact) return base + n;
    // The reference's float32 steps, in its order: log(n / exact),
    // divided by the double log(max_distance / exact) rounded to
    // float, times (half - exact), truncated. At n = 16, 32 and 64 the
    // quotient is an exact integer, which float32 gets right.
    const float lg = std::log(static_cast<float>(n) /
                              static_cast<float>(exact));
    const float den = static_cast<float>(
        std::log(static_cast<double>(max_distance) / exact));
    const float v = lg / den * static_cast<float>(half - exact);
    const std::uint32_t large = exact + static_cast<std::uint32_t>(v);
    return base + std::min(large, half - 1);
}

T5Encoder T5Encoder::load(const gguf::GgufFile& g) {
    if (g.get_string("general.architecture") != "t5encoder") {
        throw gguf::Error("t5: general.architecture is not t5encoder");
    }
    T5Encoder e;
    auto u = [&](std::string_view k) -> std::uint32_t {
        auto x = g.get_uint(k);
        if (!x || *x == 0) {
            throw gguf::Error("t5: missing " + std::string(k));
        }
        return static_cast<std::uint32_t>(*x);
    };
    e.width_ = u("t5encoder.embedding_length");
    e.ffn_ = u("t5encoder.feed_forward_length");
    e.heads_ = u("t5encoder.attention.head_count");
    e.head_dim_ = u("t5encoder.attention.key_length");
    e.buckets_ = u("t5encoder.attention.relative_buckets_count");
    const std::uint32_t layers = u("t5encoder.block_count");
    // Not written by llama.cpp's converter; T5 always uses 128.
    e.max_distance_ = static_cast<std::uint32_t>(
        g.get_uint("t5encoder.attention.relative_max_distance")
            .value_or(128));
    const gguf::Value* eps =
        g.find("t5encoder.attention.layer_norm_rms_epsilon");
    if (eps == nullptr || eps->type != gguf::ValueType::kFloat32) {
        throw gguf::Error("t5: missing layer_norm_rms_epsilon");
    }
    e.eps_ = static_cast<float>(eps->f);
    if (e.buckets_ < 4 || e.buckets_ % 4 != 0 ||
        e.max_distance_ <= e.buckets_ / 4) {
        throw gguf::Error("t5: bad relative attention buckets");
    }
    // In 64 bits, so a hostile pair cannot wrap to a small width.
    if (std::uint64_t{e.heads_} * e.head_dim_ > (1u << 20) ||
        std::uint64_t{e.width_} > (1u << 20) ||
        std::uint64_t{e.ffn_} > (1u << 20)) {
        throw gguf::Error("t5: sizes too large");
    }
    const std::uint32_t inner = e.heads_ * e.head_dim_;

    const auto info = [&](const std::string& name) {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr) {
            throw gguf::Error("t5: missing tensor " + name);
        }
        return ti;
    };
    // A small F32 or F16 tensor, as floats.
    const auto vec = [&](const std::string& name, std::size_t n) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->nelements() != n) {
            throw gguf::Error("t5: " + name + " must have " +
                              std::to_string(n) + " values");
        }
        const auto bytes = g.tensor_data(*ti);
        std::vector<float> out(n);
        if (ti->type == gguf::TensorType::kF32) {
            std::memcpy(out.data(), bytes.data(), n * sizeof(float));
        } else if (ti->type == gguf::TensorType::kF16) {
            for (std::size_t i = 0; i < n; ++i) {
                std::uint16_t h;
                std::memcpy(&h, bytes.data() + 2 * i, 2);
                out[i] = backend::f16_to_f32(h);
            }
        } else if (ti->type == gguf::TensorType::kBF16) {
            for (std::size_t i = 0; i < n; ++i) {
                std::uint16_t h;
                std::memcpy(&h, bytes.data() + 2 * i, 2);
                const std::uint32_t u = std::uint32_t{h} << 16;
                std::memcpy(&out[i], &u, sizeof(float));
            }
        } else {
            throw gguf::Error("t5: " + name + " must be F32, F16 or BF16");
        }
        return out;
    };
    const auto mat = [&](const std::string& name, std::uint32_t rows,
                         std::uint32_t cols) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->ne[0] != cols || ti->ne[1] != rows || ti->n_dims != 2) {
            throw gguf::Error("t5: " + name + " is not " +
                              std::to_string(rows) + " x " +
                              std::to_string(cols));
        }
        return backend::Mat{ti->type, g.tensor_data(*ti).data(), rows, cols};
    };

    const gguf::TensorInfo* embd = info("token_embd.weight");
    if (embd->n_dims != 2 || embd->ne[0] != e.width_ || embd->ne[1] == 0 ||
        embd->ne[1] > 0x7fffffff) {
        throw gguf::Error("t5: token_embd.weight has the wrong shape");
    }
    e.vocab_ = static_cast<std::uint32_t>(embd->ne[1]);
    e.embd_ = mat("token_embd.weight", e.vocab_, e.width_);
    e.rel_bias_ = vec("enc.blk.0.attn_rel_b.weight",
                      std::size_t{e.buckets_} * e.heads_);
    for (std::uint32_t i = 0; i < layers; ++i) {
        const std::string b = "enc.blk." + std::to_string(i) + ".";
        Block k;
        k.attn_norm = vec(b + "attn_norm.weight", e.width_);
        k.q = mat(b + "attn_q.weight", inner, e.width_);
        k.k = mat(b + "attn_k.weight", inner, e.width_);
        k.v = mat(b + "attn_v.weight", inner, e.width_);
        k.o = mat(b + "attn_o.weight", e.width_, inner);
        k.ffn_norm = vec(b + "ffn_norm.weight", e.width_);
        k.gate = mat(b + "ffn_gate.weight", e.ffn_, e.width_);
        k.up = mat(b + "ffn_up.weight", e.ffn_, e.width_);
        k.down = mat(b + "ffn_down.weight", e.width_, e.ffn_);
        e.blocks_.push_back(std::move(k));
    }
    e.out_norm_ = vec("enc.output_norm.weight", e.width_);
    return e;
}

std::vector<float> T5Encoder::encode(std::span<const tok::TokenId> ids,
                                     const backend::Ops& op) const {
    if (ids.empty()) {
        throw std::invalid_argument("t5: no tokens");
    }
    const std::uint32_t n = static_cast<std::uint32_t>(ids.size());
    const std::uint32_t w = width_, hd = head_dim_,
                        inner = heads_ * head_dim_;
    std::vector<float> x(std::size_t{n} * w);
    for (std::uint32_t t = 0; t < n; ++t) {
        if (ids[t] < 0 || static_cast<std::uint32_t>(ids[t]) >= vocab_) {
            throw std::invalid_argument("t5: token id out of range");
        }
        op.dequant_row(embd_, static_cast<std::uint32_t>(ids[t]),
                       std::span<float>(x).subspan(std::size_t{t} * w, w));
    }

    // bias[h][d + n - 1] for d = j - i in (-n, n), shared by all blocks.
    const std::uint32_t span = 2 * n - 1;
    std::vector<float> bias(std::size_t{heads_} * span);
    for (std::uint32_t r = 0; r < span; ++r) {
        const std::int32_t d =
            static_cast<std::int32_t>(r) - static_cast<std::int32_t>(n - 1);
        const std::uint32_t bk = relative_bucket(d, buckets_, max_distance_);
        for (std::uint32_t h = 0; h < heads_; ++h) {
            bias[std::size_t{h} * span + r] =
                rel_bias_[std::size_t{bk} * heads_ + h];
        }
    }

    std::vector<float> y(std::size_t{n} * w), q(std::size_t{n} * inner),
        k(q.size()), v(q.size()), att(q.size()), o(std::size_t{n} * w),
        gate(std::size_t{n} * ffn_), up(gate.size());
    constexpr std::uint32_t kQueryBlock = 32;
    const std::uint32_t qblocks = (n + kQueryBlock - 1) / kQueryBlock;
    for (const Block& b : blocks_) {
        // x += O * attention(rms(x)); head h is dims [h*hd, (h+1)*hd).
        rms_norm(x, b.attn_norm, eps_, n, w, y);
        linear(op, b.q, y, n, q);
        linear(op, b.k, y, n, k);
        linear(op, b.v, y, n, v);
        parallel(std::size_t{heads_} * qblocks, [&](std::size_t item) {
            const std::uint32_t h = static_cast<std::uint32_t>(item / qblocks);
            const std::uint32_t q0 =
                static_cast<std::uint32_t>(item % qblocks) * kQueryBlock;
            const std::uint32_t q1 = std::min(n, q0 + kQueryBlock);
            const float* hb = &bias[std::size_t{h} * span];
            std::vector<float> s(n);
            for (std::uint32_t i = q0; i < q1; ++i) {
                const float* qi = &q[std::size_t{i} * inner + h * hd];
                for (std::uint32_t j = 0; j < n; ++j) {
                    const float* kj = &k[std::size_t{j} * inner + h * hd];
                    float dot = 0.0f;
                    for (std::uint32_t e = 0; e < hd; ++e) dot += qi[e] * kj[e];
                    s[j] = dot + hb[j + n - 1 - i];
                }
                backend::softmax_inplace(s);
                float* out = &att[std::size_t{i} * inner + h * hd];
                std::fill(out, out + hd, 0.0f);
                for (std::uint32_t j = 0; j < n; ++j) {
                    const float* vj = &v[std::size_t{j} * inner + h * hd];
                    for (std::uint32_t e = 0; e < hd; ++e) {
                        out[e] += s[j] * vj[e];
                    }
                }
            }
        });
        linear(op, b.o, att, n, o);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];
        // x += down * (gelu_tanh(gate * rms(x)) * (up * rms(x)))
        rms_norm(x, b.ffn_norm, eps_, n, w, y);
        linear(op, b.gate, y, n, gate);
        linear(op, b.up, y, n, up);
        gelu_tanh(gate);
        for (std::size_t i = 0; i < gate.size(); ++i) gate[i] *= up[i];
        linear(op, b.down, gate, n, o);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];
    }
    rms_norm(x, out_norm_, eps_, n, w, x);
    return x;
}

}  // namespace locus::image
