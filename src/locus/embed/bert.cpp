#include "locus/embed/bert.hpp"

#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

#include "locus/image/dense.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::embed {

namespace {

/** Runs fn(i) for i in [0, n) on the thread pool. */
void parallel(std::size_t n, const std::function<void(std::size_t)>& fn) {
    sys::ThreadPool::instance().parallel_for(n, fn);
}

/** LayerNorm with bias over each of n rows of width w, into out (which
 *  may alias x); the moments in double. */
void layer_norm(std::span<const float> x, std::uint32_t n, std::uint32_t w,
                const std::vector<float>& g, const std::vector<float>& b,
                float eps, std::span<float> out) {
    parallel(n, [&](std::size_t t) {
        const float* r = x.data() + t * w;
        float* o = out.data() + t * w;
        double mean = 0.0;
        for (std::uint32_t i = 0; i < w; ++i) mean += r[i];
        mean /= w;
        double var = 0.0;
        for (std::uint32_t i = 0; i < w; ++i) {
            const double d = r[i] - mean;
            var += d * d;
        }
        var /= w;
        const auto inv = static_cast<float>(1.0 / std::sqrt(var + eps));
        const auto m = static_cast<float>(mean);
        for (std::uint32_t i = 0; i < w; ++i) {
            o[i] = (r[i] - m) * inv * g[i] + b[i];
        }
    });
}

/** BERT's GELU, the exact erf form. */
void gelu(std::span<float> x) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    for (float& v : x) v = 0.5f * v * (1.0f + std::erf(v * kInvSqrt2));
}

}  // namespace

BertEncoder BertEncoder::load(const gguf::GgufFile& g) {
    if (g.get_string("general.architecture") != "bert") {
        throw gguf::Error("bert: general.architecture is not bert");
    }
    const auto u = [&](const std::string& key) {
        const auto v = g.get_uint("bert." + key);
        if (!v || *v == 0 || *v > (1u << 20)) {
            throw gguf::Error("bert: missing or bad bert." + key);
        }
        return static_cast<std::uint32_t>(*v);
    };
    BertEncoder e;
    e.width_ = u("embedding_length");
    e.heads_ = u("attention.head_count");
    e.ffn_ = u("feed_forward_length");
    e.context_ = u("context_length");
    const std::uint32_t blocks = u("block_count");
    if (blocks > 1024) throw gguf::Error("bert: too many blocks");
    if (e.width_ % e.heads_ != 0) {
        throw gguf::Error("bert: width is not a multiple of the heads");
    }
    const gguf::Value* eps = g.find("bert.attention.layer_norm_epsilon");
    if (eps == nullptr || eps->type != gguf::ValueType::kFloat32 ||
        !(eps->f > 0.0f)) {
        throw gguf::Error("bert: missing or bad layer_norm_epsilon");
    }
    e.eps_ = static_cast<float>(eps->f);
    const auto pool = g.get_uint("bert.pooling_type");
    if (!pool) throw gguf::Error("bert: bert.pooling_type missing");
    e.pooling_ = static_cast<Pooling>(*pool);
    if (e.pooling_ != Pooling::kMean && e.pooling_ != Pooling::kCls &&
        e.pooling_ != Pooling::kLast) {
        throw gguf::Error("bert: pooling_type " + std::to_string(*pool) +
                          " is not mean (1), cls (2) or last (3)");
    }

    const auto info = [&](const std::string& name) {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr) throw gguf::Error("bert: missing " + name);
        return ti;
    };
    // A small F32 or F16 tensor, as floats.
    const auto vec = [&](const std::string& name, std::size_t n) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->nelements() != n) {
            throw gguf::Error("bert: " + name + " must have " +
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
        } else {
            throw gguf::Error("bert: " + name + " must be F32 or F16");
        }
        return out;
    };
    const auto mat = [&](const std::string& name, std::uint32_t rows,
                         std::uint32_t cols) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->n_dims != 2 || ti->ne[0] != cols || ti->ne[1] != rows) {
            throw gguf::Error("bert: " + name + " is not " +
                              std::to_string(rows) + " x " +
                              std::to_string(cols));
        }
        return backend::Mat{ti->type, g.tensor_data(*ti).data(), rows, cols};
    };
    const auto lin = [&](const std::string& p, std::uint32_t rows,
                         std::uint32_t cols) {
        Linear l;
        l.w = mat(p + ".weight", rows, cols);
        l.b = vec(p + ".bias", rows);
        return l;
    };
    const auto norm = [&](const std::string& p) {
        return Norm{vec(p + ".weight", e.width_), vec(p + ".bias", e.width_)};
    };

    const gguf::TensorInfo* tok = info("token_embd.weight");
    if (tok->n_dims != 2 || tok->ne[0] != e.width_ || tok->ne[1] == 0 ||
        tok->ne[1] > 0x7fffffff) {
        throw gguf::Error("bert: token_embd.weight has the wrong shape");
    }
    e.vocab_ = static_cast<std::uint32_t>(tok->ne[1]);
    e.tok_embd_ = mat("token_embd.weight", e.vocab_, e.width_);
    e.pos_embd_ = mat("position_embd.weight", e.context_, e.width_);
    // Token types are all 0 ("sentence A"), as in llama.cpp: row 0.
    const gguf::TensorInfo* types = info("token_types.weight");
    if (types->n_dims != 2 || types->ne[0] != e.width_ || types->ne[1] == 0) {
        throw gguf::Error("bert: token_types.weight has the wrong shape");
    }
    e.type0_ = vec("token_types.weight", types->nelements());
    e.type0_.resize(e.width_);
    e.embd_norm_ = norm("token_embd_norm");
    const std::uint32_t w = e.width_;
    for (std::uint32_t i = 0; i < blocks; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        Layer l;
        l.q = lin(p + "attn_q", w, w);
        l.k = lin(p + "attn_k", w, w);
        l.v = lin(p + "attn_v", w, w);
        l.o = lin(p + "attn_output", w, w);
        l.attn_norm = norm(p + "attn_output_norm");
        l.up = lin(p + "ffn_up", e.ffn_, w);
        l.down = lin(p + "ffn_down", w, e.ffn_);
        l.out_norm = norm(p + "layer_output_norm");
        e.layers_.push_back(std::move(l));
    }
    return e;
}

std::vector<std::vector<float>> BertEncoder::encode(
    std::span<const std::vector<tok::TokenId>> batch,
    const backend::Ops& op) const {
    const std::uint32_t w = width_, hd = width_ / heads_;
    // Each sequence's first row in the batch, and the total.
    std::vector<std::uint32_t> start;
    std::uint32_t n = 0;
    for (const auto& ids : batch) {
        if (ids.empty() || ids.size() > context_) {
            throw std::invalid_argument(
                "bert: a sequence needs 1 to " + std::to_string(context_) +
                " tokens, not " + std::to_string(ids.size()));
        }
        for (const tok::TokenId id : ids) {
            if (id < 0 || static_cast<std::uint32_t>(id) >= vocab_) {
                throw std::invalid_argument("bert: id outside the vocabulary");
            }
        }
        start.push_back(n);
        n += static_cast<std::uint32_t>(ids.size());
    }
    if (n == 0) return {};

    const std::size_t nw = std::size_t{n} * w;
    std::vector<float> x(nw), h(nw), q(nw), k(nw), v(nw), a(nw), pe(w);
    std::vector<float> f(std::size_t{n} * ffn_);
    for (std::size_t s = 0; s < batch.size(); ++s) {
        for (std::uint32_t t = 0; t < batch[s].size(); ++t) {
            const std::span<float> row(x.data() +
                                           std::size_t{start[s] + t} * w,
                                       w);
            op.dequant_row(tok_embd_,
                           static_cast<std::uint32_t>(batch[s][t]), row);
            op.dequant_row(pos_embd_, t, pe);
            for (std::uint32_t i = 0; i < w; ++i) row[i] += type0_[i] + pe[i];
        }
    }
    layer_norm(x, n, w, embd_norm_.w, embd_norm_.b, eps_, x);
    for (const Layer& l : layers_) {
        image::dense::linear(op, l.q.w, l.q.b, x, n, q);
        image::dense::linear(op, l.k.w, l.k.b, x, n, k);
        image::dense::linear(op, l.v.w, l.v.b, x, n, v);
        for (std::size_t s = 0; s < batch.size(); ++s) {
            const std::size_t o = std::size_t{start[s]} * w;
            const auto m = static_cast<std::uint32_t>(batch[s].size());
            const std::size_t len = std::size_t{m} * w;
            image::dense::attention(
                op, std::span<const float>(q).subspan(o, len), m,
                std::span<const float>(k).subspan(o, len),
                std::span<const float>(v).subspan(o, len), m, heads_, hd,
                std::span<float>(a).subspan(o, len));
        }
        image::dense::linear(op, l.o.w, l.o.b, a, n, h);
        for (std::size_t i = 0; i < nw; ++i) x[i] += h[i];
        layer_norm(x, n, w, l.attn_norm.w, l.attn_norm.b, eps_, x);
        image::dense::linear(op, l.up.w, l.up.b, x, n, f);
        gelu(f);
        image::dense::linear(op, l.down.w, l.down.b, f, n, h);
        for (std::size_t i = 0; i < nw; ++i) x[i] += h[i];
        layer_norm(x, n, w, l.out_norm.w, l.out_norm.b, eps_, x);
    }
    std::vector<std::vector<float>> out;
    out.reserve(batch.size());
    for (std::size_t s = 0; s < batch.size(); ++s) {
        const auto first = x.begin() + static_cast<std::ptrdiff_t>(
                                           std::size_t{start[s]} * w);
        out.emplace_back(first,
                         first + static_cast<std::ptrdiff_t>(
                                     batch[s].size() * w));
    }
    return out;
}

std::vector<float> BertEncoder::pool(std::span<const float> states,
                                     std::uint32_t n, std::uint32_t width,
                                     Pooling pooling) {
    if (n == 0 || states.size() != std::size_t{n} * width) {
        throw std::invalid_argument("bert: pool needs n x width states");
    }
    std::vector<float> out(width, 0.0f);
    if (pooling == Pooling::kMean) {
        std::vector<double> sum(width, 0.0);
        for (std::uint32_t t = 0; t < n; ++t) {
            for (std::uint32_t i = 0; i < width; ++i) {
                sum[i] += states[std::size_t{t} * width + i];
            }
        }
        for (std::uint32_t i = 0; i < width; ++i) {
            out[i] = static_cast<float>(sum[i] / n);
        }
    } else if (pooling == Pooling::kCls || pooling == Pooling::kLast) {
        const std::size_t t = pooling == Pooling::kCls ? 0 : n - 1;
        std::memcpy(out.data(), states.data() + t * width,
                    width * sizeof(float));
    } else {
        throw std::invalid_argument("bert: pooling is not mean, cls or last");
    }
    double norm = 0.0;
    for (const float z : out) norm += static_cast<double>(z) * z;
    norm = std::sqrt(norm);
    if (norm > 0.0) {
        for (float& z : out) z = static_cast<float>(z / norm);
    }
    return out;
}

std::vector<std::vector<float>> BertEncoder::embed(
    std::span<const std::vector<tok::TokenId>> batch,
    const backend::Ops& op) const {
    auto states = encode(batch, op);
    std::vector<std::vector<float>> out;
    out.reserve(states.size());
    for (std::size_t s = 0; s < states.size(); ++s) {
        out.push_back(pool(states[s],
                           static_cast<std::uint32_t>(batch[s].size()),
                           width_, pooling_));
    }
    return out;
}

}  // namespace locus::embed
