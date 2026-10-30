#include "locus/image/clip_text.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "locus/image/dense.hpp"

namespace locus::image {

namespace {

/** LayerNorm over each row of x (n rows of w floats), into out. */
void layer_norm(std::span<const float> x, std::uint32_t n, std::uint32_t w,
                const std::vector<float>& g, const std::vector<float>& b,
                float eps, std::span<float> out) {
    for (std::uint32_t t = 0; t < n; ++t) {
        const float* r = x.data() + std::size_t{t} * w;
        float* o = out.data() + std::size_t{t} * w;
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
    }
}

}  // namespace

ClipTextEncoder ClipTextEncoder::load(const gguf::GgufFile& g) {
    if (g.get_string("general.architecture") != "clip_text") {
        throw gguf::Error("clip: general.architecture is not clip_text");
    }
    const auto u = [&](const std::string& key) {
        const auto v = g.get_uint("clip_text." + key);
        if (!v || *v == 0 || *v > 0xffffffffu) {
            throw gguf::Error("clip: missing or bad clip_text." + key);
        }
        return static_cast<std::uint32_t>(*v);
    };
    ClipTextEncoder e;
    e.width_ = u("embedding_length");
    e.heads_ = u("attention.head_count");
    e.ffn_ = u("feed_forward_length");
    e.context_ = u("context_length");
    const std::uint32_t blocks = u("block_count");
    e.legacy_eos_ = u("config_eos_token_id") == 2 ? 1u : 0u;
    const auto eos = g.get_uint("tokenizer.ggml.eos_token_id");
    if (!eos) throw gguf::Error("clip: tokenizer.ggml.eos_token_id missing");
    e.eos_ = static_cast<tok::TokenId>(*eos);
    const gguf::Value* eps = g.find("clip_text.norm_epsilon");
    if (eps == nullptr || eps->type != gguf::ValueType::kFloat32 ||
        !(eps->f > 0.0f)) {
        throw gguf::Error("clip: missing or bad clip_text.norm_epsilon");
    }
    e.eps_ = static_cast<float>(eps->f);
    if (e.width_ % e.heads_ != 0) {
        throw gguf::Error("clip: width is not a multiple of the heads");
    }

    const auto info = [&](const std::string& name, std::uint64_t n) {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr) throw gguf::Error("clip: missing " + name);
        if (ti->type != gguf::TensorType::kF32 || ti->nelements() != n) {
            throw gguf::Error("clip: " + name + " must be F32 with " +
                              std::to_string(n) + " values");
        }
        return ti;
    };
    const auto vec = [&](const std::string& name, std::uint32_t n) {
        const auto bytes = g.tensor_data(*info(name, n));
        std::vector<float> out(n);
        std::memcpy(out.data(), bytes.data(), std::size_t{n} * 4);
        return out;
    };
    const auto lin = [&](const std::string& p, std::uint32_t rows,
                         std::uint32_t cols) {
        const gguf::TensorInfo* ti =
            info(p + ".weight", std::uint64_t{rows} * cols);
        if (ti->n_dims != 2 || ti->ne[0] != cols) {
            throw gguf::Error("clip: " + p + ".weight is not " +
                              std::to_string(rows) + " x " +
                              std::to_string(cols));
        }
        Linear l;
        l.w = backend::Mat{gguf::TensorType::kF32,
                           g.tensor_data(*ti).data(), rows, cols};
        l.b = vec(p + ".bias", rows);
        return l;
    };

    const std::string m = "text_model.";
    const gguf::TensorInfo* tok =
        g.find_tensor(m + "embeddings.token_embedding.weight");
    if (tok == nullptr || tok->type != gguf::TensorType::kF32 ||
        tok->n_dims != 2 || tok->ne[0] != e.width_ || tok->ne[1] == 0 ||
        tok->ne[1] > 0x7fffffff) {
        throw gguf::Error("clip: bad token_embedding.weight");
    }
    e.vocab_ = static_cast<std::uint32_t>(tok->ne[1]);
    e.token_embd_ = reinterpret_cast<const float*>(g.tensor_data(*tok).data());
    e.pos_embd_ = reinterpret_cast<const float*>(
        g.tensor_data(*info(m + "embeddings.position_embedding.weight",
                            std::uint64_t{e.context_} * e.width_))
            .data());
    const std::uint32_t w = e.width_;
    for (std::uint32_t n = 0; n < blocks; ++n) {
        const std::string p = m + "encoder.layers." + std::to_string(n) + ".";
        Layer l;
        l.ln1_w = vec(p + "layer_norm1.weight", w);
        l.ln1_b = vec(p + "layer_norm1.bias", w);
        l.ln2_w = vec(p + "layer_norm2.weight", w);
        l.ln2_b = vec(p + "layer_norm2.bias", w);
        l.q = lin(p + "self_attn.q_proj", w, w);
        l.k = lin(p + "self_attn.k_proj", w, w);
        l.v = lin(p + "self_attn.v_proj", w, w);
        l.out = lin(p + "self_attn.out_proj", w, w);
        l.fc1 = lin(p + "mlp.fc1", e.ffn_, w);
        l.fc2 = lin(p + "mlp.fc2", w, e.ffn_);
        e.layers_.push_back(std::move(l));
    }
    e.final_w_ = vec(m + "final_layer_norm.weight", w);
    e.final_b_ = vec(m + "final_layer_norm.bias", w);
    return e;
}

ClipTextEncoder::Output ClipTextEncoder::encode(
    std::span<const tok::TokenId> ids, const backend::Ops& op) const {
    const auto n = static_cast<std::uint32_t>(ids.size());
    if (n == 0 || n > context_) {
        throw std::invalid_argument("clip: need 1 to context() ids");
    }
    for (const tok::TokenId id : ids) {
        if (id < 0 || static_cast<std::uint32_t>(id) >= vocab_) {
            throw std::invalid_argument("clip: id outside the vocabulary");
        }
    }
    const std::uint32_t w = width_, hd = w / heads_;
    const std::size_t nw = std::size_t{n} * w;
    std::vector<float> x(nw), h(nw), q(nw), k(nw), v(nw), a(nw);
    std::vector<float> f(std::size_t{n} * ffn_);
    for (std::uint32_t t = 0; t < n; ++t) {
        const float* te =
            token_embd_ + static_cast<std::size_t>(ids[t]) * w;
        const float* pe = pos_embd_ + std::size_t{t} * w;
        for (std::uint32_t i = 0; i < w; ++i) {
            x[std::size_t{t} * w + i] = te[i] + pe[i];
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> s(n);
    for (const Layer& l : layers_) {
        layer_norm(x, n, w, l.ln1_w, l.ln1_b, eps_, h);
        dense::linear(op, l.q.w, l.q.b, h, n, q);
        dense::linear(op, l.k.w, l.k.b, h, n, k);
        dense::linear(op, l.v.w, l.v.b, h, n, v);
        // Causal attention: query t sees keys 0..t.
        for (std::uint32_t hh = 0; hh < heads_; ++hh) {
            for (std::uint32_t t = 0; t < n; ++t) {
                const float* qr = q.data() + std::size_t{t} * w + hh * hd;
                float mx = -INFINITY;
                for (std::uint32_t j = 0; j <= t; ++j) {
                    const float* kr = k.data() + std::size_t{j} * w + hh * hd;
                    float d = 0.0f;
                    for (std::uint32_t i = 0; i < hd; ++i) d += qr[i] * kr[i];
                    s[j] = d * scale;
                    mx = std::max(mx, s[j]);
                }
                float sum = 0.0f;
                for (std::uint32_t j = 0; j <= t; ++j) {
                    s[j] = std::exp(s[j] - mx);
                    sum += s[j];
                }
                float* ar = a.data() + std::size_t{t} * w + hh * hd;
                std::fill(ar, ar + hd, 0.0f);
                for (std::uint32_t j = 0; j <= t; ++j) {
                    const float p = s[j] / sum;
                    const float* vr = v.data() + std::size_t{j} * w + hh * hd;
                    for (std::uint32_t i = 0; i < hd; ++i) ar[i] += p * vr[i];
                }
            }
        }
        dense::linear(op, l.out.w, l.out.b, a, n, h);
        for (std::size_t i = 0; i < nw; ++i) x[i] += h[i];
        layer_norm(x, n, w, l.ln2_w, l.ln2_b, eps_, h);
        dense::linear(op, l.fc1.w, l.fc1.b, h, n, f);
        for (float& z : f) z = z / (1.0f + std::exp(-1.702f * z));
        dense::linear(op, l.fc2.w, l.fc2.b, f, n, h);
        for (std::size_t i = 0; i < nw; ++i) x[i] += h[i];
    }
    Output out;
    out.hidden.resize(nw);
    layer_norm(x, n, w, final_w_, final_b_, eps_, out.hidden);
    std::size_t at = 0;
    if (legacy_eos_ != 0) {
        at = static_cast<std::size_t>(
            std::max_element(ids.begin(), ids.end()) - ids.begin());
    } else {
        const auto it = std::find(ids.begin(), ids.end(), eos_);
        at = it != ids.end() ? static_cast<std::size_t>(it - ids.begin())
                             : n - 1;
    }
    out.pooled.assign(out.hidden.begin() + static_cast<std::ptrdiff_t>(at * w),
                      out.hidden.begin() +
                          static_cast<std::ptrdiff_t>((at + 1) * w));
    return out;
}

}  // namespace locus::image
