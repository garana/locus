#include "locus/image/ltxv_dit.hpp"

#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

#include "locus/image/dense.hpp"
#include "locus/image/pixart_dit.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::image {

namespace {

using dense::attention;
using dense::linear;

/** Runs fn(i) for i in [0, n) on the thread pool. */
void parallel(std::size_t n, const std::function<void(std::size_t)>& fn) {
    sys::ThreadPool::instance().parallel_for(n, fn);
}

/** torch's F.gelu(approximate="tanh"). */
float gelu_tanh(float v) {
    constexpr float kC = 0.79788456080286535588f;  // sqrt(2 / pi)
    return 0.5f * v * (1.0f + std::tanh(kC * (v + 0.044715f * v * v * v)));
}

float silu(float v) { return v / (1.0f + std::exp(-v)); }

/** One vector: out = w x + b. */
void linear1(const backend::Ops& op, const backend::Mat& w,
             std::span<const float> b, std::span<const float> x,
             std::span<float> out) {
    op.matvec(w, x, out);
    for (std::uint32_t r = 0; r < w.rows; ++r) out[r] += b[r];
}

/**
 * diffusers' RMSNorm over each of n rows of width d, in place:
 * x / sqrt(mean(x^2) + eps), times weight when weight is not empty.
 */
void rms_norm(std::span<float> x, std::uint32_t n, std::uint32_t d,
              float eps, std::span<const float> weight) {
    parallel(n, [&](std::size_t t) {
        float* xi = &x[t * d];
        float ss = 0.0f;
        for (std::uint32_t i = 0; i < d; ++i) ss += xi[i] * xi[i];
        const float inv =
            1.0f / std::sqrt(ss / static_cast<float>(d) + eps);
        for (std::uint32_t i = 0; i < d; ++i) {
            xi[i] *= inv;
            if (!weight.empty()) xi[i] *= weight[i];
        }
    });
}

/**
 * RMSNorm (no affine) over each of n rows of width d, then
 * out = norm(x) * (1 + scale) + shift. Row t takes its shift and scale
 * at offset gid[t] * stride: one set per distinct noise level.
 */
void rms_modulate(std::span<const float> x, std::uint32_t n,
                  std::uint32_t d, float eps, const float* shift0,
                  const float* scale0, std::span<const std::uint32_t> gid,
                  std::size_t stride, std::span<float> out) {
    parallel(n, [&](std::size_t t) {
        const float* xi = &x[t * d];
        float* o = &out[t * d];
        const float* shift = shift0 + gid[t] * stride;
        const float* scale = scale0 + gid[t] * stride;
        float ss = 0.0f;
        for (std::uint32_t i = 0; i < d; ++i) ss += xi[i] * xi[i];
        const float inv =
            1.0f / std::sqrt(ss / static_cast<float>(d) + eps);
        for (std::uint32_t i = 0; i < d; ++i) {
            o[i] = xi[i] * inv * (1.0f + scale[i]) + shift[i];
        }
    });
}

/**
 * LayerNorm without affine over each of n rows of width d, then
 * out = ln(x) * (1 + scale) + shift, with shift and scale per row as
 * for rms_modulate.
 */
void ln_modulate(std::span<const float> x, std::uint32_t n, std::uint32_t d,
                 float eps, const float* shift0, const float* scale0,
                 std::span<const std::uint32_t> gid, std::size_t stride,
                 std::span<float> out) {
    parallel(n, [&](std::size_t t) {
        const float* xi = &x[t * d];
        float* o = &out[t * d];
        const float* shift = shift0 + gid[t] * stride;
        const float* scale = scale0 + gid[t] * stride;
        float mean = 0.0f;
        for (std::uint32_t i = 0; i < d; ++i) mean += xi[i];
        mean /= static_cast<float>(d);
        float var = 0.0f;
        for (std::uint32_t i = 0; i < d; ++i) {
            const float c = xi[i] - mean;
            var += c * c;
        }
        var /= static_cast<float>(d);
        const float inv = 1.0f / std::sqrt(var + eps);
        for (std::uint32_t i = 0; i < d; ++i) {
            o[i] = (xi[i] - mean) * inv * (1.0f + scale[i]) + shift[i];
        }
    });
}

/**
 * The reference's apply_rotary_emb on each of n rows of width d, in
 * place: for each pair (2 i, 2 i + 1),
 * (x0, x1) -> (x0 cos - x1 sin, x1 cos + x0 sin).
 */
void apply_rope(std::span<float> x, std::uint32_t n, std::uint32_t d,
                std::span<const float> cosv, std::span<const float> sinv) {
    parallel(n, [&](std::size_t t) {
        float* xi = &x[t * d];
        const float* c = &cosv[t * d];
        const float* s = &sinv[t * d];
        for (std::uint32_t i = 0; i + 1 < d; i += 2) {
            const float x0 = xi[i], x1 = xi[i + 1];
            xi[i] = x0 * c[i] - x1 * s[i];
            xi[i + 1] = x1 * c[i + 1] + x0 * s[i + 1];
        }
    });
}

}  // namespace

void LtxvDit::rope(std::span<const float> coords, std::uint32_t n,
                   std::uint32_t dim, float theta,
                   const float (&max_pos)[3], std::span<float> cosv,
                   std::span<float> sinv) {
    if (coords.size() != std::size_t{3} * n || dim < 12 ||
        cosv.size() != std::size_t{n} * dim ||
        sinv.size() != std::size_t{n} * dim) {
        throw std::invalid_argument("ltxv: rope sizes");
    }
    // torch, float32: theta ** linspace(0, 1, dim // 6), times pi, / 2.
    // torch's linspace steps from the start for the first half and back
    // from the end for the rest, so the exponents match it bit for bit.
    constexpr float kPi = 3.14159265358979323846f;
    const std::uint32_t nf = dim / 6, pad = dim % 6;
    const float step = 1.0f / static_cast<float>(nf - 1);
    std::vector<float> freq(nf);
    for (std::uint32_t k = 0; k < nf; ++k) {
        const float e = k < nf / 2
                            ? step * static_cast<float>(k)
                            : 1.0f - step * static_cast<float>(nf - 1 - k);
        freq[k] = std::pow(theta, e) * kPi / 2.0f;
    }
    for (std::uint32_t t = 0; t < n; ++t) {
        float frac[3];
        for (std::size_t a = 0; a < 3; ++a) {
            frac[a] = coords[a * n + t] / max_pos[a] * 2.0f - 1.0f;
        }
        float* c = &cosv[std::size_t{t} * dim];
        float* s = &sinv[std::size_t{t} * dim];
        for (std::uint32_t i = 0; i < pad; ++i) {
            c[i] = 1.0f;
            s[i] = 0.0f;
        }
        for (std::uint32_t j = 0; j < 3 * nf; ++j) {
            const float ang = freq[j / 3] * frac[j % 3];
            const float cv = std::cos(ang), sv = std::sin(ang);
            c[pad + 2 * j] = c[pad + 2 * j + 1] = cv;
            s[pad + 2 * j] = s[pad + 2 * j + 1] = sv;
        }
    }
}

LtxvDit LtxvDit::load(const gguf::GgufFile& g) {
    if (g.get_string("general.architecture") != "ltxv") {
        throw gguf::Error("ltxv: general.architecture is not ltxv");
    }
    LtxvDit d;
    auto u = [&](std::string_view k) -> std::uint32_t {
        auto x = g.get_uint(k);
        if (!x || *x == 0 || *x > (1u << 20)) {
            throw gguf::Error("ltxv: missing or bad " + std::string(k));
        }
        return static_cast<std::uint32_t>(*x);
    };
    auto f = [&](std::string_view k) {
        const gguf::Value* v = g.find(k);
        if (v == nullptr || v->type != gguf::ValueType::kFloat32) {
            throw gguf::Error("ltxv: missing " + std::string(k));
        }
        return static_cast<float>(v->f);
    };
    const std::uint32_t layers = u("ltxv.block_count");
    d.width_ = u("ltxv.embedding_length");
    d.ffn_ = u("ltxv.feed_forward_length");
    d.heads_ = u("ltxv.attention.head_count");
    d.head_dim_ = u("ltxv.attention.key_length");
    d.eps_ = f("ltxv.norm_epsilon");
    d.in_ch_ = u("ltxv.in_channels");
    d.out_ch_ = u("ltxv.out_channels");
    d.cap_ch_ = u("ltxv.caption_channels");
    d.t_ch_ = u("ltxv.timestep.channels");
    d.theta_ = f("ltxv.rope.theta");
    d.t_scale_ = f("ltxv.timestep.scale_multiplier");
    const gguf::Value* mp = g.find("ltxv.rope.max_pos");
    if (mp == nullptr || mp->type != gguf::ValueType::kArray ||
        mp->arr.size() != 3) {
        throw gguf::Error("ltxv: rope.max_pos must hold 3 values");
    }
    // The converter writes Python ints, which gguf stores as int32.
    using VT = gguf::ValueType;
    for (int a = 0; a < 3; ++a) {
        const gguf::Value& e = mp->arr[static_cast<std::size_t>(a)];
        double x = 0.0;
        if (e.type == VT::kFloat32 || e.type == VT::kFloat64) {
            x = e.f;
        } else if (e.type == VT::kInt8 || e.type == VT::kInt16 ||
                   e.type == VT::kInt32 || e.type == VT::kInt64) {
            x = static_cast<double>(e.i);
        } else if (e.type == VT::kUint8 || e.type == VT::kUint16 ||
                   e.type == VT::kUint32 || e.type == VT::kUint64) {
            x = static_cast<double>(e.u);
        }
        if (!(x > 0.0)) throw gguf::Error("ltxv: bad rope.max_pos");
        d.max_pos_[a] = static_cast<float>(x);
    }
    if (std::uint64_t{d.heads_} * d.head_dim_ != d.width_ ||
        d.in_ch_ > 4096 || d.out_ch_ > 4096 || d.width_ < 12 ||
        d.width_ % 2 != 0 || d.t_ch_ % 2 != 0 || !(d.theta_ > 1.0f) ||
        !(d.t_scale_ > 0.0f)) {
        throw gguf::Error("ltxv: inconsistent sizes");
    }

    const auto info = [&](const std::string& name) {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr) {
            throw gguf::Error("ltxv: missing tensor " + name);
        }
        return ti;
    };
    // F32, or BF16 widened exactly: the converter keeps every 2D tensor
    // BF16, which includes the [2][w] and [6][w] scale_shift_tables.
    const auto vec = [&](const std::string& name, std::size_t n) {
        const gguf::TensorInfo* ti = info(name);
        if ((ti->type != gguf::TensorType::kF32 &&
             ti->type != gguf::TensorType::kBF16) ||
            ti->nelements() != n) {
            throw gguf::Error("ltxv: " + name + " must be F32 or BF16 with " +
                              std::to_string(n) + " values");
        }
        const auto bytes = g.tensor_data(*ti);
        std::vector<float> out(n);
        if (ti->type == gguf::TensorType::kF32) {
            std::memcpy(out.data(), bytes.data(), n * sizeof(float));
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                std::uint16_t h;
                std::memcpy(&h, bytes.data() + 2 * i, 2);
                const std::uint32_t u = std::uint32_t{h} << 16;
                std::memcpy(&out[i], &u, sizeof(float));
            }
        }
        return out;
    };
    // A Linear [rows, cols], BF16 or F32.
    const auto lin = [&](const std::string& name, std::uint32_t rows,
                         std::uint32_t cols) {
        const gguf::TensorInfo* ti = info(name + ".weight");
        if ((ti->type != gguf::TensorType::kBF16 &&
             ti->type != gguf::TensorType::kF32) ||
            ti->n_dims != 2 || ti->ne[0] != cols || ti->ne[1] != rows) {
            throw gguf::Error("ltxv: " + name + ".weight is not " +
                              std::to_string(rows) + " x " +
                              std::to_string(cols));
        }
        Linear l;
        l.w = backend::Mat{ti->type, g.tensor_data(*ti).data(), rows, cols};
        l.b = vec(name + ".bias", rows);
        return l;
    };

    const std::uint32_t w = d.width_;
    d.patchify_ = lin("patchify_proj", w, d.in_ch_);
    const std::string te = "adaln_single.emb.timestep_embedder.";
    d.t1_ = lin(te + "linear_1", w, d.t_ch_);
    d.t2_ = lin(te + "linear_2", w, w);
    d.ada_ = lin("adaln_single.linear", 6 * w, w);
    d.cap1_ = lin("caption_projection.linear_1", w, d.cap_ch_);
    d.cap2_ = lin("caption_projection.linear_2", w, w);
    d.table_ = vec("scale_shift_table", std::size_t{2} * w);
    d.proj_out_ = lin("proj_out", d.out_ch_, w);
    for (std::uint32_t i = 0; i < layers; ++i) {
        const std::string b = "transformer_blocks." + std::to_string(i) + ".";
        Block k;
        k.table = vec(b + "scale_shift_table", std::size_t{6} * w);
        k.qn1 = vec(b + "attn1.q_norm.weight", w);
        k.kn1 = vec(b + "attn1.k_norm.weight", w);
        k.qn2 = vec(b + "attn2.q_norm.weight", w);
        k.kn2 = vec(b + "attn2.k_norm.weight", w);
        k.q1 = lin(b + "attn1.to_q", w, w);
        k.k1 = lin(b + "attn1.to_k", w, w);
        k.v1 = lin(b + "attn1.to_v", w, w);
        k.o1 = lin(b + "attn1.to_out.0", w, w);
        k.q2 = lin(b + "attn2.to_q", w, w);
        k.k2 = lin(b + "attn2.to_k", w, w);
        k.v2 = lin(b + "attn2.to_v", w, w);
        k.o2 = lin(b + "attn2.to_out.0", w, w);
        k.up = lin(b + "ff.net.0.proj", d.ffn_, w);
        k.down = lin(b + "ff.net.2", w, d.ffn_);
        d.blocks_.push_back(std::move(k));
    }
    return d;
}

std::vector<float> LtxvDit::forward(std::span<const float> latent,
                                    std::uint32_t n,
                                    std::span<const float> coords, float sigma,
                                    std::span<const float> text,
                                    std::uint32_t n_text,
                                    const backend::Ops& op) const {
    return forward(latent, n, coords, std::span<const float>(&sigma, 1), text,
                   n_text, op);
}

std::vector<float> LtxvDit::forward(std::span<const float> latent,
                                    std::uint32_t n,
                                    std::span<const float> coords,
                                    std::span<const float> sigmas,
                                    std::span<const float> text,
                                    std::uint32_t n_text,
                                    const backend::Ops& op) const {
    const std::uint32_t w = width_;
    if (n == 0 || latent.size() != std::size_t{in_ch_} * n) {
        throw std::invalid_argument("ltxv: latent is not C x n");
    }
    if (sigmas.size() != 1 && sigmas.size() != n) {
        throw std::invalid_argument("ltxv: need 1 or n sigmas");
    }
    if (coords.size() != std::size_t{3} * n) {
        throw std::invalid_argument("ltxv: coords is not 3 x n");
    }
    if (n_text == 0 || text.size() != std::size_t{n_text} * cap_ch_) {
        throw std::invalid_argument("ltxv: need n_text x 4096 T5 rows");
    }

    // Tokens: each latent cell's C values through patchify_proj.
    std::vector<float> x(std::size_t{n} * in_ch_), h(std::size_t{n} * w);
    for (std::uint32_t c = 0; c < in_ch_; ++c) {
        for (std::uint32_t t = 0; t < n; ++t) {
            x[std::size_t{t} * in_ch_ + c] = latent[std::size_t{c} * n + t];
        }
    }
    linear(op, patchify_.w, patchify_.b, x, n, h);

    std::vector<float> cosv(std::size_t{n} * w), sinv(cosv.size());
    rope(coords, n, w, theta_, max_pos_, cosv, sinv);

    // adaLN-single on sigma * the timestep scale, once per distinct
    // sigma (frame-0 conditioning gives two); gid[t] is token t's.
    std::vector<float> levels;
    std::vector<std::uint32_t> gid(n);
    for (std::uint32_t t = 0; t < n; ++t) {
        const float s = sigmas[sigmas.size() == 1 ? 0 : t];
        std::size_t g = 0;
        while (g < levels.size() && levels[g] != s) ++g;
        if (g == levels.size()) levels.push_back(s);
        gid[t] = static_cast<std::uint32_t>(g);
    }
    const std::size_t n_lv = levels.size(), w6 = std::size_t{6} * w;
    std::vector<float> c(n_lv * w), m(n_lv * w6);
    std::vector<float> e1(w), sc(w);
    for (std::size_t g = 0; g < n_lv; ++g) {
        const auto tf =
            PixArtDit::timestep_features(levels[g] * t_scale_, t_ch_);
        const std::span<float> cg(&c[g * w], w);
        linear1(op, t1_.w, t1_.b, tf, e1);
        for (float& v : e1) v = silu(v);
        linear1(op, t2_.w, t2_.b, e1, cg);
        for (std::uint32_t i = 0; i < w; ++i) sc[i] = silu(cg[i]);
        linear1(op, ada_.w, ada_.b, sc, std::span<float>(&m[g * w6], w6));
    }

    // Caption projection over the valid T5 rows.
    std::vector<float> u1(std::size_t{n_text} * w), u(u1.size());
    linear(op, cap1_.w, cap1_.b, text, n_text, u1);
    for (float& v : u1) v = gelu_tanh(v);
    linear(op, cap2_.w, cap2_.b, u1, n_text, u);

    std::vector<float> a(std::size_t{n} * w), q(a.size()), k(a.size()),
        v(a.size()), att(a.size()), o(a.size()), up(std::size_t{n} * ffn_),
        kc(std::size_t{n_text} * w), vc(kc.size()), mod(n_lv * w6);
    constexpr float kQkEps = 1e-5f;  // the reference's q / k RMSNorm eps
    for (const Block& b : blocks_) {
        // Rows per level: shift_msa, scale_msa, gate_msa, shift_mlp,
        // scale_mlp, gate_mlp, each the block's table plus that level's m.
        for (std::size_t g = 0; g < n_lv; ++g) {
            for (std::size_t i = 0; i < w6; ++i) {
                mod[g * w6 + i] = b.table[i] + m[g * w6 + i];
            }
        }
        const float* shift1 = &mod[0];
        const float* scale1 = &mod[w];
        const float* gate1 = &mod[2 * w];
        const float* shift2 = &mod[3 * w];
        const float* scale2 = &mod[4 * w];
        const float* gate2 = &mod[5 * w];

        const auto add_gated = [&](const float* gate0) {
            parallel(n, [&](std::size_t t) {
                float* hr = &h[t * w];
                const float* orow = &o[t * w];
                const float* gate =
                    gate0 != nullptr ? gate0 + gid[t] * w6 : nullptr;
                for (std::uint32_t i = 0; i < w; ++i) {
                    hr[i] += gate != nullptr ? gate[i] * orow[i] : orow[i];
                }
            });
        };

        rms_modulate(h, n, w, eps_, shift1, scale1, gid, w6, a);
        linear(op, b.q1.w, b.q1.b, a, n, q);
        linear(op, b.k1.w, b.k1.b, a, n, k);
        linear(op, b.v1.w, b.v1.b, a, n, v);
        rms_norm(q, n, w, kQkEps, b.qn1);
        rms_norm(k, n, w, kQkEps, b.kn1);
        apply_rope(q, n, w, cosv, sinv);
        apply_rope(k, n, w, cosv, sinv);
        attention(op, q, n, k, v, n, heads_, head_dim_, att);
        linear(op, b.o1.w, b.o1.b, att, n, o);
        add_gated(gate1);

        linear(op, b.q2.w, b.q2.b, h, n, q);
        linear(op, b.k2.w, b.k2.b, u, n_text, kc);
        linear(op, b.v2.w, b.v2.b, u, n_text, vc);
        rms_norm(q, n, w, kQkEps, b.qn2);
        rms_norm(kc, n_text, w, kQkEps, b.kn2);
        attention(op, q, n, kc, vc, n_text, heads_, head_dim_, att);
        linear(op, b.o2.w, b.o2.b, att, n, o);
        add_gated(nullptr);

        rms_modulate(h, n, w, eps_, shift2, scale2, gid, w6, a);
        linear(op, b.up.w, b.up.b, a, n, up);
        parallel(n, [&](std::size_t t) {
            float* ur = &up[t * ffn_];
            for (std::uint32_t i = 0; i < ffn_; ++i) ur[i] = gelu_tanh(ur[i]);
        });
        linear(op, b.down.w, b.down.b, up, n, o);
        add_gated(gate2);
    }

    // Final layer: LayerNorm, shift and scale from the table plus each
    // level's c.
    std::vector<float> shift(n_lv * w), scale(n_lv * w);
    for (std::size_t g = 0; g < n_lv; ++g) {
        for (std::uint32_t i = 0; i < w; ++i) {
            shift[g * w + i] = table_[i] + c[g * w + i];
            scale[g * w + i] = table_[w + i] + c[g * w + i];
        }
    }
    ln_modulate(h, n, w, 1e-6f, shift.data(), scale.data(), gid, w, a);
    std::vector<float> y(std::size_t{n} * out_ch_);
    linear(op, proj_out_.w, proj_out_.b, a, n, y);

    std::vector<float> out(std::size_t{out_ch_} * n);
    for (std::uint32_t t = 0; t < n; ++t) {
        for (std::uint32_t ch = 0; ch < out_ch_; ++ch) {
            out[std::size_t{ch} * n + t] = y[std::size_t{t} * out_ch_ + ch];
        }
    }
    return out;
}

}  // namespace locus::image
