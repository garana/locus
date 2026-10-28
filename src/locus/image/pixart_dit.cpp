#include "locus/image/pixart_dit.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

#include "locus/sys/thread_pool.hpp"

namespace locus::image {

namespace {

/** Tokens per matvec_batch call: keeps the inputs in cache. */
constexpr std::uint32_t kTokenBlock = 64;

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

/**
 * out[r * n + t] = w[r] . x[t] for n token-major inputs: the backend's
 * matvec_batch layout (row-major output), or n matvec calls when the
 * backend has none.
 */
void mat_batch(const backend::Ops& op, const backend::Mat& w,
               std::span<const float> x, std::uint32_t n,
               std::span<float> out) {
    if (op.matvec_batch != nullptr) {
        op.matvec_batch(w, x.first(std::size_t{n} * w.cols),
                        out.first(std::size_t{n} * w.rows), n);
        return;
    }
    std::vector<float> o(w.rows);
    for (std::uint32_t t = 0; t < n; ++t) {
        op.matvec(w, x.subspan(std::size_t{t} * w.cols, w.cols), o);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out[std::size_t{r} * n + t] = o[r];
        }
    }
}

/**
 * out[t][r] = w[r] . x[t] + b[r] for n tokens, token-major in and out.
 * Rows are split across threads; each slice runs over the tokens in
 * blocks of kTokenBlock.
 */
void linear(const backend::Ops& op, const backend::Mat& w,
            std::span<const float> b, std::span<const float> x,
            std::uint32_t n, std::span<float> out) {
    const std::uint32_t rows = w.rows, cols = w.cols;
    const std::uint32_t threads =
        std::max(1u, std::thread::hardware_concurrency());
    const std::uint32_t slices =
        op.mt_safe ? std::max(1u, std::min(rows / 16, 2 * threads)) : 1;
    const std::uint32_t per = (rows + slices - 1) / slices;
    parallel(slices, [&](std::size_t s) {
        const std::uint32_t r0 = static_cast<std::uint32_t>(s) * per;
        if (r0 >= rows) return;
        const std::uint32_t nr = std::min(per, rows - r0);
        const backend::Mat ws = backend::mat_rows(w, r0, nr);
        std::vector<float> tmp(std::size_t{nr} * kTokenBlock);
        for (std::uint32_t t0 = 0; t0 < n; t0 += kTokenBlock) {
            const std::uint32_t nt = std::min(kTokenBlock, n - t0);
            mat_batch(op, ws, x.subspan(std::size_t{t0} * cols), nt, tmp);
            for (std::uint32_t t = 0; t < nt; ++t) {
                float* o = &out[std::size_t{t0 + t} * rows + r0];
                for (std::uint32_t r = 0; r < nr; ++r) {
                    o[r] = tmp[std::size_t{r} * nt + t] + b[r0 + r];
                }
            }
        }
    });
}

/** One vector: out = w x + b. */
void linear1(const backend::Ops& op, const backend::Mat& w,
             std::span<const float> b, std::span<const float> x,
             std::span<float> out) {
    op.matvec(w, x, out);
    for (std::uint32_t r = 0; r < w.rows; ++r) out[r] += b[r];
}

/**
 * LayerNorm without affine over each of n rows of width d, then
 * out = ln(x) * (1 + scale) + shift (scale and shift: d floats).
 */
void ln_modulate(std::span<const float> x, std::uint32_t n, std::uint32_t d,
                 float eps, const float* shift, const float* scale,
                 std::span<float> out) {
    parallel(n, [&](std::size_t t) {
        const float* xi = &x[t * d];
        float* o = &out[t * d];
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
 * Multi-head attention without a mask: nq queries (q, token-major,
 * heads * hd wide) over nk keys and values (k, v, same layout),
 * softmax(q k^T / sqrt(hd)) v, into out (nq rows, token-major).
 *
 * Each head's keys are copied to a [nk][hd] matrix and its values to
 * [hd][nk], so the scores and the weighted sum run through the
 * backend's batched matvec, kTokenBlock queries at a time.
 */
void attention(const backend::Ops& op, std::span<const float> q,
               std::uint32_t nq, std::span<const float> k,
               std::span<const float> v, std::uint32_t nk,
               std::uint32_t heads, std::uint32_t hd, std::span<float> out) {
    const std::uint32_t w = heads * hd;
    std::vector<float> kh(std::size_t{heads} * nk * hd),
        vt(std::size_t{heads} * hd * nk);
    parallel(heads, [&](std::size_t h) {
        float* kd = &kh[h * nk * hd];
        float* vd = &vt[h * hd * nk];
        for (std::uint32_t j = 0; j < nk; ++j) {
            const float* ks = &k[std::size_t{j} * w + h * hd];
            const float* vs = &v[std::size_t{j} * w + h * hd];
            std::memcpy(kd + std::size_t{j} * hd, ks, hd * sizeof(float));
            for (std::uint32_t e = 0; e < hd; ++e) {
                vd[std::size_t{e} * nk + j] = vs[e];
            }
        }
    });
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const std::uint32_t qblocks = (nq + kTokenBlock - 1) / kTokenBlock;
    parallel(std::size_t{heads} * qblocks, [&](std::size_t item) {
        const std::uint32_t h = static_cast<std::uint32_t>(item / qblocks);
        const std::uint32_t q0 =
            static_cast<std::uint32_t>(item % qblocks) * kTokenBlock;
        const std::uint32_t m = std::min(kTokenBlock, nq - q0);
        const backend::Mat km{gguf::TensorType::kF32,
                              reinterpret_cast<const std::byte*>(
                                  &kh[std::size_t{h} * nk * hd]),
                              nk, hd};
        const backend::Mat vm{gguf::TensorType::kF32,
                              reinterpret_cast<const std::byte*>(
                                  &vt[std::size_t{h} * hd * nk]),
                              hd, nk};
        std::vector<float> qb(std::size_t{m} * hd), s(std::size_t{nk} * m),
            p(std::size_t{m} * nk), ob(std::size_t{hd} * m);
        for (std::uint32_t t = 0; t < m; ++t) {
            std::memcpy(&qb[std::size_t{t} * hd],
                        &q[std::size_t{q0 + t} * w + h * hd],
                        hd * sizeof(float));
        }
        mat_batch(op, km, qb, m, s);  // s[j * m + t]
        for (std::uint32_t t = 0; t < m; ++t) {
            float* pt = &p[std::size_t{t} * nk];
            float mx = -INFINITY;
            for (std::uint32_t j = 0; j < nk; ++j) {
                pt[j] = s[std::size_t{j} * m + t] * scale;
                mx = std::max(mx, pt[j]);
            }
            float sum = 0.0f;
            for (std::uint32_t j = 0; j < nk; ++j) {
                pt[j] = std::exp(pt[j] - mx);
                sum += pt[j];
            }
            const float inv = 1.0f / sum;
            for (std::uint32_t j = 0; j < nk; ++j) pt[j] *= inv;
        }
        mat_batch(op, vm, p, m, ob);  // ob[e * m + t]
        for (std::uint32_t t = 0; t < m; ++t) {
            float* o = &out[std::size_t{q0 + t} * w + h * hd];
            for (std::uint32_t e = 0; e < hd; ++e) {
                o[e] = ob[std::size_t{e} * m + t];
            }
        }
    });
}

}  // namespace

std::vector<float> PixArtDit::pos_embed(std::uint32_t dim, std::uint32_t grid,
                                        std::uint32_t base_size,
                                        float interpolation_scale) {
    // numpy: the grid in float32 (arange / (grid / base) / scale), the
    // frequencies and the sin/cos in float64.
    const std::uint32_t quarter = dim / 4;
    std::vector<double> omega(quarter);
    for (std::uint32_t k = 0; k < quarter; ++k) {
        omega[k] = 1.0 / std::pow(10000.0, static_cast<double>(k) /
                                               (dim / 4.0));
    }
    const float step = static_cast<float>(static_cast<double>(grid) /
                                          base_size);
    std::vector<float> coord(grid);
    for (std::uint32_t i = 0; i < grid; ++i) {
        coord[i] = static_cast<float>(i) / step / interpolation_scale;
    }
    std::vector<float> pe(std::size_t{grid} * grid * dim);
    for (std::uint32_t i = 0; i < grid; ++i) {
        for (std::uint32_t j = 0; j < grid; ++j) {
            float* row = &pe[(std::size_t{i} * grid + j) * dim];
            const double col = coord[j], line = coord[i];
            for (std::uint32_t k = 0; k < quarter; ++k) {
                row[k] = static_cast<float>(std::sin(col * omega[k]));
                row[quarter + k] = static_cast<float>(std::cos(col * omega[k]));
                row[2 * quarter + k] =
                    static_cast<float>(std::sin(line * omega[k]));
                row[3 * quarter + k] =
                    static_cast<float>(std::cos(line * omega[k]));
            }
        }
    }
    return pe;
}

std::vector<float> PixArtDit::timestep_features(float t,
                                                std::uint32_t channels) {
    // torch, all float32: exponent = -ln(10000) * k / half, exp, t * f.
    const std::uint32_t half = channels / 2;
    const float ln = static_cast<float>(-std::log(10000.0));
    std::vector<float> tf(channels);
    for (std::uint32_t k = 0; k < half; ++k) {
        const float e = ln * static_cast<float>(k) / static_cast<float>(half);
        const float a = t * std::exp(e);
        tf[k] = std::cos(a);
        tf[half + k] = std::sin(a);
    }
    return tf;
}

PixArtDit PixArtDit::load(const gguf::GgufFile& g) {
    if (g.get_string("general.architecture") != "pixart") {
        throw gguf::Error("pixart: general.architecture is not pixart");
    }
    PixArtDit d;
    auto u = [&](std::string_view k) -> std::uint32_t {
        auto x = g.get_uint(k);
        if (!x || *x == 0 || *x > (1u << 20)) {
            throw gguf::Error("pixart: missing or bad " + std::string(k));
        }
        return static_cast<std::uint32_t>(*x);
    };
    auto f = [&](std::string_view k) {
        const gguf::Value* v = g.find(k);
        if (v == nullptr || v->type != gguf::ValueType::kFloat32) {
            throw gguf::Error("pixart: missing " + std::string(k));
        }
        return static_cast<float>(v->f);
    };
    const std::uint32_t layers = u("pixart.block_count");
    d.width_ = u("pixart.embedding_length");
    d.ffn_ = u("pixart.feed_forward_length");
    d.heads_ = u("pixart.attention.head_count");
    d.head_dim_ = u("pixart.attention.key_length");
    d.eps_ = f("pixart.norm_epsilon");
    d.in_ch_ = u("pixart.in_channels");
    d.out_ch_ = u("pixart.out_channels");
    d.patch_ = u("pixart.patch_size");
    d.sample_ = u("pixart.sample_size");
    d.cap_ch_ = u("pixart.caption_channels");
    d.t_ch_ = u("pixart.timestep.channels");
    const std::uint32_t base = u("pixart.pos_embed.base_size");
    const float interp = f("pixart.pos_embed.interpolation_scale");
    if (d.sample_ == 128) {
        throw gguf::Error("pixart: the 1024 model is not supported");
    }
    if (std::uint64_t{d.heads_} * d.head_dim_ != d.width_ ||
        d.sample_ % d.patch_ != 0 || d.width_ % 4 != 0 || d.t_ch_ % 2 != 0 ||
        !(interp > 0.0f) || g.get_bool("pixart.timestep.flip_sin_to_cos")
                                    .value_or(false) != true) {
        throw gguf::Error("pixart: inconsistent sizes");
    }
    d.grid_ = d.sample_ / d.patch_;

    const auto info = [&](const std::string& name) {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr) {
            throw gguf::Error("pixart: missing tensor " + name);
        }
        return ti;
    };
    const auto vec = [&](const std::string& name, std::size_t n) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->type != gguf::TensorType::kF32 || ti->nelements() != n) {
            throw gguf::Error("pixart: " + name + " must be F32 with " +
                              std::to_string(n) + " values");
        }
        const auto bytes = g.tensor_data(*ti);
        std::vector<float> out(n);
        std::memcpy(out.data(), bytes.data(), n * sizeof(float));
        return out;
    };
    // A Linear [rows, cols]; the patch conv [D, C, p, p] is one too.
    const auto lin = [&](const std::string& name, std::uint32_t rows,
                         std::uint32_t cols) {
        const gguf::TensorInfo* ti = info(name + ".weight");
        if (ti->type != gguf::TensorType::kF32 ||
            ti->nelements() != std::uint64_t{rows} * cols ||
            ti->ne[ti->n_dims - 1] != rows) {
            throw gguf::Error("pixart: " + name + ".weight is not F32 " +
                              std::to_string(rows) + " x " +
                              std::to_string(cols));
        }
        Linear l;
        l.w = backend::Mat{ti->type, g.tensor_data(*ti).data(), rows, cols};
        l.b = vec(name + ".bias", rows);
        return l;
    };

    const std::uint32_t w = d.width_;
    d.patch_embd_ = lin("pos_embed.proj", w, d.in_ch_ * d.patch_ * d.patch_);
    const std::string te = "adaln_single.emb.timestep_embedder.";
    d.t1_ = lin(te + "linear_1", w, d.t_ch_);
    d.t2_ = lin(te + "linear_2", w, w);
    d.ada_ = lin("adaln_single.linear", 6 * w, w);
    d.cap1_ = lin("caption_projection.linear_1", w, d.cap_ch_);
    d.cap2_ = lin("caption_projection.linear_2", w, w);
    d.table_ = vec("scale_shift_table", std::size_t{2} * w);
    d.proj_out_ = lin("proj_out", d.patch_ * d.patch_ * d.out_ch_, w);
    for (std::uint32_t i = 0; i < layers; ++i) {
        const std::string b = "transformer_blocks." + std::to_string(i) + ".";
        Block k;
        k.table = vec(b + "scale_shift_table", std::size_t{6} * w);
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
    d.pos_ = pos_embed(w, d.grid_, base, interp);
    return d;
}

std::vector<float> PixArtDit::forward(std::span<const float> latent, float t,
                                      std::span<const float> text,
                                      std::uint32_t n_text,
                                      const backend::Ops& op) const {
    const std::uint32_t w = width_, g = grid_, p = patch_, s = sample_;
    const std::uint32_t n = g * g;
    if (latent.size() != std::size_t{in_ch_} * s * s) {
        throw std::invalid_argument("pixart: latent is not C x S x S");
    }
    if (n_text == 0 || text.size() != std::size_t{n_text} * cap_ch_) {
        throw std::invalid_argument("pixart: need n_text x 4096 T5 rows");
    }

    // Patch embed (conv as a Linear over each patch's c, p, q values)
    // plus the position table.
    const std::uint32_t pv = in_ch_ * p * p;
    std::vector<float> patches(std::size_t{n} * pv), h(std::size_t{n} * w);
    for (std::uint32_t i = 0; i < g; ++i) {
        for (std::uint32_t j = 0; j < g; ++j) {
            float* o = &patches[(std::size_t{i} * g + j) * pv];
            for (std::uint32_t c = 0; c < in_ch_; ++c) {
                for (std::uint32_t a = 0; a < p; ++a) {
                    for (std::uint32_t b = 0; b < p; ++b) {
                        *o++ = latent[(std::size_t{c} * s + p * i + a) * s +
                                      p * j + b];
                    }
                }
            }
        }
    }
    linear(op, patch_embd_.w, patch_embd_.b, patches, n, h);
    for (std::size_t i = 0; i < h.size(); ++i) h[i] += pos_[i];

    // adaLN-single: c = embedded timestep, m = the shared [6][w] vector.
    const auto tf = timestep_features(t, t_ch_);
    std::vector<float> e1(w), c(w), sc(w), m(std::size_t{6} * w);
    linear1(op, t1_.w, t1_.b, tf, e1);
    for (float& v : e1) v = silu(v);
    linear1(op, t2_.w, t2_.b, e1, c);
    for (std::uint32_t i = 0; i < w; ++i) sc[i] = silu(c[i]);
    linear1(op, ada_.w, ada_.b, sc, m);

    // Caption projection over the valid T5 rows.
    std::vector<float> u1(std::size_t{n_text} * w), u(u1.size());
    linear(op, cap1_.w, cap1_.b, text, n_text, u1);
    for (float& v : u1) v = gelu_tanh(v);
    linear(op, cap2_.w, cap2_.b, u1, n_text, u);

    std::vector<float> a(std::size_t{n} * w), q(a.size()), k(a.size()),
        v(a.size()), att(a.size()), o(a.size()), up(std::size_t{n} * ffn_),
        kc(std::size_t{n_text} * w), vc(kc.size()), mod(std::size_t{6} * w);
    for (const Block& b : blocks_) {
        // Rows: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
        // gate_mlp, each the block's table plus the shared m.
        for (std::size_t i = 0; i < mod.size(); ++i) mod[i] = b.table[i] + m[i];
        const float* shift1 = &mod[0];
        const float* scale1 = &mod[w];
        const float* gate1 = &mod[2 * w];
        const float* shift2 = &mod[3 * w];
        const float* scale2 = &mod[4 * w];
        const float* gate2 = &mod[5 * w];

        ln_modulate(h, n, w, eps_, shift1, scale1, a);
        linear(op, b.q1.w, b.q1.b, a, n, q);
        linear(op, b.k1.w, b.k1.b, a, n, k);
        linear(op, b.v1.w, b.v1.b, a, n, v);
        attention(op, q, n, k, v, n, heads_, head_dim_, att);
        linear(op, b.o1.w, b.o1.b, att, n, o);
        for (std::size_t i = 0; i < h.size(); ++i) h[i] += gate1[i % w] * o[i];

        linear(op, b.q2.w, b.q2.b, h, n, q);
        linear(op, b.k2.w, b.k2.b, u, n_text, kc);
        linear(op, b.v2.w, b.v2.b, u, n_text, vc);
        attention(op, q, n, kc, vc, n_text, heads_, head_dim_, att);
        linear(op, b.o2.w, b.o2.b, att, n, o);
        for (std::size_t i = 0; i < h.size(); ++i) h[i] += o[i];

        ln_modulate(h, n, w, eps_, shift2, scale2, a);
        linear(op, b.up.w, b.up.b, a, n, up);
        for (float& x : up) x = gelu_tanh(x);
        linear(op, b.down.w, b.down.b, up, n, o);
        for (std::size_t i = 0; i < h.size(); ++i) h[i] += gate2[i % w] * o[i];
    }

    // Final layer: shift and scale from the table plus c itself.
    std::vector<float> shift(w), scale(w);
    for (std::uint32_t i = 0; i < w; ++i) {
        shift[i] = table_[i] + c[i];
        scale[i] = table_[w + i] + c[i];
    }
    ln_modulate(h, n, w, 1e-6f, shift.data(), scale.data(), a);
    const std::uint32_t py = p * p * out_ch_;
    std::vector<float> y(std::size_t{n} * py);
    linear(op, proj_out_.w, proj_out_.b, a, n, y);

    // Unpatchify: out[k][p i + a][p j + b] = y[g i + j][(p a + b) C + k].
    std::vector<float> out(std::size_t{out_ch_} * s * s);
    for (std::uint32_t i = 0; i < g; ++i) {
        for (std::uint32_t j = 0; j < g; ++j) {
            const float* yr = &y[(std::size_t{i} * g + j) * py];
            for (std::uint32_t aa = 0; aa < p; ++aa) {
                for (std::uint32_t bb = 0; bb < p; ++bb) {
                    for (std::uint32_t kk = 0; kk < out_ch_; ++kk) {
                        out[(std::size_t{kk} * s + p * i + aa) * s + p * j +
                            bb] = yr[(aa * p + bb) * out_ch_ + kk];
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace locus::image
