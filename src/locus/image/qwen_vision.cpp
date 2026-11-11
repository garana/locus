#include "locus/image/qwen_vision.hpp"

#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

#include "locus/image/dense.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::image {

namespace {

/** Exact GELU (torch.nn.GELU(), approximate="none"). */
void gelu(std::span<float> x) {
    for (float& v : x) {
        v = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752440f));
    }
}

/** Runs fn(i) for i in [0, n) on the thread pool. */
void parallel(std::size_t n, const std::function<void(std::size_t)>& fn) {
    sys::ThreadPool::instance().parallel_for(n, fn);
}

/** RMSNorm with weight w over each of n rows of width d. */
void rms_norm(std::span<const float> x, std::span<const float> w, float eps,
              std::uint32_t n, std::uint32_t d, std::span<float> out) {
    parallel(n, [&](std::size_t t) {
        backend::rmsnorm(x.subspan(t * d, d), w, eps, out.subspan(t * d, d));
    });
}

}  // namespace

QwenVision QwenVision::load(const gguf::GgufFile& g) {
    QwenVision v;
    const auto proj = g.get_string("clip.projector_type");
    if (!proj || *proj != "qwen2.5vl_merger") {
        throw gguf::Error("qwen vision: projector type is not "
                          "qwen2.5vl_merger");
    }
    auto u = [&](std::string_view k) -> std::uint32_t {
        auto x = g.get_uint(k);
        if (!x || *x == 0) {
            throw gguf::Error("qwen vision: missing " + std::string(k));
        }
        return static_cast<std::uint32_t>(*x);
    };
    // Keys llama.cpp's converter does not write default to Qwen2.5-VL's
    // config values.
    auto u_or = [&](std::string_view k, std::uint32_t d) {
        auto x = g.get_uint(k);
        return x && *x != 0 ? static_cast<std::uint32_t>(*x) : d;
    };
    v.patch_ = u("clip.vision.patch_size");
    v.width_ = u("clip.vision.embedding_length");
    v.ffn_ = u("clip.vision.feed_forward_length");
    v.heads_ = u("clip.vision.attention.head_count");
    v.out_ = u("clip.vision.projection_dim");
    const std::uint32_t layers = u("clip.vision.block_count");
    const std::uint32_t wa = u("clip.vision.n_wa_pattern");
    v.merge_ = u_or("clip.vision.spatial_merge_size", 2);
    v.window_ = u_or("clip.vision.window_size", 112);
    const gguf::Value* eps =
        g.find("clip.vision.attention.layer_norm_epsilon");
    if (eps == nullptr || eps->type != gguf::ValueType::kFloat32) {
        throw gguf::Error("qwen vision: missing layer_norm_epsilon");
    }
    v.eps_ = static_cast<float>(eps->f);
    const std::pair<const char*, float*> norms[] = {
        {"clip.vision.image_mean", v.mean_}, {"clip.vision.image_std", v.std_}};
    for (const auto& [key, dst] : norms) {
        const auto* a = g.get_array(key);
        if (a == nullptr || a->size() != 3) {
            throw gguf::Error(std::string("qwen vision: missing ") + key);
        }
        for (int c = 0; c < 3; ++c) {
            dst[c] = static_cast<float>((*a)[c].f);
        }
    }
    const std::uint32_t hd = v.width_ / v.heads_;
    const std::uint32_t unit = v.patch_ * v.merge_;
    if (v.width_ % v.heads_ != 0 || hd % 4 != 0 || v.window_ % unit != 0) {
        throw gguf::Error("qwen vision: inconsistent sizes");
    }
    // The rotary table of the reference's VisionRotaryEmbedding(hd / 2):
    // 1 / 10000^(2j / (hd / 2)), in float32 as torch computes it.
    const float dim = static_cast<float>(hd / 2);
    for (std::uint32_t j = 0; j < hd / 4; ++j) {
        v.inv_freq_.push_back(
            1.0f / std::pow(10000.0f, static_cast<float>(2 * j) / dim));
    }

    const auto info = [&](const std::string& name) {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr) {
            throw gguf::Error("qwen vision: missing tensor " + name);
        }
        return ti;
    };
    const auto f32 = [&](const std::string& name, std::size_t n) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->type != gguf::TensorType::kF32 || ti->nelements() != n) {
            throw gguf::Error("qwen vision: " + name + " must be F32 with " +
                              std::to_string(n) + " values");
        }
        const auto bytes = g.tensor_data(*ti);
        return std::span<const float>(
            reinterpret_cast<const float*>(bytes.data()), n);
    };
    const auto mat = [&](const std::string& name, std::uint32_t rows,
                         std::uint32_t cols) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->ne[0] != cols || ti->ne[1] != rows || ti->n_dims != 2) {
            throw gguf::Error("qwen vision: " + name + " is not " +
                              std::to_string(rows) + " x " +
                              std::to_string(cols));
        }
        return backend::Mat{ti->type, g.tensor_data(*ti).data(), rows, cols};
    };

    // The two per-frame 2D kernels ([width][3][p][p] each) joined into
    // one width x patch_dim() matrix over [channel][frame][y][x].
    const std::uint32_t w = v.width_, p = v.patch_;
    const std::size_t pp = std::size_t{p} * p;
    v.patch_w_.resize(std::size_t{w} * v.patch_dim());
    for (std::uint32_t f = 0; f < kFrames; ++f) {
        const std::string name =
            f == 0 ? "v.patch_embd.weight" : "v.patch_embd.weight.1";
        const gguf::TensorInfo* ti = info(name);
        if (ti->ne[0] != p || ti->ne[1] != p || ti->ne[2] != 3 ||
            ti->ne[3] != w) {
            throw gguf::Error("qwen vision: " + name + " is not " +
                              std::to_string(w) + " x 3 x p x p");
        }
        const auto bytes = g.tensor_data(*ti);
        const std::size_t n = std::size_t{w} * 3 * pp;
        for (std::size_t i = 0; i < n; ++i) {
            float x = 0.0f;
            if (ti->type == gguf::TensorType::kF32) {
                std::memcpy(&x, bytes.data() + i * 4, 4);
            } else if (ti->type == gguf::TensorType::kF16) {
                std::uint16_t h = 0;
                std::memcpy(&h, bytes.data() + i * 2, 2);
                x = backend::f16_to_f32(h);
            } else {
                throw gguf::Error("qwen vision: " + name +
                                  " must be F32 or F16");
            }
            const std::size_t oc = i / pp, yx = i % pp;  // oc = o * 3 + c
            v.patch_w_[(oc * kFrames + f) * pp + yx] = x;
        }
    }
    v.patch_mat_ = backend::Mat{
        gguf::TensorType::kF32,
        reinterpret_cast<const std::byte*>(v.patch_w_.data()), w,
        v.patch_dim()};

    for (std::uint32_t i = 0; i < layers; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        Block k;
        k.ln1 = f32(b + "ln1.weight", w);
        k.ln2 = f32(b + "ln2.weight", w);
        k.q = mat(b + "attn_q.weight", w, w);
        k.k = mat(b + "attn_k.weight", w, w);
        k.v = mat(b + "attn_v.weight", w, w);
        k.out = mat(b + "attn_out.weight", w, w);
        k.q_b = f32(b + "attn_q.bias", w);
        k.k_b = f32(b + "attn_k.bias", w);
        k.v_b = f32(b + "attn_v.bias", w);
        k.out_b = f32(b + "attn_out.bias", w);
        k.gate = mat(b + "ffn_gate.weight", v.ffn_, w);
        k.up = mat(b + "ffn_up.weight", v.ffn_, w);
        k.down = mat(b + "ffn_down.weight", w, v.ffn_);
        k.gate_b = f32(b + "ffn_gate.bias", v.ffn_);
        k.up_b = f32(b + "ffn_up.bias", v.ffn_);
        k.down_b = f32(b + "ffn_down.bias", w);
        k.full = (i + 1) % wa == 0;
        v.blocks_.push_back(k);
    }
    const std::uint32_t mw = w * v.merge_ * v.merge_;
    v.post_ln_ = f32("v.post_ln.weight", w);
    v.mm0_ = mat("mm.0.weight", mw, mw);
    v.mm0_b_ = f32("mm.0.bias", mw);
    v.mm2_ = mat("mm.2.weight", v.out_, mw);
    v.mm2_b_ = f32("mm.2.bias", v.out_);
    return v;
}

std::vector<float> QwenVision::patches(std::span<const std::uint8_t> rgb,
                                       std::uint32_t h,
                                       std::uint32_t w) const {
    const std::uint32_t unit = patch_ * merge_, p = patch_;
    if (h == 0 || w == 0 || h % unit != 0 || w % unit != 0 ||
        rgb.size() != std::size_t{h} * w * 3) {
        throw std::invalid_argument(
            "qwen vision: need h x w RGB, sides multiples of 28");
    }
    const std::uint32_t gh = h / p, gw = w / p;
    const std::size_t pp = std::size_t{p} * p;
    std::vector<float> out(std::size_t{gh} * gw * patch_dim());
    std::size_t n = 0;
    for (std::uint32_t by = 0; by < gh / merge_; ++by) {
        for (std::uint32_t bx = 0; bx < gw / merge_; ++bx) {
            for (std::uint32_t my = 0; my < merge_; ++my) {
                for (std::uint32_t mx = 0; mx < merge_; ++mx, ++n) {
                    const std::uint32_t py = by * merge_ + my;
                    const std::uint32_t px = bx * merge_ + mx;
                    float* dst = &out[n * patch_dim()];
                    for (std::uint32_t c = 0; c < 3; ++c) {
                        for (std::uint32_t i = 0; i < p; ++i) {
                            for (std::uint32_t j = 0; j < p; ++j) {
                                const std::size_t s =
                                    (std::size_t{py * p + i} * w + px * p +
                                     j) * 3 + c;
                                // The processor's rescale (in float64,
                                // then float32) and normalize (float32).
                                const float x = static_cast<float>(
                                    rgb[s] * (1.0 / 255.0));
                                const float y = (x - mean_[c]) / std_[c];
                                for (std::uint32_t f = 0; f < kFrames; ++f) {
                                    dst[(c * kFrames + f) * pp + i * p + j] =
                                        y;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

std::vector<std::vector<std::uint32_t>> QwenVision::windows(
    std::uint32_t gh, std::uint32_t gw) const {
    const std::uint32_t lh = gh / merge_, lw = gw / merge_;
    const std::uint32_t win = window_ / merge_ / patch_;
    // As the reference pads: a full window of padding when the side
    // divides evenly, so the count rounds up past it.
    const std::uint32_t nh = (lh + win - lh % win) / win;
    const std::uint32_t nw = (lw + win - lw % win) / win;
    std::vector<std::vector<std::uint32_t>> out;
    for (std::uint32_t wy = 0; wy < nh; ++wy) {
        for (std::uint32_t wx = 0; wx < nw; ++wx) {
            std::vector<std::uint32_t> t;
            for (std::uint32_t r = wy * win; r < (wy + 1) * win && r < lh;
                 ++r) {
                for (std::uint32_t c = wx * win;
                     c < (wx + 1) * win && c < lw; ++c) {
                    t.push_back(r * lw + c);
                }
            }
            if (!t.empty()) out.push_back(std::move(t));
        }
    }
    return out;
}

void QwenVision::rope(float* t, std::uint32_t y, std::uint32_t x) const {
    const std::uint32_t hd = width_ / heads_, half = hd / 2,
                        quarter = hd / 4;
    // The first quarter of the pairs turns with the row, the second
    // with the column; pair i is (i, i + hd / 2), as rotate_half.
    float cs[256], sn[256];
    for (std::uint32_t i = 0; i < half; ++i) {
        const float a = i < quarter
                            ? static_cast<float>(y) * inv_freq_[i]
                            : static_cast<float>(x) * inv_freq_[i - quarter];
        cs[i] = std::cos(a);
        sn[i] = std::sin(a);
    }
    for (std::uint32_t h = 0; h < heads_; ++h) {
        float* b = t + std::size_t{h} * hd;
        for (std::uint32_t i = 0; i < half; ++i) {
            const float lo = b[i], hi = b[i + half];
            b[i] = lo * cs[i] - hi * sn[i];
            b[i + half] = hi * cs[i] + lo * sn[i];
        }
    }
}

std::vector<float> QwenVision::encode(std::span<const float> patches,
                                      std::uint32_t gh, std::uint32_t gw,
                                      const backend::Ops& op) const {
    if (gh == 0 || gw == 0 || gh % merge_ != 0 || gw % merge_ != 0 ||
        patches.size() != std::size_t{gh} * gw * patch_dim()) {
        throw std::invalid_argument("qwen vision: need gh x gw patches");
    }
    const std::uint32_t n = gh * gw, w = width_, hd = w / heads_;
    const std::uint32_t unit = merge_ * merge_;
    if (hd / 2 > 256) {
        throw std::invalid_argument("qwen vision: head too wide");
    }
    std::vector<float> x(std::size_t{n} * w);
    dense::linear(op, patch_mat_, {}, patches, n, x);

    // Each token's patch row and column, in merge order.
    std::vector<std::uint32_t> py(n), px(n);
    for (std::uint32_t t = 0; t < n; ++t) {
        const std::uint32_t g = t / unit, m = t % unit;
        py[t] = g / (gw / merge_) * merge_ + m / merge_;
        px[t] = g % (gw / merge_) * merge_ + m % merge_;
    }
    const auto wins = windows(gh, gw);

    std::vector<float> h(std::size_t{n} * w), q(h.size()), k(h.size()),
        v(h.size()), att(h.size()), o(h.size()),
        gate(std::size_t{n} * ffn_), up(gate.size());
    std::vector<float> wq, wk, wv, wo;
    for (const Block& b : blocks_) {
        // x += out(attention(rope(q), rope(k), v)) over RMSNorm(x)
        rms_norm(x, b.ln1, eps_, n, w, h);
        dense::linear(op, b.q, b.q_b, h, n, q);
        dense::linear(op, b.k, b.k_b, h, n, k);
        dense::linear(op, b.v, b.v_b, h, n, v);
        parallel(n, [&](std::size_t t) {
            rope(&q[t * w], py[t], px[t]);
            rope(&k[t * w], py[t], px[t]);
        });
        if (b.full) {
            dense::attention(op, q, n, k, v, n, heads_, hd, att);
        } else {
            for (const auto& win : wins) {
                const std::uint32_t m =
                    static_cast<std::uint32_t>(win.size()) * unit;
                wq.resize(std::size_t{m} * w);
                wk.resize(wq.size());
                wv.resize(wq.size());
                wo.resize(wq.size());
                const auto row = [&](std::uint32_t i) {
                    return std::size_t{win[i / unit] * unit + i % unit} * w;
                };
                for (std::uint32_t i = 0; i < m; ++i) {
                    const std::size_t s = row(i), d = std::size_t{i} * w;
                    std::memcpy(&wq[d], &q[s], w * sizeof(float));
                    std::memcpy(&wk[d], &k[s], w * sizeof(float));
                    std::memcpy(&wv[d], &v[s], w * sizeof(float));
                }
                dense::attention(op, wq, m, wk, wv, m, heads_, hd, wo);
                for (std::uint32_t i = 0; i < m; ++i) {
                    std::memcpy(&att[row(i)], &wo[std::size_t{i} * w],
                                w * sizeof(float));
                }
            }
        }
        dense::linear(op, b.out, b.out_b, att, n, o);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];
        // x += down(silu(gate(h)) * up(h)) over RMSNorm(x)
        rms_norm(x, b.ln2, eps_, n, w, h);
        dense::linear(op, b.gate, b.gate_b, h, n, gate);
        dense::linear(op, b.up, b.up_b, h, n, up);
        backend::silu_mul(gate, up, gate);
        dense::linear(op, b.down, b.down_b, gate, n, o);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];
    }

    // The merger: RMSNorm per patch; each 2 x 2 group (contiguous in
    // merge order) is one 4 * width row; Linear, GELU, Linear.
    rms_norm(x, post_ln_, eps_, n, w, h);
    const std::uint32_t nt = n / unit;
    std::vector<float> a(std::size_t{nt} * w * unit),
        out(std::size_t{nt} * out_);
    dense::linear(op, mm0_, mm0_b_, h, nt, a);
    gelu(a);
    dense::linear(op, mm2_, mm2_b_, a, nt, out);
    return out;
}

}  // namespace locus::image
