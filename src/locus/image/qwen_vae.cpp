#include "locus/image/qwen_vae.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/image/dense.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::image {

namespace {

using Conv = QwenVae::Conv;
using Layer = QwenVae::Layer;

/** Runs fn(i) for i in [0, n) on the thread pool. */
void parallel(std::size_t n, const std::function<void(std::size_t)>& fn) {
    sys::ThreadPool::instance().parallel_for(n, fn);
}

/** The reader of the qwen_image_vae GGUF's tensors. */
struct Reader {
    const gguf::GgufFile& g;

    const gguf::TensorInfo& info(const std::string& name) const {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr) {
            throw gguf::Error("qwen vae: missing tensor " + name);
        }
        if (ti->type != gguf::TensorType::kF32) {
            throw gguf::Error("qwen vae: " + name + " must be F32");
        }
        return *ti;
    }

    std::span<const float> f32(const std::string& name) const {
        const auto& ti = info(name);
        const auto b = g.tensor_data(ti);
        return {reinterpret_cast<const float*>(b.data()),
                static_cast<std::size_t>(ti.nelements())};
    }

    std::vector<float> vec(const std::string& name, std::size_t n) const {
        const auto v = f32(name);
        if (v.size() != n) {
            throw gguf::Error("qwen vae: " + name + " needs " +
                              std::to_string(n) + " values");
        }
        return {v.begin(), v.end()};
    }

    /**
     * A Conv3d ([out][in][kt * k * k]) or Conv2d ([out][in][k][k])
     * kernel as the 2D conv a single frame sees: the last time slice
     * (the frame itself; the slices before it see the causal zero
     * padding).
     */
    Conv conv(const std::string& p, std::uint32_t in, std::uint32_t out,
              std::uint32_t k) const {
        const auto w = f32(p + ".weight");
        const std::size_t kk = std::size_t{k} * k, per = std::size_t{in} * kk;
        if (w.size() % (std::size_t{out} * per) != 0 || w.empty()) {
            throw gguf::Error("qwen vae: " + p + ".weight is not " +
                              std::to_string(out) + " x " +
                              std::to_string(in) + " x k x " +
                              std::to_string(k) + " x " + std::to_string(k));
        }
        const std::size_t kt = w.size() / (std::size_t{out} * per);
        Conv c;
        c.in = in;
        c.out = out;
        c.k = k;
        c.w.resize(std::size_t{out} * per);
        for (std::size_t o = 0; o < out; ++o) {
            for (std::size_t i = 0; i < in; ++i) {
                const float* src = &w[((o * in + i) * kt + kt - 1) * kk];
                std::copy(src, src + kk, &c.w[(o * in + i) * kk]);
            }
        }
        c.b = vec(p + ".bias", out);
        return c;
    }

    QwenVae::Res res(const std::string& p, std::uint32_t in,
                     std::uint32_t out) const {
        QwenVae::Res r;
        r.g1 = vec(p + ".norm1.gamma", in);
        r.c1 = conv(p + ".conv1", in, out, 3);
        r.g2 = vec(p + ".norm2.gamma", out);
        r.c2 = conv(p + ".conv2", out, out, 3);
        if (in != out) r.shortcut = conv(p + ".conv_shortcut", in, out, 1);
        return r;
    }

    Layer res_layer(const std::string& p, std::uint32_t in,
                    std::uint32_t out) const {
        Layer l{Layer::Kind::kRes, res(p, in, out), {}, {}};
        return l;
    }

    Layer attn_layer(const std::string& p, std::uint32_t c) const {
        Layer l{Layer::Kind::kAttn, {}, {}, {}};
        l.attn.g = vec(p + ".norm.gamma", c);
        l.attn.qkv = conv(p + ".to_qkv", c, 3 * c, 1);
        l.attn.proj = conv(p + ".proj", c, c, 1);
        return l;
    }
};

std::vector<std::uint32_t> uints(const gguf::GgufFile& g,
                                 const std::string& key) {
    const auto* a = g.get_array(key);
    if (a == nullptr) throw gguf::Error("qwen vae: missing " + key);
    std::vector<std::uint32_t> v;
    for (const auto& x : *a) {
        const std::int64_t i = x.type == gguf::ValueType::kInt32
                                   ? x.i
                                   : static_cast<std::int64_t>(x.u);
        if (i <= 0) throw gguf::Error("qwen vae: bad " + key);
        v.push_back(static_cast<std::uint32_t>(i));
    }
    return v;
}

std::vector<float> floats(const gguf::GgufFile& g, const std::string& key,
                          std::size_t n) {
    const auto* a = g.get_array(key);
    if (a == nullptr || (n != 0 && a->size() != n)) {
        throw gguf::Error("qwen vae: missing or wrong " + key);
    }
    std::vector<float> v;
    for (const auto& x : *a) v.push_back(static_cast<float>(x.f));
    return v;
}

/** The 2D conv, [in][h][w] to [out][h'][w']: conv2d_vulkan on the
 *  Vulkan image ops (i#587), else the CPU's. */
std::vector<float> conv2d(const Conv& c, std::span<const float> x,
                          std::uint32_t h, std::uint32_t w,
                          std::uint32_t stride, std::uint32_t pad,
                          const backend::Ops& op) {
    const std::uint32_t oh = (h + 2 * pad - c.k) / stride + 1,
                        ow = (w + 2 * pad - c.k) / stride + 1;
    std::vector<float> out(std::size_t{c.out} * oh * ow);
    if (&op == &backend::vulkan_image_ops()) {
        backend::conv2d_vulkan(x, c.in, h, w, c.w, c.out, c.k, c.k, c.b,
                               stride, pad, 1, out);
    } else {
        backend::conv2d_mt(x, c.in, h, w, c.w, c.out, c.k, c.k, c.b, stride,
                           pad, 1, out);
    }
    return out;
}

/** The reference's RMS_norm then SiLU (silu false: the norm only): per
 *  pixel, x / max(|x|, 1e-12) * sqrt(C) * gamma[c]. */
std::vector<float> norm(std::span<const float> x, std::span<const float> g,
                        std::uint32_t c, std::size_t plane, bool silu) {
    std::vector<float> out(x.size());
    const float scale = std::sqrt(static_cast<float>(c));
    constexpr std::size_t kChunk = 256;
    parallel((plane + kChunk - 1) / kChunk, [&](std::size_t item) {
        const std::size_t p0 = item * kChunk,
                          p1 = std::min(plane, p0 + kChunk);
        for (std::size_t p = p0; p < p1; ++p) {
            float ss = 0.0f;
            for (std::uint32_t k = 0; k < c; ++k) {
                const float v = x[k * plane + p];
                ss += v * v;
            }
            const float d = std::max(std::sqrt(ss), 1e-12f);
            for (std::uint32_t k = 0; k < c; ++k) {
                float v = x[k * plane + p] / d * scale * g[k];
                if (silu) v = v / (1.0f + std::exp(-v));
                out[k * plane + p] = v;
            }
        }
    });
    return out;
}

/** A residual block on [in][h][w]. */
std::vector<float> res_block(const QwenVae::Res& r, std::span<const float> x,
                             std::uint32_t h, std::uint32_t w,
                             const backend::Ops& op) {
    const std::size_t plane = std::size_t{h} * w;
    auto a =
        conv2d(r.c1, norm(x, r.g1, r.c1.in, plane, true), h, w, 1, 1, op);
    auto b =
        conv2d(r.c2, norm(a, r.g2, r.c2.in, plane, true), h, w, 1, 1, op);
    if (r.shortcut.out != 0) {
        const auto s = conv2d(r.shortcut, x, h, w, 1, 0, op);
        for (std::size_t i = 0; i < b.size(); ++i) b[i] += s[i];
    } else {
        for (std::size_t i = 0; i < b.size(); ++i) b[i] += x[i];
    }
    return b;
}

/** The single-head attention block over the h * w pixels. */
std::vector<float> attn_block(const QwenVae::Attn& a,
                              std::span<const float> x, std::uint32_t h,
                              std::uint32_t w, const backend::Ops& op) {
    const std::uint32_t c = a.proj.in;
    const std::size_t plane = std::size_t{h} * w;
    const auto qkv =
        conv2d(a.qkv, norm(x, a.g, c, plane, false), h, w, 1, 0, op);
    // [3c][plane] to token-major q, k and v, [plane][c] each.
    std::vector<float> q(plane * c), k(q.size()), v(q.size()),
        o(q.size());
    parallel(plane, [&](std::size_t p) {
        for (std::uint32_t i = 0; i < c; ++i) {
            q[p * c + i] = qkv[i * plane + p];
            k[p * c + i] = qkv[(c + i) * plane + p];
            v[p * c + i] = qkv[(2 * c + i) * plane + p];
        }
    });
    const auto n = static_cast<std::uint32_t>(plane);
    dense::attention(op, q, n, k, v, n, 1, c, o);
    std::vector<float> t(plane * c);
    parallel(c, [&](std::size_t i) {
        for (std::size_t p = 0; p < plane; ++p) t[i * plane + p] = o[p * c + i];
    });
    auto out = conv2d(a.proj, t, h, w, 1, 0, op);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] += x[i];
    return out;
}

}  // namespace

QwenVae QwenVae::load(const gguf::GgufFile& g) {
    const std::string p = "qwen_image_vae.";
    auto u = [&](const std::string& k) -> std::uint32_t {
        const auto x = g.get_uint(p + k);
        if (!x || *x == 0) throw gguf::Error("qwen vae: missing " + p + k);
        return static_cast<std::uint32_t>(*x);
    };
    QwenVae v;
    const std::uint32_t base = u("base_dim");
    v.z_ = u("z_dim");
    const std::uint32_t nres = u("num_res_blocks");
    const auto mult = uints(g, p + "dim_mult");
    // An empty attn_scales (the real model's) may be left out.
    const auto attn_scales = g.get_array(p + "attn_scales") != nullptr
                                 ? floats(g, p + "attn_scales", 0)
                                 : std::vector<float>{};
    v.mean_ = floats(g, p + "latents_mean", v.z_);
    v.std_ = floats(g, p + "latents_std", v.z_);
    if (mult.size() < 2) throw gguf::Error("qwen vae: bad dim_mult");
    v.downs_ = static_cast<std::uint32_t>(mult.size() - 1);
    const Reader r{g};

    // Encoder, when present (a decoder-only GGUF is allowed).
    if (g.find_tensor("encoder.conv_in.weight") != nullptr) {
        std::vector<std::uint32_t> dims = {base};
        for (auto m : mult) dims.push_back(base * m);
        v.enc_in_ = r.conv("encoder.conv_in", 3, dims[0], 3);
        float scale = 1.0f;
        std::uint32_t idx = 0;
        for (std::size_t i = 0; i + 1 < dims.size(); ++i) {
            std::uint32_t in = dims[i];
            const std::uint32_t out = dims[i + 1];
            for (std::uint32_t j = 0; j < nres; ++j) {
                const std::string b =
                    "encoder.down_blocks." + std::to_string(idx++);
                v.enc_.push_back(r.res_layer(b, in, out));
                if (std::find(attn_scales.begin(), attn_scales.end(),
                              scale) != attn_scales.end()) {
                    v.enc_.push_back(r.attn_layer(
                        "encoder.down_blocks." + std::to_string(idx++), out));
                }
                in = out;
            }
            if (i + 1 != mult.size()) {
                Layer l{Layer::Kind::kDown, {}, {}, {}};
                l.conv = r.conv("encoder.down_blocks." +
                                    std::to_string(idx++) + ".resample.1",
                                out, out, 3);
                v.enc_.push_back(std::move(l));
                scale /= 2.0f;
            }
        }
        const std::uint32_t top = dims.back();
        v.enc_.push_back(r.res_layer("encoder.mid_block.resnets.0", top, top));
        v.enc_.push_back(r.attn_layer("encoder.mid_block.attentions.0", top));
        v.enc_.push_back(r.res_layer("encoder.mid_block.resnets.1", top, top));
        v.enc_norm_ = r.vec("encoder.norm_out.gamma", top);
        v.enc_out_ = r.conv("encoder.conv_out", top, 2 * v.z_, 3);
        v.quant_ = r.conv("quant_conv", 2 * v.z_, 2 * v.z_, 1);
    }

    // Decoder: widths dim_mult reversed, each up block's input halved
    // by the upsampler before it.
    std::vector<std::uint32_t> dims = {base * mult.back()};
    for (auto it = mult.rbegin(); it != mult.rend(); ++it) {
        dims.push_back(base * *it);
    }
    v.post_quant_ = r.conv("post_quant_conv", v.z_, v.z_, 1);
    v.dec_in_ = r.conv("decoder.conv_in", v.z_, dims[0], 3);
    v.dec_.push_back(
        r.res_layer("decoder.mid_block.resnets.0", dims[0], dims[0]));
    v.dec_.push_back(r.attn_layer("decoder.mid_block.attentions.0", dims[0]));
    v.dec_.push_back(
        r.res_layer("decoder.mid_block.resnets.1", dims[0], dims[0]));
    for (std::size_t i = 0; i + 1 < dims.size(); ++i) {
        std::uint32_t in = i > 0 ? dims[i] / 2 : dims[i];
        const std::uint32_t out = dims[i + 1];
        const std::string b = "decoder.up_blocks." + std::to_string(i);
        for (std::uint32_t j = 0; j <= nres; ++j) {
            v.dec_.push_back(r.res_layer(
                b + ".resnets." + std::to_string(j), in, out));
            in = out;
        }
        if (i + 1 != mult.size()) {
            Layer l{Layer::Kind::kUp, {}, {}, {}};
            l.conv = r.conv(b + ".upsamplers.0.resample.1", out, out / 2, 3);
            v.dec_.push_back(std::move(l));
        }
    }
    v.dec_norm_ = r.vec("decoder.norm_out.gamma", dims.back());
    v.dec_out_ = r.conv("decoder.conv_out", dims.back(), 3, 3);
    return v;
}

std::uint32_t QwenVae::scale() const { return 1u << downs_; }

QwenVae::~QwenVae() {
    if (!gpu_used_.v.load()) return;
    // vulkan_release_weights throws inside an open batch; a destructor
    // must not throw (it may run while unwinding), so such a release is
    // skipped and reported.
    try {
        each_conv([](const Conv& k) {
            for (const auto* v : {&k.w, &k.b}) {
                if (!v->empty()) {
                    backend::vulkan_release_weights(
                        v->data(), v->size() * sizeof(float));
                }
            }
        });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwen vae: GPU weights not released: %s\n",
                     e.what());
    }
}

template <class F>
void QwenVae::each_conv(F&& fn) const {
    for (const Conv* k : {&enc_in_, &enc_out_, &quant_, &post_quant_,
                          &dec_in_, &dec_out_}) {
        fn(*k);
    }
    for (const auto* layers : {&enc_, &dec_}) {
        for (const Layer& l : *layers) {
            for (const Conv* k : {&l.res.c1, &l.res.c2, &l.res.shortcut,
                                  &l.attn.qkv, &l.attn.proj, &l.conv}) {
                fn(*k);
            }
        }
    }
}

std::vector<float> QwenVae::run(const std::vector<Layer>& layers,
                                std::vector<float> x, std::uint32_t& c,
                                std::uint32_t& h, std::uint32_t& w,
                                const backend::Ops& op) const {
    for (const Layer& l : layers) {
        switch (l.kind) {
            case Layer::Kind::kRes:
                x = res_block(l.res, x, h, w, op);
                c = l.res.c2.out;
                break;
            case Layer::Kind::kAttn:
                x = attn_block(l.attn, x, h, w, op);
                break;
            case Layer::Kind::kDown: {
                // ZeroPad2d((0, 1, 0, 1)), then 3 x 3 stride 2.
                const std::size_t ph = h + 1, pw = w + 1;
                std::vector<float> p(std::size_t{c} * ph * pw, 0.0f);
                for (std::uint32_t k = 0; k < c; ++k) {
                    for (std::uint32_t y = 0; y < h; ++y) {
                        std::memcpy(&p[(k * ph + y) * pw],
                                    &x[(std::size_t{k} * h + y) * w],
                                    w * sizeof(float));
                    }
                }
                x = conv2d(l.conv, p, h + 1, w + 1, 2, 0, op);
                h /= 2;
                w /= 2;
                break;
            }
            case Layer::Kind::kUp: {
                // Nearest x 2, then 3 x 3 to half the channels.
                std::vector<float> u(std::size_t{c} * 4 * h * w);
                const std::uint32_t uh = 2 * h, uw = 2 * w;
                parallel(c, [&](std::size_t k) {
                    for (std::uint32_t y = 0; y < uh; ++y) {
                        for (std::uint32_t xx = 0; xx < uw; ++xx) {
                            u[(k * uh + y) * uw + xx] =
                                x[(k * h + y / 2) * w + xx / 2];
                        }
                    }
                });
                h = uh;
                w = uw;
                x = conv2d(l.conv, u, h, w, 1, 1, op);
                c = l.conv.out;
                break;
            }
        }
    }
    return x;
}

std::vector<float> QwenVae::encode(std::span<const float> chw,
                                   std::uint32_t h, std::uint32_t w,
                                   const backend::Ops& op) const {
    if (!has_encoder()) {
        throw std::invalid_argument("qwen vae: this GGUF has no encoder");
    }
    if (h == 0 || w == 0 || h % scale() != 0 || w % scale() != 0 ||
        chw.size() != std::size_t{3} * h * w) {
        throw std::invalid_argument(
            "qwen vae: need [3][h][w], sides multiples of scale()");
    }
    if (&op == &backend::vulkan_image_ops()) gpu_used_.v.store(true);
    std::uint32_t c = enc_in_.out;
    auto x = run(enc_, conv2d(enc_in_, chw, h, w, 1, 1, op), c, h, w, op);
    x = conv2d(enc_out_, norm(x, enc_norm_, c, std::size_t{h} * w, true),
               h, w, 1, 1, op);
    x = conv2d(quant_, x, h, w, 1, 0, op);
    x.resize(std::size_t{z_} * h * w);  // the mean: the first z channels
    return x;
}

std::vector<float> QwenVae::decode(std::span<const float> z, std::uint32_t h,
                                   std::uint32_t w,
                                   const backend::Ops& op) const {
    if (h == 0 || w == 0 || z.size() != std::size_t{z_} * h * w) {
        throw std::invalid_argument("qwen vae: need [z_dim][h][w]");
    }
    if (&op == &backend::vulkan_image_ops()) gpu_used_.v.store(true);
    auto x = conv2d(post_quant_, z, h, w, 1, 0, op);
    std::uint32_t c = dec_in_.out;
    x = run(dec_, conv2d(dec_in_, x, h, w, 1, 1, op), c, h, w, op);
    x = conv2d(dec_out_, norm(x, dec_norm_, c, std::size_t{h} * w, true),
               h, w, 1, 1, op);
    for (float& v : x) v = std::clamp(v, -1.0f, 1.0f);
    return x;
}

void QwenVae::normalize(std::span<float> z) const {
    const std::size_t plane = z.size() / z_;
    for (std::uint32_t c = 0; c < z_; ++c) {
        for (std::size_t p = 0; p < plane; ++p) {
            float& v = z[c * plane + p];
            v = (v - mean_[c]) / std_[c];
        }
    }
}

void QwenVae::denormalize(std::span<float> z) const {
    const std::size_t plane = z.size() / z_;
    for (std::uint32_t c = 0; c < z_; ++c) {
        for (std::size_t p = 0; p < plane; ++p) {
            float& v = z[c * plane + p];
            v = v * std_[c] + mean_[c];
        }
    }
}

}  // namespace locus::image
