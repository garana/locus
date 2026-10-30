#include "locus/image/janus_vision.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_seq.hpp"
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

/**
 * out[t][r] = w[r] . x[t] + bias[r] for n tokens, token-major in and
 * out. The rows are split across threads; each thread runs its slice
 * over every token, so the slice stays in cache while the tokens
 * stream past it. A backend that does not split rows (see
 * backend::splits_rows) runs as one slice. With a batched kernel
 * (Ops::matvec_batch, byte-identical to per-token matvec) a slice runs
 * all tokens in one call.
 */
void linear(const backend::Ops& op, const backend::Mat& w,
            std::span<const float> bias, std::span<const float> x,
            std::uint32_t n, std::span<float> out) {
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
        if (op.matvec_batch != nullptr) {
            // Row-major [nr][n] out of the kernel, token-major here.
            thread_local std::vector<float> rm;
            rm.resize(std::size_t{nr} * n);
            op.matvec_batch(ws, x.subspan(0, std::size_t{n} * cols), rm, n);
            for (std::uint32_t t = 0; t < n; ++t) {
                float* o = &out[std::size_t{t} * rows + r0];
                for (std::uint32_t r = 0; r < nr; ++r) {
                    o[r] = rm[std::size_t{r} * n + t] + bias[r0 + r];
                }
            }
            return;
        }
        for (std::uint32_t t = 0; t < n; ++t) {
            const auto o = out.subspan(std::size_t{t} * rows + r0, nr);
            op.matvec(ws, x.subspan(std::size_t{t} * cols, cols), o);
            for (std::uint32_t r = 0; r < nr; ++r) o[r] += bias[r0 + r];
        }
    });
}

/** LayerNorm with weight and bias over each of n rows of width d. */
void layer_norm(std::span<const float> x, std::span<const float> w,
                std::span<const float> b, float eps, std::uint32_t n,
                std::uint32_t d, std::span<float> out) {
    parallel(n, [&](std::size_t t) {
        const auto o = out.subspan(t * d, d);
        backend::layernorm(x.subspan(t * d, d), w, eps, o);
        for (std::uint32_t i = 0; i < d; ++i) o[i] += b[i];
    });
}

}  // namespace

JanusVision JanusVision::load(const gguf::GgufFile& g, std::uint32_t n_embd) {
    JanusVision v;
    auto u = [&](std::string_view k) -> std::uint32_t {
        auto x = g.get_uint(k);
        if (!x || *x == 0) {
            throw gguf::Error("janus vision: missing " + std::string(k));
        }
        return static_cast<std::uint32_t>(*x);
    };
    v.image_ = u("janus.vision.image_size");
    v.patch_ = u("janus.vision.patch_size");
    v.width_ = u("janus.vision.embedding_length");
    v.heads_ = u("janus.vision.head_count");
    v.ffn_ = u("janus.vision.feed_forward_length");
    const std::uint32_t layers = u("janus.vision.block_count");
    v.placeholder_ =
        static_cast<tok::TokenId>(u("janus.vision.image_placeholder"));
    v.end_ = static_cast<tok::TokenId>(u("janus.vision.end_of_image"));
    v.begin_ = static_cast<tok::TokenId>(u("janus.begin_of_image"));
    const gguf::Value* eps = g.find("janus.vision.layer_norm_eps");
    if (eps == nullptr || eps->type != gguf::ValueType::kFloat32) {
        throw gguf::Error("janus vision: missing layer_norm_eps");
    }
    v.eps_ = static_cast<float>(eps->f);
    if (v.image_ % v.patch_ != 0 || v.width_ % v.heads_ != 0) {
        throw gguf::Error("janus vision: inconsistent sizes");
    }
    v.grid_ = v.image_ / v.patch_;
    v.n_embd_ = n_embd;

    const auto info = [&](const std::string& name) {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr) {
            throw gguf::Error("janus vision: missing tensor " + name);
        }
        return ti;
    };
    const auto f32 = [&](const std::string& name, std::size_t n) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->type != gguf::TensorType::kF32 || ti->nelements() != n) {
            throw gguf::Error("janus vision: " + name +
                              " must be F32 with " + std::to_string(n) +
                              " values");
        }
        const auto bytes = g.tensor_data(*ti);
        return std::span<const float>(
            reinterpret_cast<const float*>(bytes.data()), n);
    };
    const auto mat = [&](const std::string& name, std::uint32_t rows,
                         std::uint32_t cols) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->ne[0] != cols || ti->ne[1] != rows || ti->n_dims != 2) {
            throw gguf::Error("janus vision: " + name + " is not " +
                              std::to_string(rows) + " x " +
                              std::to_string(cols));
        }
        return backend::Mat{ti->type, g.tensor_data(*ti).data(), rows, cols};
    };

    const std::uint32_t w = v.width_;
    v.patch_w_ =
        f32("vit.patch_embd.weight", std::size_t{w} * 3 * v.patch_ * v.patch_);
    v.patch_b_ = f32("vit.patch_embd.bias", w);
    v.pos_ = f32("vit.pos_embd", std::size_t{v.tokens()} * w);
    for (std::uint32_t i = 0; i < layers; ++i) {
        const std::string b = "vit.blk." + std::to_string(i) + ".";
        Block k;
        k.ln1_w = f32(b + "ln1.weight", w);
        k.ln1_b = f32(b + "ln1.bias", w);
        k.qkv = mat(b + "attn_qkv.weight", 3 * w, w);
        k.qkv_b = f32(b + "attn_qkv.bias", 3 * w);
        k.out = mat(b + "attn_out.weight", w, w);
        k.out_b = f32(b + "attn_out.bias", w);
        k.ln2_w = f32(b + "ln2.weight", w);
        k.ln2_b = f32(b + "ln2.bias", w);
        k.up = mat(b + "ffn_up.weight", v.ffn_, w);
        k.up_b = f32(b + "ffn_up.bias", v.ffn_);
        k.down = mat(b + "ffn_down.weight", w, v.ffn_);
        k.down_b = f32(b + "ffn_down.bias", w);
        v.blocks_.push_back(k);
    }
    v.post_w_ = f32("vit.post_ln.weight", w);
    v.post_b_ = f32("vit.post_ln.bias", w);
    v.align0_ = mat("vit.aligner.0.weight", n_embd, w);
    v.align0_b_ = f32("vit.aligner.0.bias", n_embd);
    v.align2_ = mat("vit.aligner.2.weight", n_embd, n_embd);
    v.align2_b_ = f32("vit.aligner.2.bias", n_embd);
    return v;
}

std::vector<float> JanusVision::to_pixels(
    std::span<const std::uint8_t> rgb) const {
    const std::size_t plane = std::size_t{image_} * image_;
    if (rgb.size() != plane * 3) {
        throw std::invalid_argument("janus vision: need image_size^2 RGB");
    }
    std::vector<float> chw(plane * 3);
    for (std::size_t i = 0; i < plane; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            const float x = static_cast<float>(rgb[i * 3 + c]) / 255.0f;
            chw[c * plane + i] = (x - 0.5f) / 0.5f;
        }
    }
    return chw;
}

std::vector<float> JanusVision::features(std::span<const float> chw,
                                         const backend::Ops& op) const {
    const std::uint32_t n = tokens(), w = width_, hd = w / heads_;
    if (chw.size() != std::size_t{3} * image_ * image_) {
        throw std::invalid_argument("janus vision: need 3 x size x size");
    }
    if (&op == &backend::vulkan_image_ops() &&
        backend::VulkanSeq::available()) {
        return features_vulkan(chw);
    }
    // Patch embed: [w][grid][grid] -> token-major [n][w], plus positions.
    std::vector<float> conv(std::size_t{w} * n);
    backend::conv2d_mt(chw, 3, image_, image_, patch_w_, w, patch_, patch_,
                       patch_b_, patch_, 0, 1, conv);
    std::vector<float> x(std::size_t{n} * w);
    for (std::uint32_t t = 0; t < n; ++t) {
        for (std::uint32_t c = 0; c < w; ++c) {
            x[std::size_t{t} * w + c] =
                conv[std::size_t{c} * n + t] + pos_[std::size_t{t} * w + c];
        }
    }

    std::vector<float> h(std::size_t{n} * w), qkv(std::size_t{n} * 3 * w),
        att(std::size_t{n} * w), up(std::size_t{n} * ffn_),
        o(std::size_t{n} * w);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    constexpr std::uint32_t kQueryBlock = 32;
    const std::uint32_t qblocks = (n + kQueryBlock - 1) / kQueryBlock;
    for (const Block& k : blocks_) {
        // x += proj(attention(LN1(x)))
        layer_norm(x, k.ln1_w, k.ln1_b, eps_, n, w, h);
        linear(op, k.qkv, k.qkv_b, h, n, qkv);
        // qkv[t] holds q | k | v, each heads x hd (head-major), as the
        // reference's reshape(B, N, 3, heads, hd).
        parallel(std::size_t{heads_} * qblocks, [&](std::size_t item) {
            const std::uint32_t hh = static_cast<std::uint32_t>(item / qblocks);
            const std::uint32_t q0 =
                static_cast<std::uint32_t>(item % qblocks) * kQueryBlock;
            const std::uint32_t q1 = std::min(n, q0 + kQueryBlock);
            std::vector<float> s(n);
            for (std::uint32_t i = q0; i < q1; ++i) {
                const float* q = &qkv[std::size_t{i} * 3 * w + hh * hd];
                for (std::uint32_t j = 0; j < n; ++j) {
                    const float* kj =
                        &qkv[std::size_t{j} * 3 * w + w + hh * hd];
                    float d = 0.0f;
                    for (std::uint32_t e = 0; e < hd; ++e) d += q[e] * kj[e];
                    s[j] = d * scale;
                }
                backend::softmax_inplace(s);
                float* out = &att[std::size_t{i} * w + hh * hd];
                std::fill(out, out + hd, 0.0f);
                for (std::uint32_t j = 0; j < n; ++j) {
                    const float* vj =
                        &qkv[std::size_t{j} * 3 * w + 2 * w + hh * hd];
                    for (std::uint32_t e = 0; e < hd; ++e) {
                        out[e] += s[j] * vj[e];
                    }
                }
            }
        });
        linear(op, k.out, k.out_b, att, n, o);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];
        // x += fc2(GELU(fc1(LN2(x))))
        layer_norm(x, k.ln2_w, k.ln2_b, eps_, n, w, h);
        linear(op, k.up, k.up_b, h, n, up);
        gelu(up);
        linear(op, k.down, k.down_b, up, n, o);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];
    }
    layer_norm(x, post_w_, post_b_, eps_, n, w, x);
    return x;
}

std::vector<float> JanusVision::align(std::span<const float> feat,
                                      const backend::Ops& op) const {
    const std::uint32_t n = tokens();
    if (feat.size() != std::size_t{n} * width_) {
        throw std::invalid_argument("janus vision: need tokens x width");
    }
    std::vector<float> a(std::size_t{n} * n_embd_), b(a.size());
    linear(op, align0_, align0_b_, feat, n, a);
    gelu(a);
    linear(op, align2_, align2_b_, a, n, b);
    return b;
}

}  // namespace locus::image
