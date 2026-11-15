#include "locus/image/qwen_image_dit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

#include "locus/backend/metal/matvec.hpp"
#include "locus/backend/metal/seq.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_seq.hpp"
#include "locus/image/dense.hpp"
#include "locus/image/mmdit_cpu.hpp"

namespace locus::image {

namespace {

using dense::linear;
using dense::linear1;
using dense::ln_modulate;
using dense::rms_norm;
using dense::silu;

/**
 * QwenTimestepProjEmbeddings' Timesteps(channels, flip_sin_to_cos=True,
 * downscale_freq_shift=0, scale=1000) in float32: [cos(a_k), sin(a_k)],
 * a_k = 1000 * (sigma * f_k), f_k = exp(-ln(10000) k / (channels / 2)).
 * The scale multiplies after sigma * f_k, as diffusers does (not
 * PixArtDit::timestep_features(sigma * 1000), whose float32 product
 * rounds differently at angles near 1000).
 */
std::vector<float> timestep_features(float sigma, std::uint32_t channels) {
    const std::uint32_t half = channels / 2;
    const float ln = static_cast<float>(-std::log(10000.0));
    std::vector<float> tf(channels);
    for (std::uint32_t k = 0; k < half; ++k) {
        const float e = ln * static_cast<float>(k) / static_cast<float>(half);
        const float a = 1000.0f * (sigma * std::exp(e));
        tf[k] = std::cos(a);
        tf[half + k] = std::sin(a);
    }
    return tf;
}

}  // namespace

void QwenImageDit::rope(std::span<const Segment> segs, std::uint32_t n_txt,
                        std::vector<float>& cos_out,
                        std::vector<float>& sin_out) const {
    std::size_t n_img = 0;
    std::int64_t txt0 = 0;
    for (const Segment& s : segs) {
        n_img += std::size_t{s.frames} * s.rows * s.cols;
        txt0 = std::max<std::int64_t>({txt0, s.rows / 2, s.cols / 2});
    }
    const std::size_t n = n_txt + n_img;
    cos_out.assign(n * head_dim_, 0.0f);
    sin_out.assign(cos_out.size(), 0.0f);
    // QwenEmbedRope: float32 inverse frequencies and angles.
    std::vector<std::vector<float>> inv(axes_.size());
    for (std::size_t a = 0; a < axes_.size(); ++a) {
        const std::uint32_t d = axes_[a];
        for (std::uint32_t j = 0; j < d / 2; ++j) {
            inv[a].push_back(1.0f /
                             std::pow(theta_, static_cast<float>(2 * j) /
                                                  static_cast<float>(d)));
        }
    }
    const auto put = [&](std::size_t t, const std::int64_t pos[3]) {
        std::uint32_t off = 0;
        for (std::size_t a = 0; a < axes_.size(); ++a) {
            for (std::size_t j = 0; j < inv[a].size(); ++j) {
                const float ang = static_cast<float>(pos[a]) * inv[a][j];
                const std::size_t k = t * head_dim_ + off + 2 * j;
                cos_out[k] = cos_out[k + 1] = std::cos(ang);
                sin_out[k] = sin_out[k + 1] = std::sin(ang);
            }
            off += axes_[a];
        }
    };
    for (std::uint32_t j = 0; j < n_txt; ++j) {
        const std::int64_t p[3] = {txt0 + j, txt0 + j, txt0 + j};
        put(j, p);
    }
    std::size_t t = n_txt;
    for (std::size_t s = 0; s < segs.size(); ++s) {
        const Segment& g = segs[s];
        const std::int64_t r0 = g.rows - g.rows / 2, c0 = g.cols - g.cols / 2;
        for (std::uint32_t f = 0; f < g.frames; ++f) {
            for (std::uint32_t r = 0; r < g.rows; ++r) {
                for (std::uint32_t c = 0; c < g.cols; ++c) {
                    const std::int64_t p[3] = {
                        static_cast<std::int64_t>(s) + f,
                        static_cast<std::int64_t>(r) - r0,
                        static_cast<std::int64_t>(c) - c0};
                    put(t++, p);
                }
            }
        }
    }
}

QwenImageDit QwenImageDit::load(const gguf::GgufFile& g) {
    if (g.get_string("general.architecture") != "qwen_image") {
        throw gguf::Error("qwen_image: general.architecture is not "
                          "qwen_image");
    }
    QwenImageDit d;
    const std::string p = "qwen_image.";
    auto u = [&](const std::string& k) -> std::uint32_t {
        auto x = g.get_uint(p + k);
        if (!x || *x == 0 || *x > (1u << 20)) {
            throw gguf::Error("qwen_image: missing or bad " + p + k);
        }
        return static_cast<std::uint32_t>(*x);
    };
    auto f = [&](const std::string& k) {
        const gguf::Value* v = g.find(p + k);
        if (v == nullptr || v->type != gguf::ValueType::kFloat32) {
            throw gguf::Error("qwen_image: missing " + p + k);
        }
        return static_cast<float>(v->f);
    };
    const std::uint32_t n_blocks = u("block_count");
    d.width_ = u("embedding_length");
    d.ffn_ = u("feed_forward_length");
    d.heads_ = u("attention.head_count");
    d.head_dim_ = u("attention.key_length");
    d.eps_ = f("norm_epsilon");
    d.in_ch_ = u("in_channels");
    const std::uint32_t patch = u("patch_size");
    d.out_ = u("out_channels") * patch * patch;
    d.joint_ = u("joint_attention_dim");
    d.t_ch_ = u("timestep.channels");
    d.theta_ = f("rope.theta");
    const auto scale = g.get_bool(p + "rope.scale");
    if (!scale || !*scale) {
        throw gguf::Error("qwen_image: only rope.scale true is supported");
    }
    if (g.get_string(p + "activation") != "gelu_tanh") {
        throw gguf::Error("qwen_image: only the gelu_tanh FFN is supported");
    }
    const gguf::Value* ax = g.find(p + "rope.axes");
    if (ax == nullptr || ax->type != gguf::ValueType::kArray ||
        ax->arr.size() != 3) {
        throw gguf::Error("qwen_image: rope.axes must be 3 values");
    }
    std::uint32_t axes_sum = 0;
    for (const gguf::Value& e : ax->arr) {
        using VT = gguf::ValueType;
        const std::int64_t x = e.type == VT::kInt32 || e.type == VT::kInt64
                                   ? e.i
                                   : static_cast<std::int64_t>(e.u);
        if (x <= 0 || x % 2 != 0 || x > 4096) {
            throw gguf::Error("qwen_image: bad rope.axes entry");
        }
        d.axes_.push_back(static_cast<std::uint32_t>(x));
        axes_sum += static_cast<std::uint32_t>(x);
    }
    if (std::uint64_t{d.heads_} * d.head_dim_ != d.width_ ||
        axes_sum != d.head_dim_ || d.t_ch_ % 2 != 0 ||
        !(d.theta_ > 1.0f) || !(d.eps_ > 0.0f)) {
        throw gguf::Error("qwen_image: inconsistent sizes");
    }

    const auto info = [&](const std::string& name) {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr) {
            throw gguf::Error("qwen_image: missing tensor " + name);
        }
        return ti;
    };
    const auto vec = [&](const std::string& name, std::size_t n) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->type != gguf::TensorType::kF32 || ti->nelements() != n) {
            throw gguf::Error("qwen_image: " + name + " must be F32 with " +
                              std::to_string(n) + " values");
        }
        std::vector<float> out(n);
        std::memcpy(out.data(), g.tensor_data(*ti).data(), n * sizeof(float));
        return out;
    };
    // A Linear [rows, cols], BF16, Q8_0 (--dit-type q8_0) or F32.
    const auto lin = [&](const std::string& name, std::uint32_t rows,
                         std::uint32_t cols) {
        const gguf::TensorInfo* ti = info(name + ".weight");
        if ((ti->type != gguf::TensorType::kBF16 &&
             ti->type != gguf::TensorType::kQ8_0 &&
             ti->type != gguf::TensorType::kF32) ||
            ti->n_dims != 2 || ti->ne[0] != cols || ti->ne[1] != rows) {
            throw gguf::Error("qwen_image: " + name + ".weight is not " +
                              std::to_string(rows) + " x " +
                              std::to_string(cols));
        }
        Linear l;
        l.w = backend::Mat{ti->type, g.tensor_data(*ti).data(), rows, cols};
        l.b = vec(name + ".bias", rows);
        return l;
    };

    const std::uint32_t w = d.width_, hd = d.head_dim_;
    d.img_in_ = lin("img_in", w, d.in_ch_);
    d.txt_in_ = lin("txt_in", w, d.joint_);
    d.txt_norm_ = vec("txt_norm.weight", d.joint_);
    d.t1_ = lin("time_text_embed.timestep_embedder.linear_1", w, d.t_ch_);
    d.t2_ = lin("time_text_embed.timestep_embedder.linear_2", w, w);
    d.norm_out_ = lin("norm_out.linear", 2 * w, w);
    d.proj_out_ = lin("proj_out", d.out_, w);
    for (std::uint32_t i = 0; i < n_blocks; ++i) {
        const std::string b = "transformer_blocks." + std::to_string(i) + ".";
        Block k;
        k.img.mod = lin(b + "img_mod.1", 6 * w, w);
        k.txt.mod = lin(b + "txt_mod.1", 6 * w, w);
        k.img.q = lin(b + "attn.to_q", w, w);
        k.img.k = lin(b + "attn.to_k", w, w);
        k.img.v = lin(b + "attn.to_v", w, w);
        k.img.out = lin(b + "attn.to_out.0", w, w);
        k.img.q_norm = vec(b + "attn.norm_q.weight", hd);
        k.img.k_norm = vec(b + "attn.norm_k.weight", hd);
        k.txt.q = lin(b + "attn.add_q_proj", w, w);
        k.txt.k = lin(b + "attn.add_k_proj", w, w);
        k.txt.v = lin(b + "attn.add_v_proj", w, w);
        k.txt.out = lin(b + "attn.to_add_out", w, w);
        k.txt.q_norm = vec(b + "attn.norm_added_q.weight", hd);
        k.txt.k_norm = vec(b + "attn.norm_added_k.weight", hd);
        k.img.ff1 = lin(b + "img_mlp.net.0.proj", d.ffn_, w);
        k.img.ff2 = lin(b + "img_mlp.net.2", w, d.ffn_);
        k.txt.ff1 = lin(b + "txt_mlp.net.0.proj", d.ffn_, w);
        k.txt.ff2 = lin(b + "txt_mlp.net.2", w, d.ffn_);
        d.blocks_.push_back(std::move(k));
    }
    return d;
}

std::vector<float> QwenImageDit::forward(std::span<const float> x,
                                         std::span<const Segment> segs,
                                         std::span<const float> text,
                                         std::uint32_t n_txt, float sigma,
                                         const backend::Ops& op,
                                         Trace* trace) const {
    std::size_t n_img_sz = 0;
    for (const Segment& s : segs) {
        if (s.frames == 0 || s.rows == 0 || s.cols == 0) {
            throw std::invalid_argument("qwen_image: empty segment");
        }
        n_img_sz += std::size_t{s.frames} * s.rows * s.cols;
    }
    if (segs.empty() || n_txt == 0 || n_img_sz > (1u << 20) ||
        x.size() != n_img_sz * in_ch_ ||
        text.size() != std::size_t{n_txt} * joint_) {
        throw std::invalid_argument("qwen_image: forward sizes");
    }
    const auto n_img = static_cast<std::uint32_t>(n_img_sz);
    const std::uint32_t w = width_, hd = head_dim_, nh = heads_;

    // The conditioning vector, then silu of it for every modulation.
    std::vector<float> h1(w), temb(w), st(w);
    linear1(op, t1_.w, t1_.b, timestep_features(sigma, t_ch_), h1);
    for (float& v : h1) v = silu(v);
    linear1(op, t2_.w, t2_.b, h1, temb);
    for (std::uint32_t i = 0; i < w; ++i) st[i] = silu(temb[i]);

    std::vector<float> cosv, sinv;
    rope(segs, n_txt, cosv, sinv);

    std::vector<float> tn(text.begin(), text.end());
    rms_norm(tn, txt_norm_, eps_, n_txt, joint_, tn);
    const bool vk = &op == &backend::vulkan_image_ops() &&
                    backend::VulkanSeq::available();
    const bool mt = &op == &backend::metal::image_ops() &&
                    backend::metal::Seq::available();
    if (vk || mt) {
        const char* env = std::getenv("LOCUS_QWEN_IMAGE_GPU");
        if (env == nullptr || std::string(env) != "0") {
            return vk ? forward_gpu<backend::VulkanSeq>(x, n_img, tn, n_txt, st,
                                                        cosv, sinv, trace)
                      : forward_gpu<backend::metal::Seq>(x, n_img, tn, n_txt,
                                                         st, cosv, sinv, trace);
        }
    }

    std::vector<float> img(std::size_t{n_img} * w), txt(std::size_t{n_txt} * w);
    linear(op, img_in_.w, img_in_.b, x, n_img, img);
    linear(op, txt_in_.w, txt_in_.b, tn, n_txt, txt);

    const mmdit::CpuShape sh{n_txt, n_img, w, nh, hd, ffn_, eps_};
    mmdit::CpuBufs bufs(sh);
    const auto lin = [&](const Linear& l, std::span<const float> in,
                         std::uint32_t rows, std::span<float> out) {
        linear(op, l.w, l.b, in, rows, out);
    };
    const auto lin1 = [&](const Linear& l, std::span<const float> in,
                          std::span<float> out) {
        linear1(op, l.w, l.b, in, out);
    };

    for (std::size_t bi = 0; bi < blocks_.size(); ++bi) {
        const Block& b = blocks_[bi];
        mmdit::double_block_cpu(op, b.img, b.txt, sh, st, cosv, sinv, img, txt,
                                bufs, lin, lin1);
        if (trace != nullptr && bi == 0) {
            trace->block0_img = img;
            trace->block0_txt = txt;
        }
    }

    // norm_out (AdaLayerNormContinuous: scale, then shift) on the image
    // rows, then proj_out.
    std::vector<float> ss(std::size_t{2} * w);
    linear1(op, norm_out_.w, norm_out_.b, st, ss);
    auto ni = std::span<float>(bufs.nrm).subspan(0, std::size_t{n_img} * w);
    ln_modulate(img, n_img, w, eps_, &ss[w], &ss[0], ni);
    std::vector<float> out(std::size_t{n_img} * out_);
    linear(op, proj_out_.w, proj_out_.b, ni, n_img, out);
    return out;
}

}  // namespace locus::image
