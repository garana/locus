#include "locus/image/sd_vae.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "locus/backend/variants.hpp"
#include "locus/image/dense.hpp"

namespace locus::image {

namespace {

/** A [ch][h][w] activation. */
struct Act {
    std::vector<float> data;
    std::uint32_t ch = 0, h = 0, w = 0;
    std::size_t plane() const { return std::size_t{h} * w; }
};

}  // namespace

std::span<const float> SdVaeDecoder::t(const std::string& name,
                                       std::size_t n) const {
    const gguf::TensorInfo* ti = g_->find_tensor(name);
    if (ti == nullptr) {
        throw gguf::Error("vae: missing tensor " + name);
    }
    if (ti->type != gguf::TensorType::kF32 || ti->nelements() != n) {
        throw gguf::Error("vae: " + name + " must be F32 with " +
                          std::to_string(n) + " values");
    }
    const auto bytes = g_->tensor_data(*ti);
    return {reinterpret_cast<const float*>(bytes.data()), n};
}

SdVaeDecoder SdVaeDecoder::load(const gguf::GgufFile& g) {
    if (g.get_string("general.architecture") != "autoencoder_kl") {
        throw gguf::Error("vae: general.architecture is not autoencoder_kl");
    }
    SdVaeDecoder d;
    d.g_ = &g;
    auto u = [&](std::string_view k) -> std::uint32_t {
        auto x = g.get_uint(k);
        if (!x || *x == 0 || *x > 4096) {
            throw gguf::Error("vae: missing or bad " + std::string(k));
        }
        return static_cast<std::uint32_t>(*x);
    };
    auto f = [&](std::string_view k) {
        const gguf::Value* v = g.find(k);
        if (v == nullptr || v->type != gguf::ValueType::kFloat32) {
            throw gguf::Error("vae: missing " + std::string(k));
        }
        return static_cast<float>(v->f);
    };
    d.latent_ = u("autoencoder_kl.latent_channels");
    d.out_ch_ = u("autoencoder_kl.out_channels");
    d.layers_ = u("autoencoder_kl.layers_per_block");
    d.groups_ = u("autoencoder_kl.norm_groups");
    d.eps_ = f("autoencoder_kl.norm_epsilon");
    d.scaling_ = f("autoencoder_kl.scaling_factor");
    // FLUX's VAE: a shift as well as a scale, and no post-quant conv.
    // The SD VAE GGUFs predate both keys.
    if (g.find("autoencoder_kl.shift_factor") != nullptr) {
        d.shift_ = f("autoencoder_kl.shift_factor");
    }
    d.post_quant_ =
        g.get_bool("autoencoder_kl.use_post_quant_conv").value_or(true);
    if (g.get_bool("autoencoder_kl.mid_block_attention").value_or(false) !=
        true) {
        throw gguf::Error("vae: the decoder must have the mid attention");
    }
    const auto* arr = g.get_array("autoencoder_kl.block_out_channels");
    if (arr == nullptr || arr->empty() || arr->size() > 8) {
        throw gguf::Error("vae: missing block_out_channels");
    }
    for (const auto& v : *arr) {
        const std::int64_t c = v.type == gguf::ValueType::kInt32
                                   ? v.i
                                   : static_cast<std::int64_t>(v.u);
        if (c <= 0 || c > 4096 || c % d.groups_ != 0) {
            throw gguf::Error("vae: bad block_out_channels");
        }
        d.chans_.push_back(static_cast<std::uint32_t>(c));
    }
    if (!(d.scaling_ > 0.0f) || d.layers_ > 16) {
        throw gguf::Error("vae: bad scaling_factor or layers_per_block");
    }
    // Check every tensor now, so decode cannot fail halfway.
    const std::uint32_t top = d.chans_.back();
    const auto res = [&](const std::string& p, std::uint32_t in,
                         std::uint32_t out) {
        d.t(p + ".norm1.weight", in);
        d.t(p + ".norm1.bias", in);
        d.t(p + ".conv1.weight", std::size_t{out} * in * 9);
        d.t(p + ".conv1.bias", out);
        d.t(p + ".norm2.weight", out);
        d.t(p + ".norm2.bias", out);
        d.t(p + ".conv2.weight", std::size_t{out} * out * 9);
        d.t(p + ".conv2.bias", out);
        if (in != out) {
            d.t(p + ".conv_shortcut.weight", std::size_t{out} * in);
            d.t(p + ".conv_shortcut.bias", out);
        }
    };
    if (d.post_quant_) {
        d.t("post_quant_conv.weight", std::size_t{d.latent_} * d.latent_);
        d.t("post_quant_conv.bias", d.latent_);
    }
    d.t("decoder.conv_in.weight", std::size_t{top} * d.latent_ * 9);
    d.t("decoder.conv_in.bias", top);
    res("decoder.mid_block.resnets.0", top, top);
    res("decoder.mid_block.resnets.1", top, top);
    const std::string a = "decoder.mid_block.attentions.0.";
    d.t(a + "group_norm.weight", top);
    d.t(a + "group_norm.bias", top);
    for (const char* m : {"to_q", "to_k", "to_v", "to_out.0"}) {
        d.t(a + m + ".weight", std::size_t{top} * top);
        d.t(a + m + ".bias", top);
    }
    std::uint32_t prev = top;
    const std::size_t nb = d.chans_.size();
    for (std::size_t b = 0; b < nb; ++b) {
        const std::uint32_t out = d.chans_[nb - 1 - b];
        const std::string p = "decoder.up_blocks." + std::to_string(b);
        for (std::uint32_t r = 0; r <= d.layers_; ++r) {
            res(p + ".resnets." + std::to_string(r), r == 0 ? prev : out, out);
        }
        if (b + 1 < nb) {
            d.t(p + ".upsamplers.0.conv.weight", std::size_t{out} * out * 9);
            d.t(p + ".upsamplers.0.conv.bias", out);
        }
        prev = out;
    }
    const std::uint32_t last = d.chans_.front();
    d.t("decoder.conv_norm_out.weight", last);
    d.t("decoder.conv_norm_out.bias", last);
    d.t("decoder.conv_out.weight", std::size_t{d.out_ch_} * last * 9);
    d.t("decoder.conv_out.bias", d.out_ch_);
    return d;
}

std::vector<float> SdVaeDecoder::decode(std::span<const float> latent,
                                        std::uint32_t h, std::uint32_t w,
                                        const backend::Ops& op) const {
    // 128 is a 1024 px image; larger latents make the mid attention
    // (one head over h * w tokens) and the activations unreasonable.
    if (h == 0 || w == 0 || h > kMaxLatent || w > kMaxLatent ||
        latent.size() != std::size_t{latent_} * h * w) {
        throw std::invalid_argument("vae: latent is not C x h x w");
    }
    const std::uint32_t groups = groups_;
    const float eps = eps_;

    // A conv with bias; k is 1 or 3 (padding k / 2), stride 1.
    auto conv = [&](const Act& in, const std::string& name,
                    std::uint32_t out_ch, std::uint32_t k) {
        Act o{std::vector<float>(std::size_t{out_ch} * in.plane()), out_ch,
              in.h, in.w};
        const auto wt = t(name + ".weight", std::size_t{out_ch} * in.ch * k * k);
        const auto bs = t(name + ".bias", out_ch);
        if (&op == &backend::vulkan_image_ops()) {
            backend::conv2d_vulkan(in.data, in.ch, in.h, in.w, wt, out_ch, k,
                                   k, bs, 1, k / 2, 1, o.data);
        } else {
            backend::conv2d_mt(in.data, in.ch, in.h, in.w, wt, out_ch, k, k,
                               bs, 1, k / 2, 1, o.data);
        }
        return o;
    };
    auto norm = [&](Act& x, const std::string& name) {
        backend::group_norm(x.data, x.ch, static_cast<std::uint32_t>(x.plane()),
                            groups, t(name + ".weight", x.ch),
                            t(name + ".bias", x.ch), eps);
    };
    // ResnetBlock2D: two GN / SiLU / conv3 steps, plus x (through a 1x1
    // conv_shortcut when the width changes).
    auto res = [&](const Act& x, const std::string& p, std::uint32_t out) {
        Act hh = x;
        norm(hh, p + ".norm1");
        backend::silu(hh.data);
        hh = conv(hh, p + ".conv1", out, 3);
        norm(hh, p + ".norm2");
        backend::silu(hh.data);
        hh = conv(hh, p + ".conv2", out, 3);
        if (x.ch != out) {
            const Act s = conv(x, p + ".conv_shortcut", out, 1);
            for (std::size_t i = 0; i < hh.data.size(); ++i) {
                hh.data[i] += s.data[i];
            }
        } else {
            for (std::size_t i = 0; i < hh.data.size(); ++i) {
                hh.data[i] += x.data[i];
            }
        }
        return hh;
    };
    // Mid attention: GroupNorm, then one head over the h * w positions
    // (token-major), out projection, plus the input from before the
    // norm.
    auto attention = [&](Act& x, const std::string& p) {
        const std::uint32_t c = x.ch;
        const std::uint32_t n = static_cast<std::uint32_t>(x.plane());
        Act hn = x;
        norm(hn, p + "group_norm");
        std::vector<float> tok(std::size_t{n} * c);
        for (std::uint32_t ch = 0; ch < c; ++ch) {
            for (std::uint32_t i = 0; i < n; ++i) {
                tok[std::size_t{i} * c + ch] = hn.data[std::size_t{ch} * n + i];
            }
        }
        const auto mat = [&](const char* m) {
            const auto wt = t(p + m + ".weight", std::size_t{c} * c);
            return backend::Mat{gguf::TensorType::kF32,
                                reinterpret_cast<const std::byte*>(wt.data()),
                                c, c};
        };
        const auto bias = [&](const char* m) { return t(p + m + ".bias", c); };
        std::vector<float> q(tok.size()), k(tok.size()), v(tok.size()),
            o(tok.size());
        dense::linear(op, mat("to_q"), bias("to_q"), tok, n, q);
        dense::linear(op, mat("to_k"), bias("to_k"), tok, n, k);
        dense::linear(op, mat("to_v"), bias("to_v"), tok, n, v);
        dense::attention(op, q, n, k, v, n, 1, c, o);
        dense::linear(op, mat("to_out.0"), bias("to_out.0"), o, n, q);
        for (std::uint32_t ch = 0; ch < c; ++ch) {
            for (std::uint32_t i = 0; i < n; ++i) {
                x.data[std::size_t{ch} * n + i] += q[std::size_t{i} * c + ch];
            }
        }
    };

    Act z{std::vector<float>(latent.begin(), latent.end()), latent_, h, w};
    for (float& x : z.data) x = x / scaling_ + shift_;
    if (post_quant_) z = conv(z, "post_quant_conv", latent_, 1);
    const std::uint32_t top = chans_.back();
    Act s = conv(z, "decoder.conv_in", top, 3);
    s = res(s, "decoder.mid_block.resnets.0", top);
    attention(s, "decoder.mid_block.attentions.0.");
    s = res(s, "decoder.mid_block.resnets.1", top);
    const std::size_t nb = chans_.size();
    for (std::size_t b = 0; b < nb; ++b) {
        const std::uint32_t out = chans_[nb - 1 - b];
        const std::string p = "decoder.up_blocks." + std::to_string(b);
        for (std::uint32_t r = 0; r <= layers_; ++r) {
            s = res(s, p + ".resnets." + std::to_string(r), out);
        }
        if (b + 1 < nb) {
            Act up{std::vector<float>(std::size_t{s.ch} * s.plane() * 4), s.ch,
                   s.h * 2, s.w * 2};
            backend::upsample_nearest2x(s.data, s.ch, s.h, s.w, up.data);
            s = conv(up, p + ".upsamplers.0.conv", out, 3);
        }
    }
    norm(s, "decoder.conv_norm_out");
    backend::silu(s.data);
    return conv(s, "decoder.conv_out", out_ch_, 3).data;
}

std::vector<std::uint8_t> SdVaeDecoder::to_rgb8(std::span<const float> image,
                                                std::uint32_t h,
                                                std::uint32_t w) {
    const std::size_t plane = std::size_t{h} * w;
    if (image.size() != plane * 3) {
        throw std::invalid_argument("vae: image is not 3 x h x w");
    }
    std::vector<std::uint8_t> rgb(plane * 3);
    for (std::size_t i = 0; i < plane; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            const float y = std::clamp(image[c * plane + i] / 2.0f + 0.5f,
                                       0.0f, 1.0f);
            rgb[i * 3 + c] =
                static_cast<std::uint8_t>(std::nearbyint(y * 255.0f));
        }
    }
    return rgb;
}

}  // namespace locus::image
