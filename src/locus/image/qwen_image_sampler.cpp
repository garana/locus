#include "locus/image/qwen_image_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "locus/image/flux_sampler.hpp"

namespace locus::image {

QwenImageSampler QwenImageSampler::load(const gguf::GgufFile& g) {
    const std::string p = "qwen_image.sampler.";
    const auto f = [&](const std::string& k) {
        const gguf::Value* v = g.find(p + k);
        if (v == nullptr || (v->type != gguf::ValueType::kFloat64 &&
                             v->type != gguf::ValueType::kFloat32)) {
            throw gguf::Error("qwen image: missing " + p + k);
        }
        return v->f;
    };
    const auto u = [&](const std::string& k) {
        const auto v = g.get_uint(p + k);
        if (!v || *v == 0 || *v > (1u << 24)) {
            throw gguf::Error("qwen image: missing or bad " + p + k);
        }
        return static_cast<std::uint32_t>(*v);
    };
    QwenImageSampler s;
    const auto dyn = g.get_bool(p + "use_dynamic_shifting");
    if (!dyn) {
        throw gguf::Error("qwen image: missing " + p +
                          "use_dynamic_shifting");
    }
    s.dynamic_ = *dyn;
    s.shift_ = f("shift");
    s.base_shift_ = f("base_shift");
    s.max_shift_ = f("max_shift");
    s.terminal_ = f("shift_terminal");
    s.base_seq_ = u("base_image_seq_len");
    s.max_seq_ = u("max_image_seq_len");
    if (s.max_seq_ <= s.base_seq_ || !(s.shift_ > 0.0) ||
        !(s.terminal_ >= 0.0 && s.terminal_ < 1.0)) {
        throw gguf::Error("qwen image: bad sampler keys");
    }
    return s;
}

std::vector<float> QwenImageSampler::sigmas(std::uint32_t steps,
                                            std::uint32_t image_tokens) const {
    if (steps == 0 || (steps == 1 && terminal_ > 0.0)) {
        throw std::invalid_argument(
            "qwen image: need 2 or more steps with a shift_terminal");
    }
    // calculate_shift, in double like the pipeline's Python floats.
    const double seq_span = static_cast<double>(max_seq_) - base_seq_;
    const double m = (max_shift_ - base_shift_) / seq_span;
    const double b = base_shift_ - m * base_seq_;
    const double mu = image_tokens * m + b;
    // np.linspace(1.0, 1 / steps, steps) in double, then float32; the
    // scheduler's shift and stretch run on that float32 array, Python
    // scalars becoming float32.
    const double lo = 1.0 / steps;
    const double dstep = steps > 1 ? (lo - 1.0) / (steps - 1.0) : 0.0;
    const float e = static_cast<float>(std::exp(mu));
    const float shift = static_cast<float>(shift_);
    std::vector<float> out(steps + 1, 0.0f);
    for (std::uint32_t i = 0; i < steps; ++i) {
        const float s = static_cast<float>(
            i + 1 == steps && steps > 1 ? lo : i * dstep + 1.0);
        if (dynamic_) {
            const float r = 1.0f / s - 1.0f;
            out[i] = e / (e + r);
        } else {
            const float num = shift * s;
            out[i] = num / (1.0f + (shift - 1.0f) * s);
        }
    }
    if (terminal_ > 0.0) {
        const float scale = (1.0f - out[steps - 1]) /
                            static_cast<float>(1.0 - terminal_);
        for (std::uint32_t i = 0; i < steps; ++i) {
            out[i] = 1.0f - (1.0f - out[i]) / scale;
        }
    }
    return out;
}

void qwen_true_cfg(std::span<float> cond, std::span<const float> neg,
                   float scale, std::uint32_t channels) {
    if (channels == 0 || cond.size() != neg.size() ||
        cond.size() % channels != 0) {
        throw std::invalid_argument("qwen image: true cfg sizes");
    }
    for (std::size_t t = 0; t < cond.size(); t += channels) {
        float* c = &cond[t];
        const float* n = &neg[t];
        double cn = 0.0, mn = 0.0;
        for (std::uint32_t k = 0; k < channels; ++k) {
            const float d = c[k] - n[k];
            const float sd = scale * d;
            const float comb = n[k] + sd;
            cn += double{c[k]} * c[k];
            mn += double{comb} * comb;
            c[k] = comb;
        }
        const float ratio = static_cast<float>(std::sqrt(cn)) /
                            static_cast<float>(std::sqrt(mn));
        for (std::uint32_t k = 0; k < channels; ++k) c[k] *= ratio;
    }
}

std::vector<float> qwen_edit_denoise(
    const QwenImageDit& dit, const QwenImageSampler& sampler,
    std::span<const float> noise, QwenImageDit::Segment noise_seg,
    std::span<const float> ref, QwenImageDit::Segment ref_seg,
    QwenEditText prompt, QwenEditText negative, float cfg_scale,
    std::uint32_t steps, const backend::Ops& op,
    const std::function<bool()>& alive) {
    const std::size_t ch = dit.channels();
    const std::size_t n_noise =
        std::size_t{noise_seg.frames} * noise_seg.rows * noise_seg.cols;
    const std::size_t n_ref =
        std::size_t{ref_seg.frames} * ref_seg.rows * ref_seg.cols;
    if (n_noise == 0 || noise.size() != n_noise * ch ||
        ref.size() != n_ref * ch || dit.out_channels() != ch) {
        throw std::invalid_argument("qwen edit: denoise sizes");
    }
    const bool cfg = negative.n > 0 && cfg_scale > 1.0f;
    const auto sig =
        sampler.sigmas(steps, static_cast<std::uint32_t>(n_noise));
    const QwenImageDit::Segment segs[2] = {noise_seg, ref_seg};
    const std::span<const QwenImageDit::Segment> seg_span(segs,
                                                          n_ref ? 2 : 1);
    // The MMDiT's input: the noise tokens, then the reference's.
    std::vector<float> x(noise.begin(), noise.end());
    x.insert(x.end(), ref.begin(), ref.end());
    const std::span<float> lat(x.data(), n_noise * ch);
    for (std::uint32_t i = 0; i < steps; ++i) {
        if (alive && !alive()) return {};
        const float t = FluxSampler::model_sigma(sig[i]);
        auto v = dit.forward(x, seg_span, prompt.rows, prompt.n, t, op);
        v.resize(n_noise * ch);
        if (cfg) {
            auto vn =
                dit.forward(x, seg_span, negative.rows, negative.n, t, op);
            vn.resize(n_noise * ch);
            qwen_true_cfg(v, vn, cfg_scale, static_cast<std::uint32_t>(ch));
        }
        FluxSampler::step(lat, v, sig[i], sig[i + 1]);
    }
    x.resize(n_noise * ch);
    return x;
}

std::vector<float> qwen_edit_decode(const QwenVae& vae,
                                    std::span<const float> tokens,
                                    std::uint32_t h, std::uint32_t w,
                                    const backend::Ops& op) {
    const std::uint32_t s = vae.scale();
    if (h == 0 || w == 0 || h % (2 * s) != 0 || w % (2 * s) != 0) {
        throw std::invalid_argument(
            "qwen edit: decode sides must be multiples of 2 x the VAE "
            "scale");
    }
    const std::uint32_t zh = h / s, zw = w / s;
    auto z = FluxSampler::unpack(tokens, vae.z_dim(), zh, zw);
    vae.denormalize(z);
    const auto chw = vae.decode(z, zh, zw, op);
    const std::size_t plane = std::size_t{h} * w;
    std::vector<float> hwc(3 * plane);
    for (std::size_t p = 0; p < plane; ++p) {
        for (std::size_t c = 0; c < 3; ++c) {
            const float v = chw[c * plane + p] / 2.0f + 0.5f;
            hwc[p * 3 + c] = std::clamp(v, 0.0f, 1.0f);
        }
    }
    return hwc;
}

}  // namespace locus::image
