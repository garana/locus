#include "locus/image/flux_sampler.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace locus::image {

FluxSampler FluxSampler::load(const gguf::GgufFile& g) {
    const std::string p = "flux.sampler.";
    const auto f = [&](const std::string& k) {
        const gguf::Value* v = g.find(p + k);
        if (v == nullptr || (v->type != gguf::ValueType::kFloat64 &&
                             v->type != gguf::ValueType::kFloat32)) {
            throw gguf::Error("flux: missing " + p + k);
        }
        return v->f;
    };
    const auto u = [&](const std::string& k) {
        const auto v = g.get_uint(p + k);
        if (!v || *v == 0 || *v > (1u << 24)) {
            throw gguf::Error("flux: missing or bad " + p + k);
        }
        return static_cast<std::uint32_t>(*v);
    };
    FluxSampler s;
    const auto dyn = g.get_bool(p + "use_dynamic_shifting");
    if (!dyn) throw gguf::Error("flux: missing " + p + "use_dynamic_shifting");
    s.dynamic_ = *dyn;
    s.shift_ = f("shift");
    s.base_shift_ = f("base_shift");
    s.max_shift_ = f("max_shift");
    s.base_seq_ = u("base_image_seq_len");
    s.max_seq_ = u("max_image_seq_len");
    if (s.max_seq_ <= s.base_seq_ || !(s.shift_ > 0.0)) {
        throw gguf::Error("flux: bad sampler keys");
    }
    return s;
}

std::vector<float> FluxSampler::sigmas(std::uint32_t steps,
                                       std::uint32_t image_tokens) const {
    if (steps == 0) throw std::invalid_argument("flux: need 1 or more steps");
    // calculate_shift, in double like the pipeline's Python floats.
    const double seq_span = static_cast<double>(max_seq_) - base_seq_;
    const double m = (max_shift_ - base_shift_) / seq_span;
    const double b = base_shift_ - m * base_seq_;
    const double mu = image_tokens * m + b;
    std::vector<float> out;
    out.reserve(steps + 1);
    // np.linspace(1.0, 1 / steps, steps): i * step + start, the last one
    // set to stop exactly.
    const double lo = 1.0 / steps;
    const double dstep = steps > 1 ? (lo - 1.0) / (steps - 1.0) : 0.0;
    for (std::uint32_t i = 0; i < steps; ++i) {
        const double s = i + 1 == steps && steps > 1 ? lo : i * dstep + 1.0;
        double t;
        if (dynamic_) {
            t = std::exp(mu) / (std::exp(mu) + (1.0 / s - 1.0));
        } else {
            t = shift_ * s / (1.0 + (shift_ - 1.0) * s);
        }
        out.push_back(static_cast<float>(t));
    }
    out.push_back(0.0f);
    return out;
}

float FluxSampler::model_sigma(float sigma) {
    const float t = sigma * 1000.0f;
    return t / 1000.0f;
}

void FluxSampler::step(std::span<float> x, std::span<const float> v,
                       float sigma, float sigma_next) {
    if (x.size() != v.size()) throw std::invalid_argument("flux: step sizes");
    const float d = sigma_next - sigma;
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += d * v[i];
}

std::vector<float> FluxSampler::pack(std::span<const float> latent,
                                     std::uint32_t c, std::uint32_t h,
                                     std::uint32_t w) {
    if (h % 2 != 0 || w % 2 != 0 ||
        latent.size() != std::size_t{c} * h * w) {
        throw std::invalid_argument("flux: pack needs [c][h][w], h, w even");
    }
    const std::uint32_t h2 = h / 2, w2 = w / 2;
    std::vector<float> out(latent.size());
    // view(c, h2, 2, w2, 2).permute(h2, w2, c, 2, 2).
    for (std::uint32_t y = 0; y < h2; ++y) {
        for (std::uint32_t x = 0; x < w2; ++x) {
            float* o = &out[(std::size_t{y} * w2 + x) * c * 4];
            for (std::uint32_t ch = 0; ch < c; ++ch) {
                for (std::uint32_t dy = 0; dy < 2; ++dy) {
                    for (std::uint32_t dx = 0; dx < 2; ++dx) {
                        *o++ = latent[(std::size_t{ch} * h + 2 * y + dy) * w +
                                      2 * x + dx];
                    }
                }
            }
        }
    }
    return out;
}

std::vector<float> FluxSampler::unpack(std::span<const float> tokens,
                                       std::uint32_t c, std::uint32_t h,
                                       std::uint32_t w) {
    if (h % 2 != 0 || w % 2 != 0 ||
        tokens.size() != std::size_t{c} * h * w) {
        throw std::invalid_argument("flux: unpack sizes");
    }
    const std::uint32_t h2 = h / 2, w2 = w / 2;
    std::vector<float> out(tokens.size());
    for (std::uint32_t y = 0; y < h2; ++y) {
        for (std::uint32_t x = 0; x < w2; ++x) {
            const float* t = &tokens[(std::size_t{y} * w2 + x) * c * 4];
            for (std::uint32_t ch = 0; ch < c; ++ch) {
                for (std::uint32_t dy = 0; dy < 2; ++dy) {
                    for (std::uint32_t dx = 0; dx < 2; ++dx) {
                        out[(std::size_t{ch} * h + 2 * y + dy) * w + 2 * x +
                            dx] = *t++;
                    }
                }
            }
        }
    }
    return out;
}

std::vector<float> FluxSampler::image_ids(std::uint32_t h2, std::uint32_t w2,
                                          float first) {
    std::vector<float> ids;
    ids.reserve(std::size_t{h2} * w2 * 3);
    for (std::uint32_t y = 0; y < h2; ++y) {
        for (std::uint32_t x = 0; x < w2; ++x) {
            ids.push_back(first);
            ids.push_back(static_cast<float>(y));
            ids.push_back(static_cast<float>(x));
        }
    }
    return ids;
}

}  // namespace locus::image
