#include "locus/image/pixart_sampler.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace locus::image {

namespace {

/** _sigma_to_alpha_sigma_t: the signal and noise levels of sigma. */
struct Levels {
    float alpha, sigma, lambda;
};

Levels levels(float s) {
    Levels l;
    l.alpha = 1.0f / std::sqrt(s * s + 1.0f);
    l.sigma = s * l.alpha;
    l.lambda = std::log(l.alpha) - std::log(l.sigma);
    return l;
}

}  // namespace

DpmSolverPP::DpmSolverPP(std::uint32_t steps, std::uint32_t train_steps,
                         float beta_start, float beta_end) {
    if (steps == 0 || train_steps < 2 || steps > train_steps ||
        !(beta_start > 0.0f) || !(beta_end > beta_start) ||
        !(beta_end < 1.0f)) {
        throw std::invalid_argument("dpm-solver++: bad schedule");
    }
    // torch.linspace (float32): start + step i for the first half,
    // end - step (n - 1 - i) for the rest.
    const float step = (beta_end - beta_start) /
                       static_cast<float>(train_steps - 1);
    const std::uint32_t half = train_steps / 2;
    std::vector<float> train_sigma(train_steps);
    double cum = 1.0;  // torch's CPU cumprod accumulates in double
    for (std::uint32_t k = 0; k < train_steps; ++k) {
        const float beta =
            k < half ? beta_start + step * static_cast<float>(k)
                     : beta_end - step * static_cast<float>(train_steps - k - 1);
        cum *= static_cast<double>(1.0f - beta);
        const float abar = static_cast<float>(cum);
        train_sigma[k] = std::sqrt((1.0f - abar) / abar);
    }
    // linspace timestep spacing: round(linspace(0, T - 1, steps + 1)),
    // high to low, the 0 dropped; numpy rounds half to even.
    for (std::uint32_t i = steps; i >= 1; --i) {
        const double v = static_cast<double>(train_steps - 1) * i / steps;
        t_.push_back(static_cast<std::int32_t>(std::nearbyint(v)));
    }
    for (const std::int32_t t : t_) {
        sigma_.push_back(train_sigma[static_cast<std::uint32_t>(t)]);
    }
    sigma_.push_back(0.0f);  // final_sigmas_type "zero"
}

DpmSolverPP DpmSolverPP::from_gguf(const gguf::GgufFile& g,
                                   std::uint32_t steps) {
    const std::string k = "pixart.sampler.";
    const auto str = [&](const std::string& key, std::string_view want) {
        if (g.get_string(k + key) != want) {
            throw gguf::Error("pixart: sampler " + key + " must be " +
                              std::string(want));
        }
    };
    str("algorithm", "dpmsolver++");
    str("solver_type", "midpoint");
    str("beta_schedule", "linear");
    str("timestep_spacing", "linspace");
    str("final_sigmas_type", "zero");
    if (g.get_uint(k + "solver_order").value_or(0) != 2 ||
        g.get_bool(k + "lower_order_final").value_or(false) != true) {
        throw gguf::Error("pixart: sampler must be order 2, lower order "
                          "final");
    }
    const auto f = [&](const std::string& key) {
        const gguf::Value* v = g.find(k + key);
        if (v == nullptr || v->type != gguf::ValueType::kFloat32) {
            throw gguf::Error("pixart: missing sampler " + key);
        }
        return static_cast<float>(v->f);
    };
    const auto train = g.get_uint(k + "num_train_timesteps");
    if (!train || *train > 100000) {
        throw gguf::Error("pixart: missing sampler num_train_timesteps");
    }
    try {
        return DpmSolverPP(steps, static_cast<std::uint32_t>(*train),
                           f("beta_start"), f("beta_end"));
    } catch (const std::invalid_argument& e) {
        throw gguf::Error(std::string("pixart: ") + e.what());
    }
}

void DpmSolverPP::step(std::span<const float> eps, std::span<float> x) {
    const std::uint32_t n = static_cast<std::uint32_t>(t_.size());
    if (index_ >= n) {
        throw std::logic_error("dpm-solver++: no steps left");
    }
    if (eps.size() != x.size() ||
        (!prev_x0_.empty() && prev_x0_.size() != x.size())) {
        throw std::invalid_argument("dpm-solver++: size mismatch");
    }
    const std::uint32_t i = index_;
    const Levels s0 = levels(sigma_[i]);
    // convert_model_output: x0 = (x - sigma_s eps) / alpha_s.
    std::vector<float> x0(x.size());
    for (std::size_t j = 0; j < x.size(); ++j) {
        x0[j] = (x[j] - s0.sigma * eps[j]) / s0.alpha;
    }
    if (i + 1 == n) {
        // To sigma 0: alpha 1, sigma 0, exp(-h) 0, so x' = x0.
        for (std::size_t j = 0; j < x.size(); ++j) x[j] = x0[j];
    } else {
        const Levels t = levels(sigma_[i + 1]);
        const float h = t.lambda - s0.lambda;
        const float a = t.sigma / s0.sigma;
        const float b = t.alpha * (std::exp(-h) - 1.0f);
        if (i == 0) {
            // First order: x' = a x - b x0.
            for (std::size_t j = 0; j < x.size(); ++j) {
                const float ax = a * x[j];
                const float bx = b * x0[j];
                x[j] = ax - bx;
            }
        } else {
            // Second order, midpoint: D1 = (m0 - m1) / r0,
            // x' = a x - b D0 - 0.5 b D1.
            const Levels s1 = levels(sigma_[i - 1]);
            const float h0 = s0.lambda - s1.lambda;
            const float r0 = h0 / h;
            const float inv_r0 = 1.0f / r0;
            const float hb = 0.5f * b;
            for (std::size_t j = 0; j < x.size(); ++j) {
                const float d1 = inv_r0 * (x0[j] - prev_x0_[j]);
                const float ax = a * x[j];
                const float bx = b * x0[j];
                const float cx = hb * d1;
                x[j] = (ax - bx) - cx;
            }
        }
    }
    prev_x0_ = std::move(x0);
    ++index_;
}

void pixart_denoise(const PixArtDit& dit, DpmSolverPP& sched,
                    std::span<float> latent, std::span<const float> neg,
                    std::uint32_t n_neg, std::span<const float> pos,
                    std::uint32_t n_pos, float guidance,
                    const backend::Ops& op,
                    const std::function<void(std::uint32_t)>& on_step) {
    const std::size_t plane = latent.size() / dit.in_channels();
    const std::size_t noise = plane * dit.in_channels();
    if (sched.index() != 0 || dit.out_channels() < dit.in_channels()) {
        throw std::invalid_argument("pixart: need a fresh sampler");
    }
    std::vector<float> eps(noise);
    for (const std::int32_t t : sched.timesteps()) {
        const float tf = static_cast<float>(t);
        const auto e0 = dit.forward(latent, tf, neg, n_neg, op);
        const auto e1 = dit.forward(latent, tf, pos, n_pos, op);
        // The first in_channels output channels are the noise.
        for (std::size_t j = 0; j < noise; ++j) {
            const float d = e1[j] - e0[j];
            const float gd = guidance * d;
            eps[j] = e0[j] + gd;
        }
        const std::uint32_t i = sched.index();
        sched.step(eps, latent);
        if (on_step) on_step(i);
    }
}

}  // namespace locus::image
