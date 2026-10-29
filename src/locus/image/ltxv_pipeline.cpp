#include "locus/image/ltxv_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "locus/image/pixart_pipeline.hpp"
#include "locus/image/sd_vae.hpp"

namespace locus::image {

namespace {

using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
           c == '\f';
}

constexpr std::uint32_t kSteps = 8;
constexpr float kDecodeTimestep = 0.05f, kDecodeNoise = 0.025f;

}  // namespace

std::vector<float> LtxvPipeline::schedule(std::uint32_t n) {
    // rf.py's linear_quadratic_schedule(n, threshold_noise=0.025), in
    // Python floats, each term in its order, then torch.tensor (float32).
    if (n == 0) throw std::invalid_argument("ltxv: schedule needs n >= 1");
    if (n == 1) return {1.0f};
    constexpr double kThreshold = 0.025;
    const std::uint32_t lin = n / 2, quad = n - lin;
    std::vector<double> sigma;
    for (std::uint32_t i = 0; i < lin; ++i) {
        sigma.push_back(i * kThreshold / lin);
    }
    const double diff = lin - kThreshold * n;
    const double qc = diff / (lin * static_cast<double>(quad) * quad);
    const double lc = kThreshold / lin - 2 * diff / (static_cast<double>(quad) *
                                                      quad);
    const double c = qc * (static_cast<double>(lin) * lin);
    for (std::uint32_t i = lin; i < n; ++i) {
        const double di = i;
        sigma.push_back(qc * (di * di) + lc * di + c);
    }
    std::vector<float> out(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(1.0 - sigma[i]);
    }
    return out;
}

std::vector<float> LtxvPipeline::coords(std::uint32_t f, std::uint32_t h,
                                        std::uint32_t w, float frame_rate) {
    // latent_to_pixel_coords without causal_fix (integer latent corner
    // times the VAE scale), to float32, the time times 1 / frame_rate.
    const std::size_t n = std::size_t{f} * h * w;
    const auto inv = static_cast<float>(1.0 / double{frame_rate});
    std::vector<float> out(3 * n);
    std::size_t t = 0;
    for (std::uint32_t d = 0; d < f; ++d) {
        for (std::uint32_t y = 0; y < h; ++y) {
            for (std::uint32_t x = 0; x < w; ++x, ++t) {
                out[t] = static_cast<float>(d * 8) * inv;
                out[n + t] = static_cast<float>(y * 32);
                out[2 * n + t] = static_cast<float>(x * 32);
            }
        }
    }
    return out;
}

std::string LtxvPipeline::preprocess(std::string_view text) {
    while (!text.empty() && is_space(text.front())) text.remove_prefix(1);
    while (!text.empty() && is_space(text.back())) text.remove_suffix(1);
    return std::string(text);
}

std::vector<tok::TokenId> LtxvPipeline::prompt_ids(const tok::Tokenizer& tok,
                                                   std::string_view text,
                                                   std::size_t max_length) {
    auto ids = tok.encode(preprocess(text), true);
    if (max_length > 0 && ids.size() > max_length) {
        ids.resize(max_length - 1);
        ids.push_back(tok.eos_id());
    }
    return ids;
}

void LtxvPipeline::check(const Request& req) {
    const auto side = [](std::uint32_t v) {
        return v >= 32 && v <= kMaxSide && v % 32 == 0;
    };
    if (!side(req.width)) {
        throw std::invalid_argument(
            "width must be a multiple of 32 from 32 to 1280");
    }
    if (!side(req.height)) {
        throw std::invalid_argument(
            "height must be a multiple of 32 from 32 to 1280");
    }
    if (req.frames == 0 || req.frames > kMaxFrames || (req.frames - 1) % 8) {
        throw std::invalid_argument(
            "frames must be 8 k + 1, from 1 to 257");
    }
    if (!(req.frame_rate > 0.0f && req.frame_rate <= 1000.0f)) {
        throw std::invalid_argument("frame_rate must be in (0, 1000]");
    }
}

std::unique_ptr<LtxvPipeline> LtxvPipeline::load_dir(const std::string& dir) {
    namespace fs = std::filesystem;
    std::vector<fs::path> paths;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (e.is_regular_file() && e.path().extension() == ".gguf") {
            paths.push_back(e.path());
        }
    }
    if (ec) {
        throw gguf::Error("ltxv: cannot list " + dir + ": " + ec.message());
    }
    std::sort(paths.begin(), paths.end());
    auto p = std::unique_ptr<LtxvPipeline>(new LtxvPipeline());
    const gguf::GgufFile *t5 = nullptr, *dit = nullptr, *vae = nullptr;
    for (const auto& path : paths) {
        auto g = std::make_unique<gguf::GgufFile>(
            gguf::GgufFile::open(path.string()));
        const auto arch = g->get_string("general.architecture");
        const gguf::GgufFile** slot = nullptr;
        if (arch == "t5encoder") {
            slot = &t5;
        } else if (arch == "ltxv") {
            slot = &dit;
        } else if (arch == "ltxv_vae") {
            slot = &vae;
        } else {
            continue;  // not a part of the pipeline
        }
        if (*slot != nullptr) {
            throw gguf::Error("ltxv: two " + std::string(*arch) +
                              " GGUFs in " + dir);
        }
        *slot = g.get();
        p->files_.push_back(std::move(g));
    }
    if (t5 == nullptr || dit == nullptr || vae == nullptr) {
        throw gguf::Error("ltxv: " + dir +
                          " needs a t5encoder, an ltxv and an ltxv_vae GGUF");
    }
    p->tok_ = std::make_unique<tok::UgmTokenizer>(
        tok::UgmTokenizer::from_gguf(*t5));
    p->t5_ = std::make_unique<T5Encoder>(T5Encoder::load(*t5));
    p->dit_ = std::make_unique<LtxvDit>(LtxvDit::load(*dit));
    p->vae_ = std::make_unique<LtxvVae>(LtxvVae::load(*vae));
    if (p->t5_->width() != p->dit_->caption_channels() ||
        p->dit_->channels() != p->vae_->latent_channels() ||
        p->vae_->time_scale() != 8 || p->vae_->space_scale() != 32) {
        throw gguf::Error(
            "ltxv: the T5, DiT and VAE GGUFs do not fit together");
    }
    // The schedule must be the checkpoint's: the reference asserts each
    // step, rounded to 4 places, is one of allowed_inference_steps.
    p->steps_ = schedule(kSteps);
    if (dit->get_string("ltxv.sampler.sampler") != "LinearQuadratic") {
        throw gguf::Error("ltxv: the sampler is not LinearQuadratic");
    }
    const gguf::Value* al = dit->find("ltxv.sampler.allowed_timesteps");
    if (al == nullptr || al->type != gguf::ValueType::kArray) {
        throw gguf::Error("ltxv: missing sampler.allowed_timesteps");
    }
    for (const float t : p->steps_) {
        const double r = std::round(double{t} * 1e4) / 1e4;
        const bool ok = std::any_of(
            al->arr.begin(), al->arr.end(), [&](const gguf::Value& v) {
                return std::fabs(v.f - r) < 1e-6;
            });
        if (!ok) {
            throw gguf::Error("ltxv: step " + std::to_string(t) +
                              " is not an allowed timestep");
        }
    }
    return p;
}

std::vector<float> LtxvPipeline::denoise(
    std::span<const float> noise, const Request& req, const backend::Ops& op,
    const std::function<bool()>& alive) const {
    check(req);
    const std::uint32_t f = (req.frames - 1) / 8 + 1, h = req.height / 32,
                        w = req.width / 32;
    const std::uint32_t n = f * h * w;
    if (noise.size() != std::size_t{dit_->channels()} * n) {
        throw std::invalid_argument("ltxv: noise is not 128 x f x h x w");
    }
    const auto ids = prompt_ids(*tok_, req.prompt);
    const auto text = t5_->encode(ids, op);
    const auto pos = coords(f, h, w, req.frame_rate);
    std::vector<float> x(noise.begin(), noise.end());
    for (std::uint32_t i = 0; i < kSteps; ++i) {
        if (alive && !alive()) return {};
        const float t = steps_[i];
        const float t_next = i + 1 < kSteps ? steps_[i + 1] : 0.0f;
        const auto v = dit_->forward(x, n, pos, t, text,
                                     static_cast<std::uint32_t>(ids.size()),
                                     op);
        const float dt = t - t_next;
        for (std::size_t j = 0; j < x.size(); ++j) x[j] = x[j] - dt * v[j];
    }
    return x;
}

std::vector<float> LtxvPipeline::decode(std::span<const float> latent,
                                        std::span<const float> decode_noise,
                                        const Request& req) const {
    check(req);
    const std::uint32_t f = (req.frames - 1) / 8 + 1, h = req.height / 32,
                        w = req.width / 32;
    const std::size_t n = std::size_t{vae_->latent_channels()} * f * h * w;
    if (latent.size() != n || decode_noise.size() != n) {
        throw std::invalid_argument("ltxv: latent is not 128 x f x h x w");
    }
    std::vector<float> mixed(n);
    const float keep = 1.0f - kDecodeNoise;
    for (std::size_t i = 0; i < n; ++i) {
        mixed[i] = latent[i] * keep + decode_noise[i] * kDecodeNoise;
    }
    return vae_->decode(mixed, f, h, w, kDecodeTimestep);
}

std::vector<std::vector<std::uint8_t>> LtxvPipeline::generate(
    const Request& req, const backend::Ops& op,
    const std::function<bool()>& alive, Timing* timing) const {
    check(req);
    const std::uint32_t f = (req.frames - 1) / 8 + 1, h = req.height / 32,
                        w = req.width / 32;
    const std::size_t n = std::size_t{dit_->channels()} * f * h * w;
    const auto t0 = Clock::now();
    const auto noise = PixArtPipeline::normal_latent(req.seed, n);
    const auto x = denoise(noise, req, op, alive);
    if (x.empty()) return {};
    const auto t1 = Clock::now();
    if (alive && !alive()) return {};
    const auto dnoise = PixArtPipeline::normal_latent(req.seed + 1, n);
    const auto video = decode(x, dnoise, req);
    const auto t2 = Clock::now();
    if (timing != nullptr) *timing = {ms(t0, t1), ms(t1, t2)};
    return to_frames(video, req);
}

std::vector<std::vector<std::uint8_t>> LtxvPipeline::to_frames(
    std::span<const float> video, const Request& req) {
    const std::size_t plane = std::size_t{req.height} * req.width;
    if (video.size() != 3 * plane * req.frames) {
        throw std::invalid_argument("ltxv: video is not 3 x frames x h x w");
    }
    std::vector<std::vector<std::uint8_t>> frames;
    std::vector<float> one(3 * plane);
    for (std::uint32_t d = 0; d < req.frames; ++d) {
        for (std::size_t c = 0; c < 3; ++c) {
            std::copy_n(&video[(c * req.frames + d) * plane], plane,
                        &one[c * plane]);
        }
        frames.push_back(SdVaeDecoder::to_rgb8(one, req.height, req.width));
    }
    return frames;
}

}  // namespace locus::image
