#include "locus/image/pixart_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <random>
#include <stdexcept>

#include "locus/image/pixart_sampler.hpp"

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

}  // namespace

std::unique_ptr<PixArtPipeline> PixArtPipeline::load_dir(
    const std::string& dir) {
    namespace fs = std::filesystem;
    std::vector<fs::path> paths;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (e.is_regular_file() && e.path().extension() == ".gguf") {
            paths.push_back(e.path());
        }
    }
    if (ec) {
        throw gguf::Error("pixart: cannot list " + dir + ": " + ec.message());
    }
    std::sort(paths.begin(), paths.end());
    auto p = std::unique_ptr<PixArtPipeline>(new PixArtPipeline());
    const gguf::GgufFile *t5 = nullptr, *dit = nullptr, *vae = nullptr;
    for (const auto& path : paths) {
        auto g = std::make_unique<gguf::GgufFile>(
            gguf::GgufFile::open(path.string()));
        const auto arch = g->get_string("general.architecture");
        const gguf::GgufFile** slot = nullptr;
        if (arch == "t5encoder") {
            slot = &t5;
        } else if (arch == "pixart") {
            slot = &dit;
        } else if (arch == "autoencoder_kl") {
            slot = &vae;
        } else {
            continue;  // not a part of the pipeline
        }
        if (*slot != nullptr) {
            throw gguf::Error("pixart: two " + std::string(*arch) +
                              " GGUFs in " + dir);
        }
        *slot = g.get();
        p->files_.push_back(std::move(g));
    }
    if (t5 == nullptr || dit == nullptr || vae == nullptr) {
        throw gguf::Error("pixart: " + dir +
                          " needs a t5encoder, a pixart and an "
                          "autoencoder_kl GGUF");
    }
    p->tok_ = std::make_unique<tok::UgmTokenizer>(
        tok::UgmTokenizer::from_gguf(*t5));
    p->t5_ = std::make_unique<T5Encoder>(T5Encoder::load(*t5));
    p->dit_ = std::make_unique<PixArtDit>(PixArtDit::load(*dit));
    p->vae_ = std::make_unique<SdVaeDecoder>(SdVaeDecoder::load(*vae));
    p->dit_file_ = dit;
    if (p->t5_->width() != p->dit_->caption_channels() ||
        p->dit_->in_channels() != p->vae_->latent_channels() ||
        p->dit_->out_channels() < p->dit_->in_channels()) {
        throw gguf::Error("pixart: the T5, DiT and VAE GGUFs do not fit "
                          "together");
    }
    DpmSolverPP::from_gguf(*dit, 1);  // the sampler keys are valid
    if (const gguf::Value* v = dit->find("pixart.guidance_scale");
        v != nullptr && v->type == gguf::ValueType::kFloat32) {
        p->guidance_ = static_cast<float>(v->f);
    }
    return p;
}

std::string PixArtPipeline::preprocess(std::string_view text) {
    while (!text.empty() && is_space(text.front())) text.remove_prefix(1);
    while (!text.empty() && is_space(text.back())) text.remove_suffix(1);
    std::string out(text);
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return out;
}

std::vector<tok::TokenId> PixArtPipeline::prompt_ids(
    const tok::Tokenizer& tok, std::string_view text,
    std::size_t max_length) {
    auto ids = tok.encode(preprocess(text), true);
    if (max_length > 0 && ids.size() > max_length) {
        ids.resize(max_length - 1);
        ids.push_back(tok.eos_id());
    }
    return ids;
}

std::vector<float> PixArtPipeline::normal_latent(std::uint64_t seed,
                                                 std::size_t n) {
    // Box-Muller on 53-bit uniforms from mt19937_64, so the latent is
    // the same on every platform (std::normal_distribution is not).
    std::mt19937_64 rng(seed);
    const auto uniform = [&] {
        return static_cast<double>(rng() >> 11) * 0x1.0p-53;
    };
    std::vector<float> v(n);
    constexpr double kTwoPi = 6.283185307179586476925;
    for (std::size_t i = 0; i < n; i += 2) {
        const double u1 = 1.0 - uniform();  // (0, 1]
        const double u2 = uniform();
        const double r = std::sqrt(-2.0 * std::log(u1));
        v[i] = static_cast<float>(r * std::cos(kTwoPi * u2));
        if (i + 1 < n) v[i + 1] = static_cast<float>(r * std::sin(kTwoPi * u2));
    }
    return v;
}

std::vector<float> PixArtPipeline::generate_from(
    std::span<const float> latent, const Request& req, const backend::Ops& op,
    const std::function<bool()>& alive, Timing* timing) const {
    const std::uint32_t s = dit_->latent_size();
    const std::size_t n = std::size_t{dit_->in_channels()} * s * s;
    if (latent.size() != n) {
        throw std::invalid_argument("pixart: latent is not C x S x S");
    }
    if (req.steps == 0 || req.steps > 1000) {
        throw std::invalid_argument("steps must be 1 to 1000");
    }
    if (!std::isfinite(req.guidance) || req.guidance < 0.0f) {
        throw std::invalid_argument("guidance must be finite and >= 0");
    }
    const auto t0 = Clock::now();
    const auto pos_ids = prompt_ids(*tok_, req.prompt);
    const auto neg_ids = prompt_ids(*tok_, req.negative_prompt);
    const auto pos = t5_->encode(pos_ids, op);
    if (alive && !alive()) return {};
    const auto neg = t5_->encode(neg_ids, op);
    const auto t1 = Clock::now();
    auto sched = DpmSolverPP::from_gguf(*dit_file_, req.steps);
    std::vector<float> x(latent.begin(), latent.end());
    const bool done = pixart_denoise(
        *dit_, sched, x, neg, static_cast<std::uint32_t>(neg_ids.size()), pos,
        static_cast<std::uint32_t>(pos_ids.size()), req.guidance, op,
        [&](std::uint32_t) { return !alive || alive(); });
    if (!done) return {};
    const auto t2 = Clock::now();
    auto img = vae_->decode(x, s, s, op);
    const auto t3 = Clock::now();
    if (timing != nullptr) {
        *timing = {ms(t0, t1), ms(t1, t2), ms(t2, t3)};
    }
    return img;
}

std::vector<std::uint8_t> PixArtPipeline::generate(
    const Request& req, const backend::Ops& op,
    const std::function<bool()>& alive, Timing* timing) const {
    const std::uint32_t s = dit_->latent_size();
    const auto latent = normal_latent(
        req.seed, std::size_t{dit_->in_channels()} * s * s);
    const auto img = generate_from(latent, req, op, alive, timing);
    if (img.empty()) return {};
    const std::uint32_t side = image_size();
    return SdVaeDecoder::to_rgb8(img, side, side);
}

}  // namespace locus::image
