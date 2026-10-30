#include "locus/image/flux_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "locus/image/pixart_pipeline.hpp"

namespace locus::image {

namespace {

using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

/** The latent channels FLUX packs (16, into 64 per 2x2 token). */
constexpr std::uint32_t kLatentCh = 16;

}  // namespace

void FluxPipeline::validate(const Request& req) {
    // The VAE decodes latents up to kMaxLatent (128) a side: 1024 px.
    constexpr std::uint32_t kMax = SdVaeDecoder::kMaxLatent * 8;
    if (req.width < 16 || req.height < 16 || req.width > kMax ||
        req.height > kMax || req.width % 16 != 0 || req.height % 16 != 0) {
        throw std::invalid_argument(
            "width and height must be multiples of 16 from 16 to 1024");
    }
    if (req.steps > 1000) throw std::invalid_argument("steps must be <= 1000");
    if (req.max_sequence_length == 0 || req.max_sequence_length > 512) {
        throw std::invalid_argument("max_sequence_length must be 1 to 512");
    }
    if (!std::isfinite(req.guidance) || req.guidance < 0.0f) {
        throw std::invalid_argument("guidance must be finite and >= 0");
    }
}

std::unique_ptr<FluxPipeline> FluxPipeline::load_dir(const std::string& dir) {
    namespace fs = std::filesystem;
    std::vector<fs::path> paths;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (e.is_regular_file() && e.path().extension() == ".gguf") {
            paths.push_back(e.path());
        }
    }
    if (ec) {
        throw gguf::Error("flux: cannot list " + dir + ": " + ec.message());
    }
    std::sort(paths.begin(), paths.end());
    auto p = std::unique_ptr<FluxPipeline>(new FluxPipeline());
    const gguf::GgufFile *t5 = nullptr, *clip = nullptr, *dit = nullptr,
                         *vae = nullptr;
    for (const auto& path : paths) {
        auto g = std::make_unique<gguf::GgufFile>(
            gguf::GgufFile::open(path.string()));
        const auto arch = g->get_string("general.architecture");
        const gguf::GgufFile** slot = nullptr;
        if (arch == "t5encoder") {
            slot = &t5;
        } else if (arch == "clip_text") {
            slot = &clip;
        } else if (arch == "flux") {
            slot = &dit;
        } else if (arch == "autoencoder_kl") {
            slot = &vae;
        } else {
            continue;  // not a part of the pipeline
        }
        if (*slot != nullptr) {
            throw gguf::Error("flux: two " + std::string(*arch) +
                              " GGUFs in " + dir);
        }
        *slot = g.get();
        p->files_.push_back(std::move(g));
    }
    if (t5 == nullptr || clip == nullptr || dit == nullptr || vae == nullptr) {
        throw gguf::Error("flux: " + dir +
                          " needs a t5encoder, a clip_text, a flux and an "
                          "autoencoder_kl GGUF");
    }
    p->t5_tok_ = std::make_unique<tok::UgmTokenizer>(
        tok::UgmTokenizer::from_gguf(*t5));
    p->clip_tok_ = std::make_unique<tok::ClipTokenizer>(
        tok::ClipTokenizer::from_gguf(*clip));
    p->t5_ = std::make_unique<T5Encoder>(T5Encoder::load(*t5));
    p->clip_ = std::make_unique<ClipTextEncoder>(ClipTextEncoder::load(*clip));
    p->dit_ = std::make_unique<FluxDit>(FluxDit::load(*dit));
    p->sampler_ = std::make_unique<FluxSampler>(FluxSampler::load(*dit));
    p->vae_ = std::make_unique<SdVaeDecoder>(SdVaeDecoder::load(*vae));
    if (p->t5_->width() != p->dit_->text_width() ||
        p->clip_->width() != p->dit_->pooled_width() ||
        p->vae_->latent_channels() != kLatentCh ||
        p->dit_->channels() != kLatentCh * 4 || p->vae_->upscale() != 8) {
        throw gguf::Error("flux: the T5, CLIP, transformer and VAE GGUFs "
                          "do not fit together");
    }
    return p;
}

std::vector<tok::TokenId> FluxPipeline::t5_ids(
    const std::string& prompt, std::uint32_t max_length) const {
    auto ids = t5_tok_->encode(prompt, true);
    if (ids.size() > max_length) {
        ids.resize(max_length - 1);
        ids.push_back(t5_tok_->eos_id());
    }
    ids.resize(max_length, t5_tok_->pad_id());
    return ids;
}

std::vector<float> FluxPipeline::generate_from(
    std::span<const float> latent, const Request& req, const backend::Ops& op,
    const std::function<bool()>& alive, Timing* timing) const {
    validate(req);
    const std::uint32_t h = req.height / 8, w = req.width / 8;
    if (latent.size() != std::size_t{kLatentCh} * h * w) {
        throw std::invalid_argument("flux: latent is not 16 x h/8 x w/8");
    }
    const std::uint32_t steps = req.steps != 0 ? req.steps : default_steps();

    const auto t0 = Clock::now();
    const auto clip_ids = clip_tok_->encode(req.prompt, clip_->context());
    const auto pooled = clip_->encode(clip_ids, op).pooled;
    if (alive && !alive()) return {};
    const auto ids = t5_ids(req.prompt, req.max_sequence_length);
    const auto text = t5_->encode(ids, op);
    const auto n_txt = static_cast<std::uint32_t>(ids.size());
    const std::vector<float> txt_ids(std::size_t{n_txt} * 3, 0.0f);
    const auto t1 = Clock::now();

    auto x = FluxSampler::pack(latent, kLatentCh, h, w);
    const std::uint32_t n_img = (h / 2) * (w / 2);
    const auto img_ids = FluxSampler::image_ids(h / 2, w / 2, 0.0f);
    const auto sig = sampler_->sigmas(steps, n_img);
    for (std::uint32_t i = 0; i < steps; ++i) {
        if (alive && !alive()) return {};
        const auto v = dit_->forward(x, img_ids, n_img, text, txt_ids, n_txt,
                                     pooled, FluxSampler::model_sigma(sig[i]),
                                     req.guidance, op);
        FluxSampler::step(x, v, sig[i], sig[i + 1]);
    }
    const auto t2 = Clock::now();
    const auto lat = FluxSampler::unpack(x, kLatentCh, h, w);
    auto img = vae_->decode(lat, h, w, op);
    const auto t3 = Clock::now();
    if (timing != nullptr) *timing = {ms(t0, t1), ms(t1, t2), ms(t2, t3)};
    return img;
}

std::vector<std::uint8_t> FluxPipeline::generate(
    const Request& req, const backend::Ops& op,
    const std::function<bool()>& alive, Timing* timing) const {
    validate(req);  // before the noise: a bad size costs nothing
    const std::size_t n =
        std::size_t{kLatentCh} * (req.height / 8) * (req.width / 8);
    const auto latent = PixArtPipeline::normal_latent(req.seed, n);
    const auto img = generate_from(latent, req, op, alive, timing);
    if (img.empty()) return {};
    return SdVaeDecoder::to_rgb8(img, req.height, req.width);
}

}  // namespace locus::image
