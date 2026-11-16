#include "locus/image/qwen_image_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "locus/image/flux_sampler.hpp"
#include "locus/image/pixart_pipeline.hpp"

namespace locus::image {

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

}  // namespace

std::vector<tok::TokenId> qwen_image_prompt(const tok::Tokenizer& tok,
                                            std::string_view prompt) {
    std::string text(kImageTemplateHead);
    text += prompt;
    text += kImageTemplateTail;
    auto ids = tok.encode(text, false);
    // truncation=True at max_length: the tail goes, the template's
    // head stays.
    const std::size_t cap = kImageTokenizerMax + kImageDropTokens;
    if (ids.size() > cap) ids.resize(cap);
    return ids;
}

std::vector<float> qwen_image_text_embeds(const model::TransformerModel& lm,
                                          std::span<const tok::TokenId> ids,
                                          std::uint32_t drop,
                                          std::uint32_t max_rows) {
    const std::uint32_t E = lm.hparams().n_embd;
    const auto n = static_cast<std::uint32_t>(ids.size());
    if (n <= drop) {
        throw std::invalid_argument("qwen image: the prompt is too short");
    }
    // Text only: get_rope_index gives every axis the token's position.
    const auto pos = model::mrope_positions(n, 0, 0, 0);
    auto cache = lm.make_cache();
    auto ws = lm.make_workspace();
    kv::PagedKvCache::Seq seq;
    if (!cache.ensure_capacity(seq, n)) {
        throw std::invalid_argument("qwen image: prompt exceeds the cache");
    }
    std::vector<float> hidden(std::size_t{n} * E);
    lm.forward_batch(ids, cache, seq, ws, hidden, false, {}, pos, true);
    cache.release(seq);
    hidden.erase(hidden.begin(),
                 hidden.begin() + static_cast<std::ptrdiff_t>(
                                      std::size_t{drop} * E));
    const std::uint32_t rows = std::min(n - drop, max_rows);
    hidden.resize(std::size_t{rows} * E);
    return hidden;
}

std::unique_ptr<QwenImagePipeline> QwenImagePipeline::load_dir(
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
        throw gguf::Error("qwen image: cannot list " + dir + ": " +
                          ec.message());
    }
    std::sort(paths.begin(), paths.end());
    std::vector<std::unique_ptr<gguf::GgufFile>> files;
    const gguf::GgufFile *lm = nullptr, *dit = nullptr, *vae = nullptr;
    for (const auto& path : paths) {
        auto g = std::make_unique<gguf::GgufFile>(
            gguf::GgufFile::open(path.string()));
        const auto arch = g->get_string("general.architecture");
        const gguf::GgufFile** slot = nullptr;
        if (arch == "qwen2vl") {
            slot = &lm;
        } else if (arch == "qwen_image") {
            slot = &dit;
        } else if (arch == "qwen_image_vae") {
            slot = &vae;
        } else {
            continue;  // not a part of the pipeline
        }
        if (*slot != nullptr) {
            throw gguf::Error("qwen image: two " + std::string(*arch) +
                              " GGUFs in " + dir);
        }
        *slot = g.get();
        files.push_back(std::move(g));
    }
    if (lm == nullptr || dit == nullptr || vae == nullptr) {
        throw gguf::Error("qwen image: " + dir +
                          " needs a qwen2vl, a qwen_image and a "
                          "qwen_image_vae GGUF");
    }
    auto p = from_parts(*lm, *dit, *vae, *lm);
    p->files_ = std::move(files);
    return p;
}

std::unique_ptr<QwenImagePipeline> QwenImagePipeline::from_parts(
    const gguf::GgufFile& lm, const gguf::GgufFile& dit,
    const gguf::GgufFile& vae, const gguf::GgufFile& tok) {
    // The checkpoint kind first (i#1099), before any part loads.
    qwen_image_check_pipeline(dit, "text-to-image");
    auto p = std::unique_ptr<QwenImagePipeline>(new QwenImagePipeline());
    p->tok_ = tok::tokenizer_from_gguf(tok);
    p->lm_ = std::make_unique<model::TransformerModel>(
        model::TransformerModel::load(lm));
    p->dit_ = std::make_unique<QwenImageDit>(QwenImageDit::load(dit));
    p->sampler_ =
        std::make_unique<QwenImageSampler>(QwenImageSampler::load(dit));
    p->vae_ = std::make_unique<QwenVae>(QwenVae::load(vae));
    if (p->dit_->text_width() != p->lm_->hparams().n_embd) {
        throw gguf::Error(
            "qwen image: the language model and the MMDiT's text width "
            "do not match");
    }
    if (p->vae_->z_dim() * 4 != p->dit_->channels() ||
        p->dit_->out_channels() != p->dit_->channels()) {
        throw gguf::Error(
            "qwen image: the VAE and the MMDiT's channels do not match");
    }
    return p;
}

std::pair<std::uint32_t, std::uint32_t> QwenImagePipeline::output_size(
    const Request& req) const {
    const std::uint32_t m = 2 * vae_->scale();
    const std::uint32_t w = req.width / m * m, h = req.height / m * m;
    if (w == 0 || h == 0) {
        throw std::invalid_argument("qwen image: the size is too small");
    }
    return {w, h};
}

std::vector<float> QwenImagePipeline::text_rows(const std::string& prompt,
                                                std::uint32_t max_rows,
                                                std::uint32_t& n) const {
    const auto ids = qwen_image_prompt(*tok_, prompt);
    auto rows =
        qwen_image_text_embeds(*lm_, ids, kImageDropTokens, max_rows);
    n = static_cast<std::uint32_t>(rows.size() / lm_->hparams().n_embd);
    return rows;
}

std::vector<float> QwenImagePipeline::generate_from(
    std::span<const float> noise, const Request& req,
    const backend::Ops& op, const std::function<bool()>& alive,
    Timing* timing) const {
    if (!std::isfinite(req.true_cfg_scale) || req.steps > 1000 ||
        req.max_sequence_length == 0 ||
        req.max_sequence_length > kImageTokenizerMax) {
        throw std::invalid_argument(
            "qwen image: steps must be at most 1000, true_cfg_scale "
            "finite and max_sequence_length from 1 to 1024");
    }
    const auto [w, h] = output_size(req);
    const std::uint32_t s = vae_->scale();
    const QwenImageDit::Segment noise_seg{1, h / s / 2, w / s / 2};
    if (noise.size() != std::size_t{noise_seg.rows} * noise_seg.cols *
                            dit_->channels()) {
        throw std::invalid_argument("qwen image: noise size");
    }
    auto t0 = std::chrono::steady_clock::now();
    std::uint32_t n_pos = 0, n_neg = 0;
    const auto pos = text_rows(req.prompt, req.max_sequence_length, n_pos);
    std::vector<float> neg;
    if (req.negative_prompt && req.true_cfg_scale > 1.0f) {
        neg = text_rows(*req.negative_prompt, req.max_sequence_length,
                        n_neg);
    }
    if (timing != nullptr) timing->encode = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    // No reference: the MMDiT sees the noise tokens alone.
    const auto out = qwen_edit_denoise(
        *dit_, *sampler_, noise, noise_seg, {}, {0, 0, 0},
        QwenEditText{pos, n_pos}, QwenEditText{neg, n_neg},
        req.true_cfg_scale, req.steps, op, alive);
    if (timing != nullptr) timing->denoise = ms_since(t0);
    if (out.empty()) return {};
    t0 = std::chrono::steady_clock::now();
    auto img = qwen_edit_decode(*vae_, out, h, w, op);
    if (timing != nullptr) timing->decode = ms_since(t0);
    return img;
}

std::vector<std::uint8_t> QwenImagePipeline::generate(
    const Request& req, const backend::Ops& op,
    const std::function<bool()>& alive, std::uint32_t& out_w,
    std::uint32_t& out_h, Timing* timing) const {
    const auto [w, h] = output_size(req);
    const std::uint32_t s = vae_->scale(), c = vae_->z_dim();
    // prepare_latents: randn of [c][h / s][w / s], then packed.
    const auto z = PixArtPipeline::normal_latent(
        req.seed, std::size_t{c} * (h / s) * (w / s));
    const auto noise = FluxSampler::pack(z, c, h / s, w / s);
    const auto img = generate_from(noise, req, op, alive, timing);
    if (img.empty()) return {};
    out_w = w;
    out_h = h;
    // numpy_to_pil: (x * 255).round() as uint8.
    std::vector<std::uint8_t> rgb(img.size());
    for (std::size_t i = 0; i < img.size(); ++i) {
        rgb[i] = static_cast<std::uint8_t>(std::nearbyint(img[i] * 255.0f));
    }
    return rgb;
}

}  // namespace locus::image
