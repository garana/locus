#include "locus/image/qwen_edit_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "locus/image/flux_sampler.hpp"
#include "locus/image/pixart_pipeline.hpp"
#include "locus/image/qwen_edit_image.hpp"
#include "locus/image/qwen_edit_text.hpp"

namespace locus::image {

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

}  // namespace

std::unique_ptr<QwenEditPipeline> QwenEditPipeline::load_dir(
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
        throw gguf::Error("qwen edit: cannot list " + dir + ": " +
                          ec.message());
    }
    std::sort(paths.begin(), paths.end());
    std::vector<std::unique_ptr<gguf::GgufFile>> files;
    const gguf::GgufFile *lm = nullptr, *vision = nullptr, *dit = nullptr,
                         *vae = nullptr;
    for (const auto& path : paths) {
        auto g = std::make_unique<gguf::GgufFile>(
            gguf::GgufFile::open(path.string()));
        const auto arch = g->get_string("general.architecture");
        const gguf::GgufFile** slot = nullptr;
        if (arch == "qwen2vl") {
            slot = &lm;
        } else if (arch == "clip") {
            slot = &vision;
        } else if (arch == "qwen_image") {
            slot = &dit;
        } else if (arch == "qwen_image_vae") {
            slot = &vae;
        } else {
            continue;  // not a part of the pipeline
        }
        if (*slot != nullptr) {
            throw gguf::Error("qwen edit: two " + std::string(*arch) +
                              " GGUFs in " + dir);
        }
        *slot = g.get();
        files.push_back(std::move(g));
    }
    if (lm == nullptr || vision == nullptr || dit == nullptr ||
        vae == nullptr) {
        throw gguf::Error("qwen edit: " + dir +
                          " needs a qwen2vl, a clip (mmproj), a qwen_image "
                          "and a qwen_image_vae GGUF");
    }
    auto p = from_parts(*lm, *vision, *dit, *vae, *lm);
    p->files_ = std::move(files);
    return p;
}

std::unique_ptr<QwenEditPipeline> QwenEditPipeline::from_parts(
    const gguf::GgufFile& lm, const gguf::GgufFile& vision,
    const gguf::GgufFile& dit, const gguf::GgufFile& vae,
    const gguf::GgufFile& tok) {
    auto p = std::unique_ptr<QwenEditPipeline>(new QwenEditPipeline());
    p->tok_ = tok::tokenizer_from_gguf(tok);
    p->lm_ = std::make_unique<model::TransformerModel>(
        model::TransformerModel::load(lm));
    p->vision_ = std::make_unique<QwenVision>(QwenVision::load(vision));
    p->dit_ = std::make_unique<QwenImageDit>(QwenImageDit::load(dit));
    p->sampler_ =
        std::make_unique<QwenImageSampler>(QwenImageSampler::load(dit));
    p->vae_ = std::make_unique<QwenVae>(QwenVae::load(vae));
    const std::uint32_t e = p->lm_->hparams().n_embd;
    if (p->vision_->out_dim() != e || p->dit_->text_width() != e) {
        throw gguf::Error(
            "qwen edit: the vision tower, the language model and the "
            "MMDiT's text width do not match");
    }
    if (!p->vae_->has_encoder() ||
        p->vae_->z_dim() * 4 != p->dit_->channels() ||
        p->dit_->out_channels() != p->dit_->channels()) {
        throw gguf::Error(
            "qwen edit: the VAE (with its encoder) and the MMDiT's "
            "channels do not match");
    }
    return p;
}

std::pair<std::uint32_t, std::uint32_t> QwenEditPipeline::output_size(
    const Request& req) const {
    if (req.reference.w == 0 || req.reference.h == 0) {
        throw std::invalid_argument("qwen edit: no reference image");
    }
    const auto [vw, vh] =
        qwen_edit_dimensions(req.reference.w, req.reference.h, req.area);
    const std::uint32_t m = 2 * vae_->scale();
    const std::uint32_t w = (req.width ? req.width : vw) / m * m;
    const std::uint32_t h = (req.height ? req.height : vh) / m * m;
    if (w == 0 || h == 0) {
        throw std::invalid_argument("qwen edit: the size is too small");
    }
    return {w, h};
}

std::vector<float> QwenEditPipeline::text_rows(
    const std::string& prompt, std::span<const float> patches,
    std::uint32_t gh, std::uint32_t gw, const backend::Ops& op,
    std::uint32_t& n) const {
    const std::uint32_t merge = vision_->merge_size();
    const auto p =
        qwen_edit_prompt(*tok_, prompt, gh / merge * (gw / merge));
    if (p.ids.size() <= kEditDropTokens) {
        throw std::invalid_argument("qwen edit: the prompt is too short");
    }
    n = static_cast<std::uint32_t>(p.ids.size() - kEditDropTokens);
    return qwen_edit_text_embeds(*lm_, *vision_, p.ids, p.image_at, patches,
                                 gh, gw, kEditDropTokens, op);
}

std::vector<float> QwenEditPipeline::edit_from(
    std::span<const float> noise, const Request& req,
    const backend::Ops& op, const std::function<bool()>& alive,
    Timing* timing) const {
    if (!std::isfinite(req.true_cfg_scale) || req.steps > 1000) {
        throw std::invalid_argument(
            "qwen edit: steps must be at most 1000 and true_cfg_scale "
            "finite");
    }
    const auto [w, h] = output_size(req);
    const std::uint32_t s = vae_->scale();
    const QwenImageDit::Segment noise_seg{1, h / s / 2, w / s / 2};
    if (noise.size() != std::size_t{noise_seg.rows} * noise_seg.cols *
                            dit_->channels()) {
        throw std::invalid_argument("qwen edit: noise size");
    }
    auto t0 = std::chrono::steady_clock::now();
    const auto r = qwen_edit_reference(req.reference, req.area);
    const std::uint32_t ps = vision_->patch_size();
    const std::uint32_t th = r.text_image.h, tw = r.text_image.w;
    const auto patches = vision_->patches(r.text_image.rgb, th, tw);
    std::uint32_t n_pos = 0, n_neg = 0;
    const auto pos = text_rows(req.prompt, patches, th / ps, tw / ps, op,
                               n_pos);
    std::vector<float> neg;
    if (req.negative_prompt && req.true_cfg_scale > 1.0f) {
        neg = text_rows(*req.negative_prompt, patches, th / ps, tw / ps, op,
                        n_neg);
    }
    const auto lat = qwen_edit_reference_latents(
        *vae_, r.vae_input, r.vae_image.h, r.vae_image.w, op);
    if (timing != nullptr) timing->encode = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    const auto out = qwen_edit_denoise(
        *dit_, *sampler_, noise, noise_seg, lat.tokens,
        {1, lat.rows, lat.cols}, QwenEditText{pos, n_pos},
        QwenEditText{neg, n_neg}, req.true_cfg_scale, req.steps, op, alive);
    if (timing != nullptr) timing->denoise = ms_since(t0);
    if (out.empty()) return {};
    t0 = std::chrono::steady_clock::now();
    auto img = qwen_edit_decode(*vae_, out, h, w, op);
    if (timing != nullptr) timing->decode = ms_since(t0);
    return img;
}

std::vector<std::uint8_t> QwenEditPipeline::edit(
    const Request& req, const backend::Ops& op,
    const std::function<bool()>& alive, std::uint32_t& out_w,
    std::uint32_t& out_h, Timing* timing) const {
    const auto [w, h] = output_size(req);
    const std::uint32_t s = vae_->scale(), c = vae_->z_dim();
    // prepare_latents: randn of [c][h / s][w / s], then packed.
    const auto z = PixArtPipeline::normal_latent(
        req.seed, std::size_t{c} * (h / s) * (w / s));
    const auto noise = FluxSampler::pack(z, c, h / s, w / s);
    const auto img = edit_from(noise, req, op, alive, timing);
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
