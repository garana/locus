#include "locus/image/qwen_vl_chat.hpp"

#include <algorithm>
#include <stdexcept>

#include "locus/image/qwen_edit_image.hpp"

namespace locus::image {

std::string qwen_vl_chat_text(std::span<const ChatTurn> turns,
                              const std::optional<std::string>& system) {
    if (turns.empty()) {
        throw std::invalid_argument("qwen vl chat: no turns");
    }
    if (turns.back().role != "user") {
        throw std::invalid_argument(
            "qwen vl chat: the last turn must be the user's");
    }
    std::string out = "<|im_start|>system\n";
    out += system ? std::string_view(*system) : kQwenVlSystemPrompt;
    out += "<|im_end|>\n";
    for (const ChatTurn& t : turns) {
        if (t.role != "user" && t.role != "assistant") {
            throw std::invalid_argument("qwen vl chat: unsupported role " +
                                        t.role);
        }
        out += "<|im_start|>";
        out += t.role;
        out += "\n";
        std::string_view rest = t.text;
        for (;;) {
            const auto at = rest.find(kImagePlaceholder);
            out += rest.substr(0, at);
            if (at == std::string_view::npos) break;
            out += "<|vision_start|><|image_pad|><|vision_end|>";
            rest.remove_prefix(at + kImagePlaceholder.size());
        }
        out += "<|im_end|>\n";
    }
    out += "<|im_start|>assistant\n";
    return out;
}

std::pair<std::uint32_t, std::uint32_t> qwen_vl_size(
    const QwenVision& v, const Rgb8& img, std::uint64_t max_pixels) {
    if (img.w == 0 || img.h == 0 ||
        img.rgb.size() != std::size_t{img.w} * img.h * 3) {
        throw std::invalid_argument("qwen vl chat: empty or bad image");
    }
    return qwen_smart_resize(img.h, img.w, v.patch_size() * v.merge_size(),
                             kQwenVlMinPixels, max_pixels);
}

QwenVlImage qwen_vl_image(const QwenVision& v, const Rgb8& img,
                          std::uint64_t max_pixels) {
    const auto [h, w] = qwen_vl_size(v, img, max_pixels);
    const Rgb8 sized =
        (w == img.w && h == img.h) ? img : resize_bicubic(img, w, h);
    QwenVlImage out;
    out.patches = v.patches(sized.rgb, h, w);
    out.gh = h / v.patch_size();
    out.gw = w / v.patch_size();
    return out;
}

QwenVlPrompt qwen_vl_chat_ids(
    const tok::Tokenizer& tok, std::string_view text,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> grids) {
    const auto pad = tok.encode("<|image_pad|>", false);
    if (pad.size() != 1) {
        throw std::invalid_argument(
            "qwen vl chat: the tokenizer has no <|image_pad|> token");
    }
    const auto ids = tok.encode(text, false);
    QwenVlPrompt p;
    p.ids.reserve(ids.size());
    std::uint32_t pos = 0;
    std::size_t k = 0;
    for (const tok::TokenId id : ids) {
        if (id != pad[0]) {
            p.ids.push_back(id);
            p.pos.push_back({pos, pos, pos});
            ++pos;
            continue;
        }
        if (k == grids.size()) {
            throw std::invalid_argument(
                "qwen vl chat: more image pads than images");
        }
        const auto [rows, cols] = grids[k++];
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("qwen vl chat: empty image grid");
        }
        p.image_at.push_back(static_cast<std::uint32_t>(p.ids.size()));
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (std::uint32_t c = 0; c < cols; ++c) {
                p.ids.push_back(pad[0]);
                p.pos.push_back({pos, pos + r, pos + c});
            }
        }
        pos += std::max(rows, cols);
    }
    if (k != grids.size()) {
        throw std::invalid_argument(
            "qwen vl chat: fewer image pads than images");
    }
    p.next = pos;
    return p;
}

bool qwen_vl_prefill(const model::TransformerModel& lm,
                     const QwenVision& vision, const QwenVlPrompt& p,
                     std::span<const QwenVlImage> images,
                     kv::PagedKvCache& cache, kv::PagedKvCache::Seq& seq,
                     model::TransformerModel::Workspace& ws,
                     std::span<float> logits, const backend::Ops& op,
                     const std::function<bool()>& alive) {
    const std::uint32_t E = lm.hparams().n_embd;
    if (vision.out_dim() != E) {
        throw std::invalid_argument(
            "qwen vl chat: the vision tower's width is not the model's");
    }
    if (images.size() != p.image_at.size() || p.pos.size() != p.ids.size()) {
        throw std::invalid_argument(
            "qwen vl chat: the images do not match the prompt");
    }
    const std::uint32_t m = vision.merge_size();
    std::vector<std::vector<float>> rows;
    std::vector<const float*> embd(p.ids.size(), nullptr);
    for (std::size_t k = 0; k < images.size(); ++k) {
        const QwenVlImage& im = images[k];
        const std::uint32_t n = (im.gh / m) * (im.gw / m);
        if (p.image_at[k] + n > p.ids.size()) {
            throw std::invalid_argument(
                "qwen vl chat: an image runs past the prompt");
        }
        rows.push_back(vision.encode(im.patches, im.gh, im.gw, op));
        if (rows.back().size() != std::size_t{n} * E) {
            throw std::invalid_argument(
                "qwen vl chat: the tower's rows do not match the image's "
                "grid");
        }
        for (std::uint32_t i = 0; i < n; ++i) {
            embd[p.image_at[k] + i] = rows.back().data() + std::size_t{i} * E;
        }
    }
    const std::size_t n = p.ids.size();
    const std::size_t chunk = std::max<std::size_t>(lm.prefill_chunk(), 1);
    for (std::size_t at = 0; at < n; at += chunk) {
        if (alive && !alive()) return false;
        const std::size_t nb = std::min(chunk, n - at);
        lm.forward_batch(
            std::span<const tok::TokenId>(p.ids.data() + at, nb), cache,
            seq, ws, logits, false,
            std::span<const float* const>(embd.data() + at, nb),
            std::span<const model::MropePos>(p.pos.data() + at, nb));
    }
    return true;
}

void qwen_vl_step(const model::TransformerModel& lm, tok::TokenId token,
                  std::uint32_t pos, kv::PagedKvCache& cache,
                  kv::PagedKvCache::Seq& seq,
                  model::TransformerModel::Workspace& ws,
                  std::span<float> logits) {
    const model::MropePos mp{pos, pos, pos};
    lm.forward_batch(std::span<const tok::TokenId>(&token, 1), cache, seq,
                     ws, logits, false, {},
                     std::span<const model::MropePos>(&mp, 1));
}

}  // namespace locus::image
