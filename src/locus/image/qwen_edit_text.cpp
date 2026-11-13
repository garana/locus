#include "locus/image/qwen_edit_text.hpp"

#include <stdexcept>
#include <string>

namespace locus::image {

EditPrompt qwen_edit_prompt(const tok::Tokenizer& tok,
                            std::string_view instruction,
                            std::uint32_t n_image) {
    if (n_image == 0) {
        throw std::invalid_argument("qwen edit: no image tokens");
    }
    const auto pad = tok.encode("<|image_pad|>", false);
    if (pad.size() != 1) {
        throw std::invalid_argument(
            "qwen edit: the tokenizer has no <|image_pad|> token");
    }
    // One string, as the processor tokenizes it: the special tokens
    // split the text, so the pads stay one token each.
    std::string text(kEditTemplateHead);
    for (std::uint32_t i = 0; i < n_image; ++i) text += "<|image_pad|>";
    text += kEditTemplateMid;
    text += instruction;
    text += kEditTemplateTail;
    EditPrompt p;
    p.ids = tok.encode(text, false);
    p.n_image = n_image;
    std::uint32_t at = 0;
    while (at < p.ids.size() && p.ids[at] != pad[0]) ++at;
    if (at + n_image > p.ids.size()) {
        throw std::invalid_argument("qwen edit: image span not found");
    }
    for (std::uint32_t i = 0; i < n_image; ++i) {
        if (p.ids[at + i] != pad[0]) {
            throw std::invalid_argument("qwen edit: image span split");
        }
    }
    p.image_at = at;
    return p;
}

EditImageRows qwen_edit_image_rows(const model::TransformerModel& lm,
                                   const QwenVision& vision,
                                   std::span<const float> patches,
                                   std::uint32_t gh, std::uint32_t gw,
                                   const backend::Ops& op) {
    if (vision.out_dim() != lm.hparams().n_embd) {
        throw std::invalid_argument(
            "qwen edit: the vision tower's width is not the model's");
    }
    const std::uint32_t mg = vision.merge_size();
    if (gh == 0 || gw == 0 || gh % mg != 0 || gw % mg != 0) {
        throw std::invalid_argument("qwen edit: bad patch grid");
    }
    return {vision.encode(patches, gh, gw, op), gh / mg, gw / mg};
}

std::vector<float> qwen_edit_text_embeds(const model::TransformerModel& lm,
                                         const QwenVision& vision,
                                         std::span<const tok::TokenId> ids,
                                         std::uint32_t image_at,
                                         std::span<const float> patches,
                                         std::uint32_t gh, std::uint32_t gw,
                                         std::uint32_t drop,
                                         const backend::Ops& op) {
    const auto img = qwen_edit_image_rows(lm, vision, patches, gh, gw, op);
    return qwen_edit_text_embeds(lm, img.rows, ids, image_at, img.grid_h,
                                 img.grid_w, drop);
}

std::vector<float> qwen_edit_text_embeds(const model::TransformerModel& lm,
                                         std::span<const float> image_rows,
                                         std::span<const tok::TokenId> ids,
                                         std::uint32_t image_at,
                                         std::uint32_t grid_h,
                                         std::uint32_t grid_w,
                                         std::uint32_t drop) {
    const std::uint32_t E = lm.hparams().n_embd;
    const std::uint32_t n_image = grid_h * grid_w;
    if (n_image == 0 || image_rows.size() != std::size_t{n_image} * E) {
        throw std::invalid_argument(
            "qwen edit: the image rows do not match the grid and the "
            "model's width");
    }
    const auto n = static_cast<std::uint32_t>(ids.size());
    if (image_at == 0 || image_at + n_image > n || drop >= n) {
        throw std::invalid_argument(
            "qwen edit: the image span or the drop does not fit the ids");
    }
    std::vector<const float*> embd(n, nullptr);
    for (std::uint32_t i = 0; i < n_image; ++i) {
        embd[image_at + i] = image_rows.data() + std::size_t{i} * E;
    }
    const auto pos = model::mrope_positions(image_at, grid_h, grid_w,
                                            n - image_at - n_image);
    auto cache = lm.make_cache();
    auto ws = lm.make_workspace();
    kv::PagedKvCache::Seq seq;
    if (!cache.ensure_capacity(seq, n)) {
        throw std::invalid_argument("qwen edit: prompt exceeds the cache");
    }
    std::vector<float> hidden(std::size_t{n} * E);
    lm.forward_batch(ids, cache, seq, ws, hidden, false, embd, pos, true);
    cache.release(seq);
    hidden.erase(hidden.begin(),
                 hidden.begin() + static_cast<std::ptrdiff_t>(
                                      std::size_t{drop} * E));
    return hidden;
}

}  // namespace locus::image
