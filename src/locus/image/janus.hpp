#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::image {

/**
 * Python's str.strip(): drops leading and trailing whitespace from
 * str.isspace()'s set (ASCII and Unicode), decoding UTF-8 at both
 * ends. The Janus reference strips every message with it.
 */
std::string_view python_strip(std::string_view s);

/**
 * Janus-Pro text-to-image generation (i#221 sections 2-3) on a llama
 * backbone loaded from the same GGUF (scripts/convert-janus.py). Image
 * tokens do not live in the text vocabulary: each one enters the
 * backbone through gen_embed + gen_aligner as an input embedding, and
 * the image logits come from gen_head on the final normed hidden state.
 * Generation runs a conditional and an unconditional sequence (the
 * prompt with its interior replaced by the pad id) and samples from
 * u + cfg * (c - u), feeding the chosen code to both.
 */
class JanusImageGen {
  public:
    /** Sampling settings; the defaults are the reference's. */
    struct Params {
        float cfg = 5.0f;          ///< classifier-free guidance scale
        float temperature = 1.0f;  ///< 0 picks the argmax (the golden)
        std::uint64_t seed = 0;    ///< for temperature > 0
    };

    /**
     * Loads the janus.* tensors (F32) and keys from the model's GGUF.
     * @throws gguf::Error when they are missing or the wrong size.
     */
    static JanusImageGen load(const gguf::GgufFile& g, std::uint32_t n_embd);

    /** @returns Codes per image (grid * grid, 576 for Janus-Pro). */
    std::uint32_t image_tokens() const { return grid_ * grid_; }
    /** @returns The code grid side (24). */
    std::uint32_t grid() const { return grid_; }
    /** @returns The default guidance scale from the GGUF (5). */
    float cfg_scale() const { return cfg_scale_; }
    /** @returns The <begin_of_image> id that ends the prompt. */
    tok::TokenId begin_of_image() const { return begin_of_image_; }

    /** The input embedding of image code `id`: gen_aligner(gen_embed). */
    void image_embed(std::int32_t id, std::span<float> out) const;

    /** Image-code logits from a final normed hidden state (gen_head). */
    void image_logits(std::span<const float> h, std::span<float> out) const;

    /**
     * Generates one image's codes, row-major, on the model's CPU ops.
     *
     * @param m The Janus-Pro backbone.
     * @param prompt The conditional prompt ids, BOS first and
     *     begin_of_image() last (the reference's wrapping).
     * @param p Guidance and sampling.
     * @param alive Polled each step; return false to stop early (the
     *     codes so far are returned). May be empty.
     * @returns image_tokens() codes (fewer if stopped).
     */
    std::vector<std::int32_t> generate(
        const model::TransformerModel& m,
        std::span<const tok::TokenId> prompt, const Params& p,
        const std::function<bool()>& alive = {}) const;

    /**
     * Wraps a text prompt as the reference does (Janus's "deepseek" SFT
     * template, empty system prompt) and tokenizes it: BOS, then
     * "<|User|>: <text>\n\n<|Assistant|>:", then begin_of_image(). The
     * roles are Janus-Pro's special tokens (not Janus-1.3B's "User").
     * The text is stripped of leading and trailing whitespace first, as
     * Python's str.strip() does in the reference.
     *
     * @param tok The model's tokenizer (from the same GGUF).
     * @param text The image description.
     * @returns The conditional prompt for generate().
     */
    std::vector<tok::TokenId> prompt_ids(const tok::Tokenizer& tok,
                                         std::string_view text) const;

    /** @returns The unconditional prompt for `prompt`: the same ids with
     * every position but the first and last set to the pad id. */
    std::vector<tok::TokenId> unconditional(
        std::span<const tok::TokenId> prompt) const;

  private:
    std::span<const float> t(const std::string& name, std::size_t n) const;

    std::unordered_map<std::string, std::vector<float>> w_;
    std::uint32_t n_embd_ = 0;
    std::uint32_t codes_ = 0;  ///< codebook size (16384)
    std::uint32_t grid_ = 0;
    tok::TokenId begin_of_image_ = 0;
    tok::TokenId pad_ = 0;
    float cfg_scale_ = 0.0f;
};

}  // namespace locus::image
