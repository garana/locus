#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/image/cnn.hpp"
#include "locus/image/image_input.hpp"

namespace locus::image {

/**
 * Inception-v3 as an image classifier and embedder (i#576), as timm's
 * inception_v3 (torchvision's weights): a stem of 3 x 3 convs and max
 * pools to 35 x 35, the Inception blocks (A x 3, B, C x 4, D, E x 2),
 * whose parallel branches run on one input and are concatenated on the
 * channel axis, then global average pooling to the 2048-wide embedding
 * and the classifier. Every conv has its batch norm folded in and a
 * ReLU after it.
 *
 * Weights come from the "inception" GGUF of scripts/convert-cnn.py,
 * under timm's names (Mixed_6b.branch7x7_2, ...). The tensors stay in
 * the GGUF mapping, so the GgufFile must outlive the model.
 */
class Inception {
  public:
    /** @throws gguf::Error on a missing key or tensor, or a wrong size. */
    static Inception load(const gguf::GgufFile& g);

    /** @returns The network's input side (299). */
    std::uint32_t input_size() const { return pre_.size; }
    /** @returns The embedding width (2048). */
    std::uint32_t embedding_size() const { return 2048; }
    /** @returns The number of classes (1000). */
    std::uint32_t num_classes() const {
        return static_cast<std::uint32_t>(labels_.size());
    }
    /** @returns Class i's label. */
    const std::string& label(std::uint32_t i) const { return labels_.at(i); }

    /** The reference eval transform (cnn::preprocess).
     *  @returns [3][input_size][input_size]. */
    std::vector<float> preprocess(const Rgb8& img) const {
        return cnn::preprocess(img, pre_);
    }

    struct Output {
        std::vector<float> logits;     ///< num_classes()
        std::vector<float> embedding;  ///< 2048, pooled
    };

    /**
     * @param chw [3][input_size][input_size], as preprocess() gives it.
     * @param gpu The convs on Vulkan (cnn::conv); the rest on the CPU.
     * @throws std::invalid_argument on a wrong size.
     */
    Output run(std::span<const float> chw, bool gpu = false) const;

  private:
    /** A feature map: [c][h][w]. */
    struct Map {
        std::vector<float> x;
        std::uint32_t c = 0, h = 0, w = 0;
    };

    /** The conv NAME with ReLU: stride 2 unpadded, stride 1 either
     *  "same" (k / 2 each way) or unpadded. */
    Map conv(const std::string& name, const Map& in, std::uint32_t stride,
             bool same) const;
    Map block_a(const std::string& p, const Map& x) const;
    Map block_b(const std::string& p, const Map& x) const;
    Map block_c(const std::string& p, const Map& x) const;
    Map block_d(const std::string& p, const Map& x) const;
    Map block_e(const std::string& p, const Map& x) const;

    cnn::Preprocess pre_;
    std::map<std::string, cnn::Conv> convs_;
    std::span<const float> fc_w_, fc_b_;
    std::vector<std::string> labels_;
};

}  // namespace locus::image
