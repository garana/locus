#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/image/image_input.hpp"

namespace locus::image {

/**
 * A ResNet image classifier and embedder (i#575), as timm's ResNet-50
 * (torchvision's weights): the 7 x 7 stride-2 stem, a 3 x 3 stride-2
 * max pool, four stages of bottleneck blocks (1 x 1, 3 x 3 with the
 * stage's stride, 1 x 1 to four times the width, plus the identity or a
 * strided 1 x 1 projection, then ReLU), global average pooling to the
 * embedding, and the linear classifier.
 *
 * Weights come from the "resnet" GGUF of scripts/convert-cnn.py, with
 * each batch norm folded into its conv. The tensors stay in the GGUF
 * mapping, so the GgufFile must outlive the model.
 */
class ResNet {
  public:
    /** @throws gguf::Error on a missing key or tensor, or a wrong size. */
    static ResNet load(const gguf::GgufFile& g);

    /** @returns The network's input side (224). */
    std::uint32_t input_size() const { return size_; }
    /** @returns The embedding width (2048 for ResNet-50). */
    std::uint32_t embedding_size() const { return feat_; }
    /** @returns The number of classes (1000). */
    std::uint32_t num_classes() const {
        return static_cast<std::uint32_t>(labels_.size());
    }
    /** @returns Class i's label ("Granny Smith"). */
    const std::string& label(std::uint32_t i) const { return labels_.at(i); }

    /**
     * The reference eval transform: the shorter side resized to resize
     * (the GGUF's interpolation, Pillow's filter), a center crop of
     * input_size() (offsets rounded half to even, as Python's round),
     * then x / 255 and (x - mean) / std.
     *
     * @returns [3][input_size][input_size].
     * @throws std::invalid_argument on an empty image.
     */
    std::vector<float> preprocess(const Rgb8& img) const;

    struct Output {
        std::vector<float> logits;     ///< num_classes()
        std::vector<float> embedding;  ///< embedding_size(), pooled
    };

    /**
     * @param chw [3][input_size][input_size], as preprocess() gives it.
     * @throws std::invalid_argument on a wrong size.
     */
    Output run(std::span<const float> chw) const;

    /**
     * @returns The k classes with the highest softmax probability, as
     *     (class, probability), highest first.
     */
    static std::vector<std::pair<std::uint32_t, float>> top_k(
        std::span<const float> logits, std::uint32_t k);

    /** A conv with its batch norm folded in: [out][in][k][k], bias. */
    struct Conv {
        std::span<const float> w, b;
        std::uint32_t in = 0, out = 0, k = 0;
    };
    struct Block {
        Conv c1, c2, c3, down;  ///< down.out == 0: the identity
        std::uint32_t stride = 1;
    };

  private:
    std::uint32_t size_ = 0, resize_ = 0, feat_ = 0;
    bool bicubic_ = false;
    float mean_[3] = {}, std_[3] = {};
    Conv stem_;
    std::vector<Block> blocks_;
    std::span<const float> fc_w_, fc_b_;
    std::vector<std::string> labels_;
};

}  // namespace locus::image
