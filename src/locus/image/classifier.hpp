#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/image/image_input.hpp"
#include "locus/image/inception.hpp"
#include "locus/image/resnet.hpp"

namespace locus::image {

/**
 * A CNN image classifier and embedder for the server (i#577): one of
 * the "resnet" or "inception" GGUFs of scripts/convert-cnn.py, picked by
 * general.architecture, with its preprocessing. Owns the GGUF the model
 * maps.
 */
class ImageClassifier {
  public:
    /** @throws gguf::Error on another architecture or a bad file. */
    static std::unique_ptr<ImageClassifier> open(const std::string& path);

    /** @returns The model id: general.name (e.g. resnet50.tv_in1k). */
    const std::string& name() const { return name_; }
    std::uint32_t num_classes() const;
    std::uint32_t embedding_size() const;
    const std::string& label(std::uint32_t i) const;

    struct Output {
        std::vector<float> logits, embedding;
    };

    /** The eval transform, then the network, on one image; with gpu
     *  the convs run on Vulkan. */
    Output run(const Rgb8& img, bool gpu = false) const;

  private:
    std::unique_ptr<gguf::GgufFile> g_;
    std::string name_;
    std::optional<ResNet> resnet_;
    std::optional<Inception> inception_;
};

}  // namespace locus::image
