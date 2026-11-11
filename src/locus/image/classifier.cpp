#include "locus/image/classifier.hpp"

namespace locus::image {

std::unique_ptr<ImageClassifier> ImageClassifier::open(
    const std::string& path) {
    auto c = std::unique_ptr<ImageClassifier>(new ImageClassifier());
    c->g_ = std::make_unique<gguf::GgufFile>(gguf::GgufFile::open(path));
    const auto arch = c->g_->get_string("general.architecture");
    const auto name = c->g_->get_string("general.name");
    c->name_ = name ? std::string(*name) : path;
    if (arch && *arch == "resnet") {
        c->resnet_.emplace(ResNet::load(*c->g_));
    } else if (arch && *arch == "inception") {
        c->inception_.emplace(Inception::load(*c->g_));
    } else {
        throw gguf::Error("image classifier: " + path +
                          " is not a resnet or inception GGUF "
                          "(scripts/convert-cnn.py)");
    }
    return c;
}

std::uint32_t ImageClassifier::num_classes() const {
    return resnet_ ? resnet_->num_classes() : inception_->num_classes();
}

std::uint32_t ImageClassifier::embedding_size() const {
    return resnet_ ? resnet_->embedding_size()
                   : inception_->embedding_size();
}

const std::string& ImageClassifier::label(std::uint32_t i) const {
    return resnet_ ? resnet_->label(i) : inception_->label(i);
}

ImageClassifier::Output ImageClassifier::run(const Rgb8& img,
                                             bool gpu) const {
    if (resnet_) {
        auto o = resnet_->run(resnet_->preprocess(img), gpu);
        return {std::move(o.logits), std::move(o.embedding)};
    }
    auto o = inception_->run(inception_->preprocess(img), gpu);
    return {std::move(o.logits), std::move(o.embedding)};
}

}  // namespace locus::image
