#include "locus/image/resnet.hpp"

#include <algorithm>
#include <stdexcept>

namespace locus::image {

namespace {

/** The conv with torchvision's padding (k / 2 on both sides). */
std::vector<float> conv_same(const cnn::Conv& c, std::span<const float> x,
                             std::uint32_t& h, std::uint32_t& w,
                             std::uint32_t stride, bool relu) {
    return cnn::conv(c, x, h, w, stride, c.kh / 2, c.kw / 2, relu);
}

}  // namespace

ResNet ResNet::load(const gguf::GgufFile& g) {
    const std::string p = "resnet.";
    ResNet r;
    r.pre_ = cnn::read_preprocess(g, p);
    r.labels_ = cnn::read_labels(g, p);
    const auto* layers = g.get_array(p + "layers");
    if (layers == nullptr || layers->size() != 4) {
        throw gguf::Error("resnet: missing resnet.layers (4 stages)");
    }
    const auto square = [&](const std::string& name, std::uint32_t in,
                            std::uint32_t out, std::uint32_t k) {
        const cnn::Conv c = cnn::read_conv(g, name, in, out);
        if (c.kh != k || c.kw != k) {
            throw gguf::Error("resnet: " + name + " is not " +
                              std::to_string(k) + " x " + std::to_string(k));
        }
        return c;
    };
    r.stem_ = square("conv1", 3, 64, 7);
    std::uint32_t in = 64;
    for (std::uint32_t s = 0; s < 4; ++s) {
        const gguf::Value& v = (*layers)[s];
        const auto n = static_cast<std::uint32_t>(
            v.type == gguf::ValueType::kInt32 ? v.i
                                              : static_cast<std::int64_t>(v.u));
        const std::uint32_t width = 64u << s, out = width * 4;
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::string b = "layer" + std::to_string(s + 1) + "." +
                                  std::to_string(i) + ".";
            Block k;
            k.stride = s > 0 && i == 0 ? 2 : 1;
            k.c1 = square(b + "conv1", in, width, 1);
            k.c2 = square(b + "conv2", width, width, 3);
            k.c3 = square(b + "conv3", width, out, 1);
            if (i == 0) k.down = square(b + "downsample", in, out, 1);
            r.blocks_.push_back(k);
            in = out;
        }
    }
    r.feat_ = in;
    r.fc_w_ = cnn::read_f32(g, "fc.weight",
                            std::size_t{r.num_classes()} * in);
    r.fc_b_ = cnn::read_f32(g, "fc.bias", r.num_classes());
    return r;
}

ResNet::Output ResNet::run(std::span<const float> chw) const {
    if (chw.size() != std::size_t{3} * pre_.size * pre_.size) {
        throw std::invalid_argument("resnet: need [3][size][size]");
    }
    std::uint32_t h = pre_.size, w = pre_.size;
    auto x = conv_same(stem_, chw, h, w, 2, true);
    x = cnn::max_pool3(x, stem_.out, h, w, 2, 1);
    for (const Block& b : blocks_) {
        std::uint32_t h1 = h, w1 = w;
        const auto a = conv_same(b.c1, x, h1, w1, 1, true);
        const auto m = conv_same(b.c2, a, h1, w1, b.stride, true);
        auto o = conv_same(b.c3, m, h1, w1, 1, false);
        if (b.down.out != 0) {
            std::uint32_t h2 = h, w2 = w;
            const auto s = conv_same(b.down, x, h2, w2, b.stride, false);
            for (std::size_t i = 0; i < o.size(); ++i) o[i] += s[i];
        } else {
            for (std::size_t i = 0; i < o.size(); ++i) o[i] += x[i];
        }
        for (float& v : o) v = std::max(v, 0.0f);
        x = std::move(o);
        h = h1;
        w = w1;
    }
    Output out;
    out.embedding = cnn::global_avg(x, feat_, std::size_t{h} * w);
    out.logits = cnn::linear(fc_w_, fc_b_, out.embedding);
    return out;
}

}  // namespace locus::image
