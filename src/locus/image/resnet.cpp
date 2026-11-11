#include "locus/image/resnet.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "locus/backend/cpu_ops.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::image {

namespace {

using Conv = ResNet::Conv;

/** Runs fn(i) for i in [0, n) on the thread pool. */
void parallel(std::size_t n, const std::function<void(std::size_t)>& fn) {
    sys::ThreadPool::instance().parallel_for(n, fn);
}

/** The conv (padding k / 2, as torchvision's), [in][h][w] to
 *  [out][h'][w']; h and w are updated. */
std::vector<float> conv(const Conv& c, std::span<const float> x,
                        std::uint32_t& h, std::uint32_t& w,
                        std::uint32_t stride, bool relu) {
    const std::uint32_t pad = c.k / 2;
    const std::uint32_t oh = (h + 2 * pad - c.k) / stride + 1,
                        ow = (w + 2 * pad - c.k) / stride + 1;
    std::vector<float> out(std::size_t{c.out} * oh * ow);
    backend::conv2d_mt(x, c.in, h, w, c.w, c.out, c.k, c.k, c.b, stride, pad,
                       1, out);
    if (relu) {
        for (float& v : out) v = std::max(v, 0.0f);
    }
    h = oh;
    w = ow;
    return out;
}

/** torch.nn.MaxPool2d(3, stride 2, padding 1), the padding never
 *  chosen. */
std::vector<float> max_pool(std::span<const float> x, std::uint32_t c,
                            std::uint32_t& h, std::uint32_t& w) {
    const std::uint32_t oh = (h + 2 - 3) / 2 + 1, ow = (w + 2 - 3) / 2 + 1;
    std::vector<float> out(std::size_t{c} * oh * ow);
    parallel(c, [&](std::size_t k) {
        const float* in = &x[k * h * w];
        for (std::uint32_t y = 0; y < oh; ++y) {
            for (std::uint32_t xx = 0; xx < ow; ++xx) {
                float m = -std::numeric_limits<float>::infinity();
                for (int dy = -1; dy <= 1; ++dy) {
                    const int iy = static_cast<int>(2 * y) + dy;
                    if (iy < 0 || iy >= static_cast<int>(h)) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int ix = static_cast<int>(2 * xx) + dx;
                        if (ix < 0 || ix >= static_cast<int>(w)) continue;
                        m = std::max(m, in[iy * w + ix]);
                    }
                }
                out[(k * oh + y) * ow + xx] = m;
            }
        }
    });
    h = oh;
    w = ow;
    return out;
}

}  // namespace

ResNet ResNet::load(const gguf::GgufFile& g) {
    const std::string p = "resnet.";
    auto u = [&](const std::string& k) -> std::uint32_t {
        const auto x = g.get_uint(p + k);
        if (!x || *x == 0) throw gguf::Error("resnet: missing " + p + k);
        return static_cast<std::uint32_t>(*x);
    };
    ResNet r;
    r.size_ = u("input_size");
    r.resize_ = u("resize");
    const auto interp = g.get_string(p + "interpolation");
    if (!interp || (*interp != "bilinear" && *interp != "bicubic")) {
        throw gguf::Error("resnet: interpolation must be bilinear or "
                          "bicubic");
    }
    r.bicubic_ = *interp == "bicubic";
    const std::pair<const char*, float*> norms[] = {{"mean", r.mean_},
                                                    {"std", r.std_}};
    for (const auto& [key, dst] : norms) {
        const auto* a = g.get_array(p + key);
        if (a == nullptr || a->size() != 3) {
            throw gguf::Error(std::string("resnet: missing ") + p + key);
        }
        for (int c = 0; c < 3; ++c) dst[c] = static_cast<float>((*a)[c].f);
    }
    const auto* layers = g.get_array(p + "layers");
    if (layers == nullptr || layers->size() != 4) {
        throw gguf::Error("resnet: missing resnet.layers (4 stages)");
    }
    const auto* labels = g.get_array(p + "labels");
    if (labels == nullptr || labels->size() != u("num_classes")) {
        throw gguf::Error("resnet: labels do not match num_classes");
    }
    for (const auto& l : *labels) r.labels_.push_back(l.s);
    if (r.resize_ < r.size_) {
        throw gguf::Error("resnet: resize smaller than input_size");
    }

    const auto f32 = [&](const std::string& name, std::size_t n) {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr || ti->type != gguf::TensorType::kF32 ||
            ti->nelements() != n) {
            throw gguf::Error("resnet: " + name + " missing or not F32 x " +
                              std::to_string(n));
        }
        const auto b = g.tensor_data(*ti);
        return std::span<const float>(
            reinterpret_cast<const float*>(b.data()), n);
    };
    const auto conv_at = [&](const std::string& name, std::uint32_t in,
                             std::uint32_t out, std::uint32_t k) {
        Conv c;
        c.in = in;
        c.out = out;
        c.k = k;
        c.w = f32(name + ".weight", std::size_t{out} * in * k * k);
        c.b = f32(name + ".bias", out);
        return c;
    };
    r.stem_ = conv_at("conv1", 3, 64, 7);
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
            k.c1 = conv_at(b + "conv1", in, width, 1);
            k.c2 = conv_at(b + "conv2", width, width, 3);
            k.c3 = conv_at(b + "conv3", width, out, 1);
            if (i == 0) k.down = conv_at(b + "downsample", in, out, 1);
            r.blocks_.push_back(k);
            in = out;
        }
    }
    r.feat_ = in;
    r.fc_w_ = f32("fc.weight", std::size_t{r.num_classes()} * in);
    r.fc_b_ = f32("fc.bias", r.num_classes());
    return r;
}

std::vector<float> ResNet::preprocess(const Rgb8& img) const {
    if (img.w == 0 || img.h == 0) {
        throw std::invalid_argument("resnet: empty image");
    }
    // torchvision's Resize(int): the shorter side to resize_, the other
    // scaled and truncated.
    std::uint32_t w = resize_, h = resize_;
    if (img.w < img.h) {
        h = static_cast<std::uint32_t>(std::uint64_t{resize_} * img.h / img.w);
    } else if (img.h < img.w) {
        w = static_cast<std::uint32_t>(std::uint64_t{resize_} * img.w / img.h);
    }
    const Rgb8 r = bicubic_ ? resize_bicubic(img, w, h)
                            : resize_bilinear(img, w, h);
    // CenterCrop: int(round((side - size) / 2.0)), Python's round.
    const auto off = [&](std::uint32_t side) {
        return static_cast<std::uint32_t>(
            std::nearbyint((static_cast<double>(side) - size_) / 2.0));
    };
    const std::uint32_t top = off(h), left = off(w);
    const std::size_t plane = std::size_t{size_} * size_;
    std::vector<float> chw(3 * plane);
    for (std::uint32_t y = 0; y < size_; ++y) {
        for (std::uint32_t x = 0; x < size_; ++x) {
            const std::uint8_t* px =
                &r.rgb[((std::size_t{top} + y) * w + left + x) * 3];
            for (int c = 0; c < 3; ++c) {
                // ToTensor (float32 / 255), then Normalize.
                const float v = static_cast<float>(px[c]) / 255.0f;
                chw[c * plane + y * size_ + x] = (v - mean_[c]) / std_[c];
            }
        }
    }
    return chw;
}

ResNet::Output ResNet::run(std::span<const float> chw) const {
    if (chw.size() != std::size_t{3} * size_ * size_) {
        throw std::invalid_argument("resnet: need [3][size][size]");
    }
    std::uint32_t h = size_, w = size_;
    auto x = conv(stem_, chw, h, w, 2, true);
    x = max_pool(x, stem_.out, h, w);
    for (const Block& b : blocks_) {
        std::uint32_t h1 = h, w1 = w;
        const auto a = conv(b.c1, x, h1, w1, 1, true);
        const auto m = conv(b.c2, a, h1, w1, b.stride, true);
        auto o = conv(b.c3, m, h1, w1, 1, false);
        if (b.down.out != 0) {
            std::uint32_t h2 = h, w2 = w;
            const auto s = conv(b.down, x, h2, w2, b.stride, false);
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
    const std::size_t plane = std::size_t{h} * w;
    out.embedding.resize(feat_);
    for (std::uint32_t c = 0; c < feat_; ++c) {
        double s = 0.0;
        for (std::size_t i = 0; i < plane; ++i) s += x[c * plane + i];
        out.embedding[c] = static_cast<float>(s / static_cast<double>(plane));
    }
    out.logits.resize(num_classes());
    parallel(num_classes(), [&](std::size_t i) {
        double s = fc_b_[i];
        const float* row = &fc_w_[i * feat_];
        for (std::uint32_t c = 0; c < feat_; ++c) {
            s += double{row[c]} * out.embedding[c];
        }
        out.logits[i] = static_cast<float>(s);
    });
    return out;
}

std::vector<std::pair<std::uint32_t, float>> ResNet::top_k(
    std::span<const float> logits, std::uint32_t k) {
    const float mx = *std::max_element(logits.begin(), logits.end());
    double z = 0.0;
    for (float v : logits) z += std::exp(double{v} - mx);
    std::vector<std::pair<std::uint32_t, float>> all;
    for (std::uint32_t i = 0; i < logits.size(); ++i) {
        all.push_back(
            {i, static_cast<float>(std::exp(double{logits[i]} - mx) / z)});
    }
    k = std::min<std::uint32_t>(k, static_cast<std::uint32_t>(all.size()));
    std::partial_sort(all.begin(), all.begin() + k, all.end(),
                      [](const auto& a, const auto& b) {
                          return a.second > b.second ||
                                 (a.second == b.second && a.first < b.first);
                      });
    all.resize(k);
    return all;
}

}  // namespace locus::image
