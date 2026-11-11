#include "locus/image/cnn.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

#include "locus/backend/cpu_ops.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::image::cnn {

namespace {

/** Runs fn(i) for i in [0, n) on the thread pool. */
void parallel(std::size_t n, const std::function<void(std::size_t)>& fn) {
    sys::ThreadPool::instance().parallel_for(n, fn);
}

}  // namespace

std::span<const float> read_f32(const gguf::GgufFile& g,
                                const std::string& name, std::size_t n) {
    const gguf::TensorInfo* ti = g.find_tensor(name);
    if (ti == nullptr || ti->type != gguf::TensorType::kF32 ||
        ti->nelements() != n) {
        throw gguf::Error("cnn: " + name + " missing or not F32 x " +
                          std::to_string(n));
    }
    const auto b = g.tensor_data(*ti);
    return {reinterpret_cast<const float*>(b.data()), n};
}

Conv read_conv(const gguf::GgufFile& g, const std::string& name,
               std::uint32_t in, std::uint32_t out) {
    const gguf::TensorInfo* ti = g.find_tensor(name + ".weight");
    if (ti == nullptr || ti->n_dims != 4) {
        throw gguf::Error("cnn: " + name + ".weight missing or not 4D");
    }
    Conv c;
    c.kw = static_cast<std::uint32_t>(ti->ne[0]);
    c.kh = static_cast<std::uint32_t>(ti->ne[1]);
    c.in = static_cast<std::uint32_t>(ti->ne[2]);
    c.out = static_cast<std::uint32_t>(ti->ne[3]);
    if ((in != 0 && c.in != in) || (out != 0 && c.out != out)) {
        throw gguf::Error("cnn: " + name + " is " + std::to_string(c.out) +
                          " x " + std::to_string(c.in) + ", not " +
                          std::to_string(out) + " x " + std::to_string(in));
    }
    c.w = read_f32(g, name + ".weight",
                   std::size_t{c.out} * c.in * c.kh * c.kw);
    c.b = read_f32(g, name + ".bias", c.out);
    return c;
}

std::vector<float> conv(const Conv& c, std::span<const float> x,
                        std::uint32_t& h, std::uint32_t& w,
                        std::uint32_t stride, std::uint32_t ph,
                        std::uint32_t pw, bool relu) {
    // conv2d_mt pads both sides by one amount: pad the other side's
    // extra on the host first when they differ (1 x 7 / 7 x 1 kernels).
    std::vector<float> padded;
    std::span<const float> in = x;
    std::uint32_t ih = h, iw = w, pad = std::min(ph, pw);
    if (ph != pw) {
        const std::uint32_t eh = ph - pad, ew = pw - pad;
        ih = h + 2 * eh;
        iw = w + 2 * ew;
        padded.assign(std::size_t{c.in} * ih * iw, 0.0f);
        for (std::uint32_t k = 0; k < c.in; ++k) {
            for (std::uint32_t y = 0; y < h; ++y) {
                std::memcpy(&padded[(std::size_t{k} * ih + y + eh) * iw + ew],
                            &x[(std::size_t{k} * h + y) * w],
                            w * sizeof(float));
            }
        }
        in = padded;
    }
    const std::uint32_t oh = (ih + 2 * pad - c.kh) / stride + 1,
                        ow = (iw + 2 * pad - c.kw) / stride + 1;
    std::vector<float> out(std::size_t{c.out} * oh * ow);
    backend::conv2d_mt(in, c.in, ih, iw, c.w, c.out, c.kh, c.kw, c.b, stride,
                       pad, 1, out);
    if (relu) {
        for (float& v : out) v = std::max(v, 0.0f);
    }
    h = oh;
    w = ow;
    return out;
}

std::vector<float> max_pool3(std::span<const float> x, std::uint32_t c,
                             std::uint32_t& h, std::uint32_t& w,
                             std::uint32_t stride, std::uint32_t pad) {
    const std::uint32_t oh = (h + 2 * pad - 3) / stride + 1,
                        ow = (w + 2 * pad - 3) / stride + 1;
    std::vector<float> out(std::size_t{c} * oh * ow);
    const int ip = static_cast<int>(pad), ih = static_cast<int>(h),
              iw = static_cast<int>(w);
    parallel(c, [&](std::size_t k) {
        const float* src = &x[k * h * w];
        for (std::uint32_t y = 0; y < oh; ++y) {
            for (std::uint32_t xx = 0; xx < ow; ++xx) {
                float m = -std::numeric_limits<float>::infinity();
                for (int dy = 0; dy < 3; ++dy) {
                    const int iy = static_cast<int>(y * stride) + dy - ip;
                    if (iy < 0 || iy >= ih) continue;
                    for (int dx = 0; dx < 3; ++dx) {
                        const int ix = static_cast<int>(xx * stride) + dx - ip;
                        if (ix < 0 || ix >= iw) continue;
                        m = std::max(m, src[iy * iw + ix]);
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

std::vector<float> avg_pool3(std::span<const float> x, std::uint32_t c,
                             std::uint32_t h, std::uint32_t w) {
    std::vector<float> out(x.size());
    const int ih = static_cast<int>(h), iw = static_cast<int>(w);
    parallel(c, [&](std::size_t k) {
        const float* src = &x[k * h * w];
        for (int y = 0; y < ih; ++y) {
            for (int xx = 0; xx < iw; ++xx) {
                float s = 0.0f;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int iy = y + dy;
                    if (iy < 0 || iy >= ih) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int ix = xx + dx;
                        if (ix < 0 || ix >= iw) continue;
                        s += src[iy * iw + ix];
                    }
                }
                out[(k * h + y) * w + xx] = s / 9.0f;
            }
        }
    });
    return out;
}

std::vector<float> global_avg(std::span<const float> x, std::uint32_t c,
                              std::size_t plane) {
    std::vector<float> out(c);
    for (std::uint32_t k = 0; k < c; ++k) {
        double s = 0.0;
        for (std::size_t i = 0; i < plane; ++i) s += x[k * plane + i];
        out[k] = static_cast<float>(s / static_cast<double>(plane));
    }
    return out;
}

std::vector<float> linear(std::span<const float> w,
                          std::span<const float> b,
                          std::span<const float> x) {
    std::vector<float> out(b.size());
    parallel(b.size(), [&](std::size_t i) {
        double s = b[i];
        const float* row = &w[i * x.size()];
        for (std::size_t c = 0; c < x.size(); ++c) {
            s += double{row[c]} * x[c];
        }
        out[i] = static_cast<float>(s);
    });
    return out;
}

Preprocess read_preprocess(const gguf::GgufFile& g,
                           const std::string& prefix) {
    auto u = [&](const std::string& k) -> std::uint32_t {
        const auto x = g.get_uint(prefix + k);
        if (!x || *x == 0) {
            throw gguf::Error("cnn: missing " + prefix + k);
        }
        return static_cast<std::uint32_t>(*x);
    };
    Preprocess p;
    p.size = u("input_size");
    p.resize = u("resize");
    if (p.resize < p.size) {
        throw gguf::Error("cnn: resize smaller than input_size");
    }
    const auto interp = g.get_string(prefix + "interpolation");
    if (!interp || (*interp != "bilinear" && *interp != "bicubic")) {
        throw gguf::Error("cnn: interpolation must be bilinear or bicubic");
    }
    p.bicubic = *interp == "bicubic";
    const std::pair<const char*, float*> norms[] = {{"mean", p.mean},
                                                    {"std", p.std}};
    for (const auto& [key, dst] : norms) {
        const auto* a = g.get_array(prefix + key);
        if (a == nullptr || a->size() != 3) {
            throw gguf::Error(std::string("cnn: missing ") + prefix + key);
        }
        for (int c = 0; c < 3; ++c) dst[c] = static_cast<float>((*a)[c].f);
    }
    return p;
}

std::vector<std::string> read_labels(const gguf::GgufFile& g,
                                     const std::string& prefix) {
    const auto n = g.get_uint(prefix + "num_classes");
    const auto* a = g.get_array(prefix + "labels");
    if (!n || a == nullptr || a->size() != *n) {
        throw gguf::Error("cnn: labels do not match num_classes");
    }
    std::vector<std::string> labels;
    for (const auto& l : *a) labels.push_back(l.s);
    return labels;
}

std::vector<float> preprocess(const Rgb8& img, const Preprocess& p) {
    if (img.w == 0 || img.h == 0) {
        throw std::invalid_argument("cnn: empty image");
    }
    // torchvision's Resize(int): the shorter side to p.resize, the other
    // scaled and truncated.
    std::uint32_t w = p.resize, h = p.resize;
    if (img.w < img.h) {
        h = static_cast<std::uint32_t>(std::uint64_t{p.resize} * img.h /
                                       img.w);
    } else if (img.h < img.w) {
        w = static_cast<std::uint32_t>(std::uint64_t{p.resize} * img.w /
                                       img.h);
    }
    const Rgb8 r = p.bicubic ? resize_bicubic(img, w, h)
                             : resize_bilinear(img, w, h);
    // CenterCrop: int(round((side - size) / 2.0)), Python's round.
    const auto off = [&](std::uint32_t side) {
        return static_cast<std::uint32_t>(
            std::nearbyint((static_cast<double>(side) - p.size) / 2.0));
    };
    const std::uint32_t top = off(h), left = off(w);
    const std::size_t plane = std::size_t{p.size} * p.size;
    std::vector<float> chw(3 * plane);
    for (std::uint32_t y = 0; y < p.size; ++y) {
        for (std::uint32_t x = 0; x < p.size; ++x) {
            const std::uint8_t* px =
                &r.rgb[((std::size_t{top} + y) * w + left + x) * 3];
            for (int c = 0; c < 3; ++c) {
                // ToTensor (float32 / 255), then Normalize.
                const float v = static_cast<float>(px[c]) / 255.0f;
                chw[c * plane + y * p.size + x] = (v - p.mean[c]) / p.std[c];
            }
        }
    }
    return chw;
}

std::vector<std::pair<std::uint32_t, float>> top_k(
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

}  // namespace locus::image::cnn
