#include "locus/image/image_input.hpp"

#include "locus/image/jpeg_decode.hpp"
#include "locus/image/png_decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace locus::image {

namespace {

/** Pillow's bicubic kernel (a = -0.5), support 2. */
double bicubic(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

/** Pillow's lanczos kernel: sinc(x) sinc(x / 3), support 3. */
double lanczos(double x) {
    const auto sinc = [](double v) {
        if (v == 0.0) return 1.0;
        v *= 3.14159265358979323846;
        return std::sin(v) / v;
    };
    if (-3.0 <= x && x < 3.0) return sinc(x) * sinc(x / 3.0);
    return 0.0;
}

/** A resampling kernel and its support at scale 1. */
struct Filter {
    double (*fn)(double);
    double support;
};
constexpr Filter kBicubic{&bicubic, 2.0};
constexpr Filter kLanczos{&lanczos, 3.0};

/** Pillow's 8-bit fixed point: 32 - 8 - 2 fraction bits. */
constexpr int kPrecisionBits = 32 - 8 - 2;

/** One output pixel's taps: first input index, count, int weights. */
struct Taps {
    int ksize = 0;
    std::vector<int> start, count;
    std::vector<std::int32_t> k;  // count rows of ksize
};

/**
 * Pillow's precompute_coeffs + normalize_coeffs_8bpc for in_size ->
 * out_size over the box [0, in_size).
 */
Taps taps(int in_size, int out_size, const Filter& f) {
    Taps t;
    // (double)(in1 - in0) / outSize, with in0 / in1 as floats.
    const double scale =
        static_cast<double>(static_cast<float>(in_size) - 0.0f) / out_size;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = f.support * filterscale;
    t.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    t.start.resize(out_size);
    t.count.resize(out_size);
    t.k.assign(static_cast<std::size_t>(out_size) * t.ksize, 0);
    std::vector<double> w(t.ksize);
    const double inv = 1.0 / filterscale;
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            w[x] = f.fn((x + xmin - center + 0.5) * inv);
            ww += w[x];
        }
        for (int x = 0; x < xmax; ++x) {
            if (ww != 0.0) w[x] /= ww;
            const double v = w[x] * (1 << kPrecisionBits);
            t.k[static_cast<std::size_t>(xx) * t.ksize + x] =
                static_cast<std::int32_t>(w[x] < 0 ? -0.5 + v : 0.5 + v);
        }
        t.start[xx] = xmin;
        t.count[xx] = xmax;
    }
    return t;
}

/** Pillow's clip8: the accumulator's integer part, clamped. */
std::uint8_t clip8(std::int32_t ss) {
    return static_cast<std::uint8_t>(
        std::clamp(ss >> kPrecisionBits, 0, 255));
}

/**
 * Horizontal pass over input rows [y0, y0 + rows): each output pixel is
 * the taps' weighted sum of the row, per channel.
 */
Rgb8 horizontal(const Rgb8& in, std::uint32_t out_w, int y0, int rows,
                const Filter& f) {
    const Taps t = taps(static_cast<int>(in.w), static_cast<int>(out_w), f);
    Rgb8 o{out_w, static_cast<std::uint32_t>(rows), {}};
    o.rgb.resize(std::size_t{out_w} * rows * 3);
    for (int yy = 0; yy < rows; ++yy) {
        const std::uint8_t* line =
            &in.rgb[static_cast<std::size_t>(yy + y0) * in.w * 3];
        for (std::uint32_t xx = 0; xx < out_w; ++xx) {
            const std::int32_t* k = &t.k[std::size_t{xx} * t.ksize];
            for (int c = 0; c < 3; ++c) {
                std::int32_t ss = 1 << (kPrecisionBits - 1);
                for (int x = 0; x < t.count[xx]; ++x) {
                    ss += line[(x + t.start[xx]) * 3 + c] * k[x];
                }
                o.rgb[(static_cast<std::size_t>(yy) * out_w + xx) * 3 + c] =
                    clip8(ss);
            }
        }
    }
    return o;
}

/** Vertical pass with precomputed taps whose starts are relative to
 * row 0 of `in`. */
Rgb8 vertical(const Rgb8& in, const Taps& t, std::uint32_t out_h) {
    Rgb8 o{in.w, out_h, {}};
    o.rgb.resize(std::size_t{in.w} * out_h * 3);
    for (std::uint32_t yy = 0; yy < out_h; ++yy) {
        const std::int32_t* k = &t.k[std::size_t{yy} * t.ksize];
        for (std::uint32_t xx = 0; xx < in.w; ++xx) {
            for (int c = 0; c < 3; ++c) {
                std::int32_t ss = 1 << (kPrecisionBits - 1);
                for (int y = 0; y < t.count[yy]; ++y) {
                    ss += in.rgb[(static_cast<std::size_t>(y + t.start[yy]) *
                                      in.w +
                                  xx) *
                                     3 +
                                 c] *
                          k[y];
                }
                o.rgb[(std::size_t{yy} * in.w + xx) * 3 + c] = clip8(ss);
            }
        }
    }
    return o;
}

/** Pillow's ImagingResampleInner with filter f. */
Rgb8 resample(const Rgb8& in, std::uint32_t w, std::uint32_t h,
              const Filter& f) {
    if (w == 0 || h == 0 || in.w == 0 || in.h == 0) {
        throw std::invalid_argument("resize: zero size");
    }
    if (in.rgb.size() != std::size_t{in.w} * in.h * 3) {
        throw std::invalid_argument("resize: need w * h * 3 bytes");
    }
    if (w == in.w && h == in.h) return in;  // Image.resize returns a copy
    // The vertical taps first, to know which input rows the horizontal
    // pass must produce.
    Taps vt = taps(static_cast<int>(in.h), static_cast<int>(h), f);
    const int y_first = vt.start[0];
    const int y_last = vt.start[h - 1] + vt.count[h - 1];
    Rgb8 cur = in;
    if (w != in.w) {
        for (auto& s : vt.start) s -= y_first;
        cur = horizontal(in, w, y_first, y_last - y_first, f);
    }
    if (h != in.h) {
        cur = vertical(cur, vt, h);
    }
    return cur;
}

}  // namespace

Rgb8 decode_image(std::span<const std::byte> data, const DecodeLimits& lim) {
    if (data.empty()) {
        throw std::invalid_argument("image: empty data");
    }
    if (data.size() > lim.max_bytes ||
        data.size() > static_cast<std::size_t>(INT32_MAX)) {
        throw std::invalid_argument("image: over the size limit");
    }
    // locus's own decoders (i#250).
    if (is_png(data)) return decode_png(data, lim);
    if (is_jpeg(data)) return decode_jpeg(data, lim);
    throw std::invalid_argument("image: not a PNG or JPEG");
}

Rgb8 resize_bicubic(const Rgb8& in, std::uint32_t w, std::uint32_t h) {
    return resample(in, w, h, kBicubic);
}

Rgb8 resize_lanczos(const Rgb8& in, std::uint32_t w, std::uint32_t h) {
    return resample(in, w, h, kLanczos);
}

Rgb8 fit_square(const Rgb8& in, std::uint32_t side, std::uint32_t min_size,
                std::array<std::uint8_t, 3> background) {
    if (in.w == 0 || in.h == 0) {
        throw std::invalid_argument("fit_square: zero size");
    }
    // size = [max(int(h / max_size * side), min), max(int(w / ...), min)]
    const double max_side = std::max(in.w, in.h);
    const auto fit = [&](std::uint32_t s) {
        return std::max(static_cast<std::uint32_t>(s / max_side * side),
                        min_size);
    };
    const Rgb8 r = resize_bicubic(in, fit(in.w), fit(in.h));
    if (r.w == r.h) return r;
    // expand2square: a max(w, h) square of background, the image pasted
    // at ((S - w) / 2, (S - h) / 2).
    const std::uint32_t s = std::max(r.w, r.h);
    Rgb8 o{s, s, std::vector<std::uint8_t>(std::size_t{s} * s * 3)};
    for (std::size_t i = 0; i < std::size_t{s} * s; ++i) {
        for (int c = 0; c < 3; ++c) o.rgb[i * 3 + c] = background[c];
    }
    const std::uint32_t x0 = (s - r.w) / 2, y0 = (s - r.h) / 2;
    for (std::uint32_t y = 0; y < r.h; ++y) {
        std::copy_n(&r.rgb[std::size_t{y} * r.w * 3], std::size_t{r.w} * 3,
                    &o.rgb[(std::size_t{y + y0} * s + x0) * 3]);
    }
    return o;
}

}  // namespace locus::image
