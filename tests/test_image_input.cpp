#include "locus/image/image_input.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/image/png.hpp"
#include "png_parse.hpp"

using locus::image::decode_image;
using locus::image::fit_square;
using locus::image::resize_bicubic;
using locus::image::Rgb8;

namespace {

// Keep in sync with scripts/make-image-goldens.py.
constexpr std::uint32_t kW = 61, kH = 37;

const std::string kDir =
    std::string(LOCUS_SOURCE_DIR) + "/tests/golden/image_input/";

std::vector<std::byte> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> c((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<std::byte> b(c.size());
    std::memcpy(b.data(), c.data(), c.size());
    return b;
}

std::vector<std::uint8_t> read_rgb(const std::string& name,
                                   std::size_t size) {
    const auto b = read_bytes(kDir + name);
    REQUIRE(b.size() == size);
    std::vector<std::uint8_t> v(size);
    std::memcpy(v.data(), b.data(), size);
    return v;
}

/**
 * REQUIREs a == b byte for byte, reporting the size or the first
 * differing byte. (A plain REQUIRE(a == b) on 65K-element vectors
 * makes Catch2 stringify both, which its console reporter mishandles.)
 */
void require_same(const std::vector<std::uint8_t>& a,
                  const std::vector<std::uint8_t>& b) {
    REQUIRE(a.size() == b.size());
    const auto m = std::mismatch(a.begin(), a.end(), b.begin());
    const std::size_t at = static_cast<std::size_t>(m.first - a.begin());
    INFO("first difference at byte " << at << " of " << a.size());
    if (m.first != a.end()) {
        INFO("got " << int{*m.first} << ", want " << int{*m.second});
        REQUIRE(m.first == a.end());
    }
}

/** The pattern as Pillow decodes src.png: the reference's RGB. */
Rgb8 source() { return decode_image(read_bytes(kDir + "src.png")); }

/**
 * A w x h 8-bit RGB PNG whose IDAT holds raw_len bytes of zeros
 * (filter 0 rows, black pixels) in stored deflate blocks, with valid
 * CRCs and Adler-32. raw_len may exceed the (w * 3 + 1) * h the header
 * implies.
 */
std::vector<std::byte> png_with_raw(std::uint32_t w, std::uint32_t h,
                                    std::size_t raw_len) {
    std::vector<std::byte> out;
    const auto be32 = [](std::vector<std::byte>& o, std::uint32_t v) {
        for (int s = 24; s >= 0; s -= 8) o.push_back(std::byte(v >> s));
    };
    const auto chunk = [&](const char* type,
                           const std::vector<std::byte>& d) {
        be32(out, static_cast<std::uint32_t>(d.size()));
        const std::size_t at = out.size();
        for (int i = 0; i < 4; ++i) out.push_back(std::byte(type[i]));
        out.insert(out.end(), d.begin(), d.end());
        be32(out, png_test::crc32_ref(&out[at], 4 + d.size()));
    };
    static const unsigned char sig[8] = {0x89, 'P',  'N',  'G',
                                         '\r', '\n', 0x1A, '\n'};
    for (unsigned char c : sig) out.push_back(std::byte(c));
    std::vector<std::byte> ihdr;
    be32(ihdr, w);
    be32(ihdr, h);
    for (int v : {8, 2, 0, 0, 0}) ihdr.push_back(std::byte(v));
    chunk("IHDR", ihdr);
    std::vector<std::byte> z = {std::byte{0x78}, std::byte{0x01}};
    for (std::size_t at = 0; at < raw_len;) {
        const std::size_t n = std::min<std::size_t>(65535, raw_len - at);
        z.push_back(std::byte(at + n == raw_len ? 1 : 0));
        z.push_back(std::byte(n & 0xFF));
        z.push_back(std::byte(n >> 8));
        z.push_back(std::byte(~n & 0xFF));
        z.push_back(std::byte((~n >> 8) & 0xFF));
        z.insert(z.end(), n, std::byte{0});
        at += n;
    }
    be32(z, 1);  // Adler-32 of all-zero data: a = 1, b = 0
    chunk("IDAT", z);
    chunk("IEND", {});
    return out;
}

}  // namespace

TEST_CASE("decode_image reads PNGs as Pillow does", "[image][decode]") {
    const Rgb8 src = source();
    REQUIRE(src.w == kW);
    REQUIRE(src.h == kH);
    // The RGB pattern: what Pillow wrote is what Pillow reads back, and
    // the resize goldens start from these exact pixels.
    const auto rgba = decode_image(read_bytes(kDir + "src_rgba.png"));
    REQUIRE(rgba.w == kW);
    require_same(rgba.rgb, src.rgb);  // alpha dropped, as convert("RGB")
    // Gray: L -> RGB replicates the one channel.
    const auto gray = decode_image(read_bytes(kDir + "src_gray.png"));
    REQUIRE(gray.w == kW);
    for (std::size_t i = 0; i < std::size_t{kW} * kH; ++i) {
        REQUIRE(gray.rgb[i * 3] == gray.rgb[i * 3 + 1]);
        REQUIRE(gray.rgb[i * 3] == gray.rgb[i * 3 + 2]);
    }
    // locus's own stored-deflate PNG round-trips too.
    const auto ours = locus::image::encode_png(src.rgb, src.h, src.w);
    require_same(decode_image(ours).rgb, src.rgb);
}

// JPEG decoders may round the IDCT and the chroma upsampling
// differently (Pillow uses libjpeg-turbo), so this is a tolerance. On
// the M2: mean 0.084, worst 2; the limits leave room for stb's other
// SIMD paths.
TEST_CASE("decode_image reads a JPEG close to Pillow", "[image][decode]") {
    const auto jpg = decode_image(read_bytes(kDir + "src.jpg"));
    REQUIRE(jpg.w == kW);
    REQUIRE(jpg.h == kH);
    const auto ref = read_rgb("src_jpg.rgb", std::size_t{kW} * kH * 3);
    double sum = 0.0;
    int worst = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const int d = std::abs(int{jpg.rgb[i]} - int{ref[i]});
        sum += d;
        worst = std::max(worst, d);
    }
    WARN("JPEG vs Pillow: mean |diff| " << sum / ref.size() << ", worst "
                                        << worst);
    REQUIRE(sum / ref.size() <= 0.5);
    REQUIRE(worst <= 8);
}

TEST_CASE("resize_bicubic is bit-exact with Pillow", "[image][decode]") {
    const Rgb8 src = source();
    for (const auto& [w, h] : {std::pair<std::uint32_t, std::uint32_t>{23, 17},
                               {97, 59},
                               {61, 20}}) {
        INFO(w << " x " << h);
        const auto r = resize_bicubic(src, w, h);
        REQUIRE(r.w == w);
        REQUIRE(r.h == h);
        require_same(r.rgb, read_rgb("resize_" + std::to_string(w) + "x" +
                                         std::to_string(h) + ".rgb",
                                     std::size_t{w} * h * 3));
    }
    // The same size is a copy.
    require_same(resize_bicubic(src, kW, kH).rgb, src.rgb);
}

TEST_CASE("fit_square matches the Janus reference preprocessing",
          "[image][decode]") {
    const Rgb8 src = source();
    const auto wide = fit_square(src, 32, 4, {127, 127, 127});
    REQUIRE(wide.w == 32);
    REQUIRE(wide.h == 32);
    require_same(wide.rgb, read_rgb("fit32_wide.rgb", 32 * 32 * 3));
    // The transpose (37 wide, 61 tall): the pad goes left and right.
    Rgb8 tall{kH, kW, std::vector<std::uint8_t>(src.rgb.size())};
    for (std::uint32_t y = 0; y < kH; ++y) {
        for (std::uint32_t x = 0; x < kW; ++x) {
            for (int c = 0; c < 3; ++c) {
                tall.rgb[(std::size_t{x} * kH + y) * 3 + c] =
                    src.rgb[(std::size_t{y} * kW + x) * 3 + c];
            }
        }
    }
    require_same(fit_square(tall, 32, 4, {127, 127, 127}).rgb,
                 read_rgb("fit32_tall.rgb", 32 * 32 * 3));
}

// Request bytes are untrusted: every bad input is an exception, never
// a crash, and the caps hold before any pixel buffer is allocated.
TEST_CASE("decode_image refuses bad and oversized input",
          "[image][decode]") {
    const auto png = read_bytes(kDir + "src.png");
    const auto jpg = read_bytes(kDir + "src.jpg");
    REQUIRE_THROWS_AS(decode_image({}), std::invalid_argument);
    // Truncated at every length a prefix could stop at, both formats.
    for (const auto* f : {&png, &jpg}) {
        for (std::size_t n = 1; n < f->size(); n += 7) {
            INFO("prefix " << n << " of " << f->size());
            try {
                (void)decode_image(std::span(f->data(), n));
            } catch (const std::invalid_argument&) {
            }
        }
    }
    // Corrupted: 2000 copies with 1 to 8 random bytes overwritten each
    // (headers, Huffman tables, IDAT), which must decode or throw.
    std::uint32_t state = 7;
    const auto next = [&] {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    };
    for (int round = 0; round < 2000; ++round) {
        auto f = (round % 2 == 0) ? png : jpg;
        const int flips = 1 + static_cast<int>(next() % 8);
        for (int i = 0; i < flips; ++i) {
            f[next() % f.size()] = std::byte(next() & 0xFF);
        }
        try {
            (void)decode_image(f);
        } catch (const std::invalid_argument&) {
        }
    }
    // Not PNG or JPEG: a GIF header, and text.
    const std::string gif = "GIF89a\x01\x00\x01\x00\x80\x00\x00";
    REQUIRE_THROWS_AS(decode_image(std::as_bytes(std::span(gif))),
                      std::invalid_argument);
    const std::string text = "not an image at all";
    REQUIRE_THROWS_AS(decode_image(std::as_bytes(std::span(text))),
                      std::invalid_argument);
    // A valid header claiming 20000 x 20000: refused on the header
    // (over STBI_MAX_DIMENSIONS and over max_pixels).
    std::vector<std::byte> huge(png.begin(), png.begin() + 33);
    for (int i = 0; i < 4; ++i) {
        huge[16 + i] = std::byte((20000u >> (24 - 8 * i)) & 0xFF);
        huge[20 + i] = std::byte((20000u >> (24 - 8 * i)) & 0xFF);
    }
    REQUIRE_THROWS_AS(decode_image(huge), std::invalid_argument);
    // A 16 x 16 PNG whose data inflates to 4 MB instead of the 784
    // bytes its header implies. stb accepts excess data, and its
    // inflate buffer grows by doubling, so without the allocation cap a
    // shaped input could make it allocate gigabytes; the cap (twice the
    // header's need plus slack) refuses it. The exact size decodes.
    const std::uint32_t pw = 16, ph = 16;
    const std::size_t need = (std::size_t{pw} * 3 + 1) * ph;
    REQUIRE(decode_image(png_with_raw(pw, ph, need)).w == pw);
    REQUIRE_THROWS_AS(decode_image(png_with_raw(pw, ph, std::size_t{4} << 20)),
                      std::invalid_argument);
    // The caps themselves.
    locus::image::DecodeLimits small;
    small.max_pixels = 100;
    REQUIRE_THROWS_AS(decode_image(png, small), std::invalid_argument);
    small = {};
    small.max_bytes = 100;
    REQUIRE_THROWS_AS(decode_image(png, small), std::invalid_argument);
}
