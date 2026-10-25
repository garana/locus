#include "locus/image/png.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"

namespace {

std::uint32_t be32(const std::vector<std::byte>& b, std::size_t at) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v = (v << 8) | std::to_integer<std::uint32_t>(b[at + i]);
    }
    return v;
}

std::uint32_t crc32_ref(const std::byte* p, std::size_t n) {
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i) {
        c ^= std::to_integer<std::uint32_t>(p[i]);
        for (int k = 0; k < 8; ++k) {
            c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        }
    }
    return c ^ 0xFFFFFFFFu;
}

/** A decoded 8-bit RGB PNG whose IDAT is stored (uncompressed) deflate. */
struct Png {
    std::uint32_t w = 0, h = 0;
    std::vector<std::uint8_t> rgb;
};

/**
 * Parses a PNG written with stored deflate blocks (locus's encoder, or
 * zlib level 0): checks the signature, every chunk CRC and the
 * Adler-32, and undoes filter 0. REQUIREs on anything else.
 */
Png parse_stored_png(const std::vector<std::byte>& b) {
    static const unsigned char sig[8] = {0x89, 'P', 'N', 'G',
                                         '\r', '\n', 0x1A, '\n'};
    REQUIRE(b.size() > 8);
    REQUIRE(std::memcmp(b.data(), sig, 8) == 0);
    Png png;
    std::vector<std::byte> z;
    for (std::size_t at = 8; at < b.size();) {
        const std::uint32_t len = be32(b, at);
        const std::string type(reinterpret_cast<const char*>(&b[at + 4]), 4);
        REQUIRE(be32(b, at + 8 + len) == crc32_ref(&b[at + 4], 4 + len));
        if (type == "IHDR") {
            png.w = be32(b, at + 8);
            png.h = be32(b, at + 12);
            REQUIRE(std::to_integer<int>(b[at + 16]) == 8);  // bit depth
            REQUIRE(std::to_integer<int>(b[at + 17]) == 2);  // RGB
        } else if (type == "IDAT") {
            z.insert(z.end(), b.begin() + at + 8, b.begin() + at + 8 + len);
        }
        at += 12 + len;
    }
    REQUIRE(z.size() > 6);
    std::vector<std::byte> raw;
    std::size_t p = 2;  // zlib header
    for (bool final = false; !final;) {
        const int hdr = std::to_integer<int>(z[p]);
        final = (hdr & 1) != 0;
        REQUIRE(((hdr >> 1) & 3) == 0);  // stored block
        const std::size_t n = std::to_integer<std::size_t>(z[p + 1]) |
                              (std::to_integer<std::size_t>(z[p + 2]) << 8);
        const std::size_t nn = std::to_integer<std::size_t>(z[p + 3]) |
                               (std::to_integer<std::size_t>(z[p + 4]) << 8);
        REQUIRE((n ^ nn) == 0xFFFF);
        raw.insert(raw.end(), z.begin() + p + 5, z.begin() + p + 5 + n);
        p += 5 + n;
    }
    std::uint32_t a = 1, s = 0;
    for (std::byte v : raw) {
        a = (a + std::to_integer<std::uint32_t>(v)) % 65521u;
        s = (s + a) % 65521u;
    }
    REQUIRE(be32(z, p) == ((s << 16) | a));
    const std::size_t row = std::size_t{png.w} * 3;
    REQUIRE(raw.size() == (row + 1) * png.h);
    for (std::uint32_t y = 0; y < png.h; ++y) {
        REQUIRE(std::to_integer<int>(raw[y * (row + 1)]) == 0);  // filter
        for (std::size_t x = 0; x < row; ++x) {
            png.rgb.push_back(
                std::to_integer<std::uint8_t>(raw[y * (row + 1) + 1 + x]));
        }
    }
    return png;
}

std::vector<std::byte> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> c((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<std::byte> b(c.size());
    std::memcpy(b.data(), c.data(), c.size());
    return b;
}

}  // namespace

TEST_CASE("png encodes RGB with valid CRCs and Adler-32", "[image][png]") {
    std::mt19937 rng(3);
    // 37 x 5 fits one stored block; 220 x 200 needs three (132,200 raw
    // bytes, blocks of at most 65535).
    for (const auto& [w, h] : {std::pair<std::uint32_t, std::uint32_t>{37, 5},
                               {220, 200}}) {
        INFO(w << " x " << h);
        std::vector<std::uint8_t> rgb(std::size_t{w} * h * 3);
        for (auto& v : rgb) v = static_cast<std::uint8_t>(rng());
        const auto png = locus::image::encode_png(rgb, w, h);
        const auto back = parse_stored_png(png);
        REQUIRE(back.w == w);
        REQUIRE(back.h == h);
        REQUIRE(back.rgb == rgb);
    }
}

TEST_CASE("to_rgb8 clips and truncates like the reference",
          "[image][png]") {
    // (x + 1) / 2 * 255: -1 -> 0, 1 -> 255, 0 -> 127.5 -> 127
    // (truncated, not rounded), out of range clipped.
    const std::vector<float> chw = {-1.0f, 0.0f, 1.0f, 2.0f,  // R
                                    -3.0f, 0.5f, 0.999f, -0.5f,  // G
                                    0.1f, -0.1f, 0.25f, 0.75f};  // B
    const auto rgb = locus::image::to_rgb8(chw, 2, 2);
    REQUIRE(rgb == std::vector<std::uint8_t>{0,   0,   140, 127, 191,
                                             114, 255, 254, 159, 255, 63,
                                             223});
}

// The reference's own conversion (scripts/make-janus-goldens.py decode:
// numpy clip then uint8 assignment) of its f32 decode, against locus's
// to_rgb8 of the same floats: every pixel identical.
TEST_CASE("to_rgb8 reproduces the reference PNG's pixels",
          "[image][png]") {
    const std::string g = std::string(LOCUS_SOURCE_DIR) + "/tests/golden/";
    if (!std::filesystem::exists(g + "janus_decode.f32") ||
        !std::filesystem::exists(g + "janus_decode.png")) {
        SKIP("janus_decode goldens absent");
    }
    const auto f = read_file(g + "janus_decode.f32");
    std::vector<float> chw(f.size() / 4);
    std::memcpy(chw.data(), f.data(), chw.size() * 4);
    const auto ref = parse_stored_png(read_file(g + "janus_decode.png"));
    REQUIRE(ref.w == 384);
    REQUIRE(ref.h == 384);
    REQUIRE(locus::image::to_rgb8(chw, 384, 384) == ref.rgb);
}
