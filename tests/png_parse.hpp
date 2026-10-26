#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"

namespace png_test {

inline std::uint32_t be32(const std::vector<std::byte>& b, std::size_t at) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v = (v << 8) | std::to_integer<std::uint32_t>(b[at + i]);
    }
    return v;
}

inline std::uint32_t crc32_ref(const std::byte* p, std::size_t n) {
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
inline Png parse_stored_png(const std::vector<std::byte>& b) {
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

}  // namespace png_test
