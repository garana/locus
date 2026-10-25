#include "locus/image/png.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace locus::image {

namespace {

const std::array<std::uint32_t, 256>& crc_table() {
    static const std::array<std::uint32_t, 256> t = [] {
        std::array<std::uint32_t, 256> a{};
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }
            a[n] = c;
        }
        return a;
    }();
    return t;
}

std::uint32_t crc32(std::span<const std::byte> d) {
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::byte b : d) {
        c = crc_table()[(c ^ std::to_integer<std::uint32_t>(b)) & 0xFFu] ^
            (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

void put_be32(std::vector<std::byte>& o, std::uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) {
        o.push_back(static_cast<std::byte>((v >> s) & 0xFFu));
    }
}

/** Appends a chunk: length, type, data, CRC over type + data. */
void chunk(std::vector<std::byte>& o, const char* type,
           std::span<const std::byte> data) {
    put_be32(o, static_cast<std::uint32_t>(data.size()));
    const std::size_t start = o.size();
    for (int i = 0; i < 4; ++i) o.push_back(static_cast<std::byte>(type[i]));
    o.insert(o.end(), data.begin(), data.end());
    put_be32(o, crc32(std::span<const std::byte>(o).subspan(start)));
}

}  // namespace

std::vector<std::uint8_t> to_rgb8(std::span<const float> chw,
                                  std::uint32_t h, std::uint32_t w) {
    const std::size_t plane = std::size_t{h} * w;
    if (chw.size() != 3 * plane) {
        throw std::invalid_argument("to_rgb8: need 3 * h * w floats");
    }
    std::vector<std::uint8_t> rgb(3 * plane);
    for (std::size_t i = 0; i < plane; ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            const float v = std::clamp((chw[c * plane + i] + 1.0f) / 2.0f *
                                           255.0f,
                                       0.0f, 255.0f);
            rgb[i * 3 + c] = static_cast<std::uint8_t>(v);  // truncates
        }
    }
    return rgb;
}

std::vector<std::byte> encode_png(std::span<const std::uint8_t> rgb,
                                  std::uint32_t w, std::uint32_t h) {
    if (w == 0 || h == 0 || rgb.size() != std::size_t{w} * h * 3) {
        throw std::invalid_argument("encode_png: need w * h * 3 bytes");
    }
    // Scanlines, each with filter byte 0.
    const std::size_t row = std::size_t{w} * 3;
    std::vector<std::byte> raw;
    raw.reserve((row + 1) * h);
    for (std::uint32_t y = 0; y < h; ++y) {
        raw.push_back(std::byte{0});
        for (std::size_t x = 0; x < row; ++x) {
            raw.push_back(static_cast<std::byte>(rgb[y * row + x]));
        }
    }
    // zlib: header (deflate, 32K window, no dictionary, FCHECK), stored
    // blocks of at most 65535 bytes, Adler-32 of the raw data.
    std::vector<std::byte> z = {std::byte{0x78}, std::byte{0x01}};
    for (std::size_t at = 0; at < raw.size() || at == 0;) {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - at);
        const bool final = at + n == raw.size();
        z.push_back(static_cast<std::byte>(final ? 1 : 0));
        z.push_back(static_cast<std::byte>(n & 0xFFu));
        z.push_back(static_cast<std::byte>(n >> 8));
        z.push_back(static_cast<std::byte>(~n & 0xFFu));
        z.push_back(static_cast<std::byte>((~n >> 8) & 0xFFu));
        z.insert(z.end(), raw.begin() + at, raw.begin() + at + n);
        at += n;
        if (final) break;
    }
    std::uint32_t a = 1, b = 0;
    for (std::byte v : raw) {
        a = (a + std::to_integer<std::uint32_t>(v)) % 65521u;
        b = (b + a) % 65521u;
    }
    put_be32(z, (b << 16) | a);

    std::vector<std::byte> png = {
        std::byte{0x89}, std::byte{'P'},  std::byte{'N'},  std::byte{'G'},
        std::byte{'\r'}, std::byte{'\n'}, std::byte{0x1A}, std::byte{'\n'}};
    std::vector<std::byte> ihdr;
    put_be32(ihdr, w);
    put_be32(ihdr, h);
    for (std::uint8_t v : {8, 2, 0, 0, 0}) {  // 8-bit RGB, no interlace
        ihdr.push_back(static_cast<std::byte>(v));
    }
    chunk(png, "IHDR", ihdr);
    chunk(png, "IDAT", z);
    chunk(png, "IEND", {});
    return png;
}

}  // namespace locus::image
