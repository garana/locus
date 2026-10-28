#include "locus/image/png_decode.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "locus/image/inflate.hpp"
#include "locus/image/png.hpp"

namespace locus::image {

namespace {

constexpr std::uint8_t kSig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};

[[noreturn]] void bad(const std::string& what) {
    throw std::invalid_argument("png: " + what);
}

std::uint32_t be32(const std::uint8_t* p) {
    return (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) |
           (std::uint32_t{p[2]} << 8) | std::uint32_t{p[3]};
}

/** The header fields the decoder needs. */
struct Header {
    std::uint32_t w = 0, h = 0;
    int depth = 0;
    int color = 0;
    int channels = 0;  ///< samples per pixel
    bool interlace = false;
};

/** @returns The bytes of one filtered row of w pixels (no filter
 *     byte). */
std::uint64_t row_bytes(const Header& hd, std::uint64_t w) {
    return (w * std::uint64_t(hd.channels) * std::uint64_t(hd.depth) + 7) /
           8;
}

// Adam7 passes: first column and row, column and row steps.
constexpr int kX0[7] = {0, 4, 0, 2, 0, 1, 0};
constexpr int kY0[7] = {0, 0, 4, 0, 2, 0, 1};
constexpr int kDx[7] = {8, 8, 4, 4, 2, 2, 1};
constexpr int kDy[7] = {8, 8, 8, 4, 4, 2, 2};

/** One pass's (or the whole image's) size in pixels. */
struct Pass {
    std::uint32_t x0 = 0, y0 = 0, dx = 1, dy = 1, w = 0, h = 0;
};

std::vector<Pass> passes(const Header& hd) {
    if (!hd.interlace) return {Pass{0, 0, 1, 1, hd.w, hd.h}};
    std::vector<Pass> out;
    for (int p = 0; p < 7; ++p) {
        Pass s;
        s.x0 = kX0[p];
        s.y0 = kY0[p];
        s.dx = kDx[p];
        s.dy = kDy[p];
        s.w = hd.w > s.x0 ? (hd.w - s.x0 + s.dx - 1) / s.dx : 0;
        s.h = hd.h > s.y0 ? (hd.h - s.y0 + s.dy - 1) / s.dy : 0;
        out.push_back(s);
    }
    return out;
}

std::uint8_t paeth(int a, int b, int c) {
    const int p = a + b - c;
    const int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return static_cast<std::uint8_t>(a);
    if (pb <= pc) return static_cast<std::uint8_t>(b);
    return static_cast<std::uint8_t>(c);
}

/** Undoes the row filters of one pass in place; rows start with their
 *  filter byte. */
void unfilter(std::uint8_t* d, std::size_t rows, std::size_t rb,
              std::size_t bpp) {
    const std::vector<std::uint8_t> zero(rb, 0);
    const std::uint8_t* prev = zero.data();
    for (std::size_t y = 0; y < rows; ++y) {
        std::uint8_t* row = d + y * (rb + 1);
        const std::uint8_t f = row[0];
        std::uint8_t* r = row + 1;
        switch (f) {
            case 0: break;
            case 1:
                for (std::size_t i = bpp; i < rb; ++i) r[i] += r[i - bpp];
                break;
            case 2:
                for (std::size_t i = 0; i < rb; ++i) r[i] += prev[i];
                break;
            case 3:
                for (std::size_t i = 0; i < rb; ++i) {
                    const int a = i >= bpp ? r[i - bpp] : 0;
                    r[i] += static_cast<std::uint8_t>((a + prev[i]) / 2);
                }
                break;
            case 4:
                for (std::size_t i = 0; i < rb; ++i) {
                    const int a = i >= bpp ? r[i - bpp] : 0;
                    const int c = i >= bpp ? prev[i - bpp] : 0;
                    r[i] += paeth(a, prev[i], c);
                }
                break;
            default: bad("bad row filter " + std::to_string(f));
        }
        prev = r;
    }
}

}  // namespace

bool is_png(std::span<const std::byte> data) {
    return data.size() >= 8 && std::memcmp(data.data(), kSig, 8) == 0;
}

Rgb8 decode_png(std::span<const std::byte> data, const DecodeLimits& lim) {
    if (!is_png(data)) bad("no PNG signature");
    if (data.size() > lim.max_bytes) bad("over the size limit");
    const auto* p = reinterpret_cast<const std::uint8_t*>(data.data());
    const std::size_t n = data.size();

    Header hd;
    bool have_hdr = false, have_end = false;
    std::vector<std::array<std::uint8_t, 3>> palette;
    std::vector<std::uint8_t> idat;
    bool idat_done = false;  ///< a chunk followed the IDAT run
    std::size_t at = 8;
    while (!have_end) {
        if (n - at < 12) bad("truncated");
        const std::uint32_t len = be32(p + at);
        if (len > 0x7FFFFFFFu || len > n - at - 12) bad("truncated");
        const std::uint8_t* type = p + at + 4;
        const std::uint8_t* body = type + 4;
        const std::uint32_t crc = be32(body + len);
        if (crc != crc32(std::span<const std::byte>(
                       reinterpret_cast<const std::byte*>(type), len + 4))) {
            bad("chunk CRC mismatch");
        }
        const std::string t(reinterpret_cast<const char*>(type), 4);
        at += 12 + len;
        if (!have_hdr && t != "IHDR") bad("first chunk is not IHDR");
        if (t != "IDAT" && !idat.empty()) idat_done = true;
        if (t == "IHDR") {
            if (have_hdr) bad("second IHDR");
            if (len != 13) bad("bad IHDR length");
            hd.w = be32(body);
            hd.h = be32(body + 4);
            hd.depth = body[8];
            hd.color = body[9];
            if (hd.w == 0 || hd.h == 0 || hd.w > 0x7FFFFFFFu ||
                hd.h > 0x7FFFFFFFu) {
                bad("bad dimensions");
            }
            if (std::uint64_t(hd.w) * hd.h > lim.max_pixels) {
                bad("dimensions over the limit");
            }
            if (body[10] != 0 || body[11] != 0) bad("bad compression/filter");
            if (body[12] > 1) bad("bad interlace method");
            hd.interlace = body[12] == 1;
            const int d = hd.depth;
            switch (hd.color) {
                case 0:
                    hd.channels = 1;
                    if (d != 1 && d != 2 && d != 4 && d != 8 && d != 16) {
                        bad("bad gray depth");
                    }
                    break;
                case 2:
                    hd.channels = 3;
                    if (d != 8 && d != 16) bad("bad RGB depth");
                    break;
                case 3:
                    hd.channels = 1;
                    if (d != 1 && d != 2 && d != 4 && d != 8) {
                        bad("bad palette depth");
                    }
                    break;
                case 4:
                    hd.channels = 2;
                    if (d != 8 && d != 16) bad("bad gray-alpha depth");
                    break;
                case 6:
                    hd.channels = 4;
                    if (d != 8 && d != 16) bad("bad RGBA depth");
                    break;
                default: bad("bad color type");
            }
            have_hdr = true;
        } else if (t == "PLTE") {
            if (!palette.empty()) bad("second PLTE");
            if (!idat.empty()) bad("PLTE after IDAT");
            if (len == 0 || len % 3 != 0 || len > 256 * 3) {
                bad("bad PLTE length");
            }
            for (std::uint32_t i = 0; i < len; i += 3) {
                palette.push_back({body[i], body[i + 1], body[i + 2]});
            }
        } else if (t == "IDAT") {
            if (idat_done) bad("IDAT chunks not consecutive");
            if (hd.color == 3 && palette.empty()) bad("palette image without PLTE");
            idat.insert(idat.end(), body, body + len);
        } else if (t == "IEND") {
            have_end = true;
        } else if ((type[0] & 0x20u) == 0) {
            bad("unknown critical chunk " + t);
        }
        // Other ancillary chunks (tRNS, gAMA, text, ...) do not change
        // the RGB result and are skipped.
    }
    if (idat.empty()) bad("no image data");

    // The exact inflated size: each pass's rows, each with a filter byte.
    const std::vector<Pass> ps = passes(hd);
    std::uint64_t raw = 0;
    for (const Pass& s : ps) {
        if (s.w == 0 || s.h == 0) continue;
        raw += std::uint64_t(s.h) * (row_bytes(hd, s.w) + 1);
    }
    if (raw > (std::uint64_t{1} << 31)) bad("image too large");
    std::vector<std::uint8_t> buf =  // unfiltered in place below
        zlib_inflate(idat, static_cast<std::size_t>(raw));
    if (buf.size() != raw) bad("image data too short");
    Rgb8 out{hd.w, hd.h,
             std::vector<std::uint8_t>(std::size_t{hd.w} * hd.h * 3)};
    const std::size_t bpp =
        std::max<std::size_t>(1, std::size_t(hd.channels) * hd.depth / 8);
    const std::uint32_t mask = (1u << std::min(hd.depth, 8)) - 1u;
    const std::uint32_t gray_scale = hd.depth < 8 ? 255u / mask : 1u;
    std::size_t off = 0;
    for (const Pass& s : ps) {
        if (s.w == 0 || s.h == 0) continue;
        const auto rb = static_cast<std::size_t>(row_bytes(hd, s.w));
        std::uint8_t* d = buf.data() + off;
        unfilter(d, s.h, rb, bpp);
        for (std::uint32_t y = 0; y < s.h; ++y) {
            const std::uint8_t* r = d + std::size_t{y} * (rb + 1) + 1;
            for (std::uint32_t x = 0; x < s.w; ++x) {
                // Sample c of pixel x, reduced to 8 bits as stb_image.
                auto sample = [&](int c) -> std::uint32_t {
                    if (hd.depth == 8) return r[std::size_t{x} * hd.channels + c];
                    if (hd.depth == 16) {
                        return r[(std::size_t{x} * hd.channels + c) * 2];
                    }
                    const std::size_t bit = std::size_t{x} * hd.depth;
                    return (r[bit / 8] >> (8 - hd.depth - bit % 8)) & mask;
                };
                std::uint8_t* o =
                    out.rgb.data() +
                    (std::size_t{s.y0 + y * s.dy} * hd.w + s.x0 + x * s.dx) * 3;
                switch (hd.color) {
                    case 0:
                    case 4: {
                        const auto g =
                            static_cast<std::uint8_t>(sample(0) * gray_scale);
                        o[0] = o[1] = o[2] = g;
                        break;
                    }
                    case 3: {
                        const std::uint32_t i = sample(0);
                        if (i >= palette.size()) bad("palette index out of range");
                        o[0] = palette[i][0];
                        o[1] = palette[i][1];
                        o[2] = palette[i][2];
                        break;
                    }
                    default:  // 2 and 6: RGB, alpha dropped
                        o[0] = static_cast<std::uint8_t>(sample(0));
                        o[1] = static_cast<std::uint8_t>(sample(1));
                        o[2] = static_cast<std::uint8_t>(sample(2));
                }
            }
        }
        off += std::size_t{s.h} * (rb + 1);
    }
    return out;
}

}  // namespace locus::image
