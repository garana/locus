#include "locus/http/base64.hpp"

#include <cstdint>
#include <stdexcept>

namespace locus::http {

std::string base64_encode(std::span<const std::byte> data) {
    static constexpr char kAlpha[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    std::size_t i = 0;
    for (; i + 3 <= data.size(); i += 3) {
        const std::uint32_t v =
            (std::to_integer<std::uint32_t>(data[i]) << 16) |
            (std::to_integer<std::uint32_t>(data[i + 1]) << 8) |
            std::to_integer<std::uint32_t>(data[i + 2]);
        out += kAlpha[(v >> 18) & 63];
        out += kAlpha[(v >> 12) & 63];
        out += kAlpha[(v >> 6) & 63];
        out += kAlpha[v & 63];
    }
    // The 1 or 2 trailing bytes, zero-padded to 3, then '=' for each
    // missing byte.
    const std::size_t rest = data.size() - i;
    if (rest > 0) {
        std::uint32_t v = std::to_integer<std::uint32_t>(data[i]) << 16;
        if (rest == 2) v |= std::to_integer<std::uint32_t>(data[i + 1]) << 8;
        out += kAlpha[(v >> 18) & 63];
        out += kAlpha[(v >> 12) & 63];
        out += rest == 2 ? kAlpha[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

std::vector<std::byte> base64_decode(std::string_view text,
                                     std::size_t max_bytes) {
    const auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    if (text.size() % 4 != 0) {
        throw std::invalid_argument("base64: length is not a multiple of 4");
    }
    const std::size_t pad = text.ends_with("==") ? 2
                            : text.ends_with('=') ? 1
                                                  : 0;
    const std::size_t n = text.size() / 4 * 3 - pad;
    if (n > max_bytes) {
        throw std::invalid_argument("base64: decoded data over the limit");
    }
    std::vector<std::byte> out;
    out.reserve(n);
    for (std::size_t i = 0; i < text.size(); i += 4) {
        const bool last = i + 4 == text.size();
        const std::size_t keep = last ? 4 - pad : 4;  // data chars here
        std::uint32_t v = 0;
        for (std::size_t k = 0; k < 4; ++k) {
            int d = 0;
            if (k < keep) {
                d = val(text[i + k]);
                if (d < 0) {
                    throw std::invalid_argument("base64: bad character");
                }
            } else if (text[i + k] != '=') {
                throw std::invalid_argument("base64: bad padding");
            }
            v = (v << 6) | static_cast<std::uint32_t>(d);
        }
        // The bits the padding leaves over must be zero (what the
        // encoder writes), so each byte string has one encoding.
        if ((pad == 1 && last && (v & 0xFF) != 0) ||
            (pad == 2 && last && (v & 0xFFFF) != 0)) {
            throw std::invalid_argument("base64: nonzero trailing bits");
        }
        out.push_back(static_cast<std::byte>(v >> 16));
        if (keep > 2) out.push_back(static_cast<std::byte>((v >> 8) & 0xFF));
        if (keep > 3) out.push_back(static_cast<std::byte>(v & 0xFF));
    }
    return out;
}

}  // namespace locus::http
