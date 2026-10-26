#include "locus/http/base64.hpp"

#include <cstdint>

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

}  // namespace locus::http
