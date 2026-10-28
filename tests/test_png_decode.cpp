#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/image/image_input.hpp"
#include "locus/image/inflate.hpp"
#include "locus/image/png.hpp"
#include "locus/image/png_decode.hpp"

using locus::image::decode_png;
using locus::image::zlib_inflate;

namespace {

const std::string kCorpus =
    std::string(LOCUS_SOURCE_DIR) + "/tests/golden/png/";

std::vector<std::byte> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    const std::string s((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    const auto* p = reinterpret_cast<const std::byte*>(s.data());
    return {p, p + s.size()};
}

std::vector<std::string> corpus() {
    std::vector<std::string> out;
    for (const auto& e : std::filesystem::directory_iterator(kCorpus)) {
        if (e.path().extension() == ".png") out.push_back(e.path().stem());
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::uint32_t be32(const std::vector<std::byte>& b, std::size_t at) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v = (v << 8) | std::to_integer<std::uint32_t>(b[at + i]);
    }
    return v;
}

void put_be32(std::vector<std::byte>& b, std::size_t at, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        b[at + i] = std::byte((v >> (24 - 8 * i)) & 0xFF);
    }
}

/** Rewrites every chunk CRC, so a mutation reaches the decoder past the
 *  CRC check. */
void fix_crcs(std::vector<std::byte>& f) {
    for (std::size_t at = 8; at + 12 <= f.size();) {
        const std::uint32_t len = be32(f, at);
        if (len > f.size() - at - 12) return;
        put_be32(f, at + 8 + len,
                 locus::image::crc32(
                     std::span<const std::byte>(f).subspan(at + 4, len + 4)));
        at += 12 + len;
    }
}

/** Decodes, accepting a clean refusal; any other exception fails. */
void decode_or_refuse(const std::vector<std::byte>& f) {
    try {
        (void)decode_png(f, {});
    } catch (const std::invalid_argument&) {
    }
}

}  // namespace

TEST_CASE("png decoder matches the corpus and stb_image", "[image][png]") {
    const auto names = corpus();
    REQUIRE(names.size() >= 37);
    for (const auto& n : names) {
        INFO(n);
        const auto png = read_file(kCorpus + n + ".png");
        const auto want = read_file(kCorpus + n + ".rgb");
        const auto got = decode_png(png, {});
        REQUIRE(got.rgb.size() == want.size());
        REQUIRE(std::equal(got.rgb.begin(), got.rgb.end(), want.begin(),
                           [](std::uint8_t a, std::byte b) {
                               return a == std::to_integer<std::uint8_t>(b);
                           }));
        const auto ref = locus::image::detail::decode_image_stb(png);
        REQUIRE(got.w == ref.w);
        REQUIRE(got.h == ref.h);
        REQUIRE(got.rgb == ref.rgb);
        // decode_image routes PNG here.
        REQUIRE(locus::image::decode_image(png).rgb == got.rgb);
    }
}

TEST_CASE("png decoder refuses broken files cleanly", "[image][png]") {
    const auto good = read_file(kCorpus + "c2_d8_plain_fixed.png");
    REQUIRE_NOTHROW(decode_png(good, {}));

    SECTION("a flipped byte breaks a CRC") {
        auto f = good;
        f[40] ^= std::byte{1};
        REQUIRE_THROWS_WITH(decode_png(f, {}),
                            Catch::Matchers::ContainsSubstring("CRC"));
    }
    SECTION("dimensions over the cap, refused on IHDR") {
        auto f = good;
        put_be32(f, 16, 100000);
        put_be32(f, 20, 100000);
        fix_crcs(f);
        REQUIRE_THROWS_WITH(decode_png(f, {}),
                            Catch::Matchers::ContainsSubstring("limit"));
        locus::image::DecodeLimits lim;
        lim.max_pixels = 13 * 7 - 1;
        REQUIRE_THROWS_AS(decode_png(good, lim), std::invalid_argument);
    }
    SECTION("bad header fields") {
        for (const auto& [off, v] :
             std::vector<std::pair<std::size_t, int>>{
                 {24, 3},    // RGB at depth 3
                 {25, 5},    // color type 5
                 {26, 1},    // compression method 1
                 {27, 1},    // filter method 1
                 {28, 2}}) { // interlace 2
            auto f = good;
            f[off] = std::byte(v);
            fix_crcs(f);
            INFO("offset " << off);
            REQUIRE_THROWS_AS(decode_png(f, {}), std::invalid_argument);
        }
    }
    SECTION("every truncation") {
        for (std::size_t n = 0; n < good.size(); ++n) {
            INFO("prefix " << n);
            std::vector<std::byte> f(good.begin(), good.begin() + n);
            REQUIRE_THROWS_AS(decode_png(f, {}), std::invalid_argument);
        }
    }
    SECTION("mutations past the CRC check decode or refuse") {
        // Random byte overwrites with the CRCs rewritten, over every
        // corpus file: they reach IHDR, PLTE, inflate and the filters.
        std::uint32_t state = 250;
        const auto next = [&] {
            state = state * 1664525u + 1013904223u;
            return state >> 8;
        };
        for (const auto& n : corpus()) {
            const auto base = read_file(kCorpus + n + ".png");
            for (int round = 0; round < 300; ++round) {
                auto f = base;
                const int flips = 1 + static_cast<int>(next() % 4);
                for (int i = 0; i < flips; ++i) {
                    f[8 + next() % (f.size() - 8)] =
                        std::byte(next() & 0xFF);
                }
                fix_crcs(f);
                decode_or_refuse(f);
            }
        }
    }
}

TEST_CASE("zlib inflate checks and caps", "[image][png]") {
    using Bytes = std::vector<std::uint8_t>;
    // zlib.compress(b"hello hello hello", 9).
    const Bytes hello = {0x78, 0xda, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57,
                         0xc8, 0x40, 0x90, 0x00, 0x3a, 0x2e, 0x06, 0x7d};
    const auto out = zlib_inflate(hello, 100);
    REQUIRE(std::string(out.begin(), out.end()) == "hello hello hello");
    REQUIRE(locus::image::adler32(out) == 0x3a2e067du);

    REQUIRE_THROWS_WITH(zlib_inflate(hello, 16),
                        Catch::Matchers::ContainsSubstring("limit"));
    Bytes adler = hello;
    adler.back() ^= 1;
    REQUIRE_THROWS_WITH(zlib_inflate(adler, 100),
                        Catch::Matchers::ContainsSubstring("Adler"));
    Bytes hdr = hello;
    hdr[1] ^= 1;  // header check
    REQUIRE_THROWS_AS(zlib_inflate(hdr, 100), std::invalid_argument);
    const Bytes dict = {0x78, 0xbb, 0, 0, 0, 1};  // FDICT set
    REQUIRE_THROWS_WITH(zlib_inflate(dict, 100),
                        Catch::Matchers::ContainsSubstring("dictionary"));
    const Bytes btype3 = {0x78, 0x01, 0x07};  // final block, type 3
    REQUIRE_THROWS_WITH(zlib_inflate(btype3, 100),
                        Catch::Matchers::ContainsSubstring("block type"));
    // A stored block whose LEN / NLEN disagree.
    const Bytes nlen = {0x78, 0x01, 0x01, 0x05, 0x00, 0xfa, 0xfe};
    REQUIRE_THROWS_WITH(zlib_inflate(nlen, 100),
                        Catch::Matchers::ContainsSubstring("stored"));
    // A fixed block whose first symbol is a match: distance before the
    // start of the output (literal/length 257 = 0000001, distance 0).
    const Bytes far = {0x78, 0x01, 0x03, 0x02, 0x00};
    REQUIRE_THROWS_WITH(zlib_inflate(far, 100),
                        Catch::Matchers::ContainsSubstring("distance"));
    for (std::size_t n = 0; n < hello.size(); ++n) {
        INFO("prefix " << n);
        REQUIRE_THROWS_AS(
            zlib_inflate(Bytes(hello.begin(), hello.begin() + n), 100),
            std::invalid_argument);
    }
}
