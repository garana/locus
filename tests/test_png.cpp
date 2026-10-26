#include "locus/image/png.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "png_parse.hpp"

namespace {

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
        const auto png = locus::image::encode_png(rgb, h, w);
        const auto back = png_test::parse_stored_png(png);
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
    const auto ref =
        png_test::parse_stored_png(read_file(g + "janus_decode.png"));
    REQUIRE(ref.w == 384);
    REQUIRE(ref.h == 384);
    REQUIRE(locus::image::to_rgb8(chw, 384, 384) == ref.rgb);
}
