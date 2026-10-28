#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/image/image_input.hpp"
#include "locus/image/jpeg_decode.hpp"

using locus::image::decode_jpeg;

namespace {

const std::string kCorpus =
    std::string(LOCUS_SOURCE_DIR) + "/tests/golden/jpeg/";

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
        if (e.path().extension() == ".jpg") out.push_back(e.path().stem());
    }
    std::sort(out.begin(), out.end());
    return out;
}

/** Decodes, accepting a clean refusal; any other exception fails. */
void decode_or_refuse(std::span<const std::byte> f) {
    try {
        (void)decode_jpeg(f, {});
    } catch (const std::invalid_argument&) {
    }
}

}  // namespace

TEST_CASE("jpeg decoder matches stb_image on the corpus", "[image][jpeg]") {
    const auto names = corpus();
    REQUIRE(names.size() >= 27);
    for (const auto& n : names) {
        INFO(n);
        const auto jpg = read_file(kCorpus + n + ".jpg");
        if (n.rfind("arith_", 0) == 0) {
            REQUIRE_THROWS_WITH(
                decode_jpeg(jpg, {}),
                Catch::Matchers::ContainsSubstring("unsupported"));
            continue;
        }
        const auto got = decode_jpeg(jpg, {});
        const auto ref = locus::image::detail::decode_image_stb(jpg);
        REQUIRE(got.w == ref.w);
        REQUIRE(got.h == ref.h);
        int worst = 0;
        std::size_t diffs = 0;
        for (std::size_t i = 0; i < got.rgb.size(); ++i) {
            const int d = std::abs(int(got.rgb[i]) - int(ref.rgb[i]));
            worst = std::max(worst, d);
            diffs += d != 0;
        }
        INFO("bytes that differ: " << diffs << ", worst " << worst);
        REQUIRE(worst == 0);
        // decode_image routes JPEG here.
        REQUIRE(locus::image::decode_image(jpg).rgb == got.rgb);
    }
}

TEST_CASE("jpeg decoder refuses broken files cleanly", "[image][jpeg]") {
    const auto base = read_file(kCorpus + "s420_q75.jpg");
    const auto prog = read_file(kCorpus + "s420_prog_restart.jpg");
    REQUIRE_NOTHROW(decode_jpeg(base, {}));

    SECTION("not a JPEG, and an empty one") {
        REQUIRE_THROWS_AS(decode_jpeg({}, {}), std::invalid_argument);
        const std::vector<std::byte> soi_only = {std::byte{0xFF},
                                                 std::byte{0xD8}};
        REQUIRE_THROWS_AS(decode_jpeg(soi_only, {}), std::invalid_argument);
        auto f = base;
        f[1] = std::byte{0xD9};
        REQUIRE_THROWS_WITH(decode_jpeg(f, {}),
                            Catch::Matchers::ContainsSubstring("SOI"));
    }
    SECTION("dimensions over the cap, refused on the frame header") {
        locus::image::DecodeLimits lim;
        lim.max_pixels = 37 * 29 - 1;
        REQUIRE_THROWS_WITH(decode_jpeg(base, lim),
                            Catch::Matchers::ContainsSubstring("limit"));
        // A frame header claiming 60000 x 60000.
        auto f = base;
        for (std::size_t i = 2; i + 8 < f.size(); ++i) {
            if (f[i] == std::byte{0xFF} && f[i + 1] == std::byte{0xC0}) {
                f[i + 5] = f[i + 7] = std::byte{0xEA};
                f[i + 6] = f[i + 8] = std::byte{0x60};
                break;
            }
        }
        REQUIRE_THROWS_WITH(decode_jpeg(f, {}),
                            Catch::Matchers::ContainsSubstring("limit"));
    }
    SECTION("every truncation decodes or refuses") {
        for (const auto* file : {&base, &prog}) {
            for (std::size_t n = 0; n < file->size(); ++n) {
                INFO("prefix " << n);
                decode_or_refuse(std::span(file->data(), n));
            }
        }
    }
    SECTION("random mutations decode or refuse") {
        std::uint32_t state = 250;
        const auto next = [&] {
            state = state * 1664525u + 1013904223u;
            return state >> 8;
        };
        for (const auto& n : corpus()) {
            const auto f0 = read_file(kCorpus + n + ".jpg");
            for (int round = 0; round < 200; ++round) {
                auto f = f0;
                const int flips = 1 + static_cast<int>(next() % 6);
                for (int i = 0; i < flips; ++i) {
                    f[2 + next() % (f.size() - 2)] = std::byte(next() & 0xFF);
                }
                decode_or_refuse(f);
            }
        }
    }
}
