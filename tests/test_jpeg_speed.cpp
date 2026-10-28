#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/image/image_input.hpp"
#include "locus/image/jpeg_decode.hpp"

// Hidden bench: decode time of locus's JPEG decoder against stb_image
// on a large file. LOCUS_JPEG_BENCH=<path to a .jpg>.
TEST_CASE("bench: jpeg decode against stb_image", "[.][bench][jpeg]") {
    const char* path = std::getenv("LOCUS_JPEG_BENCH");
    if (path == nullptr) SKIP("set LOCUS_JPEG_BENCH to a JPEG file");
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    const std::string s((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    const auto* p = reinterpret_cast<const std::byte*>(s.data());
    const std::vector<std::byte> data(p, p + s.size());
    locus::image::DecodeLimits lim;
    lim.max_bytes = data.size();
    using Clock = std::chrono::steady_clock;
    auto time = [&](auto fn) {
        double best = 1e9;
        for (int i = 0; i < 3; ++i) {
            const auto t0 = Clock::now();
            fn();
            best = std::min(
                best, std::chrono::duration<double, std::milli>(Clock::now() - t0)
                          .count());
        }
        return best;
    };
    const double ours = time([&] { (void)locus::image::decode_jpeg(data, lim); });
    const double stb =
        time([&] { (void)locus::image::detail::decode_image_stb(data, lim); });
    WARN("locus " << ours << " ms, stb_image " << stb << " ms");
}
