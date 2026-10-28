#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/image/image_input.hpp"
#include "locus/image/jpeg_decode.hpp"

// Hidden bench: decode time of locus's JPEG decoder on a large file
// (best of 3). LOCUS_JPEG_BENCH=<path to a .jpg>. On M2 a 4000 x 3000
// 4:2:0 file took 171 ms (stb_image 106 ms), progressive 210 ms (199).
TEST_CASE("bench: jpeg decode", "[.][bench][jpeg]") {
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
    const double ms = time([&] { (void)locus::image::decode_jpeg(data, lim); });
    WARN("decode_jpeg " << ms << " ms");
}
