// Micro-benchmarks for the matvec paths. Hidden ([.]) so they never
// run in the normal suite; invoke explicitly:
//   ./build/tests/locus_tests "[bench]"
// Reports ns per matvec so the Q8_K opt-in path (DESIGN.md R16) can be
// compared to the f32-dequant default -- Q8_K exists for integer-dot
// throughput, and this is where that win is verified.

#include <chrono>
#include <cstring>
#include <random>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/variants.hpp"
#include "locus/sys/features.hpp"

using namespace locus::backend;

namespace {

// A rows x cols Q4_K weight with plausible per-block scales.
std::vector<std::byte> make_q4k(std::uint32_t rows, std::uint32_t cols,
                                std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::vector<std::byte> w(static_cast<std::size_t>(rows) *
                             (cols / 256) * 144);
    for (auto& b : w) {
        b = static_cast<std::byte>(byte_d(rng));
    }
    for (std::size_t bl = 0; bl < w.size() / 144; ++bl) {
        std::byte* blk = w.data() + bl * 144;
        const std::uint16_t d = f32_to_f16(0.01f);
        const std::uint16_t dmin = f32_to_f16(0.005f);
        std::memcpy(blk, &d, 2);
        std::memcpy(blk + 2, &dmin, 2);
    }
    return w;
}

double time_matvec(void (*mv)(const Mat&, std::span<const float>,
                              std::span<float>),
                   const Mat& m, std::span<const float> x,
                   std::span<float> out, int iters) {
    for (int i = 0; i < 3; ++i) {
        mv(m, x, out);  // warm up
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        mv(m, x, out);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns =
        std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / iters;
}

}  // namespace

TEST_CASE("bench: Q4_K matvec f32 vs Q8_K", "[.][bench]") {
    const std::uint32_t rows = 4096, cols = 4096;  // ~9.4MB Q4_K
    auto w = make_q4k(rows, cols, 7);
    Mat m{locus::gguf::TensorType::kQ4_K, w.data(), rows, cols};
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(cols), out(rows);
    for (auto& v : x) {
        v = dist(rng);
    }
    const int iters = 200;

    const double s_f32 = time_matvec(&matvec, m, x, out, iters);
    const double s_q8k = time_matvec(&matvec_q8k, m, x, out, iters);
    WARN("scalar  Q4_K: f32=" << s_f32 << "ns  q8k=" << s_q8k
                              << "ns  speedup=" << s_f32 / s_q8k
                              << "x");
#if defined(__x86_64__)
    if (locus::sys::detect().sse4) {
        const double x_f32 =
            time_matvec(&matvec_sse4, m, x, out, iters);
        const double x_q8k =
            time_matvec(&matvec_sse4_q8k, m, x, out, iters);
        WARN("sse4    Q4_K: f32=" << x_f32 << "ns  q8k=" << x_q8k
                                  << "ns  speedup=" << x_f32 / x_q8k
                                  << "x");
    }
#endif
    SUCCEED();
}
