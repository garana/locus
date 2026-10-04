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

// A rows x cols k-quant weight with plausible per-block scales.
std::vector<std::byte> make_kquant(std::uint32_t rows,
                                   std::uint32_t cols, std::size_t bytes,
                                   std::size_t d_off, int dmin_off,
                                   std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::vector<std::byte> w(static_cast<std::size_t>(rows) *
                             (cols / 256) * bytes);
    for (auto& b : w) {
        b = static_cast<std::byte>(byte_d(rng));
    }
    for (std::size_t bl = 0; bl < w.size() / bytes; ++bl) {
        std::byte* blk = w.data() + bl * bytes;
        const std::uint16_t d = f32_to_f16(0.01f);
        const std::uint16_t dmin = f32_to_f16(0.005f);
        std::memcpy(blk + d_off, &d, 2);
        if (dmin_off >= 0) {
            std::memcpy(blk + dmin_off, &dmin, 2);
        }
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

TEST_CASE("bench: k-quant matvec f32 vs Q8_K", "[.][bench]") {
    using TT = locus::gguf::TensorType;
    struct Q {
        TT type;
        const char* name;
        std::size_t bytes;
        std::size_t d_off;
        int dmin_off;
    };
    const std::uint32_t rows = 4096, cols = 4096;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(cols), out(rows);
    for (auto& v : x) {
        v = dist(rng);
    }
    const int iters = 200;
    for (const Q& q : {Q{TT::kQ4_K, "Q4_K", 144, 0, 2},
                       Q{TT::kQ5_K, "Q5_K", 176, 0, 2},
                       Q{TT::kQ6_K, "Q6_K", 210, 208, -1}}) {
        auto w = make_kquant(rows, cols, q.bytes, q.d_off, q.dmin_off,
                             7);
        Mat m{q.type, w.data(), rows, cols};
        const double s_f32 = time_matvec(&matvec, m, x, out, iters);
        const double s_q8k = time_matvec(&matvec_q8k, m, x, out, iters);
        WARN("scalar " << q.name << ": f32=" << s_f32 << "ns q8k="
                       << s_q8k << "ns speedup=" << s_f32 / s_q8k
                       << "x");
#if defined(__x86_64__)
        if (locus::sys::detect().sse4) {
            const double x_f32 =
                time_matvec(&matvec_sse4, m, x, out, iters);
            const double x_q8k =
                time_matvec(&matvec_sse4_q8k, m, x, out, iters);
            WARN("sse4   " << q.name << ": f32=" << x_f32 << "ns q8k="
                           << x_q8k << "ns speedup=" << x_f32 / x_q8k
                           << "x");
        }
#endif
    }
    SUCCEED();
}

// i#52 measure-first: the last stage projects out_w_ (the LM head, the
// biggest tensor) once PER token via matvec. Amortizing it into one
// matvec_batch reads the weight once for the whole batch; this measures
// whether that actually wins on THIS GPU at the batch widths we run,
// before committing a bigger activation buffer + a transpose to it.
// Synthetic Q6_K LM head (the usual output.weight type) at vocab x embd.
TEST_CASE("bench: out_w_ n x matvec_cuda vs matvec_batch_cuda",
          "[.][bench][cuda]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    using TT = locus::gguf::TensorType;
    const std::uint32_t rows = 32000;  // vocab
    const std::uint32_t cols = 2048;   // embd
    auto w = make_kquant(rows, cols, /*bytes=*/210, /*d_off=*/208,
                         /*dmin_off=*/-1, 7);
    Mat m{TT::kQ6_K, w.data(), rows, cols};
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int iters = 50;
    for (std::uint32_t n : {1u, 4u, 8u, 16u}) {
        std::vector<float> xb(static_cast<std::size_t>(n) * cols);
        for (auto& v : xb) {
            v = dist(rng);
        }
        std::vector<float> col(rows);
        std::vector<float> outb(static_cast<std::size_t>(rows) * n);
        cuda_pool_reset();
        auto per_token = [&] {
            for (std::uint32_t t = 0; t < n; ++t) {
                matvec_cuda(m,
                            {xb.data() + static_cast<std::size_t>(t) *
                                             cols,
                             cols},
                            col);
            }
        };
        auto batched = [&] { matvec_batch_cuda(m, xb, outb, n); };
        for (int i = 0; i < 3; ++i) {  // warm (upload + JIT)
            per_token();
            batched();
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            per_token();
        }
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            batched();
        }
        const auto t2 = std::chrono::steady_clock::now();
        const double per_ns =
            std::chrono::duration<double, std::nano>(t1 - t0).count() /
            iters;
        const double bat_ns =
            std::chrono::duration<double, std::nano>(t2 - t1).count() /
            iters;
        WARN("cuda out_w_ n=" << n << ": per-token=" << per_ns
                              << "ns batched=" << bat_ns
                              << "ns speedup=" << per_ns / bat_ns << "x");
    }
    SUCCEED();
}
