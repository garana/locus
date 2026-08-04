#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/sys/features.hpp"

using namespace locus::backend;

TEST_CASE("registry lists and resolves backends", "[backend]") {
    REQUIRE(!backends().empty());
    REQUIRE(find_backend("scalar") != nullptr);
    REQUIRE(find_backend("vulkan") != nullptr);
    REQUIRE(find_backend("nope") == nullptr);

    const Backend& best = best_backend();
    REQUIRE(best.available);
    REQUIRE(best.selectable);
#if defined(__aarch64__)
    REQUIRE(best.name == "neon");
#endif

    REQUIRE(&resolve_backend("") == &best);
    REQUIRE(&resolve_backend("scalar") ==
            find_backend("scalar"));
    REQUIRE_THROWS_AS(resolve_backend("nope"),
                      std::invalid_argument);
    // Selectable exactly when a usable device + kernels exist.
    if (vulkan_backend_usable()) {
        REQUIRE(&resolve_backend("vulkan") ==
                find_backend("vulkan"));
    } else {
        REQUIRE_THROWS_AS(resolve_backend("vulkan"),
                          std::invalid_argument);
    }
}

#if defined(__aarch64__)
TEST_CASE("neon matvec matches scalar", "[backend]") {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    SECTION("f32, with a non-multiple-of-4 tail") {
        const std::uint32_t rows = 5, cols = 67;
        std::vector<float> w(static_cast<std::size_t>(rows) *
                             cols);
        std::vector<float> x(cols);
        for (auto& v : w) {
            v = dist(rng);
        }
        for (auto& v : x) {
            v = dist(rng);
        }
        Mat m{locus::gguf::TensorType::kF32,
              reinterpret_cast<const std::byte*>(w.data()), rows,
              cols};
        std::vector<float> a(rows), b(rows);
        matvec(m, x, a);
        matvec_neon(m, x, b);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(b[r] == Catch::Approx(a[r]).margin(1e-4));
        }
    }

    SECTION("q8_0 blocks") {
        const std::uint32_t rows = 3, cols = 64;  // 2 blocks/row
        std::vector<std::byte> w(rows * (cols / 32) * 34);
        std::uniform_int_distribution<int> qd(-127, 127);
        for (std::size_t b = 0; b < rows * (cols / 32); ++b) {
            std::byte* blk = w.data() + b * 34;
            const std::uint16_t d = f32_to_f16(0.02f);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 32; ++i) {
                blk[2 + i] = static_cast<std::byte>(
                    static_cast<std::int8_t>(qd(rng)));
            }
        }
        std::vector<float> x(cols);
        for (auto& v : x) {
            v = dist(rng);
        }
        Mat m{locus::gguf::TensorType::kQ8_0, w.data(), rows,
              cols};
        std::vector<float> a(rows), b(rows);
        matvec(m, x, a);
        matvec_neon(m, x, b);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(b[r] == Catch::Approx(a[r]).margin(1e-4));
        }
    }
}
#endif

namespace {

/** Drives one matvec variant vs the scalar reference over an f32
 *  tail case and a multi-block q8_0 case. `variant` is only invoked
 *  after the caller has confirmed the device/CPU supports it. */
void check_matvec_variant_matches_scalar(
    void (*variant)(const Mat&, std::span<const float>,
                    std::span<float>)) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    SECTION("f32, with a non-multiple-of-4 tail") {
        const std::uint32_t rows = 5, cols = 67;
        std::vector<float> w(static_cast<std::size_t>(rows) *
                             cols);
        std::vector<float> x(cols);
        for (auto& v : w) {
            v = dist(rng);
        }
        for (auto& v : x) {
            v = dist(rng);
        }
        Mat m{locus::gguf::TensorType::kF32,
              reinterpret_cast<const std::byte*>(w.data()), rows,
              cols};
        std::vector<float> a(rows), b(rows);
        matvec(m, x, a);
        variant(m, x, b);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(b[r] == Catch::Approx(a[r]).margin(1e-4));
        }
    }

    SECTION("q8_0 blocks") {
        const std::uint32_t rows = 3, cols = 64;  // 2 blocks/row
        std::vector<std::byte> w(rows * (cols / 32) * 34);
        std::uniform_int_distribution<int> qd(-127, 127);
        for (std::size_t b = 0; b < rows * (cols / 32); ++b) {
            std::byte* blk = w.data() + b * 34;
            const std::uint16_t d = f32_to_f16(0.02f);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 32; ++i) {
                blk[2 + i] = static_cast<std::byte>(
                    static_cast<std::int8_t>(qd(rng)));
            }
        }
        std::vector<float> x(cols);
        for (auto& v : x) {
            v = dist(rng);
        }
        Mat m{locus::gguf::TensorType::kQ8_0, w.data(), rows,
              cols};
        std::vector<float> a(rows), b(rows);
        matvec(m, x, a);
        variant(m, x, b);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(b[r] == Catch::Approx(a[r]).margin(1e-4));
        }
    }

    SECTION("q4_k super-block") {
        const std::uint32_t rows = 3, cols = 256;  // 1 sblk/row
        const std::uint32_t nsb = cols / 256;
        std::vector<std::byte> w(
            static_cast<std::size_t>(rows) * nsb * 144);
        std::uniform_int_distribution<int> bd(0, 255);
        for (auto& by : w) {
            by = static_cast<std::byte>(bd(rng));
        }
        // Sane f16 d/dmin per super-block (random qs/scales).
        for (std::uint32_t s = 0; s < rows * nsb; ++s) {
            std::byte* blk =
                w.data() + static_cast<std::size_t>(s) * 144;
            const std::uint16_t d = f32_to_f16(0.01f);
            const std::uint16_t dmin = f32_to_f16(0.005f);
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &dmin, 2);
        }
        std::vector<float> x(cols);
        for (auto& v : x) {
            v = dist(rng);
        }
        Mat m{locus::gguf::TensorType::kQ4_K, w.data(), rows, cols};
        std::vector<float> a(rows), b(rows);
        matvec(m, x, a);
        variant(m, x, b);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(b[r] == Catch::Approx(a[r]).margin(1e-3));
        }
    }

    SECTION("q5_k super-block") {
        const std::uint32_t rows = 3, cols = 256;  // 1 sblk/row
        const std::uint32_t nsb = cols / 256;
        std::vector<std::byte> w(
            static_cast<std::size_t>(rows) * nsb * 176);
        std::uniform_int_distribution<int> bd(0, 255);
        for (auto& by : w) {
            by = static_cast<std::byte>(bd(rng));
        }
        // Sane f16 d/dmin at the block head (random qh/ql/scales).
        for (std::uint32_t s = 0; s < rows * nsb; ++s) {
            std::byte* blk =
                w.data() + static_cast<std::size_t>(s) * 176;
            const std::uint16_t d = f32_to_f16(0.01f);
            const std::uint16_t dmin = f32_to_f16(0.005f);
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &dmin, 2);
        }
        std::vector<float> x(cols);
        for (auto& v : x) {
            v = dist(rng);
        }
        Mat m{locus::gguf::TensorType::kQ5_K, w.data(), rows, cols};
        std::vector<float> a(rows), b(rows);
        matvec(m, x, a);
        variant(m, x, b);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(b[r] == Catch::Approx(a[r]).margin(1e-3));
        }
    }

    SECTION("q6_k super-block") {
        const std::uint32_t rows = 3, cols = 256;  // 1 sblk/row
        const std::uint32_t nsb = cols / 256;
        std::vector<std::byte> w(
            static_cast<std::size_t>(rows) * nsb * 210);
        std::uniform_int_distribution<int> bd(0, 255);
        for (auto& by : w) {
            by = static_cast<std::byte>(bd(rng));
        }
        // Q6_K's f16 d lives at the block tail (blk+208); sc[16] and
        // ql/qh stay random.
        for (std::uint32_t s = 0; s < rows * nsb; ++s) {
            std::byte* blk =
                w.data() + static_cast<std::size_t>(s) * 210;
            const std::uint16_t d = f32_to_f16(0.01f);
            std::memcpy(blk + 208, &d, 2);
        }
        std::vector<float> x(cols);
        for (auto& v : x) {
            v = dist(rng);
        }
        Mat m{locus::gguf::TensorType::kQ6_K, w.data(), rows, cols};
        std::vector<float> a(rows), b(rows);
        matvec(m, x, a);
        variant(m, x, b);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(b[r] == Catch::Approx(a[r]).margin(1e-2));
        }
    }

    SECTION("iq1_s super-block") {
        const std::uint32_t rows = 3, cols = 256;  // 1 sblk/row
        const std::uint32_t nsb = cols / 256;
        std::vector<std::byte> w(
            static_cast<std::size_t>(rows) * nsb * 50);
        std::uniform_int_distribution<int> bd(0, 255);
        for (auto& by : w) {
            by = static_cast<std::byte>(bd(rng));
        }
        // Sane f16 d per super-block; random qs/qh index the grid
        // (max index 255|(7<<8)=2047 < 2048, always in range).
        for (std::uint32_t s = 0; s < rows * nsb; ++s) {
            std::byte* blk =
                w.data() + static_cast<std::size_t>(s) * 50;
            const std::uint16_t d = f32_to_f16(0.01f);
            std::memcpy(blk, &d, 2);
        }
        std::vector<float> x(cols);
        for (auto& v : x) {
            v = dist(rng);
        }
        Mat m{locus::gguf::TensorType::kIQ1_S, w.data(), rows, cols};
        std::vector<float> a(rows), b(rows);
        matvec(m, x, a);
        variant(m, x, b);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(b[r] == Catch::Approx(a[r]).margin(1e-2));
        }
    }
}

}  // namespace

#if defined(__x86_64__)
TEST_CASE("sse4 matvec matches scalar", "[backend]") {
    if (!locus::sys::detect().sse4) {
        SKIP("CPU lacks SSE4.1");
    }
    check_matvec_variant_matches_scalar(&matvec_sse4);
}

TEST_CASE("avx2 matvec matches scalar", "[backend]") {
    if (!locus::sys::detect().avx2) {
        SKIP("CPU lacks AVX2 (running matvec_avx2 would fault)");
    }
    check_matvec_variant_matches_scalar(&matvec_avx2);
}
#endif

TEST_CASE("cuda matvec matches scalar", "[backend]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    cuda_pool_reset();  // drop any stale pages from prior test cases
    check_matvec_variant_matches_scalar(&matvec_cuda);
}

// R16 ggml-parity: the CUDA Q8_K-activation dot must match the scalar
// matvec_q8k bit-for-bit (same host quant, same accumulation order).
TEST_CASE("cuda q8k matvec matches scalar q8k (k-quants)",
          "[backend]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    cuda_pool_reset();
    using TT = locus::gguf::TensorType;
    // {type, super-block bytes, d offset, dmin offset (-1: none)}.
    struct Q {
        TT type;
        std::size_t bytes;
        std::size_t d_off;
        int dmin_off;
    };
    const std::uint32_t rows = 4, cols = 512;  // 2 blocks/row
    std::mt19937 rng(29);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    for (const Q& q : {Q{TT::kQ2_K, 84, 80, 82}, Q{TT::kQ4_K, 144, 0, 2},
                       Q{TT::kQ5_K, 176, 0, 2},
                       Q{TT::kQ6_K, 210, 208, -1}}) {
        std::vector<std::byte> w(static_cast<std::size_t>(rows) *
                                 (cols / 256) * q.bytes);
        for (auto& b : w) {
            b = static_cast<std::byte>(byte_d(rng));
        }
        for (std::size_t bl = 0; bl < w.size() / q.bytes; ++bl) {
            std::byte* blk = w.data() + bl * q.bytes;
            const std::uint16_t d = f32_to_f16(0.01f);
            const std::uint16_t dmin = f32_to_f16(0.005f);
            std::memcpy(blk + q.d_off, &d, 2);
            if (q.dmin_off >= 0) {
                std::memcpy(blk + q.dmin_off, &dmin, 2);
            }
        }
        Mat m{q.type, w.data(), rows, cols};
        std::vector<float> a(rows), b(rows);
        matvec_q8k(m, x, a);
        matvec_cuda_q8k(m, x, b);
        for (std::uint32_t r = 0; r < rows; ++r) {
            INFO("type " << static_cast<int>(q.type) << " row " << r);
            REQUIRE(b[r] == Catch::Approx(a[r]).margin(1e-4));
        }
    }
}

// Phase 3: prefetch uploads the weight async on a second stream;
// the following matvec must wait on that copy and stay token-exact.
TEST_CASE("cuda prefetch then matvec is token-exact", "[backend]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    cuda_pool_reset();  // drop any stale pages from prior test cases
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const std::uint32_t rows = 4, cols = 64;  // 2 q8_0 blocks/row
    std::vector<std::byte> w(rows * (cols / 32) * 34);
    std::uniform_int_distribution<int> qd(-127, 127);
    for (std::size_t bl = 0; bl < rows * (cols / 32); ++bl) {
        std::byte* blk = w.data() + bl * 34;
        const std::uint16_t d = f32_to_f16(0.02f);
        std::memcpy(blk, &d, 2);
        for (int i = 0; i < 32; ++i) {
            blk[2 + i] = static_cast<std::byte>(
                static_cast<std::int8_t>(qd(rng)));
        }
    }
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    Mat m{locus::gguf::TensorType::kQ8_0, w.data(), rows, cols};
    std::vector<float> a(rows), b(rows);
    matvec(m, x, a);
    cuda_prefetch(m);       // async upload into the pool
    matvec_cuda(m, x, b);   // must consume the in-flight page
    for (std::uint32_t r = 0; r < rows; ++r) {
        REQUIRE(b[r] == Catch::Approx(a[r]).margin(1e-4));
    }
}
