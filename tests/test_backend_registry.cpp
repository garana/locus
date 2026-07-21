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
    check_matvec_variant_matches_scalar(&matvec_cuda);
}
