#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/variants.hpp"

using namespace locus::backend;
using locus::gguf::TensorType;
using Catch::Approx;

TEST_CASE("f16 conversion round-trips", "[ops]") {
    for (float v : {0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, 1e-4f}) {
        REQUIRE(f16_to_f32(f32_to_f16(v)) ==
                Approx(v).margin(v == 0 ? 0 : std::abs(v) * 1e-3));
    }
    // Subnormal range: absolute precision is one f16 ulp (~6e-8).
    REQUIRE(f16_to_f32(f32_to_f16(1e-6f)) ==
            Approx(1e-6f).margin(6e-8));
    REQUIRE(f16_to_f32(0x3800) == Approx(0.5f));
    REQUIRE(std::isinf(f16_to_f32(0x7c00)));
}

TEST_CASE("rmsnorm matches the closed form", "[ops]") {
    const float x[] = {3.0f, 4.0f};
    const float w[] = {1.0f, 2.0f};
    float out[2];
    rmsnorm(x, w, 0.0f, out);
    const float scale = 1.0f / std::sqrt((9.0f + 16.0f) / 2.0f);
    REQUIRE(out[0] == Approx(3.0f * scale));
    REQUIRE(out[1] == Approx(4.0f * 2.0f * scale));
}

TEST_CASE("softmax normalizes and orders", "[ops]") {
    float x[] = {0.0f, std::log(3.0f)};
    softmax_inplace(x);
    REQUIRE(x[0] == Approx(0.25f));
    REQUIRE(x[1] == Approx(0.75f));
}

TEST_CASE("silu_mul limits", "[ops]") {
    const float gate[] = {0.0f, 100.0f, -100.0f};
    const float up[] = {5.0f, 2.0f, 3.0f};
    float out[3];
    silu_mul(gate, up, out);
    REQUIRE(out[0] == 0.0f);            // silu(0) = 0
    REQUIRE(out[1] == Approx(200.0f));  // silu(x) -> x for big x
    REQUIRE(out[2] == Approx(0.0f).margin(1e-6));
}

TEST_CASE("rope is identity at pos 0 and norm-preserving", "[ops]") {
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    auto orig = x;
    rope_norm(x, 1, 4, 0, 10000.0f);
    REQUIRE(x == orig);

    rope_norm(x, 2, 2, 17, 10000.0f);  // two heads of dim 2
    float n0 = 0, n1 = 0;
    for (int i = 0; i < 4; ++i) {
        n0 += orig[i] * orig[i];
        n1 += x[i] * x[i];
    }
    REQUIRE(n1 == Approx(n0));
    REQUIRE(x[0] != orig[0]);  // actually rotated
}

TEST_CASE("rope factors divide the frequency", "[ops]") {
    // With factor 2 at pos p, the rotation equals factor 1 at
    // pos p/2 (theta scales linearly with pos).
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = a;
    const float factors[] = {2.0f};
    rope_norm(a, 1, 2, 10, 10000.0f, factors);
    rope_norm(b, 1, 2, 5, 10000.0f);
    REQUIRE(a[0] == Approx(b[0]));
    REQUIRE(a[1] == Approx(b[1]));
}

TEST_CASE("matvec f32 and f16 agree", "[ops]") {
    const std::vector<float> wf = {1, 2, 3, 4, 5, 6};  // 2x3
    std::vector<std::uint16_t> wh(6);
    for (int i = 0; i < 6; ++i) {
        wh[i] = f32_to_f16(wf[i]);
    }
    const float x[] = {1.0f, -1.0f, 2.0f};
    float out[2];

    Mat m32{TensorType::kF32,
            reinterpret_cast<const std::byte*>(wf.data()), 2, 3};
    matvec(m32, x, out);
    REQUIRE(out[0] == Approx(1 - 2 + 6));
    REQUIRE(out[1] == Approx(4 - 5 + 12));

    Mat m16{TensorType::kF16,
            reinterpret_cast<const std::byte*>(wh.data()), 2, 3};
    matvec(m16, x, out);
    REQUIRE(out[0] == Approx(5.0f));
    REQUIRE(out[1] == Approx(11.0f));
}

namespace {

/** Builds one Q8_0 row (single block) with scale d and quants q. */
std::vector<std::byte> q8_0_row(float d,
                                const std::vector<int>& q) {
    std::vector<std::byte> row(34);
    std::uint16_t h = f32_to_f16(d);
    std::memcpy(row.data(), &h, 2);
    for (int i = 0; i < 32; ++i) {
        row[static_cast<std::size_t>(2 + i)] =
            static_cast<std::byte>(static_cast<std::int8_t>(q[
                static_cast<std::size_t>(i)]));
    }
    return row;
}

}  // namespace

TEST_CASE("matvec q8_0 dequantizes correctly", "[ops]") {
    std::vector<int> q(32, 0);
    q[0] = 4;
    q[31] = -2;
    auto row = q8_0_row(0.5f, q);

    std::vector<float> x(32, 1.0f);
    float out[1];
    Mat m{TensorType::kQ8_0, row.data(), 1, 32};
    matvec(m, x, out);
    REQUIRE(out[0] == Approx(0.5f * (4 - 2)));

    std::vector<float> dq(32);
    dequant_row(m, 0, dq);
    REQUIRE(dq[0] == Approx(2.0f));
    REQUIRE(dq[31] == Approx(-1.0f));
    REQUIRE(dq[15] == 0.0f);
}

TEST_CASE("matvec q4_0 dequantizes correctly", "[ops]") {
    // One block: scale 2.0, all nibbles 8 (-> value 0) except
    // byte 0 = 0x9A: low nibble 10 (elem 0 -> +2), high nibble 9
    // (elem 16 -> +1).
    std::vector<std::byte> row(18, std::byte{0x88});
    std::uint16_t h = f32_to_f16(2.0f);
    std::memcpy(row.data(), &h, 2);
    row[2] = std::byte{0x9A};

    std::vector<float> x(32, 1.0f);
    float out[1];
    Mat m{TensorType::kQ4_0, row.data(), 1, 32};
    matvec(m, x, out);
    REQUIRE(out[0] == Approx(2.0f * ((10 - 8) + (9 - 8))));

    std::vector<float> dq(32);
    dequant_row(m, 0, dq);
    REQUIRE(dq[0] == Approx(4.0f));
    REQUIRE(dq[16] == Approx(2.0f));
    REQUIRE(dq[1] == 0.0f);
}

TEST_CASE("matvec rejects unsupported weight types", "[ops]") {
    Mat m{TensorType::kQ4_1, nullptr, 1, 256};
    std::vector<float> x(256, 0.0f);
    float out[1];
    REQUIRE_THROWS_AS(matvec(m, x, out), locus::gguf::Error);
}

namespace {

/** Random-but-sane K-quant row: qs random, f16 fields bounded. */
std::vector<std::byte> k_quant_row(TensorType t,
                                   std::uint32_t blocks,
                                   std::uint32_t seed) {
    struct Layout {
        std::size_t bytes;
        std::size_t d_off;
        int dmin_off;  // -1: none
    };
    const Layout lay = [t]() -> Layout {
        switch (t) {
            case TensorType::kQ2_K: return {84, 80, 82};
            case TensorType::kQ3_K: return {110, 108, -1};
            case TensorType::kQ4_K: return {144, 0, 2};
            case TensorType::kQ5_K: return {176, 0, 2};
            case TensorType::kQ6_K: return {210, 208, -1};
            case TensorType::kIQ2_XXS: return {66, 0, -1};
            case TensorType::kIQ3_XXS: return {98, 0, -1};
            case TensorType::kIQ1_S: return {50, 0, -1};
            default: return {136, 0, -1};  // kIQ4_XS
        }
    }();
    std::vector<std::byte> row(blocks * lay.bytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_d(0, 255);
    for (auto& b : row) {
        b = static_cast<std::byte>(byte_d(rng));
    }
    // Overwrite the f16 scale fields with sane magnitudes.
    for (std::uint32_t b = 0; b < blocks; ++b) {
        std::byte* blk = row.data() + b * lay.bytes;
        const std::uint16_t d =
            locus::backend::f32_to_f16(0.01f);
        const std::uint16_t dmin =
            locus::backend::f32_to_f16(0.005f);
        std::memcpy(blk + lay.d_off, &d, 2);
        if (lay.dmin_off >= 0) {
            std::memcpy(blk + lay.dmin_off, &dmin, 2);
        }
    }
    return row;
}

}  // namespace

TEST_CASE("k-quant matvec is consistent with dequant_row",
          "[ops]") {
    const std::uint32_t rows = 3, cols = 512;  // 2 blocks/row
    std::vector<float> x(cols);
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) {
        v = dist(rng);
    }
    for (TensorType t :
         {TensorType::kQ2_K, TensorType::kQ3_K,
          TensorType::kQ4_K, TensorType::kQ5_K,
          TensorType::kQ6_K, TensorType::kIQ2_XXS,
          TensorType::kIQ3_XXS, TensorType::kIQ1_S,
          TensorType::kIQ4_XS}) {
        auto row0 = k_quant_row(t, 2, 40);
        auto row1 = k_quant_row(t, 2, 41);
        auto row2 = k_quant_row(t, 2, 42);
        std::vector<std::byte> w;
        for (auto* r : {&row0, &row1, &row2}) {
            w.insert(w.end(), r->begin(), r->end());
        }
        Mat m{t, w.data(), rows, cols};

        std::vector<float> mv(rows), dq(cols);
        matvec(m, x, mv);
        for (std::uint32_t r = 0; r < rows; ++r) {
            dequant_row(m, r, dq);
            float ref = 0.0f;
            for (std::uint32_t i = 0; i < cols; ++i) {
                ref += dq[i] * x[i];
            }
            INFO("type " << static_cast<int>(t) << " row " << r);
            REQUIRE(mv[r] == Approx(ref).margin(1e-4));
        }
        // Some value must be nonzero, or the test proves nothing.
        bool nonzero = false;
        for (float v : dq) {
            nonzero = nonzero || v != 0.0f;
        }
        REQUIRE(nonzero);
    }
}

#if defined(__aarch64__)
TEST_CASE("matvec_neon q4_k matches the scalar reference",
          "[ops][neon]") {
    constexpr std::uint32_t cols = 512;  // two Q4_K super-blocks
    constexpr std::uint32_t rows = 3;
    const std::size_t row_bytes = cols / 256 * 144;
    std::vector<std::byte> w(rows * row_bytes);
    std::mt19937 rng(1234);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::byte* row = w.data() + r * row_bytes;
        for (std::size_t b = 0; b < cols / 256; ++b) {
            std::byte* blk = row + b * 144;
            // Finite f16 scales; any bytes are a valid Q4_K block.
            std::uint16_t d = f32_to_f16(0.03f + 0.01f * r);
            std::uint16_t dmin = f32_to_f16(0.015f + 0.005f * b);
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &dmin, 2);
            for (std::size_t i = 4; i < 144; ++i) {
                blk[i] = std::byte{
                    static_cast<std::uint8_t>(rng() & 0xff)};
            }
        }
    }
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    Mat m{TensorType::kQ4_K, w.data(), rows, cols};
    std::vector<float> ref(rows), got(rows);
    matvec(m, x, ref);
    matvec_neon(m, x, got);
    for (std::uint32_t r = 0; r < rows; ++r) {
        INFO("row " << r);
        REQUIRE(got[r] == Approx(ref[r]).epsilon(1e-4).margin(1e-4));
    }
}
#endif
