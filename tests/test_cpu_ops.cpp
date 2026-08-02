#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/variants.hpp"
#include "locus/sys/features.hpp"

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

TEST_CASE("matvec_batch_scalar: row-major layout, byte-identical "
          "to per-token matvec (R11 Ops contract)",
          "[ops][batch]") {
    constexpr std::uint32_t rows = 3, cols = 32, n = 4;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::vector<std::byte> w;
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<int> q(32);
        for (auto& v : q) {
            v = qd(rng);
        }
        auto row = q8_0_row(0.1f + 0.05f * static_cast<float>(r), q);
        w.insert(w.end(), row.begin(), row.end());
    }
    Mat m{TensorType::kQ8_0, w.data(), rows, cols};

    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::vector<float> xb(static_cast<std::size_t>(n) * cols);
    for (auto& v : xb) {
        v = xd(rng);
    }

    std::vector<float> got(static_cast<std::size_t>(rows) * n);
    matvec_batch_scalar(m, xb, got, n);

    // Output is row-major (out[r*n + t]) and each entry is bitwise a
    // matvec() of that row against token t.
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t t = 0; t < n; ++t) {
            float ref = 0.0f;
            matvec(mat_rows(m, r, 1),
                   {xb.data() + static_cast<std::size_t>(t) * cols,
                    cols},
                   {&ref, 1});
            REQUIRE(got[static_cast<std::size_t>(r) * n + t] == ref);
        }
    }
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
            case TensorType::kIQ2_XS: return {74, 0, -1};
            case TensorType::kIQ2_S: return {82, 0, -1};
            case TensorType::kIQ3_S: return {110, 0, -1};
            case TensorType::kIQ3_XXS: return {98, 0, -1};
            case TensorType::kIQ1_S: return {50, 0, -1};
            case TensorType::kTQ1_0: return {54, 52, -1};
            case TensorType::kTQ2_0: return {66, 64, -1};
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
          TensorType::kIQ2_XS, TensorType::kIQ2_S,
          TensorType::kIQ3_XXS, TensorType::kIQ3_S,
          TensorType::kIQ1_S, TensorType::kIQ4_XS,
          TensorType::kTQ1_0, TensorType::kTQ2_0}) {
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

// Hand-built golden blocks pin the ternary arithmetic to ggml's exact
// layout: TQ2_0 is 2 bits/elem (value (v-1)*d); TQ1_0 packs base-3
// digits per byte, digit n = ((uint8)(byte*pow3[n])*3)>>8 (uint8 wrap
// is load-bearing). Fill order is sequential, matching dequant_row.
TEST_CASE("ternary TQ1_0/TQ2_0 dequant matches the ggml layout",
          "[ops][ternary]") {
    using locus::backend::f32_to_f16;
    std::vector<float> y(256);
    // TQ2_0 (66B): qs[64] | d. qs[0]=0xE4 -> pairs l=0..3 = 0,1,2,3.
    {
        std::vector<std::byte> blk(66, std::byte{0});
        const std::uint16_t d = f32_to_f16(1.0f);
        std::memcpy(blk.data() + 64, &d, 2);
        blk[0] = std::byte{0xE4};
        Mat m{TensorType::kTQ2_0, blk.data(), 1, 256};
        dequant_row(m, 0, y);
        REQUIRE(y[0] == Approx(-1.0f));   // l=0 -> 0 -> -1
        REQUIRE(y[32] == Approx(0.0f));   // l=1 -> 1 ->  0
        REQUIRE(y[64] == Approx(1.0f));   // l=2 -> 2 ->  1
        REQUIRE(y[96] == Approx(2.0f));   // l=3 -> 3 ->  2
        REQUIRE(y[1] == Approx(-1.0f));   // qs[1]=0 -> -1
    }
    // TQ1_0 (54B): qs[48] | qh[4] | d.
    {
        std::vector<std::byte> blk(54, std::byte{0});
        const std::uint16_t d = f32_to_f16(1.0f);
        std::memcpy(blk.data() + 52, &d, 2);
        Mat m{TensorType::kTQ1_0, blk.data(), 1, 256};
        dequant_row(m, 0, y);
        for (float v : y) {
            REQUIRE(v == Approx(-1.0f));  // all-zero digits -> -d
        }
        // qs[0]=200: n=0 digit ((200*3)>>8)=2 -> +d at y[0];
        // n=1 digit ((uint8)(200*3)=88; (88*3)>>8=1) -> 0 at y[32].
        blk[0] = std::byte{200};
        dequant_row(m, 0, y);
        REQUIRE(y[0] == Approx(1.0f));
        REQUIRE(y[32] == Approx(0.0f));
        REQUIRE(y[1] == Approx(-1.0f));
    }
}

#if defined(__aarch64__)
TEST_CASE("matvec_neon matches the scalar reference for K-quants",
          "[ops][neon]") {
    constexpr std::uint32_t rows = 3, nblk = 2, cols = nblk * 256;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    for (TensorType t : {TensorType::kQ4_K, TensorType::kQ6_K,
                         TensorType::kIQ1_S}) {
        std::vector<std::byte> w;
        for (std::uint32_t r = 0; r < rows; ++r) {
            auto row = k_quant_row(
                t, nblk,
                1000u + r +
                    10u * static_cast<std::uint32_t>(t));
            w.insert(w.end(), row.begin(), row.end());
        }
        Mat m{t, w.data(), rows, cols};
        std::vector<float> ref(rows), got(rows);
        matvec(m, x, ref);        // scalar reference
        matvec_neon(m, x, got);   // vectorized kernel
        for (std::uint32_t r = 0; r < rows; ++r) {
            INFO("type " << static_cast<int>(t) << " row " << r);
            REQUIRE(got[r] ==
                    Approx(ref[r]).epsilon(1e-4).margin(1e-4));
        }
    }
}
#endif

// Guard on the arch only: matvec_sse4 is defined for all x86-64
// builds (matvec_sse4.cpp carries -msse4.1); __SSE4_1__ is NOT set
// in this TU, so guarding on it would silently drop the test. Gate
// execution at runtime instead, since a non-SSE4.1 CPU would fault.
#if defined(__x86_64__)
TEST_CASE("matvec_sse4 matches the scalar reference for K-quants",
          "[ops][sse4]") {
    if (!locus::sys::detect().sse4) {
        SKIP("CPU lacks SSE4.1");
    }
    constexpr std::uint32_t rows = 3, nblk = 2, cols = nblk * 256;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    // sse4 covers Q4_K/Q6_K (no IQ1_S kernel there yet).
    for (TensorType t : {TensorType::kQ4_K, TensorType::kQ6_K}) {
        std::vector<std::byte> w;
        for (std::uint32_t r = 0; r < rows; ++r) {
            auto row = k_quant_row(
                t, nblk,
                1000u + r +
                    10u * static_cast<std::uint32_t>(t));
            w.insert(w.end(), row.begin(), row.end());
        }
        Mat m{t, w.data(), rows, cols};
        std::vector<float> ref(rows), got(rows);
        matvec(m, x, ref);        // scalar reference
        matvec_sse4(m, x, got);   // vectorized kernel
        for (std::uint32_t r = 0; r < rows; ++r) {
            INFO("type " << static_cast<int>(t) << " row " << r);
            REQUIRE(got[r] ==
                    Approx(ref[r]).epsilon(1e-4).margin(1e-4));
        }
    }
}

// R11: the register-blocked batch kernel must be BIT-identical to n
// per-token matvec_sse4() calls, in row-major layout (out[r*n + t]).
TEST_CASE("matvec_batch_sse4 Q4_K/Q6_K byte-identical to per-token "
          "matvec_sse4 (R11)",
          "[ops][batch][sse4]") {
    if (!locus::sys::detect().sse4) {
        SKIP("CPU lacks SSE4.1");
    }
    constexpr std::uint32_t rows = 3, nblk = 2, cols = nblk * 256,
                            n = 5;
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::vector<float> xb(static_cast<std::size_t>(n) * cols);
    for (auto& v : xb) {
        v = xd(rng);
    }
    for (TensorType t : {TensorType::kQ4_K, TensorType::kQ6_K}) {
        std::vector<std::byte> w;
        for (std::uint32_t r = 0; r < rows; ++r) {
            auto row = k_quant_row(
                t, nblk,
                500u + r + 7u * static_cast<std::uint32_t>(t));
            w.insert(w.end(), row.begin(), row.end());
        }
        Mat m{t, w.data(), rows, cols};
        std::vector<float> got(static_cast<std::size_t>(rows) * n);
        matvec_batch_sse4(m, xb, got, n);
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (std::uint32_t tk = 0; tk < n; ++tk) {
                float ref = 0.0f;
                matvec_sse4(mat_rows(m, r, 1),
                            {xb.data() +
                                 static_cast<std::size_t>(tk) * cols,
                             cols},
                            {&ref, 1});
                INFO("type " << static_cast<int>(t) << " row " << r
                             << " tok " << tk);
                REQUIRE(got[static_cast<std::size_t>(r) * n + tk] ==
                        ref);  // exact, not Approx
            }
        }
    }
}
#endif

// Per-token CUDA device kernels vs the scalar reference, across every
// GPU-handled K-/IQ-quant. The IQ/Q2_K kernels port dequant_block_*
// (grids + signs + codebook) onto the device; each materializes the
// super-block then dots i=0..255 matching dot_k_quant. Runtime-gated
// on a usable CUDA device (SKIP on non-CUDA builds / no GPU).
TEST_CASE("matvec_cuda matches the scalar reference for K-/IQ-quants",
          "[ops][cuda]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    cuda_pool_reset();  // drop any stale pages from prior test cases
    constexpr std::uint32_t rows = 3, nblk = 2, cols = nblk * 256;
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    for (TensorType t :
         {TensorType::kQ2_K, TensorType::kQ4_K, TensorType::kQ5_K,
          TensorType::kQ6_K, TensorType::kIQ1_S,
          TensorType::kIQ2_XXS, TensorType::kIQ2_XS,
          TensorType::kIQ2_S, TensorType::kIQ3_XXS,
          TensorType::kIQ3_S, TensorType::kIQ4_XS,
          TensorType::kTQ1_0, TensorType::kTQ2_0}) {
        std::vector<std::byte> w;
        for (std::uint32_t r = 0; r < rows; ++r) {
            auto row = k_quant_row(
                t, nblk,
                2000u + r +
                    13u * static_cast<std::uint32_t>(t));
            w.insert(w.end(), row.begin(), row.end());
        }
        Mat m{t, w.data(), rows, cols};
        std::vector<float> ref(rows), got(rows);
        matvec(m, x, ref);       // scalar reference
        // Reset per type: w's address is recycled across iterations
        // and some types share a byte size (TQ2_0 == IQ2_XXS = 396B),
        // which the pool's size-check alone would not disambiguate.
        cuda_pool_reset();
        matvec_cuda(m, x, got);  // device kernel
        for (std::uint32_t r = 0; r < rows; ++r) {
            INFO("type " << static_cast<int>(t) << " row " << r);
            REQUIRE(got[r] ==
                    Approx(ref[r]).epsilon(1e-4).margin(1e-4));
        }
    }
}

// R11 CUDA: the one-launch batched kernel must be BIT-identical to n
// per-token matvec_cuda() calls (row-major out[r*n + t]). Not
// arch-guarded -- gated at runtime on a usable CUDA device.
TEST_CASE("matvec_batch_cuda Q4_K/Q6_K byte-identical to per-token "
          "matvec_cuda (R11)",
          "[ops][batch][cuda]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    cuda_pool_reset();  // drop any stale pages from prior test cases
    constexpr std::uint32_t rows = 3, nblk = 2, cols = nblk * 256,
                            n = 5;
    std::mt19937 rng(41);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::vector<float> xb(static_cast<std::size_t>(n) * cols);
    for (auto& v : xb) {
        v = xd(rng);
    }
    for (TensorType t : {TensorType::kQ4_K, TensorType::kQ6_K}) {
        std::vector<std::byte> w;
        for (std::uint32_t r = 0; r < rows; ++r) {
            auto row = k_quant_row(
                t, nblk,
                300u + r + 9u * static_cast<std::uint32_t>(t));
            w.insert(w.end(), row.begin(), row.end());
        }
        Mat m{t, w.data(), rows, cols};
        std::vector<float> got(static_cast<std::size_t>(rows) * n);
        matvec_batch_cuda(m, xb, got, n);
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (std::uint32_t tk = 0; tk < n; ++tk) {
                float ref = 0.0f;
                matvec_cuda(mat_rows(m, r, 1),
                            {xb.data() +
                                 static_cast<std::size_t>(tk) * cols,
                             cols},
                            {&ref, 1});
                INFO("type " << static_cast<int>(t) << " row " << r
                             << " tok " << tk);
                REQUIRE(got[static_cast<std::size_t>(r) * n + tk] ==
                        ref);  // exact
            }
        }
    }
}

#if defined(__aarch64__)
TEST_CASE("matvec_batch_neon is byte-identical to per-token neon",
          "[ops][batch][neon]") {
    // R11 register-blocked kernel must be bit-for-bit equal to n
    // separate matvec_neon() calls. Q4_K/Q6_K exercise the
    // register-blocked paths; Q5_K exercises the per-token fallback
    // + row-major scatter.
    constexpr std::uint32_t rows = 5, blocks = 2,
                            cols = blocks * 256, n = 6;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::vector<float> xb(static_cast<std::size_t>(n) * cols);
    for (auto& v : xb) {
        v = xd(rng);
    }
    for (TensorType t : {TensorType::kQ4_K, TensorType::kQ6_K,
                         TensorType::kQ5_K}) {
        std::vector<std::byte> w;
        for (std::uint32_t r = 0; r < rows; ++r) {
            auto row = k_quant_row(t, blocks, 100 + r);
            w.insert(w.end(), row.begin(), row.end());
        }
        Mat m{t, w.data(), rows, cols};
        std::vector<float> got(static_cast<std::size_t>(rows) * n);
        matvec_batch_neon(m, xb, got, n);
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (std::uint32_t tk = 0; tk < n; ++tk) {
                float ref = 0.0f;
                matvec_neon(
                    mat_rows(m, r, 1),
                    {xb.data() +
                         static_cast<std::size_t>(tk) * cols,
                     cols},
                    {&ref, 1});
                INFO("type " << static_cast<int>(t) << " r=" << r
                             << " t=" << tk);
                REQUIRE(got[static_cast<std::size_t>(r) * n + tk] ==
                        ref);
            }
        }
    }
}
#endif
