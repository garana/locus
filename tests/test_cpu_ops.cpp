#include <cmath>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
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

TEST_CASE("f16 conversion rounds ties to even including subnormals",
          "[ops]") {
    // An exact tie must land on the neighbour with an even pattern,
    // which is what the GPU's packHalf2x16 does. Asserting "even"
    // rather than a reference value keeps this independent of any
    // second implementation.
    const float kSubUlp = std::ldexp(1.0f, -24);  // f16 subnormal ulp
    for (std::uint32_t k = 0; k < 1024; ++k) {
        const float tie = (static_cast<float>(k) + 0.5f) * kSubUlp;
        const std::uint16_t got = f32_to_f16(tie);
        INFO("subnormal tie k=" << k);
        REQUIRE((got == k || got == k + 1));  // a neighbour
        REQUIRE((got & 1u) == 0u);            // the even one
    }
    // The smallest disagreeing case before the fix: 1.5 ulp rounded
    // toward zero (0x0001) instead of to even (0x0002).
    REQUIRE(f32_to_f16(1.5f * kSubUlp) == 0x0002);

    // Away from a tie, subnormals still round to the nearer value.
    for (std::uint32_t k = 1; k < 64; ++k) {
        INFO("subnormal non-tie k=" << k);
        REQUIRE(f32_to_f16((static_cast<float>(k) + 0.25f) * kSubUlp) ==
                k);
        REQUIRE(f32_to_f16((static_cast<float>(k) + 0.75f) * kSubUlp) ==
                k + 1);
    }

    // Normal range, unchanged by the fix but worth pinning: halves
    // near 1.0 step by 2^-10, so these two are exact ties.
    REQUIRE(f32_to_f16(1.0f + std::ldexp(1.0f, -11)) == 0x3c00);
    REQUIRE(f32_to_f16(1.0f + 1.5f * std::ldexp(1.0f, -10)) == 0x3c02);

    // Signs are symmetric.
    REQUIRE(f32_to_f16(-1.5f * kSubUlp) == (0x8000 | 0x0002));
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

TEST_CASE("layernorm matches the closed form", "[ops][layernorm]") {
    const float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float w[] = {1.0f, 1.0f, 1.0f, 2.0f};
    float out[4];
    layernorm(x, w, 0.0f, out);
    // mean 2.5, variance ((1.5^2)*2 + (0.5^2)*2)/4 = 1.25.
    const float mean = 2.5f;
    const float scale = 1.0f / std::sqrt(1.25f);
    REQUIRE(out[0] == Approx((1.0f - mean) * scale));
    REQUIRE(out[1] == Approx((2.0f - mean) * scale));
    REQUIRE(out[2] == Approx((3.0f - mean) * scale));
    REQUIRE(out[3] == Approx((4.0f - mean) * scale * 2.0f));
    // Centered output sums to ~0 (before per-channel weights).
    float uw[4];
    const float ones[] = {1.0f, 1.0f, 1.0f, 1.0f};
    layernorm(x, ones, 0.0f, uw);
    REQUIRE(uw[0] + uw[1] + uw[2] + uw[3] == Approx(0.0f).margin(1e-6));
}

TEST_CASE("layernorm differs from rmsnorm on a non-zero mean",
          "[ops][layernorm]") {
    // A constant-offset input: rmsnorm keeps the offset, layernorm
    // removes it (mean-centering), so they must diverge.
    const float x[] = {5.0f, 5.0f, 5.0f, 5.0f};
    const float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float rn[4], ln[4];
    rmsnorm(x, w, 1e-5f, rn);
    layernorm(x, w, 1e-5f, ln);
    for (float v : ln) {
        REQUIRE(v == Approx(0.0f).margin(1e-3));  // centered -> ~0
    }
    REQUIRE(rn[0] > 0.9f);  // rms keeps the constant
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

TEST_CASE("bf16 matvec, batch matvec and dequant read the high half",
          "[ops]") {
    // bfloat16 is the top 16 bits of an f32: these values (and the
    // sign / exponent edge ones) are exact in bf16.
    const std::vector<float> wf = {1.5f, -2.0f, 0.15625f,
                                   65536.0f, -0.0f, 3.0e-39f};  // 2x3
    std::vector<std::uint16_t> wb(wf.size());
    for (std::size_t i = 0; i < wf.size(); ++i) {
        std::uint32_t u;
        std::memcpy(&u, &wf[i], 4);
        wb[i] = static_cast<std::uint16_t>(u >> 16);
    }
    Mat m{TensorType::kBF16, reinterpret_cast<const std::byte*>(wb.data()),
          2, 3};
    std::vector<float> row(3);
    for (std::uint32_t r = 0; r < 2; ++r) {
        dequant_row(m, r, row);
        for (std::uint32_t c = 0; c < 3; ++c) {
            float want = wf[r * 3 + c];
            std::uint32_t u;
            std::memcpy(&u, &want, 4);
            u &= 0xffff0000u;  // 3e-39 (subnormal) truncates
            std::memcpy(&want, &u, 4);
            REQUIRE(row[c] == want);
        }
    }
    const float x[] = {2.0f, 1.0f, -4.0f};
    float out[2];
    matvec(m, x, out);
    REQUIRE(out[0] == Approx(3.0f - 2.0f - 0.625f));
    REQUIRE(out[1] == Approx(131072.0f));
    // Two tokens through the batch path: rows x tokens, row-major.
    const float xs[] = {2.0f, 1.0f, -4.0f, 1.0f, 1.0f, 1.0f};
    float ob[4];
    matvec_batch_scalar(m, xs, ob, 2);
    REQUIRE(ob[0] == Approx(out[0]));
    REQUIRE(ob[1] == Approx(1.5f - 2.0f + 0.15625f));
    REQUIRE(ob[2] == Approx(out[1]));
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
            case TensorType::kIQ1_M: return {56, 0, -1};
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

// IQ4_NL is a 32-ELEMENT block (18B), not a 256 super-block, so it
// rides the Q4_0-style row path (dot_iq4_nl) rather than dot_k_quant.
// Check matvec == dequant_row dot (scalar) and, when present, that the
// CUDA kernel agrees.
TEST_CASE("IQ4_NL 32-elem matvec matches dequant_row and CUDA",
          "[ops][iq4nl]") {
    constexpr std::uint32_t rows = 3, nblk = 4, cols = nblk * 32;
    std::mt19937 rng(88);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = xd(rng);
    }
    std::vector<std::byte> w(static_cast<std::size_t>(rows) * nblk *
                             18);
    for (auto& b : w) {
        b = static_cast<std::byte>(byte_d(rng));
    }
    for (std::uint32_t i = 0; i < rows * nblk; ++i) {  // sane scales
        const std::uint16_t d = locus::backend::f32_to_f16(0.05f);
        std::memcpy(w.data() + i * 18, &d, 2);
    }
    Mat m{TensorType::kIQ4_NL, w.data(), rows, cols};
    std::vector<float> mv(rows), dq(cols);
    matvec(m, x, mv);
    for (std::uint32_t r = 0; r < rows; ++r) {
        dequant_row(m, r, dq);
        float ref = 0.0f;
        for (std::uint32_t i = 0; i < cols; ++i) {
            ref += dq[i] * x[i];
        }
        REQUIRE(mv[r] == Approx(ref).margin(1e-4));
    }
    if (cuda_backend_usable()) {
        cuda_pool_reset();
        std::vector<float> got(rows);
        matvec_cuda(m, x, got);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(got[r] ==
                    Approx(mv[r]).epsilon(1e-4).margin(1e-4));
        }
    }
}

// IQ1_M has no d field -- its f16 scale is assembled from the top
// nibbles of 4 scale u16s -- so random bytes could yield an inf/nan
// scale. Build a block with a sane assembled scale (0.05) and zero
// sub-scale bits, then check matvec == dequant_row dot and CUDA.
TEST_CASE("IQ1_M matvec matches dequant_row and CUDA",
          "[ops][iq1m]") {
    constexpr std::uint32_t rows = 3, nsb = 2, cols = nsb * 256;
    std::mt19937 rng(91);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = xd(rng);
    }
    const std::uint16_t T = locus::backend::f32_to_f16(0.05f);
    std::uint16_t sc[4];
    for (int k = 0; k < 4; ++k) {
        sc[k] = static_cast<std::uint16_t>(((T >> (4 * k)) & 0xf)
                                           << 12);
    }
    std::vector<std::byte> w(static_cast<std::size_t>(rows) * nsb *
                             56);
    for (auto& b : w) {
        b = static_cast<std::byte>(byte_d(rng));
    }
    for (std::uint32_t i = 0; i < rows * nsb; ++i) {
        std::memcpy(w.data() + i * 56 + 48, sc, 8);  // sane scale
    }
    Mat m{TensorType::kIQ1_M, w.data(), rows, cols};
    std::vector<float> mv(rows), dq(cols);
    matvec(m, x, mv);
    for (std::uint32_t r = 0; r < rows; ++r) {
        dequant_row(m, r, dq);
        float ref = 0.0f;
        for (std::uint32_t i = 0; i < cols; ++i) {
            ref += dq[i] * x[i];
        }
        REQUIRE(mv[r] == Approx(ref).margin(1e-4));
    }
    if (cuda_backend_usable()) {
        cuda_pool_reset();
        std::vector<float> got(rows);
        matvec_cuda(m, x, got);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(got[r] ==
                    Approx(mv[r]).epsilon(1e-4).margin(1e-4));
        }
    }
}

// R16 Q8_K activation-dot: matvec_q8k quantizes x to Q8_K and does an
// integer dot (ggml parity), matching llama.cpp bit-for-bit on the
// model goldens. Here we validate the integer dot TIGHTLY against an
// f32 dot that uses the SAME Q8_K-rounded activation (x round-tripped
// through Q8_K): the two are mathematically equal, differing only in
// summation order, so a port bug shows up immediately.
namespace {
std::vector<float> q8k_roundtrip(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (std::size_t b = 0; b * 256 < x.size(); ++b) {
        float amax = 0.0f, mx = 0.0f;
        for (int j = 0; j < 256; ++j) {
            const float a = std::abs(x[b * 256 + j]);
            if (a > amax) {
                amax = a;
                mx = x[b * 256 + j];
            }
        }
        if (amax == 0.0f) {
            continue;
        }
        const float iscale = -127.0f / mx, d = 1.0f / iscale;
        for (int j = 0; j < 256; ++j) {
            const int v = std::min(
                127, static_cast<int>(
                         std::lround(iscale * x[b * 256 + j])));
            y[b * 256 + j] = d * static_cast<float>(v);
        }
    }
    return y;
}
}  // namespace

TEST_CASE("Q8_K matvec matches the f32 dot of the Q8_K activation "
          "(K-quants)",
          "[ops][q8k]") {
    constexpr std::uint32_t rows = 4, cols = 512;  // 2 blocks/row
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    const std::vector<float> xq = q8k_roundtrip(x);
    for (TensorType t : {TensorType::kQ2_K, TensorType::kQ3_K,
                         TensorType::kQ4_K, TensorType::kQ5_K,
                         TensorType::kQ6_K, TensorType::kTQ1_0,
                         TensorType::kTQ2_0, TensorType::kIQ2_XXS,
                         TensorType::kIQ2_XS, TensorType::kIQ2_S,
                         TensorType::kIQ3_XXS, TensorType::kIQ3_S,
                         TensorType::kIQ1_S, TensorType::kIQ1_M,
                         TensorType::kIQ4_XS}) {
        std::vector<std::byte> w;
        for (std::uint32_t r = 0; r < rows; ++r) {
            auto row = k_quant_row(
                t, 2, 200u + r + 7u * static_cast<std::uint32_t>(t));
            w.insert(w.end(), row.begin(), row.end());
        }
        Mat m{t, w.data(), rows, cols};
        std::vector<float> ref(rows), got(rows);
        matvec(m, xq, ref);       // f32 dot, Q8_K-rounded activation
        matvec_q8k(m, x, got);    // integer dot, Q8_K activation
        bool nonzero = false;
        for (std::uint32_t r = 0; r < rows; ++r) {
            INFO("type " << static_cast<int>(t) << " row " << r);
            REQUIRE(got[r] ==
                    Approx(ref[r]).epsilon(1e-3).margin(1e-3));
            nonzero = nonzero || got[r] != 0.0f;
        }
        REQUIRE(nonzero);
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

TEST_CASE("conv1d dense matches a hand-computed same-padded case",
          "[ops][conv1d]") {
    // 1 -> 1 channel, k=3, stride 1, pad 1 (same length), filter
    // [1,0,-1] is a discrete first-difference.
    const float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float w[] = {1.0f, 0.0f, -1.0f};
    float out[4];
    conv1d(in, 1, 4, w, 1, 3, {}, 1, 1, 1, 1, out);
    const float want[] = {-2.0f, -2.0f, -2.0f, 3.0f};
    for (int i = 0; i < 4; ++i) {
        REQUIRE(out[i] == Approx(want[i]));
    }
}

TEST_CASE("conv1d depthwise convolves each channel with its own filter",
          "[ops][conv1d]") {
    // in_ch == out_ch == groups == 2, k=3, pad 1. ch0 filter is all
    // ones (box), ch1 filter [0,1,0] is identity.
    const float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float w[] = {1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    float out[6];
    conv1d(in, 2, 3, w, 2, 3, {}, 1, 1, 1, 2, out);
    const float want[] = {3.0f, 6.0f, 5.0f, 4.0f, 5.0f, 6.0f};
    for (int i = 0; i < 6; ++i) {
        REQUIRE(out[i] == Approx(want[i]));
    }
}

TEST_CASE("conv1d honors stride and dilation", "[ops][conv1d]") {
    // k=2, dilation 2 (taps span 3), stride 2, no pad: out_len 2.
    const float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float w[] = {1.0f, 1.0f};
    float out[2];
    conv1d(in, 1, 6, w, 1, 2, {}, 2, 2, 0, 1, out);
    REQUIRE(out[0] == Approx(4.0f));  // in[0] + in[2]
    REQUIRE(out[1] == Approx(8.0f));  // in[2] + in[4]
}

TEST_CASE("conv1d_transpose matches the scatter definition",
          "[ops][conv1d]") {
    // The hand case: 1 -> 1, k=2, stride 1. out = [a w0, a w1 + b w0,
    // b w1].
    const float in[] = {2.0f, 3.0f};
    const float w[] = {1.0f, 10.0f};
    float out[3];
    conv1d_transpose(in, 1, 2, w, 1, 2, {}, 1, 1, 0, 1, out);
    REQUIRE(out[0] == Approx(2.0f));
    REQUIRE(out[1] == Approx(23.0f));
    REQUIRE(out[2] == Approx(30.0f));

    // A SNAC-shaped upsample (k = 2*stride, pad = (k-stride)/2) scales
    // length by exactly the stride; cross-check the gather against an
    // independent scatter on random weights.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    const std::uint32_t ic = 2, oc = 2, k = 4, stride = 2, pad = 1,
                        ilen = 3;
    const std::uint32_t olen = (ilen - 1) * stride - 2 * pad + (k - 1) + 1;
    REQUIRE(olen == ilen * stride);
    std::vector<float> x(ic * ilen), wt(ic * oc * k);
    for (auto& v : x) v = d(rng);
    for (auto& v : wt) v = d(rng);
    std::vector<float> got(oc * olen), ref(oc * olen, 0.0f);
    conv1d_transpose(x, ic, ilen, wt, oc, k, {}, stride, 1, pad, 1, got);
    for (std::uint32_t c = 0; c < ic; ++c) {
        for (std::uint32_t ix = 0; ix < ilen; ++ix) {
            for (std::uint32_t t = 0; t < k; ++t) {
                const long ox = (long)ix * stride - (long)pad + (long)t;
                if (ox < 0 || ox >= (long)olen) continue;
                for (std::uint32_t o = 0; o < oc; ++o) {
                    ref[o * olen + ox] +=
                        x[c * ilen + ix] * wt[(c * oc + o) * k + t];
                }
            }
        }
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        REQUIRE(got[i] == Approx(ref[i]));
    }
}

TEST_CASE("snake1d applies the per-channel nonlinearity",
          "[ops][conv1d]") {
    // x + (1/(alpha+eps)) * sin(alpha*x)^2, one alpha per channel.
    std::vector<float> x = {0.0f, 1.0f, 2.0f, 0.5f, -1.0f, 3.0f};
    const float alpha[] = {1.0f, 2.0f};
    std::vector<float> want = x;
    for (std::uint32_t c = 0; c < 2; ++c) {
        const float a = alpha[c];
        for (std::uint32_t i = 0; i < 3; ++i) {
            const float v = x[c * 3 + i];
            const float s = std::sin(a * v);
            want[c * 3 + i] = v + (1.0f / (a + 1e-9f)) * s * s;
        }
    }
    snake1d(x, 2, 3, alpha);
    for (std::size_t i = 0; i < x.size(); ++i) {
        REQUIRE(x[i] == Approx(want[i]));
    }
}

TEST_CASE("conv1d_transpose grouped path matches the scatter definition",
          "[conv1d]") {
    // groups == 2: in-ch {0,1} -> out-ch {0}, {2,3} -> {1}. Exercises
    // the ic0 / ocl / weight[ic][ocl][k] indexing the groups==1 cases
    // leave degenerate. Weight is input-channel-first (ConvTranspose1d).
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    // oc = 4, groups = 2 -> ocg = 2, so the group-local output index
    // ocl takes both 0 and 1 (oc == 2 left it degenerate at 0).
    const std::uint32_t ic = 4, oc = 4, groups = 2, k = 3, stride = 2,
                        pad = 1, ilen = 4;
    const std::uint32_t icg = ic / groups, ocg = oc / groups;
    const std::uint32_t olen =
        conv1d_transpose_out_len(ilen, k, stride, 1, pad);
    std::vector<float> x(ic * ilen), wt(static_cast<std::size_t>(ic) *
                                        ocg * k);
    for (auto& v : x) v = d(rng);
    for (auto& v : wt) v = d(rng);
    std::vector<float> got(oc * olen), ref(oc * olen, 0.0f);
    conv1d_transpose(x, ic, ilen, wt, oc, k, {}, stride, 1, pad, groups,
                     got);
    for (std::uint32_t g = 0; g < groups; ++g) {
        for (std::uint32_t ci = 0; ci < icg; ++ci) {
            const std::uint32_t c = g * icg + ci;
            for (std::uint32_t ix = 0; ix < ilen; ++ix) {
                for (std::uint32_t t = 0; t < k; ++t) {
                    const long ox = (long)ix * stride - (long)pad + (long)t;
                    if (ox < 0 || ox >= (long)olen) continue;
                    for (std::uint32_t ol = 0; ol < ocg; ++ol) {
                        const std::uint32_t o = g * ocg + ol;
                        ref[o * olen + ox] +=
                            x[c * ilen + ix] * wt[(c * ocg + ol) * k + t];
                    }
                }
            }
        }
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        REQUIRE(got[i] == Approx(ref[i]));
    }
}

TEST_CASE("conv1d out_len helpers match the kernels' shapes",
          "[conv1d]") {
    // The shared shape definition #151/#152 also use.
    REQUIRE(conv1d_out_len(4, 3, 1, 1, 1) == 4);   // same padding
    REQUIRE(conv1d_out_len(6, 2, 2, 2, 0) == 2);   // stride+dilation
    REQUIRE(conv1d_out_len(3, 8, 1, 1, 0) == 0);   // kernel wider -> 0
    REQUIRE(conv1d_transpose_out_len(3, 4, 2, 1, 1) == 6);  // upsample x2
    REQUIRE(conv1d_transpose_out_len(2, 2, 1, 1, 0) == 3);
}

namespace {

/** Runs one conv (forward or transpose) on the GPU and on the scalar
 *  reference over random data and asserts they agree. */
void check_conv1d_cuda(bool transpose, std::uint32_t in_ch,
                       std::uint32_t in_len, std::uint32_t out_ch,
                       std::uint32_t k, std::uint32_t stride,
                       std::uint32_t dilation, std::uint32_t pad,
                       std::uint32_t groups, bool with_bias,
                       std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    const std::uint32_t icg = in_ch / groups, ocg = out_ch / groups;
    const std::uint32_t out_len =
        transpose
            ? conv1d_transpose_out_len(in_len, k, stride, dilation, pad)
            : conv1d_out_len(in_len, k, stride, dilation, pad);
    std::vector<float> in(static_cast<std::size_t>(in_ch) * in_len);
    // Forward weight is [out_ch][icg][k]; transpose is [in_ch][ocg][k].
    std::vector<float> w(transpose
                             ? static_cast<std::size_t>(in_ch) * ocg * k
                             : static_cast<std::size_t>(out_ch) * icg * k);
    std::vector<float> bias(with_bias ? out_ch : 0u);
    for (auto& v : in) v = d(rng);
    for (auto& v : w) v = d(rng);
    for (auto& v : bias) v = d(rng);
    std::vector<float> ref(static_cast<std::size_t>(out_ch) * out_len);
    std::vector<float> got(static_cast<std::size_t>(out_ch) * out_len);
    if (transpose) {
        conv1d_transpose(in, in_ch, in_len, w, out_ch, k, bias, stride,
                         dilation, pad, groups, ref);
        conv1d_transpose_cuda(in, in_ch, in_len, w, out_ch, k, bias, stride,
                              dilation, pad, groups, got);
    } else {
        conv1d(in, in_ch, in_len, w, out_ch, k, bias, stride, dilation, pad,
               groups, ref);
        conv1d_cuda(in, in_ch, in_len, w, out_ch, k, bias, stride, dilation,
                    pad, groups, got);
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        INFO("i=" << i);
        REQUIRE(got[i] == Approx(ref[i]).epsilon(1e-5).margin(1e-5));
    }
}

}  // namespace

// The threaded kernels sum each output in the scalar kernels' order, so
// they must match them bit for bit, not just within a tolerance. Shapes
// span several 1024-wide tiles (and a ragged last one), padding on both
// edges, groups and every SNAC upsample stride (i#173).
TEST_CASE("conv1d_mt and conv1d_transpose_mt are bit-identical to the "
          "scalar kernels",
          "[ops][conv1d]") {
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    auto fill = [&](std::size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = d(rng);
        return v;
    };
    struct C {
        bool transpose;
        std::uint32_t in_ch, in_len, out_ch, k, stride, dil, pad, groups;
        bool bias;
    };
    const C cases[] = {
        // conv1d: SNAC's depthwise k7 at dilations 1/3/9, the pointwise
        // 1x1, the 64 -> 1 tail, and a strided one (scalar fallback).
        {false, 8, 2500, 8, 7, 1, 1, 3, 8, true},
        {false, 8, 2500, 8, 7, 1, 3, 9, 8, true},
        {false, 8, 2500, 8, 7, 1, 9, 27, 8, true},
        {false, 16, 3000, 24, 1, 1, 1, 0, 1, true},
        {false, 64, 1100, 1, 7, 1, 1, 3, 1, true},
        {false, 6, 40, 4, 3, 1, 1, 1, 2, false},
        {false, 4, 300, 4, 3, 2, 1, 1, 1, true},
        // conv1d_transpose: k = 2s, pad = s/2 for s = 2/4/8, grouped,
        // no bias, and a dilated one (scalar fallback).
        {true, 6, 700, 4, 4, 2, 1, 1, 1, true},
        {true, 6, 300, 4, 8, 4, 1, 2, 1, true},
        {true, 6, 200, 4, 16, 8, 1, 4, 1, true},
        {true, 4, 900, 2, 3, 2, 1, 1, 2, false},
        // Several m-tiles per phase (m_end 2501, 3 tiles of 1024), so
        // the tile bounds run without the SNAC GGUFs.
        {true, 4, 2500, 2, 4, 2, 1, 1, 1, true},
        {true, 3, 50, 2, 5, 3, 1, 0, 1, true},
        {true, 2, 40, 2, 3, 2, 2, 1, 1, true},
    };
    for (const auto& c : cases) {
        INFO("transpose " << c.transpose << " in " << c.in_ch << "x"
                          << c.in_len << " out_ch " << c.out_ch << " k "
                          << c.k << " s " << c.stride << " d " << c.dil
                          << " p " << c.pad << " g " << c.groups);
        const auto in = fill(std::size_t{c.in_ch} * c.in_len);
        const auto bias = c.bias ? fill(c.out_ch) : std::vector<float>{};
        std::vector<float> ref, got;
        if (!c.transpose) {
            const auto w = fill(std::size_t{c.out_ch} *
                                (c.in_ch / c.groups) * c.k);
            const std::uint32_t ol = locus::backend::conv1d_out_len(
                c.in_len, c.k, c.stride, c.dil, c.pad);
            ref.assign(std::size_t{c.out_ch} * ol, 0.0f);
            got.assign(ref.size(), -1.0f);
            conv1d(in, c.in_ch, c.in_len, w, c.out_ch, c.k, bias, c.stride,
                   c.dil, c.pad, c.groups, ref);
            locus::backend::conv1d_mt(in, c.in_ch, c.in_len, w, c.out_ch,
                                      c.k, bias, c.stride, c.dil, c.pad,
                                      c.groups, got);
        } else {
            const auto w = fill(std::size_t{c.in_ch} *
                                (c.out_ch / c.groups) * c.k);
            const std::uint32_t ol =
                locus::backend::conv1d_transpose_out_len(
                    c.in_len, c.k, c.stride, c.dil, c.pad);
            ref.assign(std::size_t{c.out_ch} * ol, 0.0f);
            got.assign(ref.size(), -1.0f);
            conv1d_transpose(in, c.in_ch, c.in_len, w, c.out_ch, c.k, bias,
                             c.stride, c.dil, c.pad, c.groups, ref);
            locus::backend::conv1d_transpose_mt(
                in, c.in_ch, c.in_len, w, c.out_ch, c.k, bias, c.stride,
                c.dil, c.pad, c.groups, got);
        }
        REQUIRE(!ref.empty());
        // memcmp: bit-identical, and every output written (got starts
        // at -1, which no output here equals bit for bit by chance).
        REQUIRE(std::memcmp(ref.data(), got.data(),
                            ref.size() * sizeof(float)) == 0);
    }
}

TEST_CASE("snake1d_mt is bit-identical to snake1d", "[ops][conv1d]") {
    constexpr std::uint32_t ch = 5, len = 2600;
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> d(-3.0f, 3.0f);
    std::uniform_real_distribution<float> da(0.2f, 2.0f);
    std::vector<float> x(static_cast<std::size_t>(ch) * len), alpha(ch);
    for (auto& v : x) v = d(rng);
    for (auto& v : alpha) v = da(rng);
    std::vector<float> ref = x, got = x;
    snake1d(ref, ch, len, alpha);
    locus::backend::snake1d_mt(got, ch, len, alpha);
    REQUIRE(std::memcmp(ref.data(), got.data(),
                        ref.size() * sizeof(float)) == 0);
}

// i#222: the scalar conv2d is the oracle for every other variant, so
// it is pinned against hand-computed values first.
TEST_CASE("conv2d matches a hand-computed 3x3 same-padded case",
          "[ops][conv2d]") {
    // One 3x3 input channel, one 3x3 all-ones filter, padding 1: each
    // output is the sum of the input's in-bounds 3x3 neighbourhood.
    const std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<float> w(9, 1.0f);
    const std::vector<float> bias = {0.5f};
    std::vector<float> out(9);
    locus::backend::conv2d(in, 1, 3, 3, w, 1, 3, 3, bias, 1, 1, 1, out);
    const std::vector<float> want = {12.5f, 21.5f, 16.5f, 27.5f, 45.5f,
                                     33.5f, 24.5f, 39.5f, 28.5f};
    REQUIRE(out == want);

    // Stride 2, no padding, 2 groups of 1 in / 1 out channel, 2x2
    // filters (+1 and -1): each group reads only its own channel.
    // 2 channels of 3 rows x 4 columns.
    const std::vector<float> in2 = {1,  2,  3,  4,   // channel 0
                                    5,  6,  7,  8,
                                    9,  10, 11, 12,
                                    13, 14, 15, 16,  // channel 1
                                    0,  0,  0,  0,
                                    1,  1,  1,  1};
    const std::vector<float> w2 = {1, 1, 1, 1, -1, -1, -1, -1};
    std::vector<float> out2(2 * 1 * 2);  // out 1 x 2 (k2, s2, no pad)
    locus::backend::conv2d(in2, 2, 3, 4, w2, 2, 2, 2, {}, 2, 0, 2, out2);
    // Channel 0, rows 0-1: windows {1,2,5,6} = 14 and {3,4,7,8} = 22.
    // Channel 1, rows 0-1: -(13+14+0+0) and -(15+16+0+0).
    REQUIRE(out2 == std::vector<float>{14, 22, -27, -31});
}

// The threaded conv2d sums each output in the reference's order, so it
// must match bit for bit: memcmp, over decoder-shaped cases (3x3 and
// 1x1, padding on every edge, odd sizes, groups, no bias) and the
// stride-2 fallback.
TEST_CASE("conv2d_mt is bit-identical to conv2d", "[ops][conv2d]") {
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    auto fill = [&](std::size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = d(rng);
        return v;
    };
    struct C {
        std::uint32_t in_ch, in_h, in_w, out_ch, k, stride, pad, groups;
        bool bias;
    };
    const C cases[] = {
        {8, 17, 23, 12, 3, 1, 1, 1, true},   // 3x3 same
        {16, 9, 31, 8, 1, 1, 0, 1, true},    // 1x1
        {6, 12, 12, 6, 3, 1, 1, 3, false},   // grouped, no bias
        {4, 5, 7, 4, 3, 1, 0, 1, true},      // valid (no padding)
        {3, 40, 40, 5, 3, 1, 2, 1, true},    // padding wider than 1
        {4, 11, 13, 6, 3, 2, 1, 1, true},    // stride 2 (fallback)
    };
    for (const auto& c : cases) {
        INFO("in " << c.in_ch << "x" << c.in_h << "x" << c.in_w
                   << " out_ch " << c.out_ch << " k " << c.k << " s "
                   << c.stride << " p " << c.pad << " g " << c.groups);
        const auto in = fill(std::size_t{c.in_ch} * c.in_h * c.in_w);
        const auto w =
            fill(std::size_t{c.out_ch} * (c.in_ch / c.groups) * c.k * c.k);
        const auto bias = c.bias ? fill(c.out_ch) : std::vector<float>{};
        const std::uint32_t oh = locus::backend::conv1d_out_len(
            c.in_h, c.k, c.stride, 1, c.pad);
        const std::uint32_t ow = locus::backend::conv1d_out_len(
            c.in_w, c.k, c.stride, 1, c.pad);
        std::vector<float> ref(std::size_t{c.out_ch} * oh * ow, 0.0f);
        std::vector<float> got(ref.size(), -1.0f);
        locus::backend::conv2d(in, c.in_ch, c.in_h, c.in_w, w, c.out_ch,
                               c.k, c.k, bias, c.stride, c.pad, c.groups,
                               ref);
        locus::backend::conv2d_mt(in, c.in_ch, c.in_h, c.in_w, w,
                                  c.out_ch, c.k, c.k, bias, c.stride,
                                  c.pad, c.groups, got);
        REQUIRE(!ref.empty());
        REQUIRE(std::memcmp(ref.data(), got.data(),
                            ref.size() * sizeof(float)) == 0);
    }
}

TEST_CASE("conv1d_cuda matches the scalar reference", "[conv1d][cuda]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    // Dense same-padded 1 -> 1, the hand case's shape.
    check_conv1d_cuda(false, 1, 20, 1, 3, 1, 1, 1, 1, false, 1);
    // The SNAC workhorse: depthwise k7 at dilation 1, 3, 9, 'same'
    // padding = dilation * (k - 1) / 2 so length is preserved.
    check_conv1d_cuda(false, 8, 24, 8, 7, 1, 1, 3, 8, false, 2);
    check_conv1d_cuda(false, 8, 24, 8, 7, 1, 3, 9, 8, false, 3);
    check_conv1d_cuda(false, 8, 24, 8, 7, 1, 9, 27, 8, false, 4);
    // Pointwise k1 dense (channel mixing), and the final dense k7
    // 64 -> 1 reduction.
    check_conv1d_cuda(false, 16, 20, 32, 1, 1, 1, 0, 1, false, 5);
    check_conv1d_cuda(false, 64, 12, 1, 7, 1, 1, 3, 1, false, 6);
    // Stride + dilation, and a bias.
    check_conv1d_cuda(false, 1, 16, 1, 2, 2, 2, 0, 1, false, 7);
    check_conv1d_cuda(false, 4, 16, 6, 3, 1, 1, 1, 1, true, 8);
}

TEST_CASE("conv1d_transpose_cuda matches the scalar reference",
          "[conv1d][cuda]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    // SNAC upsample blocks: dense, k = 2*stride, pad = stride / 2, so
    // length scales by exactly the stride.
    for (std::uint32_t s : {2u, 4u, 8u}) {
        check_conv1d_cuda(true, 6, 10, 4, 2 * s, s, 1, s / 2, 1, false,
                          100 + s);
    }
    // Grouped transpose (groups == 2), the input-channel-first weight.
    check_conv1d_cuda(true, 4, 8, 2, 3, 2, 1, 1, 2, false, 120);
    // With a bias.
    check_conv1d_cuda(true, 2, 6, 2, 4, 2, 1, 1, 1, true, 121);
}

// conv2d_cuda sums each output in conv2d's order with no fused
// multiply-add, so on a host whose build has no FMA (the x86 baseline)
// it must match the reference bit for bit: memcmp. aarch64 and FMA x86
// builds fuse the scalar reference, so this holds only without FMA.
// Cases cover 3x3 and 1x1, a non-square kernel, padding on every edge
// and wider than 1, groups, stride 2, no bias, a kernel wider than the
// padded input (empty output), and a plane big enough for many blocks
// with a ragged tail.
TEST_CASE("conv2d_cuda is bit-identical to conv2d", "[conv2d][cuda]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    std::mt19937 rng(29);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    auto fill = [&](std::size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = d(rng);
        return v;
    };
    struct C {
        std::uint32_t in_ch, in_h, in_w, out_ch, kh, kw, stride, pad, groups;
        bool bias;
    };
    const C cases[] = {
        {8, 17, 23, 12, 3, 3, 1, 1, 1, true},     // 3x3 same
        {16, 9, 31, 8, 1, 1, 1, 0, 1, true},      // 1x1
        {6, 12, 12, 6, 3, 3, 1, 1, 3, false},     // grouped, no bias
        {6, 10, 10, 6, 3, 3, 1, 1, 6, true},      // depthwise
        {4, 5, 7, 4, 3, 3, 1, 0, 1, true},        // valid (no padding)
        {3, 40, 40, 5, 3, 3, 1, 2, 1, true},      // padding wider than 1
        {4, 11, 13, 6, 3, 3, 2, 1, 1, true},      // stride 2
        {5, 9, 14, 3, 2, 4, 1, 1, 1, true},       // non-square kernel
        {64, 24, 24, 32, 3, 3, 1, 1, 1, true},    // decoder-like width
        {3, 301, 299, 4, 3, 3, 1, 1, 1, true},    // many blocks, ragged
        {2, 2, 2, 3, 5, 5, 1, 0, 1, true},        // empty output
    };
    for (const auto& c : cases) {
        INFO("in " << c.in_ch << "x" << c.in_h << "x" << c.in_w
                   << " out_ch " << c.out_ch << " k " << c.kh << "x" << c.kw
                   << " s " << c.stride << " p " << c.pad << " g "
                   << c.groups);
        const auto in = fill(std::size_t{c.in_ch} * c.in_h * c.in_w);
        const auto w = fill(std::size_t{c.out_ch} * (c.in_ch / c.groups) *
                            c.kh * c.kw);
        const auto bias = c.bias ? fill(c.out_ch) : std::vector<float>{};
        const std::uint32_t oh = conv1d_out_len(c.in_h, c.kh, c.stride, 1,
                                                c.pad);
        const std::uint32_t ow = conv1d_out_len(c.in_w, c.kw, c.stride, 1,
                                                c.pad);
        std::vector<float> ref(std::size_t{c.out_ch} * oh * ow, 0.0f);
        std::vector<float> got(ref.size(), -1.0f);
        conv2d(in, c.in_ch, c.in_h, c.in_w, w, c.out_ch, c.kh, c.kw, bias,
               c.stride, c.pad, c.groups, ref);
        conv2d_cuda(in, c.in_ch, c.in_h, c.in_w, w, c.out_ch, c.kh, c.kw,
                    bias, c.stride, c.pad, c.groups, got);
        REQUIRE(std::memcmp(ref.data(), got.data(),
                            ref.size() * sizeof(float)) == 0);
    }
}

// A bad shape must throw, not reach the scalar fallback: conv2d only
// asserts its contract, so in Release stride 0 divides by zero and a
// short span reads or writes out of bounds. Runs on every build (the
// non-CUDA stub checks the same way), no device needed.
TEST_CASE("conv2d_cuda rejects bad shapes", "[conv2d]") {
    // A valid 2 -> 2 3x3 same-padded case on a 4x4 plane, then one
    // argument broken at a time.
    const std::vector<float> in(2 * 4 * 4, 1.0f), w(2 * 2 * 3 * 3, 1.0f),
        bias(2, 0.0f);
    std::vector<float> out(2 * 4 * 4);
    REQUIRE_NOTHROW(conv2d_cuda(in, 2, 4, 4, w, 2, 3, 3, bias, 1, 1, 1, out));
    // stride 0
    REQUIRE_THROWS_AS(
        conv2d_cuda(in, 2, 4, 4, w, 2, 3, 3, bias, 0, 1, 1, out),
        std::invalid_argument);
    // short weight span
    const std::span<const float> w_short(w.data(), w.size() - 1);
    REQUIRE_THROWS_AS(
        conv2d_cuda(in, 2, 4, 4, w_short, 2, 3, 3, bias, 1, 1, 1, out),
        std::invalid_argument);
    // groups 0, and groups not dividing the channels
    REQUIRE_THROWS_AS(
        conv2d_cuda(in, 2, 4, 4, w, 2, 3, 3, bias, 1, 1, 0, out),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        conv2d_cuda(in, 2, 4, 4, w, 2, 3, 3, bias, 1, 1, 3, out),
        std::invalid_argument);
    // short input, wrong-size bias, short output
    const std::span<const float> in_short(in.data(), in.size() - 1);
    REQUIRE_THROWS_AS(
        conv2d_cuda(in_short, 2, 4, 4, w, 2, 3, 3, bias, 1, 1, 1, out),
        std::invalid_argument);
    const std::vector<float> bias3(3, 0.0f);
    REQUIRE_THROWS_AS(
        conv2d_cuda(in, 2, 4, 4, w, 2, 3, 3, bias3, 1, 1, 1, out),
        std::invalid_argument);
    const std::span<float> out_short(out.data(), out.size() - 1);
    REQUIRE_THROWS_AS(
        conv2d_cuda(in, 2, 4, 4, w, 2, 3, 3, bias, 1, 1, 1, out_short),
        std::invalid_argument);
}

TEST_CASE("snake1d_cuda matches the scalar reference", "[conv1d][cuda]") {
    if (!cuda_backend_usable()) {
        SKIP("no CUDA device or non-CUDA build");
    }
    constexpr std::uint32_t ch = 5, len = 37;
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> d(-3.0f, 3.0f);
    std::uniform_real_distribution<float> da(0.2f, 2.0f);
    std::vector<float> x(static_cast<std::size_t>(ch) * len), alpha(ch);
    for (auto& v : x) v = d(rng);
    for (auto& v : alpha) v = da(rng);
    std::vector<float> ref = x, got = x;
    snake1d(ref, ch, len, alpha);
    snake1d_cuda(got, ch, len, alpha);
    for (std::size_t i = 0; i < got.size(); ++i) {
        INFO("i=" << i);
        REQUIRE(got[i] == Approx(ref[i]).epsilon(1e-5).margin(1e-5));
    }
}

TEST_CASE("matvec_q8k dots block-32 weights against a Q8_0 activation",
          "[ops]") {
    // llama.cpp's vec_dot_type for Q8_0 / Q4_0 / IQ4_NL is Q8_0: the
    // activation is quantized per 32 (d = amax / 127 as f16, quants
    // rounded to nearest even) and integer-dotted.
    std::mt19937 rng(73);
    std::uniform_int_distribution<int> qd(-127, 127), bd(0, 255);
    SECTION("an exactly representable activation matches the f32 dot") {
        // amax 127 per block: d = 1, so x quantizes losslessly.
        std::vector<float> x(96);
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = static_cast<float>(i % 32 == 0 ? 127 : qd(rng));
        }
        std::vector<std::byte> w(5 * 3 * 34);
        for (std::size_t b = 0; b < 15; ++b) {
            const std::uint16_t h = f32_to_f16(0.0625f);
            std::memcpy(&w[b * 34], &h, 2);
            for (int j = 0; j < 32; ++j) {
                w[b * 34 + 2 + j] = static_cast<std::byte>(
                    static_cast<std::int8_t>(qd(rng)));
            }
        }
        const Mat m{TensorType::kQ8_0, w.data(), 5, 96};
        std::vector<float> f32(5), q8(5);
        matvec(m, x, f32);
        matvec_q8k(m, x, q8);
        for (int r = 0; r < 5; ++r) {
            REQUIRE(q8[r] == f32[r]);  // small exact integers
        }
    }
    SECTION("quants round half to even") {
        // d = 1: 2.5 -> 2 and 0.5 -> 0 (round-away would give 3 and 1).
        std::vector<float> x(32, 0.0f);
        x[0] = 127.0f;
        x[1] = 2.5f;
        x[2] = 0.5f;
        std::vector<std::byte> w(34);
        const std::uint16_t one = f32_to_f16(1.0f);
        std::memcpy(w.data(), &one, 2);
        for (int j = 0; j < 32; ++j) {
            w[2 + j] = static_cast<std::byte>(1);
        }
        const Mat m{TensorType::kQ8_0, w.data(), 1, 32};
        float out = 0.0f;
        matvec_q8k(m, x, {&out, 1});
        REQUIRE(out == 129.0f);
    }
    SECTION("Q8_0, Q4_0 and IQ4_NL track the f32 dot") {
        std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
        std::vector<float> x(96);  // 3 blocks: the odd tail runs too
        for (auto& v : x) {
            v = xd(rng);
        }
        struct C {
            TensorType type;
            std::size_t block;
        };
        for (const C c : {C{TensorType::kQ8_0, 34}, C{TensorType::kQ4_0, 18},
                          C{TensorType::kIQ4_NL, 18}}) {
            INFO("type " << static_cast<int>(c.type));
            std::vector<std::byte> w(7 * 3 * c.block);
            for (auto& b : w) {
                b = static_cast<std::byte>(bd(rng));
            }
            for (std::size_t b = 0; b < 21; ++b) {
                const std::uint16_t h = f32_to_f16(0.01f);
                std::memcpy(&w[b * c.block], &h, 2);
            }
            const Mat m{c.type, w.data(), 7, 96};
            std::vector<float> f32(7), q8(7);
            matvec(m, x, f32);
            matvec_q8k(m, x, q8);
            for (int r = 0; r < 7; ++r) {
                // Q8_0 activation error: about amax / 254 per element,
                // so about 1% of a 96-term dot.
                REQUIRE(q8[r] == Approx(f32[r]).epsilon(0.01).margin(0.05));
            }
        }
    }
}
