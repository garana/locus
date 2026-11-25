// Device primitives of the CUDA forward (i#517): rmsnorm, RoPE and add
// equal the host ops bit for bit; the device matvecs equal matvec_cuda.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/variants.hpp"

using namespace locus::backend;
using locus::gguf::TensorType;

namespace {

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) !=
            std::bit_cast<std::uint32_t>(b[i])) {
            return false;
        }
    }
    return true;
}

std::vector<float> random_floats(std::size_t n, std::uint32_t seed,
                                 float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = d(rng);
    }
    return v;
}

/** A device buffer for one test, freed at scope exit. */
struct Dev {
    float* p = nullptr;
    std::size_t n = 0;
    explicit Dev(std::size_t count) : p(cuda_dev_alloc(count)), n(count) {
        REQUIRE(p != nullptr);
    }
    Dev(const std::vector<float>& h) : Dev(h.size()) {
        REQUIRE(cuda_dev_upload(p, h.data(), n));
    }
    ~Dev() { cuda_dev_free(p); }
    std::vector<float> get() const {
        std::vector<float> h(n);
        REQUIRE(cuda_dev_download(h.data(), p, n));
        return h;
    }
};

/**
 * Random-but-sane rows of a Q4_K, Q6_K, Q8_0, IQ4_NL or IQ4_XS weight
 * (random bytes, bounded f16 scales), as test_cpu_ops does.
 * @param cols A multiple of the type's block size (256, or 32 for
 *     Q8_0 and IQ4_NL).
 */
std::vector<std::byte> quant_rows(TensorType t, std::uint32_t rows,
                                  std::uint32_t cols, std::uint32_t seed) {
    const std::size_t bytes = t == TensorType::kQ4_K    ? 144
                              : t == TensorType::kQ6_K   ? 210
                              : t == TensorType::kQ8_0   ? 34
                              : t == TensorType::kIQ4_XS ? 136
                                                         : 18;
    const std::uint32_t per =
        t == TensorType::kIQ4_NL || t == TensorType::kQ8_0 ? 32 : 256;
    const std::size_t blocks = std::size_t{rows} * (cols / per);
    std::vector<std::byte> w(blocks * bytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_d(0, 255);
    for (auto& b : w) {
        b = static_cast<std::byte>(byte_d(rng));
    }
    const std::uint16_t d = f32_to_f16(0.01f), dmin = f32_to_f16(0.005f);
    for (std::size_t b = 0; b < blocks; ++b) {
        if (t == TensorType::kQ6_K) {
            // ql, qh and the int8 scales are any bytes; d comes last.
            std::memcpy(w.data() + b * bytes + 208, &d, 2);
            continue;
        }
        std::memcpy(w.data() + b * bytes, &d, 2);
        if (t == TensorType::kQ4_K) {
            std::memcpy(w.data() + b * bytes + 2, &dmin, 2);
        }
    }
    return w;
}

}  // namespace

TEST_CASE("cuda device rmsnorm, rope and add equal the host ops",
          "[cuda][device]") {
    if (!cuda_backend_usable()) {
        SKIP("no usable CUDA device");
    }
    constexpr std::uint32_t n = 2048, tokens = 3;
    const auto x = random_floats(std::size_t{n} * tokens, 1, 3.0f);
    const auto w = random_floats(n, 2);
    {
        Dev dx(x), dw(w), dout(x.size());
        REQUIRE(cuda_dev_rmsnorm(dx.p, dw.p, 1e-5f, dout.p, n, tokens));
        REQUIRE(cuda_dev_finish());
        std::vector<float> ref(x.size());
        for (std::uint32_t t = 0; t < tokens; ++t) {
            rmsnorm({x.data() + t * n, n}, w, 1e-5f,
                    {ref.data() + t * n, n});
        }
        REQUIRE(same_bits(dout.get(), ref));
    }
    {
        // 32 heads of 64 at three positions, llama3-style factors.
        constexpr std::uint32_t heads = 32, hd = 64;
        const auto q = random_floats(std::size_t{heads} * hd * tokens, 3);
        std::vector<float> factors(hd / 2);
        for (std::uint32_t i = 0; i < hd / 2; ++i) {
            factors[i] = 1.0f + 0.25f * static_cast<float>(i % 5);
        }
        const std::uint32_t pos[tokens] = {0, 17, 4093};
        std::vector<float> cs(std::size_t{hd} * tokens);
        std::vector<float> ref = q;
        for (std::uint32_t t = 0; t < tokens; ++t) {
            rope_norm_table(hd, pos[t], 500000.0f, factors,
                            {cs.data() + t * hd, hd});
            rope_norm({ref.data() + std::size_t{t} * heads * hd, heads * hd},
                      heads, hd, pos[t], 500000.0f, factors);
        }
        Dev dq(q), dcs(cs);
        REQUIRE(cuda_dev_rope(dq.p, dcs.p, heads, hd, heads * hd, tokens));
        REQUIRE(cuda_dev_finish());
        REQUIRE(same_bits(dq.get(), ref));
    }
    {
        const auto y = random_floats(x.size(), 4);
        Dev dx(x), dy(y);
        REQUIRE(cuda_dev_add(dx.p, dy.p, x.size()));
        REQUIRE(cuda_dev_finish());
        std::vector<float> ref = x;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            ref[i] += y[i];
        }
        REQUIRE(same_bits(dx.get(), ref));
    }
    {
        // silu_mul uses the device expf: close to the host's, not equal.
        const auto g = random_floats(8192, 5, 6.0f);
        const auto u = random_floats(8192, 6);
        Dev dg(g), du(u);
        REQUIRE(cuda_dev_silu_mul(dg.p, du.p, g.size()));
        REQUIRE(cuda_dev_finish());
        std::vector<float> ref(g.size());
        silu_mul(g, u, ref);
        const auto got = dg.get();
        for (std::size_t i = 0; i < ref.size(); ++i) {
            REQUIRE(got[i] == Catch::Approx(ref[i]).epsilon(1e-6).margin(1e-7));
        }
    }
}

TEST_CASE("cuda device matvecs equal matvec_cuda", "[cuda][device]") {
    if (!cuda_backend_usable()) {
        SKIP("no usable CUDA device");
    }
    // 37 rows; 2048 columns (IQ4_NL: two full 32-block passes).
    constexpr std::uint32_t rows = 37, cols = 8 * 256;
    // 40 tokens: one 32-token batch launch and a partial one.
    constexpr std::uint32_t n = 40;
    const auto x = random_floats(std::size_t{n} * cols, 8);
    for (const TensorType type :
         {TensorType::kQ4_K, TensorType::kQ6_K, TensorType::kQ8_0,
          TensorType::kIQ4_NL, TensorType::kIQ4_XS}) {
        INFO("type " << static_cast<int>(type));
        // w's address may repeat across types; the pool keys by address.
        cuda_pool_reset();
        const auto wb = quant_rows(type, rows, cols, 7);
        const Mat m{type, wb.data(), rows, cols};
        std::vector<float> ref(std::size_t{n} * rows);
        for (std::uint32_t t = 0; t < n; ++t) {
            matvec_cuda(m, {x.data() + std::size_t{t} * cols, cols},
                        {ref.data() + std::size_t{t} * rows, rows});
        }
        {
            Dev dx(x), dout(ref.size());
            for (std::uint32_t t = 0; t < n; ++t) {
                REQUIRE(cuda_dev_matvec(m, dx.p + std::size_t{t} * cols,
                                        dout.p + std::size_t{t} * rows));
            }
            REQUIRE(cuda_dev_finish());
            REQUIRE(same_bits(dout.get(), ref));
        }
        // 13 and 31 tokens too: the Q4_K and Q6_K prefill tile (8 tokens
        // a warp, i#1195) then ends in a partial tile, as 37 rows do.
        // From 32 tokens the tile types run a GEMM (dequantize plus
        // cuBLAS, i#1200) where cuBLAS is built in: its own sum order,
        // so close to the per-token kernel, not its bits (the random
        // rows cancel to small sums, so about 1e-4 apart; a wrong
        // dequantize would be about 1 apart).
        const bool gemm_type = type == TensorType::kQ4_K ||
                               type == TensorType::kQ6_K ||
                               type == TensorType::kQ8_0;
        for (const std::uint32_t nb : {n, 13u, 31u}) {
            INFO("tokens " << nb);
            const std::vector<float> xb(x.begin(),
                                        x.begin() + std::size_t{nb} * cols);
            const std::vector<float> want(ref.begin(),
                                          ref.begin() + std::size_t{nb} * rows);
            Dev dx(xb), dout(want.size()), scratch(std::size_t{rows} * 32);
            REQUIRE(cuda_dev_matvec_batch(m, dx.p, dout.p, nb, scratch.p));
            REQUIRE(cuda_dev_finish());
            const auto got = dout.get();
            if (gemm_type && nb >= 32) {
                for (std::size_t i = 0; i < got.size(); ++i) {
                    REQUIRE(got[i] ==
                            Catch::Approx(want[i]).epsilon(1e-3).margin(1e-3));
                }
            } else {
                REQUIRE(same_bits(got, want));
            }
        }
    }
    cuda_pool_reset();
}

TEST_CASE("cuda prefill GEMMs on two threads keep their own weights",
          "[cuda][device]") {
    if (!cuda_backend_usable()) {
        SKIP("no usable CUDA device");
    }
    // i#1200: each GEMM dequantizes its weight into its own call's
    // scratch, so two threads at once (two engines) each get their own
    // product, the bits of a run alone. With one shared buffer, one
    // thread's GEMM could read the other's weight.
    constexpr std::uint32_t rows = 256, cols = 8 * 256, n = 48;
    const auto x = random_floats(std::size_t{n} * cols, 21);
    std::vector<std::vector<std::byte>> wb;
    std::vector<std::vector<float>> alone(2);
    for (std::uint32_t k = 0; k < 2; ++k) {
        wb.push_back(quant_rows(TensorType::kQ4_K, rows, cols, 30 + k));
    }
    const auto run = [&](std::uint32_t k) {
        const Mat m{TensorType::kQ4_K, wb[k].data(), rows, cols};
        Dev dx(x), dout(std::size_t{n} * rows),
            scratch(std::size_t{rows} * 32);
        bool ok = cuda_dev_matvec_batch(m, dx.p, dout.p, n, scratch.p) &&
                  cuda_dev_finish();
        return ok ? dout.get() : std::vector<float>{};
    };
    for (std::uint32_t k = 0; k < 2; ++k) {
        alone[k] = run(k);
        REQUIRE(alone[k].size() == std::size_t{n} * rows);
    }
    REQUIRE_FALSE(same_bits(alone[0], alone[1]));  // two distinct weights
    for (int rep = 0; rep < 20; ++rep) {
        std::vector<float> got[2];
        std::thread t1([&] { got[1] = run(1); });
        got[0] = run(0);
        t1.join();
        INFO("rep " << rep);
        REQUIRE(same_bits(got[0], alone[0]));
        REQUIRE(same_bits(got[1], alone[1]));
    }
    cuda_pool_reset();
}
