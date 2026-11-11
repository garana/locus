// Device primitives of the CUDA forward (i#517): rmsnorm, RoPE and add
// equal the host ops bit for bit; the device matvecs equal matvec_cuda.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
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

/** A random-but-sane Q4_K row (scales bounded), as test_cpu_ops does. */
std::vector<std::byte> q4k_rows(std::uint32_t rows, std::uint32_t blocks,
                                std::uint32_t seed) {
    std::vector<std::byte> w(std::size_t{rows} * blocks * 144);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_d(0, 255);
    for (auto& b : w) {
        b = static_cast<std::byte>(byte_d(rng));
    }
    const std::uint16_t d = f32_to_f16(0.01f), dmin = f32_to_f16(0.005f);
    for (std::size_t b = 0; b < std::size_t{rows} * blocks; ++b) {
        std::memcpy(w.data() + b * 144, &d, 2);
        std::memcpy(w.data() + b * 144 + 2, &dmin, 2);
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
    cuda_pool_reset();
    constexpr std::uint32_t rows = 37, blocks = 8, cols = blocks * 256;
    const auto wb = q4k_rows(rows, blocks, 7);
    const Mat m{TensorType::kQ4_K, wb.data(), rows, cols};
    // 40 tokens: one 32-token batch launch and a partial one.
    constexpr std::uint32_t n = 40;
    const auto x = random_floats(std::size_t{n} * cols, 8);
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
    {
        Dev dx(x), dout(ref.size()), scratch(std::size_t{rows} * 32);
        REQUIRE(cuda_dev_matvec_batch(m, dx.p, dout.p, n, scratch.p));
        REQUIRE(cuda_dev_finish());
        REQUIRE(same_bits(dout.get(), ref));
    }
    cuda_pool_reset();
}
