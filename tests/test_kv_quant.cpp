#include <cmath>
#include <random>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/kv/kv_quant.hpp"
#include "locus/kv/paged_cache.hpp"

using locus::kv::KvType;
using locus::kv::kv_dequantize_row;
using locus::kv::kv_quantize_row;
using locus::kv::kv_row_bytes;
using locus::kv::PagedKvCache;

namespace {

/** Root-mean-square error between two equal-length rows. */
double rmse(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s / a.size());
}

std::vector<float> round_trip(const std::vector<float>& x,
                              KvType type) {
    std::vector<std::uint8_t> packed(kv_row_bytes(x.size(), type));
    kv_quantize_row(x.data(), x.size(), packed.data(), type);
    std::vector<float> out(x.size());
    kv_dequantize_row(packed.data(), x.size(), out.data(), type);
    return out;
}

}  // namespace

TEST_CASE("kv_row_bytes matches the block layout", "[kv]") {
    // 128 elements = 4 blocks of 32.
    REQUIRE(kv_row_bytes(128, KvType::kF32) == 128 * 4);
    REQUIRE(kv_row_bytes(128, KvType::kQ8) == 4 * (2 + 32));
    REQUIRE(kv_row_bytes(128, KvType::kQ4) == 4 * (2 + 16));
}

TEST_CASE("kv F32 round-trips exactly", "[kv]") {
    std::vector<float> x{0.5f, -1.25f, 3.0f, -0.0f};
    const auto y = round_trip(x, KvType::kF32);
    REQUIRE(y == x);
}

TEST_CASE("kv Q8 round-trip is accurate", "[kv]") {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> x(256);
    for (auto& v : x) {
        v = nd(rng);
    }
    const auto y = round_trip(x, KvType::kQ8);
    // Q8 keeps ~7 bits of mantissa per block; error << the signal.
    double amax = 0.0;
    for (float v : x) {
        amax = std::max(amax, static_cast<double>(std::fabs(v)));
    }
    REQUIRE(rmse(x, y) < amax / 100.0);
}

TEST_CASE("kv Q4 round-trip is bounded and coarser than Q8", "[kv]") {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(256);
    for (auto& v : x) {
        v = nd(rng);
    }
    const auto q4 = round_trip(x, KvType::kQ4);
    const auto q8 = round_trip(x, KvType::kQ8);
    // Both bounded; Q4 (4-bit) is no more accurate than Q8 (8-bit).
    REQUIRE(rmse(x, q4) < 0.5);
    REQUIRE(rmse(x, q8) <= rmse(x, q4) + 1e-6);
}

TEST_CASE("kv quant preserves the constant-zero row", "[kv]") {
    std::vector<float> z(64, 0.0f);
    for (KvType t : {KvType::kQ8, KvType::kQ4}) {
        const auto y = round_trip(z, t);
        for (float v : y) {
            REQUIRE(v == 0.0f);
        }
    }
}

TEST_CASE("kv quant handles a single dominant spike", "[kv]") {
    // One large element among small ones stresses the block scale.
    std::vector<float> x(32, 0.01f);
    x[5] = 9.0f;
    const auto y = round_trip(x, KvType::kQ8);
    REQUIRE(std::fabs(y[5] - 9.0f) < 0.1f);
}

TEST_CASE("quantized cache wraps external GPU-mapped byte storage",
          "[kv]") {
    PagedKvCache::Geometry geom;
    geom.n_layers = 2;
    geom.kv_dim = 64;  // 2 blocks of 32
    geom.block_tokens = 16;
    geom.n_blocks = 4;
    geom.kv_type = KvType::kQ8;

    // Caller-owned byte buffer (stands in for a GPU-mapped pool).
    std::vector<std::uint8_t> storage(PagedKvCache::pool_bytes(geom));
    PagedKvCache cache(geom, storage.data());
    REQUIRE(cache.quantized());
    REQUIRE(cache.qpool_data() == storage.data());
    // Strides are exposed for the backend/shader to index the pool.
    REQUIRE(cache.q_row_bytes() == kv_row_bytes(64, KvType::kQ8));
    REQUIRE(cache.q_layer_stride() ==
            geom.block_tokens * 2 * cache.q_row_bytes());
    REQUIRE(cache.q_block_stride() ==
            geom.n_layers * cache.q_layer_stride());

    // The CPU codec round-trips through the external buffer (no GPU
    // needed to validate the host seam).
    PagedKvCache::Seq seq;
    REQUIRE(cache.ensure_capacity(seq, 1));
    std::vector<float> k(64), v(64), out(64);
    for (int i = 0; i < 64; ++i) {
        k[i] = 0.1f * static_cast<float>(i - 32);
        v[i] = -0.05f * static_cast<float>(i);
    }
    cache.store_row(seq, /*layer=*/1, /*pos=*/0, /*value=*/false, k);
    cache.store_row(seq, 1, 0, true, v);
    // The write landed in the caller's buffer, not an internal pool.
    bool nonzero = false;
    for (std::uint8_t b : storage) {
        nonzero = nonzero || b != 0;
    }
    REQUIRE(nonzero);
    cache.load_row(seq, 1, 0, false, out);
    double err = 0.0;
    for (int i = 0; i < 64; ++i) {
        err = std::max(err, std::abs(double(out[i] - k[i])));
    }
    REQUIRE(err < 0.05);  // Q8 block round-trip
    cache.release(seq);
}

TEST_CASE("byte and float external ctors reject the wrong kv_type",
          "[kv]") {
    PagedKvCache::Geometry q;
    q.n_layers = 1;
    q.kv_dim = 32;
    q.block_tokens = 16;
    q.n_blocks = 1;
    q.kv_type = KvType::kQ4;
    std::vector<float> f(PagedKvCache::pool_floats(q));
    // A quantized geometry cannot take float external storage...
    REQUIRE_THROWS_AS(PagedKvCache(q, f.data()), std::invalid_argument);

    PagedKvCache::Geometry g = q;
    g.kv_type = KvType::kF32;
    std::vector<std::uint8_t> b(64);
    // ...and an F32 geometry cannot take byte external storage.
    REQUIRE_THROWS_AS(PagedKvCache(g, b.data()), std::invalid_argument);
}
