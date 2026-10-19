#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/vulkan/context.hpp"

using locus::backend::vk::Kernel;
using locus::backend::vk::VulkanContext;

namespace {

std::uint32_t fbits(float v) { return std::bit_cast<std::uint32_t>(v); }

/** Uploads host data into a fresh buffer (at least 4 bytes). */
template <typename T>
VulkanContext::Buffer upload(VulkanContext& ctx, const std::vector<T>& v) {
    auto b = ctx.create_buffer(std::max<std::size_t>(v.size() * sizeof(T),
                                                     4));
    ctx.write_buffer(b, std::as_bytes(std::span<const T>(v)));
    return b;
}

std::vector<float> download(VulkanContext& ctx, VulkanContext::Buffer b,
                            std::size_t n) {
    std::vector<float> out(n);
    ctx.read_buffer(b, std::as_writable_bytes(std::span<float>(out)));
    return out;
}

std::vector<float> random_floats(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& f : v) {
        f = dist(rng);
    }
    return v;
}

/** A shuffled block table covering n_blocks pool blocks. */
std::vector<std::uint32_t> shuffled_table(std::uint32_t n_blocks,
                                          std::mt19937& rng) {
    std::vector<std::uint32_t> t(n_blocks);
    std::iota(t.begin(), t.end(), 0u);
    std::shuffle(t.begin(), t.end(), rng);
    return t;
}

/** Softmax-weighted sum reference: scores in, weights out. */
void softmax(std::vector<double>& s) {
    const double mx = *std::max_element(s.begin(), s.end());
    double sum = 0.0;
    for (auto& v : s) {
        v = std::exp(v - mx);
        sum += v;
    }
    for (auto& v : s) {
        v /= sum;
    }
}

}  // namespace

TEST_CASE("vulkan paged attention matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(11);

    // head_dim 160 > the 128-thread workgroup exercises the
    // one-slice path; n_pos 300 > 128 the strided position loops.
    const std::uint32_t head_dims[] = {64, 160};
    const std::uint32_t n_poss[] = {1, 37, 300};
    for (std::uint32_t head_dim : head_dims) {
        for (std::uint32_t n_pos : n_poss) {
            DYNAMIC_SECTION("head_dim " << head_dim << " n_pos "
                                        << n_pos) {
                const std::uint32_t n_heads = 4, n_kv = 2;
                const std::uint32_t group = n_heads / n_kv;
                const std::uint32_t kv_dim = n_kv * head_dim;
                const std::uint32_t bt = 16, n_layers = 2, layer = 1;
                const std::uint32_t layer_stride = bt * kv_dim * 2;
                const std::uint32_t block_stride =
                    n_layers * layer_stride;
                const std::uint32_t n_blocks = (n_pos + bt - 1) / bt;

                auto q = random_floats(n_heads * head_dim, rng);
                auto pool = random_floats(
                    static_cast<std::size_t>(n_blocks) * block_stride,
                    rng);
                auto table = shuffled_table(n_blocks, rng);

                auto kv_at = [&](std::uint32_t t, bool v,
                                 std::uint32_t i) {
                    return pool[table[t / bt] * block_stride +
                                layer * layer_stride +
                                (v ? layer_stride / 2 : 0) +
                                (t % bt) * kv_dim + i];
                };
                std::vector<float> want(n_heads * head_dim);
                for (std::uint32_t h = 0; h < n_heads; ++h) {
                    const std::uint32_t kvh = (h / group) * head_dim;
                    std::vector<double> s(n_pos);
                    for (std::uint32_t t = 0; t < n_pos; ++t) {
                        double d = 0.0;
                        for (std::uint32_t i = 0; i < head_dim; ++i) {
                            d += q[h * head_dim + i] *
                                 kv_at(t, false, kvh + i);
                        }
                        s[t] = d / std::sqrt(double(head_dim));
                    }
                    softmax(s);
                    for (std::uint32_t i = 0; i < head_dim; ++i) {
                        double acc = 0.0;
                        for (std::uint32_t t = 0; t < n_pos; ++t) {
                            acc += s[t] * kv_at(t, true, kvh + i);
                        }
                        want[h * head_dim + i] =
                            static_cast<float>(acc);
                    }
                }

                auto qb = upload(ctx, q);
                auto pb = upload(ctx, pool);
                auto tb = upload(ctx, table);
                auto ab = ctx.create_buffer(n_heads * n_pos * 4);
                auto ob = ctx.create_buffer(n_heads * head_dim * 4);
                ctx.begin_batch();
                const VulkanContext::Buffer bufs[] = {qb, pb, tb, ab,
                                                      ob};
                const std::uint32_t push[] = {
                    n_heads,      group,
                    head_dim,     kv_dim,
                    n_pos,        bt,
                    block_stride, layer * layer_stride,
                    layer_stride / 2};
                ctx.dispatch(Kernel::kAttnPaged, bufs, push, n_heads);
                ctx.end_batch();
                auto got = download(ctx, ob, want.size());
                for (std::size_t i = 0; i < want.size(); ++i) {
                    REQUIRE(got[i] == Catch::Approx(want[i])
                                          .margin(1e-5)
                                          .epsilon(1e-4));
                }
                for (auto b : {qb, pb, tb, ab, ob}) {
                    ctx.destroy_buffer(b);
                }
            }
        }
    }
}

TEST_CASE("vulkan MLA attention matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(13);

    // DeepSeek-V2-Lite geometry: rank 512 > the workgroup width.
    const std::uint32_t n_poss[] = {5, 200};
    for (std::uint32_t n_pos : n_poss) {
        DYNAMIC_SECTION("n_pos " << n_pos) {
            const std::uint32_t n_heads = 3, rank = 512, nope = 128,
                                rope = 64, qk = nope + rope;
            const std::uint32_t row = rank + rope;
            const std::uint32_t bt = 16, n_layers = 2, layer = 1;
            const std::uint32_t layer_stride = bt * row;
            const std::uint32_t block_stride = n_layers * layer_stride;
            const std::uint32_t n_blocks = (n_pos + bt - 1) / bt;
            const float kq_scale = 0.07f;

            auto q = random_floats(n_heads * qk, rng);
            auto qa = random_floats(n_heads * rank, rng);
            auto pool = random_floats(
                static_cast<std::size_t>(n_blocks) * block_stride, rng);
            auto table = shuffled_table(n_blocks, rng);

            auto lat_at = [&](std::uint32_t t, std::uint32_t i) {
                return pool[table[t / bt] * block_stride +
                            layer * layer_stride + (t % bt) * row + i];
            };
            std::vector<float> want(n_heads * rank);
            for (std::uint32_t h = 0; h < n_heads; ++h) {
                std::vector<double> s(n_pos);
                for (std::uint32_t t = 0; t < n_pos; ++t) {
                    double d = 0.0;
                    for (std::uint32_t i = 0; i < rank; ++i) {
                        d += qa[h * rank + i] * lat_at(t, i);
                    }
                    for (std::uint32_t i = 0; i < rope; ++i) {
                        d += q[h * qk + nope + i] * lat_at(t, rank + i);
                    }
                    s[t] = d * kq_scale;
                }
                softmax(s);
                for (std::uint32_t i = 0; i < rank; ++i) {
                    double acc = 0.0;
                    for (std::uint32_t t = 0; t < n_pos; ++t) {
                        acc += s[t] * lat_at(t, i);
                    }
                    want[h * rank + i] = static_cast<float>(acc);
                }
            }

            auto qb = upload(ctx, q);
            auto qab = upload(ctx, qa);
            auto pb = upload(ctx, pool);
            auto tb = upload(ctx, table);
            auto ab = ctx.create_buffer(n_heads * n_pos * 4);
            auto lb = ctx.create_buffer(n_heads * rank * 4);
            ctx.begin_batch();
            const VulkanContext::Buffer bufs[] = {qb, qab, pb,
                                                  tb, ab,  lb};
            const std::uint32_t push[] = {
                n_heads,      rank, nope, rope, n_pos, bt,
                block_stride, layer * layer_stride, fbits(kq_scale)};
            ctx.dispatch(Kernel::kAttnMla, bufs, push, n_heads);
            ctx.end_batch();
            auto got = download(ctx, lb, want.size());
            for (std::size_t i = 0; i < want.size(); ++i) {
                REQUIRE(got[i] ==
                        Catch::Approx(want[i]).margin(1e-5).epsilon(1e-4));
            }
            for (auto b : {qb, qab, pb, tb, ab, lb}) {
                ctx.destroy_buffer(b);
            }
        }
    }
}

TEST_CASE("vulkan rmsnorm matches the CPU reference", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(17);

    // Not a multiple of the 256-thread workgroup; written at an
    // offset like the MLA kv_a_norm into the KV pool.
    const std::uint32_t n = 2051, out_off = 9;
    const float eps = 1e-5f;
    auto x = random_floats(n, rng);
    auto w = random_floats(n, rng);
    double ss = 0.0;
    for (float v : x) {
        ss += double(v) * v;
    }
    const double scale = 1.0 / std::sqrt(ss / n + eps);

    auto xb = upload(ctx, x);
    auto wb = upload(ctx, w);
    auto ob = ctx.create_buffer((n + out_off) * 4);
    ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {xb, wb, ob};
    const std::uint32_t push[] = {n, fbits(eps), out_off};
    ctx.dispatch(Kernel::kRmsNorm, bufs, push, 1);
    ctx.end_batch();
    auto got = download(ctx, ob, n + out_off);
    for (std::uint32_t i = 0; i < n; ++i) {
        REQUIRE(got[out_off + i] ==
                Catch::Approx(x[i] * scale * w[i])
                    .margin(1e-5)
                    .epsilon(1e-4));
    }
    for (auto b : {xb, wb, ob}) {
        ctx.destroy_buffer(b);
    }
}
