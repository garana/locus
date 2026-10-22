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

namespace {

/** Paged-pool geometry shared by the attention cases. */
struct Paged {
    VulkanContext::Buffer pool, table;
    std::uint32_t bt, block_stride, row_w;
};

/** Split-K for a paged matmul: partial planes, then sum_splits. */
struct Splits {
    std::uint32_t n = 1;      // K splits
    std::uint32_t plane = 0;  // output floats per partial plane
    VulkanContext::Buffer scratch{};
};

/**
 * Records one paged matmul (matmul.comp PAGED / PAGED_T) the way
 * the forward does: W rows (or columns) are cache positions.
 */
void paged_matmul(VulkanContext& ctx, Kernel k, const Paged& pg,
                  std::uint32_t rows, std::uint32_t cols,
                  std::uint32_t n_tok, bool acc, std::uint32_t w_off,
                  VulkanContext::Buffer x, std::uint32_t x_off,
                  std::uint32_t x_stride, VulkanContext::Buffer out,
                  std::uint32_t out_stride, std::uint32_t n_batch = 1,
                  std::uint32_t w_bstride = 0, std::uint32_t x_bstride = 0,
                  std::uint32_t out_bstride = 0, std::uint32_t w_bdiv = 1,
                  std::uint32_t div = 1, std::uint32_t x_inner = 0,
                  std::uint32_t out_inner = 0, const Splits& sp = {}) {
    const std::uint32_t tiles = (n_tok + 63) / 64;
    const VulkanContext::Buffer dst = sp.n > 1 ? sp.scratch : out;
    const VulkanContext::Buffer bufs[] = {pg.pool, x, dst, pg.table};
    const std::uint32_t push[] = {
        rows,      cols,      n_tok,       acc ? 1u : 0u, w_off,
        x_off,     x_stride,  0,           out_stride,    fbits(1.0f),
        w_bstride, x_bstride, out_bstride, tiles,         div,
        0,         w_bdiv,    pg.bt,       pg.block_stride, pg.row_w,
        sp.n,      sp.plane,  x_inner,     out_inner};
    ctx.dispatch(k, bufs, push, (rows + 63) / 64, tiles * n_batch * sp.n);
    if (sp.n > 1) {
        const VulkanContext::Buffer sbufs[] = {sp.scratch, out};
        const std::uint32_t spush[] = {sp.plane, sp.n, 0, acc ? 1u : 0u};
        ctx.dispatch(Kernel::kSumSplits, sbufs, spush,
                     (sp.plane + 255) / 256);
    }
}

void causal_softmax(VulkanContext& ctx, VulkanContext::Buffer s,
                    std::uint32_t rows, std::uint32_t n_pos,
                    std::uint32_t heads, std::uint32_t base, float scale) {
    const VulkanContext::Buffer bufs[] = {s};
    const std::uint32_t push[] = {n_pos, heads, base, fbits(scale), rows};
    ctx.dispatch(Kernel::kAttnSoftmax, bufs, push, rows);
}

}  // namespace

// Attention as matmuls over the paged pool (S = K Q^T, causal
// softmax, O = V^T P), for n_q query tokens at the END of an n_pos
// context: token t sits at position n_pos - n_q + t and attends to
// positions 0 .. its own. n_q 1 is decode, n_q > 1 a prefill chunk.
TEST_CASE("vulkan paged attention matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(11);

    // head_dim 160 is not a 64-row tile multiple; n_pos 37 leaves a
    // ragged K step in O = V^T P; 300 spans several position tiles.
    const std::uint32_t head_dims[] = {64, 160};
    const std::uint32_t n_poss[] = {1, 37, 300};
    const std::uint32_t n_qs[] = {1, 3};
    for (std::uint32_t head_dim : head_dims) {
        for (std::uint32_t n_pos : n_poss) {
            for (std::uint32_t n_q : n_qs) {
                if (n_q > n_pos) {
                    continue;
                }
                DYNAMIC_SECTION("head_dim " << head_dim << " n_pos "
                                            << n_pos << " n_q " << n_q) {
                    const std::uint32_t n_heads = 4, n_kv = 2;
                    const std::uint32_t group = n_heads / n_kv;
                    const std::uint32_t kv_dim = n_kv * head_dim;
                    const std::uint32_t q_dim = n_heads * head_dim;
                    const std::uint32_t bt = 16, n_layers = 2, layer = 1;
                    const std::uint32_t layer_stride = bt * kv_dim * 2;
                    const std::uint32_t block_stride =
                        n_layers * layer_stride;
                    const std::uint32_t n_blocks = (n_pos + bt - 1) / bt;
                    const std::uint32_t base = n_pos - n_q;

                    auto q = random_floats(std::size_t{n_q} * q_dim, rng);
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
                    std::vector<float> want(std::size_t{n_q} * q_dim);
                    for (std::uint32_t tq = 0; tq < n_q; ++tq) {
                        const std::uint32_t span = base + tq + 1;
                        for (std::uint32_t h = 0; h < n_heads; ++h) {
                            const std::uint32_t kvh = (h / group) * head_dim;
                            const float* qh =
                                q.data() + tq * q_dim + h * head_dim;
                            std::vector<double> s(span);
                            for (std::uint32_t t = 0; t < span; ++t) {
                                double d = 0.0;
                                for (std::uint32_t i = 0; i < head_dim; ++i) {
                                    d += qh[i] * kv_at(t, false, kvh + i);
                                }
                                s[t] = d / std::sqrt(double(head_dim));
                            }
                            softmax(s);
                            for (std::uint32_t i = 0; i < head_dim; ++i) {
                                double acc = 0.0;
                                for (std::uint32_t t = 0; t < span; ++t) {
                                    acc += s[t] * kv_at(t, true, kvh + i);
                                }
                                want[tq * q_dim + h * head_dim + i] =
                                    static_cast<float>(acc);
                            }
                        }
                    }

                    auto qb = upload(ctx, q);
                    auto pb = upload(ctx, pool);
                    auto tb = upload(ctx, table);
                    auto sb = ctx.create_buffer(n_q * n_heads * n_pos * 4);
                    auto ob = ctx.create_buffer(want.size() * 4);
                    const Paged pg{pb, tb, bt, block_stride, kv_dim};
                    // Decode splits O's K (positions) three ways.
                    Splits sp;
                    if (n_q == 1) {
                        sp = {3, q_dim, ctx.create_buffer(3 * q_dim * 4)};
                    }
                    ctx.begin_batch();
                    // One batch per K/V head; its query heads are
                    // token rows t * group + i, as the forward does.
                    paged_matmul(ctx, Kernel::kMatmulPagedF32, pg, n_pos,
                                 head_dim, n_q * group, false,
                                 layer * layer_stride, qb, 0, q_dim, sb,
                                 n_heads * n_pos, n_kv, head_dim,
                                 group * head_dim, group * n_pos, 1, group,
                                 head_dim, n_pos);
                    causal_softmax(ctx, sb, n_q * n_heads, n_pos, n_heads,
                                   base,
                                   1.0f / std::sqrt(float(head_dim)));
                    paged_matmul(ctx, Kernel::kMatmulPagedTF32, pg, head_dim,
                                 n_pos, n_q * group, false,
                                 layer * layer_stride + layer_stride / 2, sb,
                                 0, n_heads * n_pos, ob, q_dim, n_kv,
                                 head_dim, group * n_pos, group * head_dim, 1,
                                 group, n_pos, head_dim, sp);
                    ctx.end_batch();
                    if (sp.n > 1) {
                        ctx.destroy_buffer(sp.scratch);
                    }
                    auto got = download(ctx, ob, want.size());
                    for (std::size_t i = 0; i < want.size(); ++i) {
                        REQUIRE(got[i] == Catch::Approx(want[i])
                                              .margin(1e-5)
                                              .epsilon(1e-4));
                    }
                    for (auto b : {qb, pb, tb, sb, ob}) {
                        ctx.destroy_buffer(b);
                    }
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

    // DeepSeek-V2-Lite geometry (rank 512, rope 64); the (token,
    // head) pairs are the query rows of one matmul.
    const std::uint32_t n_poss[] = {5, 200};
    const std::uint32_t n_qs[] = {1, 4};
    for (std::uint32_t n_pos : n_poss) {
        for (std::uint32_t n_q : n_qs) {
            DYNAMIC_SECTION("n_pos " << n_pos << " n_q " << n_q) {
                const std::uint32_t n_heads = 3, rank = 512, nope = 128,
                                    rope = 64, qk = nope + rope;
                const std::uint32_t row = rank + rope;
                const std::uint32_t bt = 16, n_layers = 2, layer = 1;
                const std::uint32_t layer_stride = bt * row;
                const std::uint32_t block_stride = n_layers * layer_stride;
                const std::uint32_t n_blocks = (n_pos + bt - 1) / bt;
                const std::uint32_t base = n_pos - n_q;
                const std::uint32_t rows = n_q * n_heads;
                const float kq_scale = 0.07f;

                auto q = random_floats(std::size_t{rows} * qk, rng);
                auto qa = random_floats(std::size_t{rows} * rank, rng);
                auto pool = random_floats(
                    static_cast<std::size_t>(n_blocks) * block_stride, rng);
                auto table = shuffled_table(n_blocks, rng);

                auto lat_at = [&](std::uint32_t t, std::uint32_t i) {
                    return pool[table[t / bt] * block_stride +
                                layer * layer_stride + (t % bt) * row + i];
                };
                std::vector<float> want(std::size_t{rows} * rank);
                for (std::uint32_t j = 0; j < rows; ++j) {
                    const std::uint32_t span = base + j / n_heads + 1;
                    std::vector<double> s(span);
                    for (std::uint32_t t = 0; t < span; ++t) {
                        double d = 0.0;
                        for (std::uint32_t i = 0; i < rank; ++i) {
                            d += qa[j * rank + i] * lat_at(t, i);
                        }
                        for (std::uint32_t i = 0; i < rope; ++i) {
                            d += q[j * qk + nope + i] * lat_at(t, rank + i);
                        }
                        s[t] = d * kq_scale;
                    }
                    softmax(s);
                    for (std::uint32_t i = 0; i < rank; ++i) {
                        double acc = 0.0;
                        for (std::uint32_t t = 0; t < span; ++t) {
                            acc += s[t] * lat_at(t, i);
                        }
                        want[j * rank + i] = static_cast<float>(acc);
                    }
                }

                auto qb = upload(ctx, q);
                auto qab = upload(ctx, qa);
                auto pb = upload(ctx, pool);
                auto tb = upload(ctx, table);
                auto sb = ctx.create_buffer(rows * n_pos * 4);
                auto lb = ctx.create_buffer(want.size() * 4);
                const Paged pg{pb, tb, bt, block_stride, row};
                const std::uint32_t lo = layer * layer_stride;
                Splits sp;
                if (n_q == 1) {
                    const std::uint32_t plane = rows * rank;
                    sp = {3, plane, ctx.create_buffer(3 * plane * 4)};
                }
                ctx.begin_batch();
                paged_matmul(ctx, Kernel::kMatmulPagedF32, pg, n_pos, rank,
                             rows, false, lo, qab, 0, rank, sb, n_pos);
                paged_matmul(ctx, Kernel::kMatmulPagedF32, pg, n_pos, rope,
                             rows, true, lo + rank, qb, nope, qk, sb, n_pos);
                causal_softmax(ctx, sb, rows, n_pos, n_heads, base, kq_scale);
                paged_matmul(ctx, Kernel::kMatmulPagedTF32, pg, rank, n_pos,
                             rows, false, lo, sb, 0, n_pos, lb, rank, 1, 0,
                             0, 0, 1, 1, 0, 0, sp);
                ctx.end_batch();
                if (sp.n > 1) {
                    ctx.destroy_buffer(sp.scratch);
                }
                auto got = download(ctx, lb, want.size());
                for (std::size_t i = 0; i < want.size(); ++i) {
                    REQUIRE(got[i] ==
                            Catch::Approx(want[i]).margin(1e-5).epsilon(1e-4));
                }
                for (auto b : {qb, qab, pb, tb, sb, lb}) {
                    ctx.destroy_buffer(b);
                }
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
    // offset like the MLA kv_a_norm into the KV pool, with a tail
    // copied through unchanged (MLA's k_pe after the latent).
    const std::uint32_t n = 2051, out_off = 9, tail = 67;
    const float eps = 1e-5f;
    auto x = random_floats(n + tail, rng);
    auto w = random_floats(n, rng);
    double ss = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        ss += double(x[i]) * x[i];
    }
    const double scale = 1.0 / std::sqrt(ss / n + eps);

    auto xb = upload(ctx, x);
    auto wb = upload(ctx, w);
    auto ob = upload(ctx, std::vector<float>(n + tail + out_off + 1, -7.0f));
    ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {xb, wb, ob};
    const std::uint32_t push[] = {n, fbits(eps), out_off, tail};
    ctx.dispatch(Kernel::kRmsNorm, bufs, push, 1);
    ctx.end_batch();
    auto got = download(ctx, ob, n + tail + out_off + 1);
    for (std::uint32_t i = 0; i < n; ++i) {
        REQUIRE(got[out_off + i] ==
                Catch::Approx(x[i] * scale * w[i])
                    .margin(1e-5)
                    .epsilon(1e-4));
    }
    for (std::uint32_t i = n; i < n + tail; ++i) {
        REQUIRE(got[out_off + i] == x[i]);
    }
    REQUIRE(got[out_off - 1] == -7.0f);
    REQUIRE(got[out_off + n + tail] == -7.0f);
    for (auto b : {xb, wb, ob}) {
        ctx.destroy_buffer(b);
    }
}

TEST_CASE("vulkan decode attention (flash) matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(43);

    struct Case {
        const char* name;
        std::uint32_t heads, group, d1, d2, dv, n_pos;
        bool shared_kv;  // MLA: one latent row, V = its first dv
    };
    const Case cases[] = {
        {"gqa one block", 4, 2, 64, 0, 64, 37, false},
        {"gqa three blocks + combine", 4, 2, 64, 0, 64, 150, false},
        {"mla two-segment key", 3, 3, 512, 64, 512, 61, true},
        {"mla blocks + combine", 3, 3, 512, 64, 512, 200, true},
    };
    for (const Case& c : cases) {
        DYNAMIC_SECTION(c.name) {
            const std::uint32_t n_kv = c.heads / c.group;
            const std::uint32_t dk = c.d1 + c.d2;
            // Pool row: GQA K for every KV head then V; MLA latent.
            const std::uint32_t kv_stride = c.shared_kv ? 0 : c.d1;
            const std::uint32_t k_off = 5;  // a layer's offset
            const std::uint32_t v_off =
                c.shared_kv ? k_off : k_off + n_kv * c.d1;
            const std::uint32_t row_w =
                c.shared_kv ? dk + 7 : k_off + 2 * n_kv * c.d1 + 3;
            const std::uint32_t bt = 16;
            const std::uint32_t n_blocks = (c.n_pos + bt - 1) / bt;
            const std::uint32_t block_stride = bt * row_w;
            auto pool = random_floats(
                static_cast<std::size_t>(n_blocks) * block_stride, rng);
            auto table = shuffled_table(n_blocks, rng);
            const std::uint32_t q1_stride = c.d1 + 3;
            const std::uint32_t q2_stride = c.d2 + 9, q2_off = 4;
            auto q1 = random_floats(c.heads * q1_stride, rng);
            auto q2 = random_floats(c.heads * q2_stride + q2_off + 1, rng);
            const float scale = 0.11f;

            auto row = [&](std::uint32_t p) {
                return table[p / bt] * block_stride + (p % bt) * row_w;
            };
            std::vector<float> want(c.heads * c.dv);
            for (std::uint32_t h = 0; h < c.heads; ++h) {
                const std::uint32_t kvh = h / c.group;
                std::vector<double> s(c.n_pos);
                for (std::uint32_t p = 0; p < c.n_pos; ++p) {
                    const std::size_t kb = row(p) + k_off + kvh * kv_stride;
                    double d = 0.0;
                    for (std::uint32_t i = 0; i < c.d1; ++i) {
                        d += q1[h * q1_stride + i] * pool[kb + i];
                    }
                    for (std::uint32_t i = 0; i < c.d2; ++i) {
                        d += q2[h * q2_stride + q2_off + i] *
                             pool[kb + c.d1 + i];
                    }
                    s[p] = d * scale;
                }
                softmax(s);
                for (std::uint32_t i = 0; i < c.dv; ++i) {
                    double acc = 0.0;
                    for (std::uint32_t p = 0; p < c.n_pos; ++p) {
                        acc += s[p] *
                               pool[row(p) + v_off + kvh * kv_stride + i];
                    }
                    want[h * c.dv + i] = static_cast<float>(acc);
                }
            }

            const std::uint32_t blocks = (c.n_pos + 63) / 64;
            auto pb = upload(ctx, pool);
            auto tb = upload(ctx, table);
            auto q1b = upload(ctx, q1);
            auto q2b = upload(ctx, q2);
            auto ob = ctx.create_buffer(want.size() * 4);
            auto partb = ctx.create_buffer(
                std::size_t{blocks} * c.heads * (c.dv + 2) * 4);
            ctx.begin_batch();
            const VulkanContext::Buffer bufs[] = {pb, tb, q1b, q2b, ob,
                                                  blocks > 1 ? partb : ob};
            const std::uint32_t push[] = {
                c.n_pos,   bt,      block_stride, row_w,
                k_off,     v_off,   kv_stride,    c.group,
                c.d1,      c.d2,    q1_stride,    q2_stride,
                q2_off,    c.dv,    fbits(scale), c.dv};
            ctx.dispatch(Kernel::kAttnDecode, bufs, push, blocks, c.heads);
            if (blocks > 1) {
                const VulkanContext::Buffer cb[] = {partb, ob};
                const std::uint32_t cp[] = {blocks, c.heads, c.dv, c.dv};
                ctx.dispatch(Kernel::kAttnCombine, cb, cp,
                             (c.dv + 63) / 64, c.heads);
            }
            ctx.end_batch();
            auto got = download(ctx, ob, want.size());
            for (std::size_t i = 0; i < want.size(); ++i) {
                REQUIRE(got[i] ==
                        Catch::Approx(want[i]).margin(1e-5).epsilon(1e-4));
            }
            for (auto b : {pb, tb, q1b, q2b, ob, partb}) {
                ctx.destroy_buffer(b);
            }
        }
    }
}

TEST_CASE("vulkan grouped decode attention matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(53);

    // attn_decode_grp.comp: 32 positions x up to 16 heads of one KV
    // head per workgroup. MLA: one shared latent row, V = its first
    // dv floats; GQA: K then V rows per KV head.
    struct Case {
        const char* name;
        std::uint32_t heads, group, d1, d2, dv, n_pos;
        bool mla;
    };
    const Case cases[] = {
        {"mla partial head chunk, one block", 3, 3, 512, 64, 512, 20, true},
        {"mla one chunk, blocks + combine", 16, 16, 512, 64, 512, 200, true},
        {"mla two chunks, ragged last block", 20, 20, 512, 64, 512, 77,
         true},
        {"mla narrow latent (8 head sets)", 16, 16, 128, 32, 128, 100, true},
        {"gqa group 4, position slices", 8, 4, 64, 0, 64, 150, false},
        {"gqa group 8, head dim 128", 16, 8, 128, 0, 128, 45, false},
    };
    for (const Case& c : cases) {
        {
            DYNAMIC_SECTION(c.name) {
                const std::uint32_t n_kv = c.heads / c.group;
                const std::uint32_t dk = c.d1 + c.d2;
                const std::uint32_t kv_stride = c.mla ? 0 : c.d1;
                const std::uint32_t k_off = 8;  // a layer's offset
                const std::uint32_t v_off =
                    c.mla ? k_off : k_off + n_kv * c.d1;
                const std::uint32_t row_w =
                    c.mla ? dk + 12 : k_off + 2 * n_kv * c.d1 + 4;
                const std::uint32_t bt = 16;
                const std::uint32_t n_blocks = (c.n_pos + bt - 1) / bt;
                const std::uint32_t block_stride = bt * row_w;
                auto pool = random_floats(
                    static_cast<std::size_t>(n_blocks) * block_stride, rng);
                auto table = shuffled_table(n_blocks, rng);
                const std::uint32_t q1_stride = c.d1 + 4;
                const std::uint32_t q2_stride = c.d2 + 40, q2_off = 40;
                auto q1 = random_floats(c.heads * q1_stride, rng);
                auto q2 = random_floats(c.heads * q2_stride + 1, rng);
                const float scale = 0.07f;

                auto row = [&](std::uint32_t p) {
                    return table[p / bt] * block_stride + (p % bt) * row_w;
                };
                std::vector<float> want(c.heads * c.dv);
                for (std::uint32_t h = 0; h < c.heads; ++h) {
                    const std::uint32_t kvh = h / c.group;
                    std::vector<double> sv(c.n_pos);
                    for (std::uint32_t p = 0; p < c.n_pos; ++p) {
                        const std::size_t kb =
                            row(p) + k_off + kvh * kv_stride;
                        double d = 0.0;
                        for (std::uint32_t i = 0; i < c.d1; ++i) {
                            d += q1[h * q1_stride + i] * pool[kb + i];
                        }
                        for (std::uint32_t i = 0; i < c.d2; ++i) {
                            d += q2[h * q2_stride + q2_off + i] *
                                 pool[kb + c.d1 + i];
                        }
                        sv[p] = d * scale;
                    }
                    softmax(sv);
                    for (std::uint32_t i = 0; i < c.dv; ++i) {
                        double acc = 0.0;
                        for (std::uint32_t p = 0; p < c.n_pos; ++p) {
                            acc += sv[p] * pool[row(p) + v_off +
                                                kvh * kv_stride + i];
                        }
                        want[h * c.dv + i] = static_cast<float>(acc);
                    }
                }

                auto pb = upload(ctx, pool);
                const std::uint32_t blocks = (c.n_pos + 31) / 32;
                const std::uint32_t chunks = (c.group + 15) / 16;
                auto tb = upload(ctx, table);
                auto q1b = upload(ctx, q1);
                auto q2b = upload(ctx, q2);
                auto ob = ctx.create_buffer(want.size() * 4);
                auto partb = ctx.create_buffer(
                    std::size_t{blocks} * c.heads * (c.dv + 2) * 4);
                ctx.begin_batch();
                const VulkanContext::Buffer bufs[] = {
                    pb, tb, q1b, q2b, ob, blocks > 1 ? partb : ob};
                const std::uint32_t push[] = {
                    c.n_pos,   bt,      block_stride, row_w,
                    k_off,     v_off,   kv_stride,    c.group,
                    c.d1,      c.d2,    q1_stride,    q2_stride,
                    q2_off,    c.dv,    fbits(scale), c.dv};
                ctx.dispatch(Kernel::kAttnDecodeGrp, bufs, push, blocks,
                             n_kv * chunks);
                if (blocks > 1) {
                    const VulkanContext::Buffer cb[] = {partb, ob};
                    const std::uint32_t cp[] = {blocks, c.heads, c.dv,
                                                c.dv};
                    ctx.dispatch(Kernel::kAttnCombine, cb, cp,
                                 (c.dv + 63) / 64, c.heads);
                }
                ctx.end_batch();
                auto got = download(ctx, ob, want.size());
                for (std::size_t i = 0; i < want.size(); ++i) {
                    REQUIRE(got[i] == Catch::Approx(want[i])
                                          .margin(1e-5)
                                          .epsilon(1e-4));
                }
                for (auto b : {pb, tb, q1b, q2b, ob, partb}) {
                    ctx.destroy_buffer(b);
                }
            }
        }
    }
}
