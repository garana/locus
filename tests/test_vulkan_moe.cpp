#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/backend/vulkan/q8_layout.hpp"
#include "locus/model/transformer.hpp"

using locus::backend::vk::Kernel;
using locus::backend::vk::VulkanContext;

namespace {

std::uint32_t fbits(float v) { return std::bit_cast<std::uint32_t>(v); }

template <typename T>
VulkanContext::Buffer upload(VulkanContext& ctx, const std::vector<T>& v) {
    auto b = ctx.create_buffer(std::max<std::size_t>(v.size() * sizeof(T),
                                                     4));
    ctx.write_buffer(b, std::as_bytes(std::span<const T>(v)));
    return b;
}

template <typename T>
std::vector<T> download(VulkanContext& ctx, VulkanContext::Buffer b,
                        std::size_t n) {
    std::vector<T> out(n);
    ctx.read_buffer(b, std::as_writable_bytes(std::span<T>(out)));
    return out;
}

/** Random Q8_0 bytes for rows x cols (34-byte blocks). */
std::vector<std::byte> random_q8(std::uint32_t rows, std::uint32_t cols,
                                 std::mt19937& rng) {
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.001f, 0.02f);
    std::vector<std::byte> w(static_cast<std::size_t>(rows) * (cols / 32) *
                             34);
    for (std::size_t b = 0; b < w.size(); b += 34) {
        const std::uint16_t h = locus::backend::f32_to_f16(sd(rng));
        std::memcpy(&w[b], &h, 2);
        for (int i = 0; i < 32; ++i) {
            w[b + 2 + i] =
                static_cast<std::byte>(static_cast<std::int8_t>(qd(rng)));
        }
    }
    return w;
}

}  // namespace

TEST_CASE("vulkan moe_route matches model::moe_select", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(29);
    std::normal_distribution<float> nd(0.0f, 2.0f);

    struct Case {
        const char* name;
        std::uint32_t n_expert, k, n_group, n_group_used;
        locus::model::GatingFunc gating;
        bool bias, norm, ties;
        float scale;
    };
    using locus::model::GatingFunc;
    const Case cases[] = {
        {"deepseek-v2-lite softmax", 64, 6, 1, 1, GatingFunc::kSoftmax,
         false, false, false, 1.0f},
        {"normalized, scaled", 60, 4, 1, 1, GatingFunc::kSoftmax, false,
         true, false, 2.5f},
        {"v3 sigmoid + bias + groups", 256, 8, 8, 4,
         GatingFunc::kSigmoid, true, true, false, 2.5f},
        {"ties pick the lowest index", 256, 5, 1, 1,
         GatingFunc::kSoftmax, false, false, true, 1.0f},
    };
    for (const Case& c : cases) {
        DYNAMIC_SECTION(c.name) {
            for (int trial = 0; trial < 8; ++trial) {
                locus::model::Hparams hp;
                hp.n_expert = c.n_expert;
                hp.n_expert_used = c.k;
                hp.n_group = c.n_group;
                hp.n_group_used = c.n_group_used;
                hp.gating = c.gating;
                hp.expert_weights_norm = c.norm;
                hp.expert_weights_scale = c.scale;
                std::vector<float> logits(c.n_expert), bias(c.n_expert);
                for (auto& v : logits) {
                    v = c.ties ? static_cast<float>(
                                     std::uniform_int_distribution<int>(
                                         0, 3)(rng))
                               : nd(rng);
                }
                for (auto& v : bias) {
                    v = nd(rng) * 0.1f;
                }
                locus::model::TransformerModel::Layer lay{};
                if (c.bias) {
                    lay.exp_probs_b = bias;
                }
                std::vector<float> cpu_logits = logits;
                const auto want =
                    locus::model::moe_select(hp, lay, cpu_logits);

                // Two partial-logit planes (split-K router): the
                // logits, then zeros that must not change them.
                std::vector<float> l2(logits);
                l2.resize(c.n_expert * 2, 0.0f);
                auto lb = upload(ctx, l2);
                auto bb = upload(ctx, bias);
                auto rb = ctx.create_buffer(c.k * 2 * 4);
                ctx.begin_batch();
                const VulkanContext::Buffer bufs[] = {lb, bb, rb};
                const std::uint32_t push[] = {
                    c.n_expert,       c.k,
                    c.gating == GatingFunc::kSigmoid ? 1u : 0u,
                    c.bias ? 1u : 0u, c.n_group,
                    c.n_group_used,   c.norm ? 1u : 0u,
                    fbits(c.scale),   2u};
                ctx.dispatch(Kernel::kMoeRoute, bufs, push, 1);
                ctx.end_batch();
                const auto route = download<std::uint32_t>(ctx, rb, c.k * 2);
                REQUIRE(want.size() == c.k);
                for (std::uint32_t j = 0; j < c.k; ++j) {
                    INFO("trial " << trial << " slot " << j);
                    REQUIRE(route[j] == want[j].first);
                    REQUIRE(std::bit_cast<float>(route[c.k + j]) ==
                            Catch::Approx(want[j].second).epsilon(1e-5));
                }
                for (auto b : {lb, bb, rb}) {
                    ctx.destroy_buffer(b);
                }
            }
        }
    }
}

TEST_CASE("vulkan batched q8_0 matvec covers split, routed and summed "
          "slots",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 5 stacked "experts" of 7 x 96; 3 slots routed to 4, 0, 2.
    const std::uint32_t rows = 7, cols = 96, n_mat = 5, slots = 3;
    const std::size_t mat_bytes = rows * (cols / 32) * 34;
    auto w = random_q8(rows * n_mat, cols, rng);
    std::vector<float> x(cols * slots);
    for (auto& v : x) {
        v = dist(rng);
    }
    const std::uint32_t ids[] = {4, 0, 2};
    const float wts[] = {0.5f, -1.25f, 2.0f};
    std::vector<std::uint32_t> route = {ids[0], ids[1], ids[2],
                                        fbits(wts[0]), fbits(wts[1]),
                                        fbits(wts[2])};

    // CPU reference: slot k = matrix ids[k] times x slice k.
    auto cpu_slot = [&](std::uint32_t id, std::uint32_t k) {
        locus::backend::Mat m{locus::gguf::TensorType::kQ8_0,
                              w.data() + id * mat_bytes, rows, cols};
        std::vector<float> out(rows);
        locus::backend::matvec(
            m, std::span<const float>(x).subspan(k * cols, cols), out);
        return out;
    };

    auto wb = ctx.create_buffer((w.size() + 3) & ~std::size_t{3});
    {
        std::vector<std::byte> wg(w.size());
        locus::backend::vk::q8_0_split_rows(w.data(), wg.data(),
                                            rows * n_mat, cols);
        ctx.write_buffer(wb, wg);
    }
    auto xb = upload(ctx, x);
    auto rb = upload(ctx, route);
    auto run = [&](std::uint32_t mode, std::uint32_t out_stride,
                   std::size_t out_n, std::vector<float> init) {
        auto ob = upload(ctx, init);
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {wb, xb, ob, rb};
        const std::uint32_t push[] = {
            rows, cols, 0, mode & 2u ? 1u : 0u, 0, 0, fbits(1.0f),
            static_cast<std::uint32_t>(mat_bytes), cols, out_stride,
            slots, mode, 0u};
        ctx.dispatch(Kernel::kMatvecQ8_0B, bufs, push,
                     VulkanContext::matvec_groups(Kernel::kMatvecQ8_0B,
                                                  rows),
                     mode & 2u ? 1u : slots);
        ctx.end_batch();
        auto got = download<float>(ctx, ob, out_n);
        ctx.destroy_buffer(ob);
        return got;
    };

    SECTION("split slots, matrix = slot") {
        auto got = run(0, rows, rows * slots,
                       std::vector<float>(rows * slots, 0.0f));
        for (std::uint32_t k = 0; k < slots; ++k) {
            auto want = cpu_slot(k, k);
            for (std::uint32_t r = 0; r < rows; ++r) {
                REQUIRE(got[k * rows + r] ==
                        Catch::Approx(want[r]).margin(1e-4));
            }
        }
    }
    SECTION("split slots, matrix = routed id") {
        auto got = run(1, rows, rows * slots,
                       std::vector<float>(rows * slots, 0.0f));
        for (std::uint32_t k = 0; k < slots; ++k) {
            auto want = cpu_slot(ids[k], k);
            for (std::uint32_t r = 0; r < rows; ++r) {
                REQUIRE(got[k * rows + r] ==
                        Catch::Approx(want[r]).margin(1e-4));
            }
        }
    }
    SECTION("weighted sum of routed slots accumulates") {
        std::vector<float> init(rows);
        for (auto& v : init) {
            v = dist(rng);
        }
        auto got = run(3, 0, rows, init);
        for (std::uint32_t r = 0; r < rows; ++r) {
            float want = init[r];
            for (std::uint32_t k = 0; k < slots; ++k) {
                want += wts[k] * cpu_slot(ids[k], k)[r];
            }
            REQUIRE(got[r] == Catch::Approx(want).margin(1e-4));
        }
    }
    for (auto b : {wb, xb, rb}) {
        ctx.destroy_buffer(b);
    }
}

TEST_CASE("vulkan batched k-quant matvecs cover routed and summed "
          "slots",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(37);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> byte_d(0, 255);

    // 5 stacked 21 x 768 experts (several 8-row workgroups); 3 slots
    // routed to 3, 0, 1. Q6_K's 13230-byte experts put odd ids off word
    // alignment.
    const std::uint32_t rows = 21, cols = 768, n_mat = 5, slots = 3;
    struct KCase {
        locus::gguf::TensorType type;
        Kernel kernel;
        std::size_t block_bytes;
        std::size_t d_off;  // f16 d inside the block (dmin follows)
        bool has_dmin;
    };
    const KCase cases[] = {
        {locus::gguf::TensorType::kQ4_K, Kernel::kMatvecQ4_KB, 144, 0,
         true},
        {locus::gguf::TensorType::kQ6_K, Kernel::kMatvecQ6_KB, 210, 208,
         false},
    };
    std::vector<float> x(cols * slots);
    for (auto& v : x) {
        v = dist(rng);
    }
    const std::uint32_t ids[] = {3, 0, 1};
    const float wts[] = {0.5f, -1.25f, 2.0f};
    std::vector<std::uint32_t> route = {ids[0], ids[1], ids[2],
                                        fbits(wts[0]), fbits(wts[1]),
                                        fbits(wts[2])};
    // x_off 1 puts every slot's activations (x_off + k * cols) off
    // 16-byte alignment: the kernels' scalar-load fallback.
    for (const std::uint32_t x_off : {0u, 1u})
    for (const auto& c : cases) {
        INFO("type " << static_cast<int>(c.type) << " x_off " << x_off);
        const std::size_t mat_bytes = rows * (cols / 256) * c.block_bytes;
        std::vector<std::byte> w(mat_bytes * n_mat);
        for (auto& b : w) {
            b = static_cast<std::byte>(byte_d(rng));
        }
        for (std::size_t at = c.d_off; at < w.size(); at += c.block_bytes) {
            const std::uint16_t d = locus::backend::f32_to_f16(0.01f);
            std::memcpy(&w[at], &d, 2);
            if (c.has_dmin) {
                const std::uint16_t dmin =
                    locus::backend::f32_to_f16(0.005f);
                std::memcpy(&w[at + 2], &dmin, 2);
            }
        }
        auto cpu_slot = [&](std::uint32_t id, std::uint32_t k) {
            locus::backend::Mat m{c.type, w.data() + id * mat_bytes, rows,
                                  cols};
            std::vector<float> out(rows);
            locus::backend::matvec(
                m, std::span<const float>(x).subspan(k * cols, cols), out);
            return out;
        };
        w.resize((w.size() + 3) & ~std::size_t{3});
        auto wb = upload(ctx, w);
        std::vector<float> xs(x_off, 0.0f);
        xs.insert(xs.end(), x.begin(), x.end());
        auto xb = upload(ctx, xs);
        auto rb = upload(ctx, route);
        auto run = [&](std::uint32_t mode, std::size_t out_n,
                       std::vector<float> init) {
            auto ob = upload(ctx, init);
            ctx.begin_batch();
            const VulkanContext::Buffer bufs[] = {wb, xb, ob, rb};
            const std::uint32_t push[] = {
                rows, cols, 0, mode & 2u ? 1u : 0u, 0, x_off, fbits(1.0f),
                static_cast<std::uint32_t>(mat_bytes), cols, rows, slots,
                mode, 0u};
            ctx.dispatch(c.kernel, bufs, push,
                         VulkanContext::matvec_groups(c.kernel, rows),
                         mode & 2u ? 1u : slots);
            ctx.end_batch();
            auto got = download<float>(ctx, ob, out_n);
            ctx.destroy_buffer(ob);
            return got;
        };

        // The type in the name: plain SECTIONs in a loop are tracked
        // by name, so they would run on the first type only.
        DYNAMIC_SECTION("split slots, matrix = routed id, type "
                        << static_cast<int>(c.type) << " x_off "
                        << x_off) {
            auto got = run(1, rows * slots,
                           std::vector<float>(rows * slots, 0.0f));
            for (std::uint32_t k = 0; k < slots; ++k) {
                auto want = cpu_slot(ids[k], k);
                for (std::uint32_t r = 0; r < rows; ++r) {
                    REQUIRE(got[k * rows + r] ==
                            Catch::Approx(want[r]).margin(1e-3));
                }
            }
        }
        DYNAMIC_SECTION("weighted sum of routed slots accumulates, type "
                        << static_cast<int>(c.type) << " x_off "
                        << x_off) {
            std::vector<float> init(rows);
            for (auto& v : init) {
                v = dist(rng);
            }
            auto got = run(3, rows, init);
            for (std::uint32_t r = 0; r < rows; ++r) {
                float want = init[r];
                for (std::uint32_t k = 0; k < slots; ++k) {
                    want += wts[k] * cpu_slot(ids[k], k)[r];
                }
                REQUIRE(got[r] == Catch::Approx(want).margin(1e-3));
            }
        }
        for (auto b : {wb, xb, rb}) {
            ctx.destroy_buffer(b);
        }
    }
}

TEST_CASE("vulkan transposed q8_0 matvec matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(37);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Two "heads" of 10 rows x 256 cols each, stacked with 3 rows of
    // padding between them (like wkv_b's W_uv rows after W_uk).
    const std::uint32_t rows = 10, cols = 256, pad = 3, heads = 2;
    const std::uint32_t stride_rows = rows + pad;
    const std::size_t row_b = (cols / 32) * 34;
    const auto w = random_q8(stride_rows * heads, cols, rng);
    std::vector<float> x(heads * 16);  // head h's x at h * 16
    for (auto& v : x) {
        v = dist(rng);
    }

    std::vector<float> want(heads * cols, 0.0f);
    std::vector<float> row(cols);
    for (std::uint32_t h = 0; h < heads; ++h) {
        for (std::uint32_t r = 0; r < rows; ++r) {
            locus::backend::Mat m{locus::gguf::TensorType::kQ8_0,
                                  w.data() + (h * stride_rows + r) * row_b,
                                  1, cols};
            locus::backend::dequant_row(m, 0, row);
            for (std::uint32_t c = 0; c < cols; ++c) {
                want[h * cols + c] += row[c] * x[h * 16 + r];
            }
        }
    }

    std::vector<std::byte> wg(w.size());
    locus::backend::vk::q8_0_split_rows(w.data(), wg.data(),
                                        stride_rows * heads, cols);
    auto wb = ctx.create_buffer(wg.size());
    ctx.write_buffer(wb, wg);
    auto xb = upload(ctx, x);
    auto ob = ctx.create_buffer(want.size() * 4);
    ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {wb, xb, ob};
    const std::uint32_t push[] = {
        rows, cols, 0, 0, 0,
        static_cast<std::uint32_t>(stride_rows * row_b), 16, cols};
    ctx.dispatch(Kernel::kMatvecTQ8, bufs, push, cols / 128, heads);
    ctx.end_batch();
    const auto got = download<float>(ctx, ob, want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
        REQUIRE(got[i] == Catch::Approx(want[i]).margin(1e-4));
    }
    for (auto b : {wb, xb, ob}) {
        ctx.destroy_buffer(b);
    }
}

TEST_CASE("vulkan fused SwiGLU q8_0 matvec matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(41);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 4 stacked "experts" of 7 x 96 per matrix; routed slots 3, 1.
    const std::uint32_t rows = 7, cols = 96, n_mat = 4, slots = 2;
    const std::size_t mat_bytes = rows * (cols / 32) * 34;
    const auto wg = random_q8(rows * n_mat, cols, rng);
    const auto wu = random_q8(rows * n_mat, cols, rng);
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    const std::uint32_t ids[] = {3, 1};
    std::vector<std::uint32_t> route = {ids[0], ids[1], fbits(1.0f),
                                        fbits(1.0f)};
    auto cpu_glu = [&](std::uint32_t id) {
        locus::backend::Mat g{locus::gguf::TensorType::kQ8_0,
                              wg.data() + id * mat_bytes, rows, cols};
        locus::backend::Mat u{locus::gguf::TensorType::kQ8_0,
                              wu.data() + id * mat_bytes, rows, cols};
        std::vector<float> a(rows), b(rows);
        locus::backend::matvec(g, x, a);
        locus::backend::matvec(u, x, b);
        for (std::uint32_t r = 0; r < rows; ++r) {
            a[r] = a[r] / (1.0f + std::exp(-a[r])) * b[r];
        }
        return a;
    };
    auto split = [&](const std::vector<std::byte>& w) {
        std::vector<std::byte> out(w.size());
        locus::backend::vk::q8_0_split_rows(w.data(), out.data(),
                                            rows * n_mat, cols);
        auto b = ctx.create_buffer(out.size());
        ctx.write_buffer(b, out);
        return b;
    };
    auto gb = split(wg);
    auto ub = split(wu);
    auto xb = upload(ctx, x);
    auto rb = upload(ctx, route);

    SECTION("plain: one matrix pair at a byte offset") {
        auto ob = ctx.create_buffer(rows * 4);
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {gb, xb, ob, ub};
        const std::uint32_t push[] = {
            rows, cols, 0, 0, static_cast<std::uint32_t>(2 * mat_bytes),
            0, fbits(1.0f)};
        ctx.dispatch(Kernel::kMatvecQ8_0Glu, bufs, push,
                     VulkanContext::matvec_groups(Kernel::kMatvecQ8_0Glu,
                                                  rows));
        ctx.end_batch();
        const auto got = download<float>(ctx, ob, rows);
        const auto want = cpu_glu(2);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(got[r] == Catch::Approx(want[r]).margin(1e-4));
        }
        ctx.destroy_buffer(ob);
    }
    SECTION("batched: routed experts, one output block per slot") {
        auto ob = ctx.create_buffer(rows * slots * 4);
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {gb, xb, ob, rb, ub};
        const std::uint32_t push[] = {
            rows, cols, 0, 0, 0, 0, fbits(1.0f),
            static_cast<std::uint32_t>(mat_bytes), 0, rows, slots, 1u,
            0u};
        ctx.dispatch(Kernel::kMatvecQ8_0GluB, bufs, push,
                     VulkanContext::matvec_groups(Kernel::kMatvecQ8_0GluB,
                                                  rows),
                     slots);
        ctx.end_batch();
        const auto got = download<float>(ctx, ob, rows * slots);
        for (std::uint32_t k = 0; k < slots; ++k) {
            const auto want = cpu_glu(ids[k]);
            for (std::uint32_t r = 0; r < rows; ++r) {
                REQUIRE(got[k * rows + r] ==
                        Catch::Approx(want[r]).margin(1e-4));
            }
        }
        ctx.destroy_buffer(ob);
    }
    for (auto b : {gb, ub, xb, rb}) {
        ctx.destroy_buffer(b);
    }
}

TEST_CASE("vulkan fused SwiGLU q4_k experts match the CPU reference",
          "[vulkan]") {
    // matvec_q4_k_glu_b: routed gate and up rows over stacked Q4_K
    // experts, written as silu(gate) * up -- what the gate, up and
    // silu_mul dispatches compute separately.
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(41);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> byte_d(0, 255);
    const std::uint32_t rows = 21, cols = 512, n_mat = 5, slots = 3;
    const std::size_t mat_bytes = rows * (cols / 256) * 144;
    auto quant = [&] {
        std::vector<std::byte> w(mat_bytes * n_mat);
        for (auto& b : w) {
            b = static_cast<std::byte>(byte_d(rng));
        }
        for (std::size_t at = 0; at < w.size(); at += 144) {
            const std::uint16_t d = locus::backend::f32_to_f16(0.01f);
            const std::uint16_t dmin = locus::backend::f32_to_f16(0.005f);
            std::memcpy(&w[at], &d, 2);
            std::memcpy(&w[at + 2], &dmin, 2);
        }
        return w;
    };
    const auto wg = quant(), wu = quant();
    std::vector<float> x(cols * slots);
    for (auto& v : x) {
        v = dist(rng);
    }
    const std::uint32_t ids[] = {3, 0, 1};
    std::vector<std::uint32_t> route = {ids[0], ids[1], ids[2],
                                        fbits(1.0f), fbits(1.0f),
                                        fbits(1.0f)};
    auto cpu_row = [&](const std::vector<std::byte>& w, std::uint32_t id,
                       std::uint32_t k) {
        locus::backend::Mat m{locus::gguf::TensorType::kQ4_K,
                              w.data() + id * mat_bytes, rows, cols};
        std::vector<float> out(rows);
        locus::backend::matvec(
            m, std::span<const float>(x).subspan(k * cols, cols), out);
        return out;
    };
    for (const std::uint32_t x_off : {0u, 1u}) {
        INFO("x_off " << x_off);
        std::vector<float> xs(x_off, 0.0f);
        xs.insert(xs.end(), x.begin(), x.end());
        auto gb = upload(ctx, wg);
        auto ub = upload(ctx, wu);
        auto xb = upload(ctx, xs);
        auto rb = upload(ctx, route);
        auto ob = upload(ctx, std::vector<float>(rows * slots, 0.0f));
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {gb, xb, ob, rb, ub};
        const std::uint32_t push[] = {
            rows, cols, 0, 0, 0, x_off, fbits(1.0f),
            static_cast<std::uint32_t>(mat_bytes), cols, rows, slots, 1u,
            0u};
        ctx.dispatch(Kernel::kMatvecQ4_KGluB, bufs, push,
                     VulkanContext::matvec_groups(Kernel::kMatvecQ4_KGluB,
                                                  rows),
                     slots);
        ctx.end_batch();
        const auto got = download<float>(ctx, ob, rows * slots);
        // The unfused GPU chain (q4_k_b gate, q4_k_b up, silu_mul) must
        // give the same bits: the fused kernel is a separate SPIR-V
        // module, so a driver could contract a*b+c differently in it.
        auto og = upload(ctx, std::vector<float>(rows * slots, 0.0f));
        auto ou = upload(ctx, std::vector<float>(rows * slots, 0.0f));
        ctx.begin_batch();
        for (const auto& [w, o] : {std::pair{gb, og}, std::pair{ub, ou}}) {
            const VulkanContext::Buffer mb[] = {w, xb, o, rb};
            ctx.dispatch(Kernel::kMatvecQ4_KB, mb, push,
                         VulkanContext::matvec_groups(Kernel::kMatvecQ4_KB,
                                                      rows),
                         slots);
        }
        const VulkanContext::Buffer sb[] = {og, ou, og};
        const std::uint32_t spush[] = {rows * slots};
        ctx.dispatch(Kernel::kSiluMul, sb, spush,
                     (rows * slots + 63) / 64);
        ctx.end_batch();
        const auto unfused = download<float>(ctx, og, rows * slots);
        REQUIRE(std::memcmp(got.data(), unfused.data(),
                            got.size() * sizeof(float)) == 0);
        ctx.destroy_buffer(og);
        ctx.destroy_buffer(ou);
        for (std::uint32_t k = 0; k < slots; ++k) {
            const auto g = cpu_row(wg, ids[k], k);
            const auto u = cpu_row(wu, ids[k], k);
            for (std::uint32_t r = 0; r < rows; ++r) {
                const float want = g[r] / (1.0f + std::exp(-g[r])) * u[r];
                REQUIRE(got[k * rows + r] ==
                        Catch::Approx(want).epsilon(1e-4).margin(1e-3));
            }
        }
        for (auto b : {gb, ub, xb, rb, ob}) {
            ctx.destroy_buffer(b);
        }
    }
}
