#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/backend/vulkan/q8_layout.hpp"
#include "locus/sys/mmap_file.hpp"

using locus::backend::vk::Kernel;
using locus::backend::vk::VulkanContext;

TEST_CASE("vulkan matvec matches the CPU reference", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;

    const std::uint32_t rows = 96;
    const std::uint32_t cols = 192;
    std::vector<float> w(static_cast<std::size_t>(rows) * cols);
    std::vector<float> x(cols);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : w) {
        v = dist(rng);
    }
    for (auto& v : x) {
        v = dist(rng);
    }

    std::vector<float> gpu(rows), cpu(rows);
    ctx.matvec_f32(w, rows, cols, x, gpu);

    locus::backend::Mat m{
        locus::gguf::TensorType::kF32,
        reinterpret_cast<const std::byte*>(w.data()), rows, cols};
    locus::backend::matvec(m, x, cpu);

    for (std::uint32_t r = 0; r < rows; ++r) {
        REQUIRE(gpu[r] ==
                Catch::Approx(cpu[r]).margin(1e-4).epsilon(1e-4));
    }
}

TEST_CASE("vulkan q8_0 matvec matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;

    const std::uint32_t rows = 7, cols = 96;  // 3 blocks per row
    std::mt19937 rng(23);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::byte> w(rows * (cols / 32) * 34);
    for (std::size_t b = 0; b < rows * (cols / 32); ++b) {
        std::byte* blk = w.data() + b * 34;
        const std::uint16_t d =
            locus::backend::f32_to_f16(0.05f);
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

    std::vector<float> cpu(rows), gpu(rows);
    locus::backend::Mat m{locus::gguf::TensorType::kQ8_0,
                           w.data(), rows, cols};
    locus::backend::matvec(m, x, cpu);

    auto wb = ctx.create_buffer((w.size() + 3) & ~std::size_t{3});
    auto xb = ctx.create_buffer(x.size() * 4);
    auto ob = ctx.create_buffer(gpu.size() * 4);
    std::vector<std::byte> wg(w.size());
    locus::backend::vk::q8_0_split_rows(w.data(), wg.data(), rows, cols);
    ctx.write_buffer(wb, wg);
    ctx.write_buffer(xb, std::as_bytes(std::span(x)));
    ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {wb, xb, ob};
    const std::uint32_t push[] = {
        rows, cols, 0, 0, 0, 0, std::bit_cast<std::uint32_t>(1.0f)};
    ctx.dispatch(Kernel::kMatvecQ8_0, bufs, push,
                 VulkanContext::matvec_groups(Kernel::kMatvecQ8_0,
                                              rows));
    ctx.end_batch();
    ctx.read_buffer(ob, std::as_writable_bytes(std::span(gpu)));

    for (std::uint32_t r = 0; r < rows; ++r) {
        REQUIRE(gpu[r] == Catch::Approx(cpu[r]).margin(1e-3));
    }
    ctx.destroy_buffer(wb);
    ctx.destroy_buffer(xb);
    ctx.destroy_buffer(ob);
}

TEST_CASE("vulkan f16 bf16 and q4_0 matvec match the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::uint32_t rows = 5, cols = 64;
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    auto run_kernel = [&](Kernel k, std::span<const std::byte> w,
                          std::span<float> out) {
        auto wb =
            ctx.create_buffer((w.size() + 3) & ~std::size_t{3});
        auto xb = ctx.create_buffer(x.size() * 4);
        auto ob = ctx.create_buffer(out.size() * 4);
        ctx.write_buffer(wb, w);
        ctx.write_buffer(xb, std::as_bytes(std::span(x)));
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {wb, xb, ob};
        const std::uint32_t push[] = {
        rows, cols, 0, 0, 0, 0, std::bit_cast<std::uint32_t>(1.0f)};
        ctx.dispatch(k, bufs, push,
                     VulkanContext::matvec_groups(k, rows));
        ctx.end_batch();
        ctx.read_buffer(ob, std::as_writable_bytes(out));
        ctx.destroy_buffer(wb);
        ctx.destroy_buffer(xb);
        ctx.destroy_buffer(ob);
    };

    SECTION("f16") {
        std::vector<std::uint16_t> w(rows * cols);
        for (auto& v : w) {
            v = locus::backend::f32_to_f16(dist(rng));
        }
        std::vector<float> cpu(rows), gpu(rows);
        locus::backend::Mat m{
            locus::gguf::TensorType::kF16,
            reinterpret_cast<const std::byte*>(w.data()), rows,
            cols};
        locus::backend::matvec(m, x, cpu);
        run_kernel(Kernel::kMatvecF16,
                   std::as_bytes(std::span(w)), gpu);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(gpu[r] ==
                    Catch::Approx(cpu[r]).margin(1e-3));
        }
    }

    SECTION("bf16") {
        // bfloat16: the top 16 bits of an f32 (truncated here), the
        // type Janus-Pro ships in (i#247).
        std::vector<std::uint16_t> w(rows * cols);
        for (auto& v : w) {
            const float f = dist(rng);
            std::uint32_t u;
            std::memcpy(&u, &f, sizeof u);
            v = static_cast<std::uint16_t>(u >> 16);
        }
        std::vector<float> cpu(rows), gpu(rows);
        locus::backend::Mat m{
            locus::gguf::TensorType::kBF16,
            reinterpret_cast<const std::byte*>(w.data()), rows,
            cols};
        locus::backend::matvec(m, x, cpu);
        run_kernel(Kernel::kMatvecBF16,
                   std::as_bytes(std::span(w)), gpu);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(gpu[r] ==
                    Catch::Approx(cpu[r]).margin(1e-3));
        }
    }

    SECTION("q4_0") {
        std::vector<std::byte> w(rows * (cols / 32) * 18);
        std::uniform_int_distribution<int> nib(0, 255);
        for (std::size_t b = 0; b < rows * (cols / 32); ++b) {
            std::byte* blk = w.data() + b * 18;
            const std::uint16_t d =
                locus::backend::f32_to_f16(0.25f);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 16; ++i) {
                blk[2 + i] =
                    static_cast<std::byte>(nib(rng));
            }
        }
        std::vector<float> cpu(rows), gpu(rows);
        locus::backend::Mat m{locus::gguf::TensorType::kQ4_0,
                               w.data(), rows, cols};
        locus::backend::matvec(m, x, cpu);
        run_kernel(Kernel::kMatvecQ4_0, w, gpu);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(gpu[r] ==
                    Catch::Approx(cpu[r]).margin(1e-3));
        }
    }

    SECTION("q5_0") {
        // 22-byte blocks: f16 d + u32 qh + 16 qs bytes; qh and qs
        // fully random so the 5th-bit path is exercised.
        std::vector<std::byte> w(rows * (cols / 32) * 22);
        std::uniform_int_distribution<int> byte(0, 255);
        for (std::size_t b = 0; b < rows * (cols / 32); ++b) {
            std::byte* blk = w.data() + b * 22;
            const std::uint16_t d =
                locus::backend::f32_to_f16(0.25f);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 20; ++i) {
                blk[2 + i] =
                    static_cast<std::byte>(byte(rng));
            }
        }
        std::vector<float> cpu(rows), gpu(rows);
        locus::backend::Mat m{locus::gguf::TensorType::kQ5_0,
                               w.data(), rows, cols};
        locus::backend::matvec(m, x, cpu);
        run_kernel(Kernel::kMatvecQ5_0, w, gpu);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(gpu[r] ==
                    Catch::Approx(cpu[r]).margin(1e-3));
        }
    }
}

// The batched f16 / bf16 kernels (matvec_f16.comp -DBATCHED), which
// only the MoE and MLA paths of the GPU forward reach: slot y reads
// matrix route[y] (mode 1), or every slot is summed into one row with
// the weights in route[n_slots..] (mode 1 | 2), as for MoE experts.
// 33 columns, so rows and matrices start on both halves of a word.
TEST_CASE("vulkan batched f16 and bf16 matvec match the CPU per slot",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(37);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::uint32_t rows = 5, cols = 33, n_mat = 3, slots = 2;
    const std::uint32_t ids[slots] = {2, 0};
    const float mix[slots] = {0.75f, -1.5f};
    std::vector<float> x(slots * cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    std::vector<std::uint32_t> route(2 * slots);
    for (std::uint32_t k = 0; k < slots; ++k) {
        route[k] = ids[k];
        route[slots + k] = std::bit_cast<std::uint32_t>(mix[k]);
    }

    struct Case {
        const char* name;
        locus::gguf::TensorType type;
        Kernel kernel;
        std::uint16_t (*to16)(float);
    };
    const Case cases[] = {
        {"f16", locus::gguf::TensorType::kF16, Kernel::kMatvecF16B,
         [](float f) { return locus::backend::f32_to_f16(f); }},
        {"bf16", locus::gguf::TensorType::kBF16, Kernel::kMatvecBF16B,
         [](float f) {
             return static_cast<std::uint16_t>(
                 std::bit_cast<std::uint32_t>(f) >> 16);
         }},
    };
    for (const auto& c : cases) {
        INFO(c.name);
        std::vector<std::uint16_t> w(n_mat * rows * cols);
        for (auto& v : w) {
            v = c.to16(dist(rng));
        }
        // CPU reference: each (matrix, x slot) product on its own.
        auto cpu_dot = [&](std::uint32_t mat, std::uint32_t slot) {
            std::vector<float> out(rows);
            const locus::backend::Mat m{
                c.type,
                reinterpret_cast<const std::byte*>(
                    w.data() + std::size_t{mat} * rows * cols),
                rows, cols};
            locus::backend::matvec(
                m, std::span(x).subspan(slot * cols, cols), out);
            return out;
        };
        auto wb = ctx.create_buffer((w.size() * 2 + 3) & ~std::size_t{3});
        auto xb = ctx.create_buffer(x.size() * 4);
        auto rb = ctx.create_buffer(route.size() * 4);
        auto ob = ctx.create_buffer(slots * rows * 4);
        ctx.write_buffer(wb, std::as_bytes(std::span(w)));
        ctx.write_buffer(xb, std::as_bytes(std::span(x)));
        ctx.write_buffer(rb, std::as_bytes(std::span(route)));
        auto run = [&](std::uint32_t mode) {
            const bool sum = (mode & 2u) != 0u;
            std::vector<float> out(sum ? rows : slots * rows);
            const VulkanContext::Buffer bufs[] = {wb, xb, ob, rb};
            // rows, cols, out_offset, accumulate, w_off, x_off,
            // scale, w_stride, x_stride, out_stride, n_slots, mode,
            // row_stride (0 = cols).
            const std::uint32_t push[] = {
                rows,       cols, 0,     0,    0, 0,
                std::bit_cast<std::uint32_t>(1.0f),
                rows * cols, cols, rows, slots, mode, 0};
            ctx.begin_batch();
            ctx.dispatch(c.kernel, bufs, push,
                         VulkanContext::matvec_groups(c.kernel, rows),
                         sum ? 1u : slots);
            ctx.end_batch();
            ctx.read_buffer(ob, std::as_writable_bytes(std::span(out)));
            return out;
        };

        const auto split = run(1u);
        for (std::uint32_t y = 0; y < slots; ++y) {
            const auto want = cpu_dot(ids[y], y);
            for (std::uint32_t r = 0; r < rows; ++r) {
                INFO("split slot " << y << " row " << r);
                REQUIRE(split[y * rows + r] ==
                        Catch::Approx(want[r]).margin(1e-3));
            }
        }
        const auto summed = run(1u | 2u);
        std::vector<float> want(rows, 0.0f);
        for (std::uint32_t k = 0; k < slots; ++k) {
            const auto part = cpu_dot(ids[k], k);
            for (std::uint32_t r = 0; r < rows; ++r) {
                want[r] += mix[k] * part[r];
            }
        }
        for (std::uint32_t r = 0; r < rows; ++r) {
            INFO("sum row " << r);
            REQUIRE(summed[r] == Catch::Approx(want[r]).margin(1e-3));
        }
        ctx.destroy_buffer(wb);
        ctx.destroy_buffer(xb);
        ctx.destroy_buffer(rb);
        ctx.destroy_buffer(ob);
    }
}

// i#267: the short-batch f16 / bf16 kernels (matvec_q8_0_n.comp
// TYPE_F16), the 8-token and the 2-token (NT2) builds. Each token's
// output row must equal the CPU matvec of that token. 64 columns take
// the word path; 33 columns with the matrix one half into the buffer
// put rows on odd halves, the per-element path.
TEST_CASE("vulkan f16 and bf16 short-batch matvec match the CPU per token",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(43);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const std::uint32_t rows = 7;
    struct Case {
        const char* name;
        locus::gguf::TensorType type;
        Kernel kernel;
        std::uint32_t n_tok;
    };
    const Case cases[] = {
        {"f16 n", locus::gguf::TensorType::kF16, Kernel::kMatvecF16N, 5},
        {"bf16 n", locus::gguf::TensorType::kBF16, Kernel::kMatvecBF16N, 5},
        {"f16 n2", locus::gguf::TensorType::kF16, Kernel::kMatvecF16N2, 2},
        {"bf16 n2", locus::gguf::TensorType::kBF16, Kernel::kMatvecBF16N2,
         2},
    };
    for (const std::uint32_t cols : {64u, 33u}) {
        const std::uint32_t lead = cols % 2;  // halves before the matrix
        for (const auto& c : cases) {
            INFO(c.name << ", " << cols << " columns");
            const bool bf = c.type == locus::gguf::TensorType::kBF16;
            std::vector<std::uint16_t> w(lead + std::size_t{rows} * cols + 1);
            for (auto& v : w) {
                const float f = dist(rng);
                v = bf ? static_cast<std::uint16_t>(
                             std::bit_cast<std::uint32_t>(f) >> 16)
                       : locus::backend::f32_to_f16(f);
            }
            std::vector<float> x(std::size_t{c.n_tok} * cols);
            for (auto& v : x) v = dist(rng);
            std::vector<float> gpu(std::size_t{c.n_tok} * rows);
            auto wb = ctx.create_buffer((w.size() * 2 + 3) & ~std::size_t{3});
            auto xb = ctx.create_buffer(x.size() * 4);
            auto ob = ctx.create_buffer(gpu.size() * 4);
            ctx.write_buffer(wb, std::as_bytes(std::span(w)));
            ctx.write_buffer(xb, std::as_bytes(std::span(x)));
            const VulkanContext::Buffer bufs[] = {wb, xb, ob};
            // rows, cols, out_offset, accumulate, w_off (halves), x_off,
            // scale, n_tok, x_stride, out_stride, w/x/out bstride.
            const std::uint32_t push[] = {
                rows, cols, 0, 0, lead, 0,
                std::bit_cast<std::uint32_t>(1.0f), c.n_tok, cols, rows,
                0, 0, 0};
            ctx.begin_batch();
            ctx.dispatch(c.kernel, bufs, push,
                         VulkanContext::matvec_groups(c.kernel, rows));
            ctx.end_batch();
            ctx.read_buffer(ob, std::as_writable_bytes(std::span(gpu)));
            const locus::backend::Mat m{
                c.type, reinterpret_cast<const std::byte*>(w.data() + lead),
                rows, cols};
            for (std::uint32_t t = 0; t < c.n_tok; ++t) {
                std::vector<float> cpu(rows);
                locus::backend::matvec(
                    m, std::span(x).subspan(std::size_t{t} * cols, cols), cpu);
                for (std::uint32_t r = 0; r < rows; ++r) {
                    INFO("token " << t << " row " << r);
                    REQUIRE(gpu[std::size_t{t} * rows + r] ==
                            Catch::Approx(cpu[r]).margin(1e-3));
                }
            }
            ctx.destroy_buffer(wb);
            ctx.destroy_buffer(xb);
            ctx.destroy_buffer(ob);
        }
    }
}

TEST_CASE("vulkan k-quant matvec matches the CPU reference",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> byte_d(0, 255);

    struct KCase {
        locus::gguf::TensorType type;
        Kernel kernel;
        std::size_t block_bytes;
        std::size_t d_off;  // f16 scale offset inside the block
        bool has_dmin;      // f16 dmin follows d at d_off + 2
    };
    const KCase cases[] = {
        {locus::gguf::TensorType::kQ4_K, Kernel::kMatvecQ4_K,
         144, 0, true},
        {locus::gguf::TensorType::kQ5_K, Kernel::kMatvecQ5_K,
         176, 0, true},
        {locus::gguf::TensorType::kQ6_K, Kernel::kMatvecQ6_K,
         210, 208, false},
        {locus::gguf::TensorType::kQ2_K, Kernel::kMatvecQ2_K,
         84, 80, true},
    };
    // 4 x 512: one workgroup, fewer units than lanes; 7 x 2304: a
    // partial workgroup (of the 8-row K-quant kernels too) and several
    // units per lane; the rest span many workgroups (1000 rows: a
    // partial last one for the 64-row kernels). x_off 1 puts the
    // activations off 16-byte alignment: the scalar-load fallback.
    // w_off 4 does the same for the weights (word loads, not quads).
    const std::uint32_t shapes[][2] = {{4, 512},    {7, 2304},
                                       {512, 2048}, {2048, 768},
                                       {1000, 2048}};
    for (const std::uint32_t w_off : {0u, 4u})
    for (const std::uint32_t x_off : {0u, 1u})
    for (const auto& [rows, cols] : shapes)
    for (const auto& c : cases) {
        INFO("type " << static_cast<int>(c.type) << " " << rows << "x"
                     << cols << " x_off " << x_off << " w_off " << w_off);
        std::vector<float> x(cols);
        for (auto& v : x) {
            v = dist(rng);
        }
        std::vector<std::byte> w(rows * (cols / 256) *
                                 c.block_bytes);
        for (auto& b : w) {
            b = static_cast<std::byte>(byte_d(rng));
        }
        for (std::size_t blk = 0; blk < rows * (cols / 256);
             ++blk) {
            std::byte* p =
                w.data() + blk * c.block_bytes + c.d_off;
            const std::uint16_t d =
                locus::backend::f32_to_f16(0.01f);
            std::memcpy(p, &d, 2);
            if (c.has_dmin) {  // dmin follows d at d_off + 2
                const std::uint16_t dmin =
                    locus::backend::f32_to_f16(0.005f);
                std::memcpy(p + 2, &dmin, 2);
            }
        }

        std::vector<float> cpu(rows), gpu(rows);
        locus::backend::Mat m{c.type, w.data(), rows, cols};
        locus::backend::matvec(m, x, cpu);

        std::vector<std::byte> ws(w_off);
        ws.insert(ws.end(), w.begin(), w.end());
        auto wb =
            ctx.create_buffer((ws.size() + 3) & ~std::size_t{3});
        std::vector<float> xs(x_off, 0.0f);
        xs.insert(xs.end(), x.begin(), x.end());
        auto xb = ctx.create_buffer(xs.size() * 4);
        auto ob = ctx.create_buffer(gpu.size() * 4);
        ctx.write_buffer(wb, ws);
        ctx.write_buffer(xb, std::as_bytes(std::span(xs)));
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {wb, xb, ob};
        const std::uint32_t push[] = {
        rows, cols, 0,     0,
        w_off, x_off, std::bit_cast<std::uint32_t>(1.0f)};
        ctx.dispatch(c.kernel, bufs, push,
                     VulkanContext::matvec_groups(c.kernel, rows));
        ctx.end_batch();
        ctx.read_buffer(ob,
                        std::as_writable_bytes(std::span(gpu)));
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(gpu[r] ==
                    Catch::Approx(cpu[r]).margin(1e-3));
        }
        ctx.destroy_buffer(wb);
        ctx.destroy_buffer(xb);
        ctx.destroy_buffer(ob);
    }
}

TEST_CASE("vulkan q4_k seg matvec routes each matrix to its output",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::mt19937 rng(53);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> byte_d(0, 255);
    const std::uint32_t cols = 768;
    auto random_q4k = [&](std::uint32_t rows) {
        std::vector<std::byte> w(rows * (cols / 256) * 144);
        for (auto& b : w) {
            b = static_cast<std::byte>(byte_d(rng));
        }
        for (std::size_t at = 0; at < w.size(); at += 144) {
            const std::uint16_t dd[] = {locus::backend::f32_to_f16(0.01f),
                                        locus::backend::f32_to_f16(0.005f)};
            std::memcpy(&w[at], dd, 4);
        }
        return w;
    };
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    auto xb = ctx.create_buffer(x.size() * 4);
    ctx.write_buffer(xb, std::as_bytes(std::span(x)));

    // Row counts off the 8-row workgroup grid, so workgroups straddle
    // segments; out at offset 3, the other two at off1 / off2 of out2.
    const std::uint32_t rows[] = {20, 12, 7};
    const std::uint32_t out_off = 3, off1 = 5, off2 = 40;
    for (const std::uint32_t n_seg : {2u, 3u}) {
        DYNAMIC_SECTION(n_seg << " segments") {
            std::vector<std::vector<std::byte>> ws;
            std::vector<VulkanContext::Buffer> wb;
            for (std::uint32_t i = 0; i < n_seg; ++i) {
                ws.push_back(random_q4k(rows[i]));
                wb.push_back(ctx.create_buffer(ws.back().size()));
                ctx.write_buffer(wb.back(), ws.back());
            }
            std::vector<float> init(64, -7.0f);
            auto ob = ctx.create_buffer(init.size() * 4);
            auto o2b = ctx.create_buffer(init.size() * 4);
            ctx.write_buffer(ob, std::as_bytes(std::span(init)));
            ctx.write_buffer(o2b, std::as_bytes(std::span(init)));
            const std::uint32_t seg1 = rows[0], seg2 = seg1 + rows[1];
            const std::uint32_t total = n_seg == 3 ? seg2 + rows[2] : seg2;
            const VulkanContext::Buffer bufs[] = {
                wb[0], xb, ob, o2b, wb[1], n_seg == 3 ? wb[2] : wb[1]};
            const std::uint32_t push[] = {
                total, cols, out_off, 0, 0, 0,
                std::bit_cast<std::uint32_t>(1.0f), seg1, off1, seg2, off2};
            ctx.begin_batch();
            ctx.dispatch(Kernel::kMatvecQ4_KSeg, bufs, push,
                         VulkanContext::matvec_groups(Kernel::kMatvecQ4_KSeg,
                                                      total));
            ctx.end_batch();
            std::vector<float> o(64), o2(64);
            ctx.read_buffer(ob, std::as_writable_bytes(std::span(o)));
            ctx.read_buffer(o2b, std::as_writable_bytes(std::span(o2)));
            std::vector<float> want_o = init, want_o2 = init;
            const std::uint32_t at[] = {out_off, off1, off2};
            for (std::uint32_t i = 0; i < n_seg; ++i) {
                std::vector<float> cpu(rows[i]);
                locus::backend::Mat m{locus::gguf::TensorType::kQ4_K,
                                      ws[i].data(), rows[i], cols};
                locus::backend::matvec(m, x, cpu);
                auto& dst = i == 0 ? want_o : want_o2;
                std::copy(cpu.begin(), cpu.end(), dst.begin() + at[i]);
            }
            for (std::size_t i = 0; i < o.size(); ++i) {
                INFO("row " << i);
                REQUIRE(o[i] == Catch::Approx(want_o[i]).margin(1e-3));
                REQUIRE(o2[i] == Catch::Approx(want_o2[i]).margin(1e-3));
            }
            for (auto b : wb) {
                ctx.destroy_buffer(b);
            }
            ctx.destroy_buffer(ob);
            ctx.destroy_buffer(o2b);
        }
    }
    ctx.destroy_buffer(xb);
}

TEST_CASE("vulkan IQ2_XXS matvec matches the CPU reference",
          "[vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    std::mt19937 rng(51);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> byte_d(0, 255);

    // IQ2_XXS: 66-byte blocks of 256 elements, f16 d at offset 0.
    const std::uint32_t rows = 4, cols = 512;
    const std::size_t nblk = rows * (cols / 256);
    std::vector<std::byte> w(nblk * 66);
    for (auto& b : w) {
        b = static_cast<std::byte>(byte_d(rng));
    }
    for (std::size_t blk = 0; blk < nblk; ++blk) {
        const std::uint16_t d = locus::backend::f32_to_f16(0.01f);
        std::memcpy(w.data() + blk * 66, &d, 2);
    }
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }

    locus::backend::Mat m{locus::gguf::TensorType::kIQ2_XXS,
                          w.data(), rows, cols};
    std::vector<float> cpu(rows), gpu(rows);
    locus::backend::matvec(m, x, cpu);
    // The op path binds the grid+signs SSBO for the IQ kernel.
    locus::backend::matvec_vulkan(m, x, gpu);
    for (std::uint32_t r = 0; r < rows; ++r) {
        REQUIRE(gpu[r] ==
                Catch::Approx(cpu[r]).margin(1e-2).epsilon(1e-3));
    }
}

TEST_CASE("vulkan IQ3_XXS and IQ4_XS matvec match the CPU "
          "reference",
          "[vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    std::mt19937 rng(53);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> byte_d(0, 255);

    struct Case {
        locus::gguf::TensorType type;
        std::size_t block_bytes;
    };
    const Case cases[] = {
        {locus::gguf::TensorType::kIQ3_XXS, 98},
        {locus::gguf::TensorType::kIQ4_XS, 136},
    };
    const std::uint32_t rows = 4, cols = 512;
    std::vector<float> x(cols);
    for (auto& v : x) {
        v = dist(rng);
    }
    for (const auto& c : cases) {
        INFO("type " << static_cast<int>(c.type));
        const std::size_t nblk = rows * (cols / 256);
        std::vector<std::byte> w(nblk * c.block_bytes);
        for (auto& b : w) {
            b = static_cast<std::byte>(byte_d(rng));
        }
        for (std::size_t blk = 0; blk < nblk; ++blk) {
            const std::uint16_t d =
                locus::backend::f32_to_f16(0.01f);
            std::memcpy(w.data() + blk * c.block_bytes, &d, 2);
        }
        locus::backend::Mat m{c.type, w.data(), rows, cols};
        std::vector<float> cpu(rows), gpu(rows);
        locus::backend::matvec(m, x, cpu);
        locus::backend::matvec_vulkan(m, x, gpu);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(gpu[r] == Catch::Approx(cpu[r])
                                  .margin(1e-2)
                                  .epsilon(1e-3));
        }
    }
}

TEST_CASE("gpu matvec beats scalar cpu at real-model sizes",
          "[vulkan][benchmark]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;

    const std::uint32_t n = 2048;  // a 7B-class weight matrix dim
    std::vector<float> w(static_cast<std::size_t>(n) * n);
    std::vector<float> x(n);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& v : w) {
        v = dist(rng);
    }
    for (auto& v : x) {
        v = dist(rng);
    }

    auto wb = ctx.create_buffer(w.size() * 4);
    auto xb = ctx.create_buffer(x.size() * 4);
    auto ob = ctx.create_buffer(x.size() * 4);
    ctx.write_buffer(wb, std::as_bytes(std::span(w)));
    ctx.write_buffer(xb, std::as_bytes(std::span(x)));

    using clock = std::chrono::steady_clock;
    const int iters = 32;

    std::vector<float> gpu(n), cpu(n);
    ctx.matvec_f32(wb, n, n, xb, ob);  // warm up
    auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        ctx.matvec_f32(wb, n, n, xb, ob);
    }
    auto gpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now() - t0)
            .count() /
        iters;
    ctx.read_buffer(ob, std::as_writable_bytes(std::span(gpu)));

    locus::backend::Mat m{
        locus::gguf::TensorType::kF32,
        reinterpret_cast<const std::byte*>(w.data()), n, n};
    t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        locus::backend::matvec(m, x, cpu);
    }
    auto cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now() - t0)
            .count() /
        iters;

    WARN("matvec 2048x2048: gpu " << gpu_us << "us, scalar cpu "
                                  << cpu_us << "us");
    for (std::uint32_t r = 0; r < n; r += 97) {
        REQUIRE(gpu[r] == Catch::Approx(cpu[r]).margin(1e-3));
    }
    // GPU-vs-scalar speed is device-dependent: a desktop or
    // MoltenVK GPU beats scalar CPU at these sizes, but a tile GPU
    // (e.g. the Pi's VideoCore VII) does not. Correctness above is
    // the invariant; the timing is reported, not gated on.
    if (gpu_us >= cpu_us) {
        WARN("vulkan matvec not faster than scalar on this device: "
             "gpu " << gpu_us << "us vs cpu " << cpu_us << "us");
    }
    ctx.destroy_buffer(wb);
    ctx.destroy_buffer(xb);
    ctx.destroy_buffer(ob);
}

TEST_CASE("vulkan matvec validates sizes", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    std::vector<float> w(6), x(3), out(1);  // out too small
    REQUIRE_THROWS_AS(ctx.matvec_f32(w, 2, 3, x, out),
                      std::runtime_error);
}

TEST_CASE("vulkan memory_available drops as buffers are allocated",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    REQUIRE(ctx.memory_available() > 0);
    // memory_used, not memory_available: on a UMA device the budget
    // follows system free memory, so another process freeing memory
    // between two reads can raise memory_available (i#362).
    const std::uint64_t before = ctx.memory_used();
    const std::size_t bytes = std::size_t{256} << 20;
    auto b = ctx.create_buffer(bytes);
    const std::uint64_t during = ctx.memory_used();
    ctx.destroy_buffer(b);
    // Budget-backed drivers may report usage coarsely; require at
    // least half the allocation to show up.
    REQUIRE(during >= before + bytes / 2);
}

TEST_CASE("LOCUS_VK_BARRIERS keeps dispatch barriers", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    ::setenv("LOCUS_VK_BARRIERS", "1", 1);
    {
        VulkanContext ctx;
        REQUIRE_FALSE(ctx.serial_dispatch());
    }
    ::unsetenv("LOCUS_VK_BARRIERS");
}

// i#268: the server reaches the backend from the engine thread and from
// HTTP threads (image, vision, embeddings, audio) at once. Every
// matvec_vulkan from four threads must still equal the CPU result.
TEST_CASE("matvec_vulkan is safe from several threads", "[vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    constexpr int kThreads = 4, kIters = 200;
    const std::uint32_t rows = 96, cols = 256;
    struct Job {
        std::vector<float> w, x, want;
    };
    std::vector<Job> jobs(kThreads);
    std::mt19937 rng(41);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& j : jobs) {
        j.w.resize(std::size_t{rows} * cols);
        j.x.resize(cols);
        j.want.resize(rows);
        for (auto& v : j.w) v = dist(rng);
        for (auto& v : j.x) v = dist(rng);
        locus::backend::matvec(
            {locus::gguf::TensorType::kF32,
             reinterpret_cast<const std::byte*>(j.w.data()), rows, cols},
            j.x, j.want);
    }
    std::atomic<int> bad{0};
    std::vector<std::thread> threads;
    for (auto& j : jobs) {
        threads.emplace_back([&j, &bad] {
            std::vector<float> got(rows);
            const locus::backend::Mat m{
                locus::gguf::TensorType::kF32,
                reinterpret_cast<const std::byte*>(j.w.data()), rows, cols};
            for (int i = 0; i < kIters; ++i) {
                locus::backend::matvec_vulkan(m, j.x, got);
                for (std::uint32_t r = 0; r < rows; ++r) {
                    if (!(std::abs(got[r] - j.want[r]) <= 1e-3f)) {
                        ++bad;
                        break;
                    }
                }
            }
        });
    }
    for (auto& t : threads) t.join();
    // The weights are freed below; drop their uploads, which are keyed
    // by host pointer.
    locus::backend::vulkan_reset_state();
    REQUIRE(bad.load() == 0);
}

TEST_CASE("vulkan import_host reads host memory in place", "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    VulkanContext ctx;
    if (!ctx.host_import()) {
        SKIP("device has no VK_EXT_external_memory_host");
    }
    // Three anonymous pages; the "tensor" starts 32 bytes into the
    // second, off the 256-byte alignment imports bind at, as GGUF
    // tensors do.
    const long page = sysconf(_SC_PAGESIZE);
    const std::size_t span = static_cast<std::size_t>(page) * 3;
    void* map = mmap(nullptr, span, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    REQUIRE(map != MAP_FAILED);
    auto* base = static_cast<std::byte*>(map);
    const std::size_t n = 1000;
    auto* src = reinterpret_cast<float*>(base + page + 32);
    for (std::size_t i = 0; i < n; ++i) {
        src[i] = static_cast<float>(i) * 0.5f - 7.0f;
    }
    {
        auto b = ctx.import_host(src, n * 4, n * 4);
        REQUIRE(b.impl != nullptr);
        REQUIRE(ctx.mapped(b) == src);
        // A kernel reads it through the descriptor offset: element 0
        // is src[0], not the 256-aligned word below it.
        auto out = ctx.create_buffer(n * 4);
        ctx.begin_batch();
        const VulkanContext::Buffer bufs[] = {b, out};
        const std::uint32_t push[] = {static_cast<std::uint32_t>(n), 0,
                                      0};
        ctx.dispatch(Kernel::kCopy, bufs, push, (n + 255) / 256);
        ctx.end_batch();
        std::vector<float> got(n);
        ctx.read_buffer(out, std::as_writable_bytes(std::span(got)));
        for (std::size_t i = 0; i < n; ++i) {
            REQUIRE(got[i] == src[i]);
        }
        ctx.destroy_buffer(out);
        ctx.destroy_buffer(b);
    }
    // Padding that would reach past the last mapped page is refused
    // (the caller then copies).
    auto* tail = base + span - 64;
    REQUIRE(ctx.import_host(tail, 128, 64).impl == nullptr);
    munmap(map, span);
}

TEST_CASE("vulkan drops imported weights when their file is unmapped",
          "[vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    // A 64 x 512 F32 matrix (128 KB, a 16-byte multiple: imported where
    // the device can) in a file mapped, used, unmapped, then rewritten
    // with other weights and mapped again (usually at the same
    // address). The second matvec must see the new weights: the unmap
    // hook dropped the first import, so no stale buffer (over freed
    // pages) is hit by the reused pointer.
    const std::uint32_t rows = 64, cols = 512;
    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("locus_import_" + std::to_string(::getpid()) + ".bin"))
            .string();
    std::vector<float> x(cols);
    std::mt19937 rng(91);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) {
        v = dist(rng);
    }
    const void* first = nullptr;
    for (const float shift : {0.0f, 3.0f}) {
        std::vector<float> w(std::size_t{rows} * cols);
        for (auto& v : w) {
            v = dist(rng) + shift;
        }
        {
            std::FILE* f = std::fopen(path.c_str(), "wb");
            REQUIRE(f != nullptr);
            REQUIRE(std::fwrite(w.data(), 4, w.size(), f) == w.size());
            std::fclose(f);
        }
        locus::sys::MappedFile mf(path);
        const void* data = mf.bytes().data();
        INFO("mapping " << data << (data == first ? " (same address)" : ""));
        first = first == nullptr ? data : first;
        const locus::backend::Mat m{locus::gguf::TensorType::kF32,
                                    static_cast<const std::byte*>(data),
                                    rows, cols};
        std::vector<float> want(rows), got(rows);
        locus::backend::matvec(m, x, want);
        locus::backend::matvec_vulkan(m, x, got);
        for (std::uint32_t r = 0; r < rows; ++r) {
            REQUIRE(got[r] == Catch::Approx(want[r]).margin(1e-3));
        }
    }
    std::remove(path.c_str());
}

TEST_CASE("vulkan reports whether a batch is open or abandoned",
          "[vulkan]") {
    if (!VulkanContext::available()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    // The unmap hook aborts while a batch is open (its weights would be
    // freed under recorded commands), so the state must be exact.
    VulkanContext ctx;
    REQUIRE_FALSE(ctx.batch_open());
    ctx.begin_batch();
    REQUIRE(ctx.batch_open());
    ctx.end_batch();
    REQUIRE_FALSE(ctx.batch_open());
    // An abandoned batch (the unwind path) closes too, and the next
    // batch runs normally.
    ctx.begin_batch();
    ctx.abandon_batch();
    REQUIRE_FALSE(ctx.batch_open());
    ctx.begin_batch();
    REQUIRE(ctx.batch_open());
    ctx.end_batch();
    REQUIRE_FALSE(ctx.batch_open());
}
