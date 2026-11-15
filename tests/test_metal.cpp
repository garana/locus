// The Metal backend's skeleton (i#865): device, shared buffers, the
// kernel library compiled from source, and the registry entry.

#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <mutex>
#include <random>
#include <tuple>
#include <utility>
#include <span>
#include <stdexcept>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/metal/context.hpp"
#include "locus/backend/metal/matvec.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"
#include "locus/sys/mmap_file.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::backend::metal::Context;

namespace {

template <typename T>
std::span<const std::byte> bytes_of(const T& v) {
    return std::as_bytes(std::span<const T>(&v, 1));
}

}  // namespace

TEST_CASE("metal is in the registry, selectable when usable",
          "[metal]") {
    const auto* b = locus::backend::find_backend("metal");
    REQUIRE(b != nullptr);
    CHECK(b->available == Context::usable());
    CHECK(b->selectable == Context::usable());
    // Never the default: the GPU backends before it stay first.
    CHECK(locus::backend::best_backend().name != "metal");
    if (Context::usable()) {
        CHECK(&locus::backend::resolve_backend("metal") == b);
        // Stage 1 runs the math on the CPU kernels.
        CHECK(b->ops.matvec != nullptr);
        CHECK(b->ops.dequant_row != nullptr);
    } else {
        CHECK_THROWS_AS(locus::backend::resolve_backend("metal"),
                        std::invalid_argument);
    }
}

TEST_CASE("metal opens the device and compiles its kernels", "[metal]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    Context& ctx = Context::get();
    CHECK(&Context::get() == &ctx);  // one per process
    CHECK_FALSE(ctx.device_name().empty());
    CHECK(ctx.has_kernel("add_f32"));
    CHECK_FALSE(ctx.has_kernel("no_such_kernel"));
}

TEST_CASE("metal buffers round-trip through shared storage", "[metal]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    Context& ctx = Context::get();
    Context::Buffer z = ctx.alloc(4096);
    REQUIRE(z.data != nullptr);
    CHECK(z.bytes == 4096);
    const auto* zb = static_cast<const unsigned char*>(z.data);
    CHECK(std::all_of(zb, zb + 4096, [](unsigned char c) { return c == 0; }));
    ctx.free(z);
    CHECK(z.handle == nullptr);
    CHECK(z.data == nullptr);
    ctx.free(z);  // freeing an empty buffer is a no-op

    std::vector<std::byte> src(1000);
    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<std::byte>(i * 7 + 3);
    }
    Context::Buffer u = ctx.upload(src);
    REQUIRE(u.bytes == src.size());
    CHECK(std::memcmp(u.data, src.data(), src.size()) == 0);
    ctx.free(u);

    Context::Buffer e = ctx.alloc(0);  // a valid, empty buffer
    CHECK(e.handle != nullptr);
    CHECK(e.bytes == 0);
    ctx.free(e);
}

TEST_CASE("metal add_f32 matches the CPU", "[metal]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    Context& ctx = Context::get();
    // Not a multiple of the group: the tail threads must not write.
    constexpr std::uint32_t n = 1000;
    std::vector<float> a(n), b(n), want(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        a[i] = 0.5f * static_cast<float>(i) - 100.0f;
        b[i] = 1.0f / static_cast<float>(i + 1);
        want[i] = a[i] + b[i];
    }
    Context::Buffer ba = ctx.upload(std::as_bytes(std::span(a)));
    Context::Buffer bb = ctx.upload(std::as_bytes(std::span(b)));
    // One float past the end, which must stay untouched.
    Context::Buffer out = ctx.alloc((n + 1) * sizeof(float));
    static_cast<float*>(out.data)[n] = 42.0f;
    const Context::Arg args[] = {{&ba, 0}, {&bb, 0}, {&out, 0}};
    const std::uint32_t params = n;
    ctx.dispatch("add_f32", args, bytes_of(params), n, 64);
    const auto* got = static_cast<const float*>(out.data);
    for (std::uint32_t i = 0; i < n; ++i) {
        REQUIRE(got[i] == want[i]);
    }
    CHECK(got[n] == 42.0f);

    // A buffer offset: add the second half of a to b's first half.
    ctx.dispatch("add_f32",
                 std::vector<Context::Arg>{{&ba, (n / 2) * sizeof(float)},
                                           {&bb, 0},
                                           {&out, 0}},
                 bytes_of(n / 2), n / 2, 32);
    for (std::uint32_t i = 0; i < n / 2; ++i) {
        REQUIRE(got[i] == a[n / 2 + i] + b[i]);
    }

    CHECK_THROWS_AS(ctx.dispatch("no_such_kernel", args, bytes_of(params),
                                 n, 64),
                    std::invalid_argument);
    CHECK_THROWS_AS(ctx.dispatch("add_f32", args, bytes_of(params), n,
                                 1u << 20),
                    std::invalid_argument);
    ctx.free(ba);
    ctx.free(bb);
    ctx.free(out);
}

namespace {

using locus::gguf::TensorType;

// rows x cols weights of type t: random bytes, with each block's f16
// scales set to small values so the dot products stay finite.
std::vector<std::byte> random_weights(TensorType t, std::uint32_t rows,
                                      std::uint32_t cols, std::uint32_t seed) {
    std::mt19937 rng(seed);
    if (t == TensorType::kF32) {
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        std::vector<float> f(static_cast<std::size_t>(rows) * cols);
        for (auto& v : f) {
            v = d(rng);
        }
        std::vector<std::byte> b(f.size() * sizeof(float));
        std::memcpy(b.data(), f.data(), b.size());
        return b;
    }
    struct Blk {
        std::uint32_t elems, bytes, d_off;
        int dmin_off;
    };
    if (t == TensorType::kBF16) {
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        std::vector<std::byte> b(static_cast<std::size_t>(rows) * cols * 2);
        for (std::size_t i = 0; i < b.size(); i += 2) {
            const float f = d(rng);
            std::uint32_t u;
            std::memcpy(&u, &f, 4);
            const auto h = static_cast<std::uint16_t>(u >> 16);
            std::memcpy(b.data() + i, &h, 2);
        }
        return b;
    }
    if (t == TensorType::kF16) {
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        std::vector<std::byte> b(static_cast<std::size_t>(rows) * cols * 2);
        for (std::size_t i = 0; i < b.size(); i += 2) {
            const std::uint16_t h = locus::backend::f32_to_f16(d(rng));
            std::memcpy(b.data() + i, &h, 2);
        }
        return b;
    }
    const Blk k = t == TensorType::kQ8_0   ? Blk{32, 34, 0, -1}
                  : t == TensorType::kQ4_0 ? Blk{32, 18, 0, -1}
                  : t == TensorType::kQ5_0 ? Blk{32, 22, 0, -1}
                  : t == TensorType::kQ4_K ? Blk{256, 144, 0, 2}
                  : t == TensorType::kQ5_K ? Blk{256, 176, 0, 2}
                                           : Blk{256, 210, 208, -1};
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<std::byte> w(static_cast<std::size_t>(rows) *
                             (cols / k.elems) * k.bytes);
    for (auto& b : w) {
        b = static_cast<std::byte>(byte(rng));
    }
    const std::uint16_t d = locus::backend::f32_to_f16(0.01f);
    const std::uint16_t dmin = locus::backend::f32_to_f16(0.005f);
    for (std::size_t at = 0; at < w.size(); at += k.bytes) {
        std::memcpy(w.data() + at + k.d_off, &d, 2);
        if (k.dmin_off >= 0) {
            std::memcpy(w.data() + at + k.dmin_off, &dmin, 2);
        }
    }
    return w;
}

}  // namespace

TEST_CASE("metal matvec matches the CPU for each weight type",
          "[metal]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    namespace mb = locus::backend::metal;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (const TensorType t :
         {TensorType::kF32, TensorType::kF16, TensorType::kBF16,
          TensorType::kQ8_0, TensorType::kQ4_0, TensorType::kQ5_0,
          TensorType::kQ4_K, TensorType::kQ5_K, TensorType::kQ6_K}) {
        CHECK(mb::matvec_supported(t));
        // Rows not a multiple of the 8 a threadgroup takes; one shape
        // with fewer blocks per row than the 16 lanes.
        for (const auto& [rows, cols] :
             {std::pair<std::uint32_t, std::uint32_t>{37, 512},
              {1027, 3072}, {5, 256}}) {
            CAPTURE(static_cast<int>(t), rows, cols);
            const auto w = random_weights(t, rows, cols, rows + cols);
            const locus::backend::Mat m{t, w.data(), rows, cols};
            std::vector<float> x(cols), want(rows), got(rows);
            for (auto& v : x) {
                v = dist(rng);
            }
            locus::backend::matvec(m, x, want);
            mb::matvec(m, x, got);
            for (std::uint32_t r = 0; r < rows; ++r) {
                REQUIRE(got[r] ==
                        Catch::Approx(want[r]).margin(1e-3).epsilon(1e-4));
            }
            // The cached weight serves a second call with new inputs.
            for (auto& v : x) {
                v = dist(rng);
            }
            locus::backend::matvec(m, x, want);
            mb::matvec(m, x, got);
            REQUIRE(got[rows - 1] ==
                    Catch::Approx(want[rows - 1]).margin(1e-3).epsilon(1e-4));
            // The repeat hook gives the same result.
            std::vector<float> rep(rows);
            CHECK(mb::matvec_repeat(m, x, rep, 3) > 0.0);
            CHECK(rep == got);
        }
    }
    CHECK_FALSE(mb::matvec_supported(TensorType::kQ2_K));
    std::vector<float> x(4), out(4);
    const std::vector<std::byte> none(64);
    const locus::backend::Mat bad{TensorType::kF32, none.data(), 4, 4};
    std::vector<float> shortx(3);
    CHECK_THROWS_AS(mb::matvec(bad, shortx, out), std::invalid_argument);
    CHECK_THROWS_AS(mb::matvec_repeat(bad, x, out, 0), std::invalid_argument);
    mb::drop_weights();
}

TEST_CASE("metal forward tracks the CPU on llama in each weight mix",
          "[metal][e2e]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    // Llama 3.2 1B in Q4_K_M (Q4_K and Q6_K), Q8_0, and two made from
    // the Q8_0 with llama-quantize --allow-requantize (i#907): Q4_0
    // (with a Q6_K head) and Q5_K_M (Q5_K and Q6_K); requantized, so
    // only fit for this GPU-against-CPU check, not for quality numbers.
    // Qwen2.5 0.5B Q8_0 is qwen2's attention (q / k / v biases, NEOX
    // RoPE); Qwen1.5-MoE-A2.7B adds routed and gated shared experts;
    // Qwen3-Coder-30B-A3B is qwen3moe (per-head q / k norm, 128 experts).
    int ran = 0;
    for (const char* name :
         {"llama-3.2-1b-q4_k_m.gguf", "llama-3.2-1b-q8_0.gguf",
          "llama-3.2-1b-q4_0.gguf", "llama-3.2-1b-q5_k_m.gguf",
          "qwen2.5-0.5b-instruct-q8_0.gguf",
          "Qwen1.5-MoE-A2.7B.Q4_K_M.gguf",
          "qwen3-coder-30b-a3b-instruct-q4_k_m.gguf"}) {
        const std::string path =
            std::string(LOCUS_SOURCE_DIR) + "/tests/models/" + name;
        if (!std::filesystem::exists(path)) {
            WARN(path << " absent: skipped");  // the others still run
            continue;
        }
        CAPTURE(name);
        ++ran;
        auto g = locus::gguf::GgufFile::open(path);
        auto model = locus::model::TransformerModel::load(g);
        // The CPU reference with f32 activations, as the GPU computes
        // (its Q8_K default rounds them, which moves the logits ~2%).
        model.use_q8k_activations(false);
        const auto tok = locus::tok::tokenizer_from_gguf(g);
        const std::uint32_t nv = model.hparams().n_vocab;
        // The same tokens through the best CPU backend and Metal: the
        // prompt, then the CPU's greedy continuation.
        std::vector<locus::tok::TokenId> ids =
            tok->encode("The capital of France is", true);
        const auto run = [&](std::string_view be) {
            model.use_backend(*locus::backend::find_backend(be));
            auto cache = model.make_cache();
            auto ws = model.make_workspace();
            locus::kv::PagedKvCache::Seq seq;
            std::vector<float> logits(nv);
            std::vector<std::vector<float>> all;
            for (std::size_t i = 0; i < ids.size(); ++i) {
                REQUIRE(cache.ensure_capacity(seq, 1));
                model.forward(ids[i], cache, seq, ws, logits);
                all.push_back(logits);
            }
            cache.release(seq);
            return all;
        };
        {
            // Extend ids with 12 greedy CPU tokens.
            model.use_backend(locus::backend::backends().front());
            auto cache = model.make_cache();
            auto ws = model.make_workspace();
            locus::kv::PagedKvCache::Seq seq;
            std::vector<float> logits(nv);
            for (std::size_t i = 0; i < ids.size(); ++i) {
                REQUIRE(cache.ensure_capacity(seq, 1));
                model.forward(ids[i], cache, seq, ws, logits);
            }
            for (int i = 0; i < 12; ++i) {
                ids.push_back(locus::model::argmax(logits));
                REQUIRE(cache.ensure_capacity(seq, 1));
                model.forward(ids.back(), cache, seq, ws, logits);
            }
            cache.release(seq);
        }
        const auto cpu = run(locus::backend::backends().front().name);
        const std::uint64_t before =
            locus::model::TransformerModel::metal_forward_tokens();
        // The one-kernel attention (the default at this length), then
        // the three passes.
        for (const bool three : {false, true}) {
            CAPTURE(three);
            if (three) {
                setenv("LOCUS_METAL_ATTN_PASSES", "3", 1);
            }
            const auto gpu = run("metal");
            unsetenv("LOCUS_METAL_ATTN_PASSES");
            // Every token took the GPU path.
            CHECK(locus::model::TransformerModel::metal_forward_tokens() -
                      before ==
                  ids.size() * (three ? 2 : 1));
            for (std::size_t t = 0; t < ids.size(); ++t) {
                CAPTURE(t);
                CHECK(locus::model::argmax(gpu[t]) ==
                      locus::model::argmax(cpu[t]));
                float worst = 0.0f, scale = 0.0f;
                for (std::uint32_t r = 0; r < nv; ++r) {
                    worst = std::max(worst, std::abs(gpu[t][r] - cpu[t][r]));
                    scale = std::max(scale, std::abs(cpu[t][r]));
                }
                CHECK(worst <= 0.01f * scale);
            }
        }
    }
    if (ran == 0) {
        SKIP("none of the llama test models is present");
    }
}

TEST_CASE("metal matmuls match the CPU matvec per token",
          "[metal]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    namespace mb = locus::backend::metal;
    Context& ctx = Context::get();
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    // Rows and tokens off the 64 x 32 tile, strided token rows.
    for (const TensorType ty :
         {TensorType::kQ4_K, TensorType::kQ6_K, TensorType::kQ8_0,
          TensorType::kQ4_0, TensorType::kQ5_0, TensorType::kF32,
          TensorType::kF16, TensorType::kBF16}) {
        CAPTURE(static_cast<int>(ty));
        for (const auto& [rows, cols, n_tok] :
             {std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>{
                  200, 512, 37},
              {64, 256, 1}, {1027, 3072, 70}}) {
            CAPTURE(rows, cols, n_tok);
            const auto w = random_weights(ty, rows, cols, rows);
            const locus::backend::Mat m{ty, w.data(), rows, cols};
            const std::uint32_t xs = cols + 8, os = rows + 4;
            std::vector<float> x(std::size_t{n_tok} * xs);
            for (auto& v : x) {
                v = dist(rng);
            }
            std::vector<float> init(std::size_t{n_tok} * os);
            for (auto& v : init) {
                v = dist(rng);
            }
            Context::Buffer wb = ctx.upload(w);
            Context::Buffer xb = ctx.upload(std::as_bytes(std::span(x)));
            for (const bool acc : {false, true}) {
                CAPTURE(acc);
                Context::Buffer ob = ctx.upload(std::as_bytes(std::span(init)));
                {
                    std::lock_guard lk(mb::device_mutex());
                    REQUIRE(mb::record_matmul(ctx, ty, rows, cols, n_tok, wb,
                                              0, xb, 0, xs, ob, 0, os, acc));
                }
                const auto* got = static_cast<const float*>(ob.data);
                std::vector<float> want(rows);
                for (std::uint32_t t = 0; t < n_tok; ++t) {
                    locus::backend::matvec(
                        m, {x.data() + std::size_t{t} * xs, cols}, want);
                    for (std::uint32_t r = 0; r < rows; ++r) {
                        const float base =
                            acc ? init[std::size_t{t} * os + r] : 0.0f;
                        REQUIRE(got[std::size_t{t} * os + r] ==
                                Catch::Approx(base + want[r])
                                    .margin(1e-3)
                                    .epsilon(1e-4));
                    }
                    // Past the rows, the stride's padding is untouched.
                    for (std::uint32_t r = rows; r < os; ++r) {
                        REQUIRE(got[std::size_t{t} * os + r] ==
                                init[std::size_t{t} * os + r]);
                    }
                }
                ctx.free(ob);
            }
            ctx.free(wb);
            ctx.free(xb);
        }
    }
    // Types without a matmul kernel say so and record nothing.
    Context::Buffer none = ctx.alloc(64);
    std::lock_guard lk(mb::device_mutex());
    CHECK_FALSE(mb::record_matmul(ctx, TensorType::kQ5_K, 4, 256, 2, none, 0,
                                  none, 0, 256, none, 0, 4, false));
    ctx.free(none);
}

TEST_CASE("metal grouped matmul matches a matvec a slot", "[metal]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    namespace mb = locus::backend::metal;
    Context& ctx = Context::get();
    std::mt19937 rng(29);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    // 23 tokens of 3 picks over 6 experts, expert 0 taking most slots
    // (past one 32-slot tile), expert 5 none.
    const std::uint32_t n_expert = 6, k = 3, n_tok = 23, slots = n_tok * k;
    std::vector<std::uint32_t> route(slots);
    std::uniform_int_distribution<std::uint32_t> pick(0, 9);
    for (auto& e : route) {
        const std::uint32_t r = pick(rng);
        e = r < 5 ? 0 : r - 5;  // 0 .. 4
    }
    for (const TensorType ty :
         {TensorType::kQ4_K, TensorType::kQ6_K, TensorType::kQ8_0,
          TensorType::kQ5_0}) {
        CAPTURE(static_cast<int>(ty));
        const std::uint32_t rows = 100, cols = 512;
        const auto w = random_weights(ty, rows * n_expert, cols, 7);
        const std::uint64_t eb = mb::matvec_weight_bytes(ty, rows, cols);
        // Gate-like (x row s / k) and down-like (x row s) inputs.
        for (const std::uint32_t div : {k, 1u}) {
            CAPTURE(div);
            const std::uint32_t xs = cols + 8, os = rows + 4;
            std::vector<float> x(std::size_t{slots} * xs);
            for (auto& v : x) {
                v = dist(rng);
            }
            std::vector<float> init(std::size_t{slots} * os, 7.0f);
            Context::Buffer wb = ctx.upload(w);
            Context::Buffer xb = ctx.upload(std::as_bytes(std::span(x)));
            Context::Buffer ob = ctx.upload(std::as_bytes(std::span(init)));
            Context::Buffer rb = ctx.upload(std::as_bytes(std::span(route)));
            Context::Buffer ib = ctx.alloc(
                (slots + 3 * std::size_t{mb::moe_max_tiles(slots, n_expert)}) *
                4);
            {
                std::lock_guard lk(mb::device_mutex());
                mb::record_moe_tiles(ctx, rb, slots, n_expert, ib);
                REQUIRE(mb::record_matmul_id(ctx, ty, rows, cols, slots,
                                             n_expert, wb, 0, eb, xb, xs, div,
                                             ob, os, ib));
            }
            const auto* got = static_cast<const float*>(ob.data);
            std::vector<float> want(rows);
            for (std::uint32_t s = 0; s < slots; ++s) {
                const locus::backend::Mat m{
                    ty, w.data() + route[s] * eb, rows, cols};
                locus::backend::matvec(
                    m, {x.data() + std::size_t{s / div} * xs, cols}, want);
                for (std::uint32_t r = 0; r < rows; ++r) {
                    REQUIRE(got[std::size_t{s} * os + r] ==
                            Catch::Approx(want[r]).margin(1e-3).epsilon(1e-4));
                }
                // The stride's padding is untouched.
                REQUIRE(got[std::size_t{s} * os + rows] == 7.0f);
            }
            for (auto* b : {&wb, &xb, &ob, &rb, &ib}) {
                ctx.free(*b);
            }
        }
    }
}

TEST_CASE("metal batched prefill tracks the CPU", "[metal][e2e]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    int ran = 0;
    // Llama Q4_K_M and Q8_0, qwen2's attention (biases, NEOX RoPE), and
    // MoE: routed and gated shared experts (Qwen1.5-MoE), qwen3moe.
    for (const char* name :
         {"llama-3.2-1b-q4_k_m.gguf", "llama-3.2-1b-q8_0.gguf",
          "qwen2.5-0.5b-instruct-q8_0.gguf", "Qwen1.5-MoE-A2.7B.Q4_K_M.gguf",
          "qwen3-coder-30b-a3b-instruct-q4_k_m.gguf"}) {
        const std::string path =
            std::string(LOCUS_SOURCE_DIR) + "/tests/models/" + name;
        if (!std::filesystem::exists(path)) {
            WARN(path << " absent: skipped");
            continue;
        }
        ++ran;
        CAPTURE(name);
        auto g = locus::gguf::GgufFile::open(path);
        auto model = locus::model::TransformerModel::load(g);
        model.use_q8k_activations(false);  // the f32 CPU reference
        const auto tok = locus::tok::tokenizer_from_gguf(g);
        const std::uint32_t nv = model.hparams().n_vocab;
        // Past two 256-token sub-chunks, so the all-logits run takes
        // three (a one-row run takes the whole prompt), and the next
        // token's decode takes the split attention (past 512).
        std::string text;
        for (int i = 0; i < 60; ++i) {
            text += "The quick brown fox jumps over the lazy dog. ";
        }
        const auto ids = tok->encode(text, true);
        REQUIRE(ids.size() > 512);
        const auto prefill = [&](std::string_view be, bool all) {
            model.use_backend(*locus::backend::find_backend(be));
            auto cache = model.make_cache();
            auto ws = model.make_workspace();
            locus::kv::PagedKvCache::Seq seq;
            REQUIRE(cache.ensure_capacity(
                seq, static_cast<std::uint32_t>(ids.size()) + 1));
            std::vector<float> logits((all ? ids.size() : 1) * nv);
            model.forward_batch(ids, cache, seq, ws, logits, all);
            REQUIRE(seq.n_tokens == ids.size());
            // One more token on top, to check the KV the prefill wrote.
            std::vector<float> next(nv);
            model.forward(locus::model::argmax(std::span<const float>(
                              logits.data() + logits.size() - nv, nv)),
                          cache, seq, ws, next);
            cache.release(seq);
            return std::pair(logits, next);
        };
        const auto close = [&](std::span<const float> a,
                               std::span<const float> b) {
            CHECK(locus::model::argmax(a) == locus::model::argmax(b));
            float worst = 0.0f, scale = 0.0f;
            for (std::size_t r = 0; r < a.size(); ++r) {
                worst = std::max(worst, std::abs(a[r] - b[r]));
                scale = std::max(scale, std::abs(b[r]));
            }
            CHECK(worst <= 0.01f * scale);
        };
        const auto cpu = prefill(locus::backend::backends().front().name,
                                 false);
        // The tiled causal attention (the default), then attn_rows.
        for (const bool tiled : {true, false}) {
            CAPTURE(tiled);
            if (!tiled) {
                setenv("LOCUS_METAL_ATTN_TILED", "0", 1);
            }
            const std::uint64_t before =
                locus::model::TransformerModel::metal_forward_tokens();
            const auto gpu = prefill("metal", false);
            // The prompt and the next token all ran on the GPU.
            CHECK(locus::model::TransformerModel::metal_forward_tokens() -
                      before ==
                  ids.size() + 1);
            close(gpu.first, cpu.first);
            close(gpu.second, cpu.second);
            // All logits: a row each, the last as the one-row run's.
            const auto all = prefill("metal", true);
            unsetenv("LOCUS_METAL_ATTN_TILED");
            REQUIRE(all.first.size() == ids.size() * nv);
            close(std::span<const float>(all.first).subspan(
                      (ids.size() - 1) * nv, nv),
                  cpu.first);
        }
        {
            // The tiled prefill attention, not the simdgroup-matrix one.
            setenv("LOCUS_METAL_ATTN_MMA", "0", 1);
            const auto tiled = prefill("metal", false);
            unsetenv("LOCUS_METAL_ATTN_MMA");
            close(tiled.first, cpu.first);
            close(tiled.second, cpu.second);
            // The decode's one-kernel attention, not split-K.
            setenv("LOCUS_METAL_ATTN_SPLIT", "0", 1);
            const auto whole = prefill("metal", false);
            unsetenv("LOCUS_METAL_ATTN_SPLIT");
            close(whole.second, cpu.second);
        }
        if (model.hparams().n_expert > 0) {
            // The experts as a matvec slot each, not grouped matmuls.
            setenv("LOCUS_METAL_MOE_MM", "0", 1);
            const auto slots = prefill("metal", false);
            unsetenv("LOCUS_METAL_MOE_MM");
            close(slots.first, cpu.first);
            close(slots.second, cpu.second);
        }
    }
    if (ran == 0) {
        SKIP("none of the llama test models is present");
    }
}

TEST_CASE("metal tiled and simdgroup-matrix attention match attn_rows",
          "[metal]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    // The parameter blocks of kernels.metal.
    struct AttnParams {
        std::uint64_t block_floats, layer_off, v_off;
        std::uint32_t n_pos, block_tokens, n_heads, n_kv_heads, head_dim;
        float kq_scale;
    };
    struct AttnRowsParams {
        AttnParams a;
        std::uint32_t base, q_stride;
    };
    Context& ctx = Context::get();
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    // Head widths 32 (64-position chunks before the cap), 64 and 128;
    // queries off the 8- and 32-query tiles (and past one 32-query
    // block); a base so chunks span blocks.
    for (const auto& [hd, m] :
         {std::pair<std::uint32_t, std::uint32_t>{32, 21}, {64, 21},
          {128, 21}, {64, 70}, {128, 70}}) {
        CAPTURE(hd, m);
        const std::uint32_t H = 4, KV = 2, bt = 16, base = 37;
        const std::uint32_t n_pos = base + m;
        const std::uint32_t n_blocks = (n_pos + bt - 1) / bt;
        const std::size_t kvd = std::size_t{KV} * hd;
        const std::size_t layer = 2 * bt * kvd;  // one layer: K then V
        // Blocks in reverse order, so the block table matters.
        std::vector<std::uint32_t> blocks(n_blocks);
        for (std::uint32_t b = 0; b < n_blocks; ++b) {
            blocks[b] = n_blocks - 1 - b;
        }
        std::vector<float> pool(std::size_t{n_blocks} * layer);
        for (auto& v : pool) {
            v = dist(rng);
        }
        const std::uint32_t qs = H * hd;
        std::vector<float> q(std::size_t{m} * qs);
        for (auto& v : q) {
            v = dist(rng);
        }
        Context::Buffer pb = ctx.upload(std::as_bytes(std::span(pool)));
        Context::Buffer bb = ctx.upload(std::as_bytes(std::span(blocks)));
        Context::Buffer qb = ctx.upload(std::as_bytes(std::span(q)));
        Context::Buffer o1 = ctx.alloc(q.size() * 4);
        Context::Buffer o2 = ctx.alloc(q.size() * 4);
        Context::Buffer o3 = ctx.alloc(q.size() * 4);
        const AttnRowsParams p{{layer, 0, bt * kvd, m, bt, H, KV, hd,
                                1.0f / std::sqrt(float(hd))},
                               base,
                               qs};
        const auto params = std::as_bytes(std::span(&p, 1));
        const Context::Arg a1[] = {{&qb, 0}, {&pb, 0}, {&bb, 0}, {&o1, 0}};
        const Context::Arg a2[] = {{&qb, 0}, {&pb, 0}, {&bb, 0}, {&o2, 0}};
        ctx.dispatch("attn_rows", a1, params, m * H * 256, 256);
        ctx.dispatch("attn_rows_tiled", a2, params, ((m + 7) / 8) * H * 256,
                     256);
        const Context::Arg a3[] = {{&qb, 0}, {&pb, 0}, {&bb, 0}, {&o3, 0}};
        ctx.dispatch("attn_rows_mma", a3, params, ((m + 31) / 32) * H * 128,
                     128);
        const auto* r = static_cast<const float*>(o1.data);
        const auto* t = static_cast<const float*>(o2.data);
        const auto* u = static_cast<const float*>(o3.data);
        for (std::size_t i = 0; i < q.size(); ++i) {
            REQUIRE(t[i] == Catch::Approx(r[i]).margin(1e-5).epsilon(1e-4));
            REQUIRE(u[i] == Catch::Approx(r[i]).margin(1e-5).epsilon(1e-4));
        }
        for (auto* b : {&pb, &bb, &qb, &o1, &o2, &o3}) {
            ctx.free(*b);
        }
    }
}

TEST_CASE("metal matvec_batch matches the CPU batch, row-major",
          "[metal]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    namespace mb = locus::backend::metal;
    std::mt19937 rng(29);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    // Types with a matmul, and Q4_0 / Q5_K, which take a matvec a token.
    for (const TensorType t :
         {TensorType::kF32, TensorType::kF16, TensorType::kBF16,
          TensorType::kQ8_0, TensorType::kQ4_K, TensorType::kQ6_K,
          TensorType::kQ4_0, TensorType::kQ5_K}) {
        CAPTURE(static_cast<int>(t));
        const std::uint32_t rows = 97, cols = 512, n = 45;
        const auto w = random_weights(t, rows, cols, 7);
        const locus::backend::Mat m{t, w.data(), rows, cols};
        std::vector<float> x(std::size_t{n} * cols);
        std::vector<float> want(std::size_t{n} * rows), got(want.size());
        for (auto& v : x) {
            v = dist(rng);
        }
        locus::backend::matvec_batch_scalar(m, x, want, n);
        mb::matvec_batch(m, x, got, n);
        for (std::size_t i = 0; i < want.size(); ++i) {
            REQUIRE(got[i] ==
                    Catch::Approx(want[i]).margin(1e-3).epsilon(1e-4));
        }
    }
    const auto& ops = mb::image_ops();
    CHECK(ops.matvec_batch == &mb::matvec_batch);
    CHECK(ops.self_parallel);
    CHECK_FALSE(ops.mt_safe);
    mb::drop_weights();
}

TEST_CASE("metal matvec re-uploads a new matrix at a reused address",
          "[metal]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    namespace mb = locus::backend::metal;
    // One buffer holding a matrix, then another of the same type and
    // shape (a temporary reused, a model mapped where one was): the
    // second matvec must see the new bytes.
    const std::uint32_t rows = 64, cols = 512;
    auto w = random_weights(TensorType::kQ8_0, rows, cols, 1);
    const locus::backend::Mat m{TensorType::kQ8_0, w.data(), rows, cols};
    std::vector<float> x(cols, 0.5f), got(rows), want(rows);
    mb::matvec(m, x, got);
    const auto w2 = random_weights(TensorType::kQ8_0, rows, cols, 2);
    std::copy(w2.begin(), w2.end(), w.begin());
    locus::backend::matvec(m, x, want);
    mb::matvec(m, x, got);
    for (std::uint32_t r = 0; r < rows; ++r) {
        REQUIRE(got[r] == Catch::Approx(want[r]).margin(1e-3).epsilon(1e-4));
    }
    mb::drop_weights();
}

TEST_CASE("metal forward on a GGUF parsed from memory copies its weights",
          "[metal][e2e]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    const std::string path = std::string(LOCUS_SOURCE_DIR) +
                             "/tests/models/qwen2.5-0.5b-instruct-q8_0.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP(path + " absent");
    }
    // A heap image, not the file's mapping: the Metal weights cannot be
    // no-copy wraps of it (page rounding would reach past the
    // allocation), so they are copies; the forward still runs on the
    // GPU and tracks the CPU.
    std::ifstream f(path, std::ios::binary);
    std::vector<std::byte> image(std::filesystem::file_size(path));
    f.read(reinterpret_cast<char*>(image.data()),
           static_cast<std::streamsize>(image.size()));
    REQUIRE(f.good());
    auto g = locus::gguf::GgufFile::parse(image);
    REQUIRE_FALSE(g.file_backed());
    auto model = locus::model::TransformerModel::load(g);
    model.use_q8k_activations(false);
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    const std::uint32_t nv = model.hparams().n_vocab;
    const auto ids = tok->encode("The capital of France is", true);
    const auto last = [&](std::string_view be) {
        model.use_backend(*locus::backend::find_backend(be));
        auto cache = model.make_cache();
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<float> logits(nv);
        for (const auto id : ids) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(id, cache, seq, ws, logits);
        }
        cache.release(seq);
        return logits;
    };
    const auto cpu = last(locus::backend::backends().front().name);
    const std::uint64_t before =
        locus::model::TransformerModel::metal_forward_tokens();
    const auto gpu = last("metal");
    CHECK(locus::model::TransformerModel::metal_forward_tokens() - before ==
          ids.size());
    CHECK(locus::model::argmax(gpu) == locus::model::argmax(cpu));
    float worst = 0.0f, scale = 0.0f;
    for (std::uint32_t r = 0; r < nv; ++r) {
        worst = std::max(worst, std::abs(gpu[r] - cpu[r]));
        scale = std::max(scale, std::abs(cpu[r]));
    }
    CHECK(worst <= 0.01f * scale);
}

namespace {

/** A temp file of n random floats, removed with the object. */
struct TempFloats {
    std::string path;
    explicit TempFloats(std::size_t n) {
        path = (std::filesystem::temp_directory_path() /
                ("locus-metal-unmap-" + std::to_string(::getpid()) +
                 ".bin"))
                   .string();
        std::vector<float> v(n);
        std::mt19937 rng(29);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (auto& x : v) {
            x = d(rng);
        }
        std::ofstream(path, std::ios::binary)
            .write(reinterpret_cast<const char*>(v.data()),
                   static_cast<std::streamsize>(v.size() * sizeof(float)));
    }
    ~TempFloats() { std::filesystem::remove(path); }
};

}  // namespace

// i#1029: unmapping a model file drops the Metal copies of its weights
// (whichever model held them), so a process that loads model after
// model keeps only the live ones on the device.
TEST_CASE("unmapping a file releases its Metal cached weights", "[metal]") {
    if (!Context::usable()) {
        SKIP("no Metal device");
    }
    namespace mb = locus::backend::metal;
    const std::uint32_t rows = 64, cols = 256;
    const TempFloats f(std::size_t{rows} * cols);
    const std::size_t base = mb::cached_weight_bytes();
    {
        const locus::sys::MappedFile map(f.path);
        const locus::backend::Mat w{TensorType::kF32, map.bytes().data(),
                                    rows, cols};
        std::vector<float> x(cols, 0.5f), out(rows);
        mb::matvec(w, x, out);
        REQUIRE(mb::cached_weight_bytes() ==
                base + std::size_t{rows} * cols * 4);
    }
    REQUIRE(mb::cached_weight_bytes() == base);
}

// The unmap hook cleans only a cache that exists: in a process that
// never ran Metal, unmapping a file makes no Metal context (meaningful
// run alone; in the whole suite an earlier test may have made it).
TEST_CASE("unmapping a file makes no Metal context", "[metal]") {
    const bool made = Context::made();
    INFO("made before: " << made);
    {
        const TempFloats f(1024);
        const locus::sys::MappedFile map(f.path);
        (void)map;
    }
    CHECK(Context::made() == made);
}
