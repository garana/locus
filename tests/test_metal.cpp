// The Metal backend's skeleton (i#865): device, shared buffers, the
// kernel library compiled from source, and the registry entry.

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <random>
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
         {TensorType::kF32, TensorType::kF16, TensorType::kQ8_0,
          TensorType::kQ4_0, TensorType::kQ5_0, TensorType::kQ4_K,
          TensorType::kQ5_K, TensorType::kQ6_K}) {
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
    // Q4_K_M (Q4_K and Q6_K), Q8_0, and two made from the Q8_0 with
    // llama-quantize --allow-requantize (i#907): Q4_0 (with a Q6_K
    // head) and Q5_K_M (Q5_K and Q6_K). Requantized, so only fit for
    // this GPU-against-CPU check, not for quality numbers.
    int ran = 0;
    for (const char* name :
         {"llama-3.2-1b-q4_k_m.gguf", "llama-3.2-1b-q8_0.gguf",
          "llama-3.2-1b-q4_0.gguf", "llama-3.2-1b-q5_k_m.gguf"}) {
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
