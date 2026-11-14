// The Metal backend's skeleton (i#865): device, shared buffers, the
// kernel library compiled from source, and the registry entry.

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/metal/context.hpp"
#include "locus/backend/registry.hpp"

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
