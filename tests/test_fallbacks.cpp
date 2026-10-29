#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/fallbacks.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"

using locus::backend::Fallback;
using locus::backend::fallback;

namespace {

/** @returns The count of (from, to, reason), or 0 when unregistered. */
std::uint64_t count_of(const char* from, const char* to,
                       const char* reason) {
    std::uint64_t n = 0;
    locus::backend::for_each_fallback([&](const Fallback& f) {
        if (f.from == from && f.to == to && f.reason == reason) {
            n = f.n.load();
        }
    });
    return n;
}

}  // namespace

TEST_CASE("fallback counters register once and count", "[metrics]") {
    // Equal text shares one counter, even from a different literal.
    const std::string reason = "test reason";
    Fallback& a = fallback("test", "cpu", "test reason");
    Fallback& b = fallback("test", "cpu", reason.c_str());
    REQUIRE(&a == &b);
    REQUIRE(&a != &fallback("test", "per_op", "test reason"));
    const std::uint64_t n0 = count_of("test", "cpu", "test reason");
    a.add();
    a.add();
    REQUIRE(count_of("test", "cpu", "test reason") == n0 + 2);

    // The registry copies the names: a first registration from a
    // temporary string stays readable after the string is gone.
    {
        std::string tmp = "temporary reason";
        fallback("test", "cpu", tmp.c_str()).add();
        tmp.assign(tmp.size(), 'x');
    }
    REQUIRE(count_of("test", "cpu", "temporary reason") == 1);
}

// Q3_K has no Vulkan matvec shader: matvec_vulkan computes it on the
// CPU and counts that.
TEST_CASE("vulkan matvec without a shader counts a fallback",
          "[metrics][vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    constexpr std::size_t kQ3KBlock = 110;  // bytes per 256 weights
    std::vector<std::byte> w(kQ3KBlock);  // all zero: d = 0
    const locus::backend::Mat m{locus::gguf::TensorType::kQ3_K, w.data(),
                                1, 256};
    std::vector<float> x(256, 1.0f), out(1, -1.0f);
    const char* reason = "no matvec shader for the weight type";
    const std::uint64_t n0 = count_of("vulkan", "cpu", reason);
    locus::backend::matvec_vulkan(m, x, out);
    REQUIRE(out[0] == 0.0f);
    REQUIRE(count_of("vulkan", "cpu", reason) == n0 + 1);
}
