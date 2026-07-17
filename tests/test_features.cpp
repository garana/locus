#include "catch_amalgamated.hpp"
#include "cppllm/sys/features.hpp"

TEST_CASE("feature detection runs and reports", "[sys]") {
    auto f = cppllm::sys::detect();

#if defined(__aarch64__)
    REQUIRE(f.neon);
    REQUIRE_FALSE(f.avx2);
#elif defined(__x86_64__)
    REQUIRE_FALSE(f.neon);
#endif

    auto s = cppllm::sys::to_string(f);
    REQUIRE(s.find("neon") != std::string::npos);
    REQUIRE(s.find("vulkan") != std::string::npos);
}
