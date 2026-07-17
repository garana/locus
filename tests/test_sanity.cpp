#include <string_view>

#include "catch_amalgamated.hpp"
#include "cppllm/version.hpp"

TEST_CASE("version string is populated", "[sanity]") {
    REQUIRE(std::string_view(cppllm::kVersion) == "0.1.0");
}
