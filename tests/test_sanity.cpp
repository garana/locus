#include <string_view>

#include "catch_amalgamated.hpp"
#include "locus/version.hpp"

TEST_CASE("version string is populated", "[sanity]") {
    REQUIRE(std::string_view(locus::kVersion) == "0.1.0");
}
