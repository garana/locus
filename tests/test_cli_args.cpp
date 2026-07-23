#include <array>

#include "catch_amalgamated.hpp"
#include "backend_cli.hpp"

TEST_CASE("--ctx caps the KV pool from the command line",
          "[cli]") {
    std::array<const char*, 4> argv1 = {"prog", "--ctx", "1024",
                                        "m.gguf"};
    auto a = locus_tools::parse_backend_args(
        static_cast<int>(argv1.size()),
        const_cast<char**>(argv1.data()));
    REQUIRE(a.ctx == 1024);
    REQUIRE(a.positional ==
            std::vector<std::string>{"m.gguf"});

    std::array<const char*, 3> argv2 = {"prog", "--ctx=64",
                                        "m.gguf"};
    auto b = locus_tools::parse_backend_args(
        static_cast<int>(argv2.size()),
        const_cast<char**>(argv2.data()));
    REQUIRE(b.ctx == 64);

    std::array<const char*, 2> argv3 = {"prog", "m.gguf"};
    auto c = locus_tools::parse_backend_args(
        static_cast<int>(argv3.size()),
        const_cast<char**>(argv3.data()));
    REQUIRE(c.ctx == 0);  // default: model-derived pool
}
