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

TEST_CASE("list flags parse and don't consume the model path",
          "[cli]") {
    std::array<const char*, 3> a1 = {"prog", "--quants", "m.gguf"};
    auto a = locus_tools::parse_backend_args(
        static_cast<int>(a1.size()),
        const_cast<char**>(a1.data()));
    REQUIRE(a.list_quants);
    REQUIRE(a.positional == std::vector<std::string>{"m.gguf"});

    std::array<const char*, 2> a2 = {"prog", "--tokenizers"};
    auto b = locus_tools::parse_backend_args(
        static_cast<int>(a2.size()),
        const_cast<char**>(a2.data()));
    REQUIRE(b.list_tokenizers);

    std::array<const char*, 2> a3 = {"prog", "--capabilities"};
    auto c = locus_tools::parse_backend_args(
        static_cast<int>(a3.size()),
        const_cast<char**>(a3.data()));
    REQUIRE(c.list_capabilities);
}

TEST_CASE("backend list includes compiled-out variants with a reason",
          "[cli]") {
    // Every canonical backend name resolves to either a real registry
    // entry or a clear "not built" reason -- the operator sees all of
    // them. (find_backend is null for a variant not built here.)
    for (const char* n : {"scalar", "neon", "sse4", "avx2", "vulkan",
                          "cuda"}) {
        const auto* b = locus::backend::find_backend(n);
        const std::string reason =
            locus_tools::backend_unavail_reason(n);
        // A missing variant must have a non-empty reason to show.
        if (b == nullptr) {
            REQUIRE_FALSE(reason.empty());
        }
    }
}

TEST_CASE("public_model_id exposes the basename, not the path",
          "[cli]") {
    REQUIRE(locus_tools::public_model_id(
                "/srv/models/llama-3.gguf") == "llama-3.gguf");
    REQUIRE(locus_tools::public_model_id("model.gguf") ==
            "model.gguf");
    REQUIRE(locus_tools::public_model_id(
                "/home/alice/secret/path/q4.gguf") == "q4.gguf");
    // Never leaks a directory component.
    REQUIRE(locus_tools::public_model_id("/srv/models/x.gguf")
                .find('/') == std::string::npos);
    // Degenerate input (trailing separator) falls back to a stable
    // id rather than an empty model name.
    REQUIRE(locus_tools::public_model_id("/tmp/dir/") == "locus");
}
