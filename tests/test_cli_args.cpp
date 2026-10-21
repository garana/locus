#include <array>
#include <cstdlib>
#include <filesystem>
#include <string>

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

TEST_CASE("--timing and --warmup parse as flags", "[cli]") {
    std::array<const char*, 5> argv1 = {"prog", "--timing", "m.gguf",
                                        "--warmup", "hi"};
    auto a = locus_tools::parse_backend_args(
        static_cast<int>(argv1.size()),
        const_cast<char**>(argv1.data()));
    REQUIRE(a.timing);
    REQUIRE(a.warmup);
    REQUIRE(a.positional == std::vector<std::string>{"m.gguf", "hi"});

    std::array<const char*, 2> argv2 = {"prog", "m.gguf"};
    auto b = locus_tools::parse_backend_args(
        static_cast<int>(argv2.size()),
        const_cast<char**>(argv2.data()));
    REQUIRE_FALSE(b.timing);
    REQUIRE_FALSE(b.warmup);
}

TEST_CASE("--prefill-f16 parses as a flag", "[cli]") {
    std::array<const char*, 3> argv1 = {"prog", "--prefill-f16",
                                        "m.gguf"};
    auto a = locus_tools::parse_backend_args(
        static_cast<int>(argv1.size()),
        const_cast<char**>(argv1.data()));
    REQUIRE(a.prefill_f16);
    REQUIRE(a.positional == std::vector<std::string>{"m.gguf"});

    std::array<const char*, 2> argv2 = {"prog", "m.gguf"};
    auto b = locus_tools::parse_backend_args(
        static_cast<int>(argv2.size()),
        const_cast<char**>(argv2.data()));
    REQUIRE_FALSE(b.prefill_f16);  // off by default: exact f32
}

TEST_CASE("locus-run starts without a Vulkan device", "[cli]") {
    // Tools set backend options at startup (vulkan_set_prefill_f16);
    // with no Vulkan driver visible that must not touch the device:
    // a CPU run works, and --prefill-f16 only warns.
    const std::string model =
        std::string(LOCUS_SOURCE_DIR) + "/tests/models/stories260K.gguf";
    if (!std::filesystem::exists(model)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    for (const char* flag : {"", "--prefill-f16 "}) {
        const std::string cmd =
            std::string("VK_ICD_FILENAMES=/nonexistent.json "
                        "VK_DRIVER_FILES=/nonexistent.json '") +
            LOCUS_RUN_BIN + "' " + flag + "--backend scalar '" + model +
            "' 'Once upon' 4 >/dev/null 2>&1";
        INFO(cmd);
        REQUIRE(std::system(cmd.c_str()) == 0);
    }
}

TEST_CASE("--speculative parses as a flag", "[cli]") {
    std::array<const char*, 3> argv1 = {"prog", "--speculative",
                                        "m.gguf"};
    auto a = locus_tools::parse_backend_args(
        static_cast<int>(argv1.size()),
        const_cast<char**>(argv1.data()));
    REQUIRE(a.speculative);
    REQUIRE(a.positional == std::vector<std::string>{"m.gguf"});

    std::array<const char*, 2> argv2 = {"prog", "m.gguf"};
    auto b = locus_tools::parse_backend_args(
        static_cast<int>(argv2.size()),
        const_cast<char**>(argv2.data()));
    REQUIRE_FALSE(b.speculative);  // off by default
}
