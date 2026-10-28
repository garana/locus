#include <array>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <sys/wait.h>

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

TEST_CASE("--image-decoder takes the VQ-16 decoder path", "[cli]") {
    std::array<const char*, 4> argv1 = {"prog", "--image-decoder",
                                        "vq.gguf", "m.gguf"};
    auto a = locus_tools::parse_backend_args(
        static_cast<int>(argv1.size()),
        const_cast<char**>(argv1.data()));
    REQUIRE(a.image_decoder == "vq.gguf");
    REQUIRE(a.positional == std::vector<std::string>{"m.gguf"});

    std::array<const char*, 3> argv2 = {"prog", "--image-decoder=v2.gguf",
                                        "m.gguf"};
    auto b = locus_tools::parse_backend_args(
        static_cast<int>(argv2.size()),
        const_cast<char**>(argv2.data()));
    REQUIRE(b.image_decoder == "v2.gguf");
    REQUIRE(b.positional == std::vector<std::string>{"m.gguf"});
}

TEST_CASE("--audio-max-tokens sets the speech token cap", "[cli]") {
    std::array<const char*, 4> argv1 = {"prog", "--audio-max-tokens",
                                        "350", "m.gguf"};
    auto a = locus_tools::parse_backend_args(
        static_cast<int>(argv1.size()),
        const_cast<char**>(argv1.data()));
    REQUIRE(a.audio_max_tokens == 350);
    REQUIRE(a.positional == std::vector<std::string>{"m.gguf"});

    std::array<const char*, 3> argv2 = {"prog", "--audio-max-tokens=70",
                                        "m.gguf"};
    auto b = locus_tools::parse_backend_args(
        static_cast<int>(argv2.size()),
        const_cast<char**>(argv2.data()));
    REQUIRE(b.audio_max_tokens == 70);

    std::array<const char*, 2> argv3 = {"prog", "m.gguf"};
    auto c = locus_tools::parse_backend_args(
        static_cast<int>(argv3.size()),
        const_cast<char**>(argv3.data()));
    REQUIRE(c.audio_max_tokens == 0);  // default: the server's 1024
}

TEST_CASE("--max-tokens sets the output token cap", "[cli]") {
    std::array<const char*, 5> argv1 = {"prog", "--max-tokens", "8192",
                                        "--audio-max-tokens=70", "m.gguf"};
    auto a = locus_tools::parse_backend_args(
        static_cast<int>(argv1.size()),
        const_cast<char**>(argv1.data()));
    REQUIRE(a.max_tokens == 8192);
    REQUIRE(a.audio_max_tokens == 70);  // a different flag
    REQUIRE(a.positional == std::vector<std::string>{"m.gguf"});

    std::array<const char*, 3> argv2 = {"prog", "--max-tokens=16",
                                        "m.gguf"};
    auto b = locus_tools::parse_backend_args(
        static_cast<int>(argv2.size()),
        const_cast<char**>(argv2.data()));
    REQUIRE(b.max_tokens == 16);
    REQUIRE(b.audio_max_tokens == 0);

    std::array<const char*, 3> argv3 = {"prog", "--max-tokens=-1",
                                        "m.gguf"};
    auto c = locus_tools::parse_backend_args(
        static_cast<int>(argv3.size()),
        const_cast<char**>(argv3.data()));
    REQUIRE(c.bad_number_flag == "--max-tokens");
}

// i#176: these flags used atol + a uint32_t cast, so "-1" wrapped to
// 4294967295 (no audio cap, a huge KV pool) instead of failing.
TEST_CASE("numeric flags reject negative, huge and junk values",
          "[cli]") {
    for (const char* flag : {"--ctx", "--concurrent",
                             "--audio-max-tokens"}) {
        for (const char* v : {"-1", "4294967297", "12abc", ""}) {
            INFO(flag << " '" << v << "'");
            std::array<const char*, 4> sep = {"prog", flag, v, "m.gguf"};
            auto a = locus_tools::parse_backend_args(
                static_cast<int>(sep.size()),
                const_cast<char**>(sep.data()));
            REQUIRE(a.bad_number_flag == flag);
            REQUIRE(a.bad_number_value == v);

            const std::string eq = std::string(flag) + "=" + v;
            std::array<const char*, 3> joined = {"prog", eq.c_str(),
                                                 "m.gguf"};
            auto b = locus_tools::parse_backend_args(
                static_cast<int>(joined.size()),
                const_cast<char**>(joined.data()));
            REQUIRE(b.bad_number_flag == flag);
            REQUIRE(b.bad_number_value == v);
        }
    }
    // The first bad flag is reported; valid flags still parse.
    std::array<const char*, 6> argv = {"prog", "--ctx",   "64",
                                       "--concurrent=-2",
                                       "--audio-max-tokens=x", "m.gguf"};
    auto c = locus_tools::parse_backend_args(
        static_cast<int>(argv.size()), const_cast<char**>(argv.data()));
    REQUIRE(c.ctx == 64);
    REQUIRE(c.bad_number_flag == "--concurrent");
    REQUIRE(c.bad_number_value == "-2");
    REQUIRE(c.positional == std::vector<std::string>{"m.gguf"});
}

TEST_CASE("--kv-type picks the KV cache precision", "[cli]") {
    using locus::kv::KvType;
    auto parse = [](std::vector<const char*> argv) {
        return locus_tools::parse_backend_args(
            static_cast<int>(argv.size()), const_cast<char**>(argv.data()));
    };
    auto a = parse({"prog", "m.gguf"});
    REQUIRE(a.kv_type_ok);
    REQUIRE(a.kv_type == KvType::kF32);  // default: exact
    auto b = parse({"prog", "--kv-type", "f16", "m.gguf"});
    REQUIRE(b.kv_type_ok);
    REQUIRE(b.kv_type == KvType::kF16);
    REQUIRE(b.positional == std::vector<std::string>{"m.gguf"});
    REQUIRE(parse({"prog", "--kv-type=q8"}).kv_type == KvType::kQ8);
    REQUIRE(parse({"prog", "--kv-type=q4"}).kv_type == KvType::kQ4);
    REQUIRE(parse({"prog", "--kv-type=f32"}).kv_type == KvType::kF32);
    auto bad = parse({"prog", "--kv-type", "bf16", "m.gguf"});
    REQUIRE_FALSE(bad.kv_type_ok);
    REQUIRE(bad.kv_type_name == "bf16");
    REQUIRE(bad.positional == std::vector<std::string>{"m.gguf"});
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

TEST_CASE("locus-run exits 2 on a negative numeric flag", "[cli]") {
    // The check runs before the model loads, so no model is needed.
    const std::string cmd = std::string("'") + LOCUS_RUN_BIN +
                            "' --ctx -1 missing.gguf 'x' >/dev/null 2>&1";
    INFO(cmd);
    const int rc = std::system(cmd.c_str());
    REQUIRE(WIFEXITED(rc));
    REQUIRE(WEXITSTATUS(rc) == 2);
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

// i#198: the positional numbers went through atoi, so "-1" tokens wrapped
// to 4294967295 and a mistyped port bound an OS-assigned one. Both
// checks run before the model loads, so no model is needed; a valid
// value with a missing model exits 1, which keeps these from passing
// emptily.
TEST_CASE("bad positional numbers exit 2", "[cli]") {
    const auto status = [](const std::string& bin,
                           const std::string& args) {
        const std::string cmd =
            "'" + bin + "' " + args + " >/dev/null 2>&1";
        INFO(cmd);
        const int rc = std::system(cmd.c_str());
        REQUIRE(WIFEXITED(rc));
        return WEXITSTATUS(rc);
    };
    for (const char* n : {"-1", "abc", "12abc", ""}) {
        REQUIRE(status(LOCUS_RUN_BIN, std::string("missing.gguf x '") +
                                          n + "'") == 2);
    }
    REQUIRE(status(LOCUS_RUN_BIN, "missing.gguf x 4") == 1);
    for (const char* p : {"-1", "0", "808O", "65536", "70000"}) {
        REQUIRE(status(LOCUS_SERVER_BIN,
                       std::string("missing.gguf '") + p + "'") == 2);
    }
    REQUIRE(status(LOCUS_SERVER_BIN, "missing.gguf 8080") == 1);
}
