#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/arch.hpp"
#include "gguf_builder.hpp"

using namespace locus::model;

TEST_CASE("arch registry lists and finds", "[arch]") {
    REQUIRE(archs().size() >= 2);
    REQUIRE(find_arch("llama") != nullptr);
    REQUIRE(find_arch("deepseek2") != nullptr);
    REQUIRE(find_arch("qwen2") == nullptr);
    for (const ArchSpec& a : archs()) {
        REQUIRE(a.load_hparams != nullptr);
        REQUIRE(a.load_attention != nullptr);
        REQUIRE(a.attention != nullptr);
        REQUIRE(a.kv_dim != nullptr);
    }
}

TEST_CASE("unsupported architectures fail with a useful error",
          "[arch]") {
    GgufBuilder b;
    b.header(0, 1).kv_string("general.architecture", "qwen2");
    auto g = locus::gguf::GgufFile::parse(b.bytes());
    try {
        TransformerModel::load(g);
        FAIL("expected gguf::Error");
    } catch (const locus::gguf::Error& e) {
        const std::string msg = e.what();
        REQUIRE(msg.find("qwen2") != std::string::npos);
        REQUIRE(msg.find("llama") != std::string::npos);
        REQUIRE(msg.find("deepseek2") != std::string::npos);
    }
}

TEST_CASE("arch hooks take the model's ops, never the raw backend's",
          "[model]") {
    // #189: TransformerModel::ops() applies use_q8k_activations; a hook
    // reading active_backend().ops would bypass it (as deepseek2's MLA,
    // glm-dsa's indexer, muse-spark and deepseek41 once did).
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(LOCUS_SOURCE_DIR) / "src/locus/model";
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".cpp" &&
            e.path().extension() != ".hpp") {
            continue;
        }
        std::ifstream f(e.path());
        const std::string text((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
        INFO(e.path().filename().string());
        REQUIRE(text.find("active_backend().ops") == std::string::npos);
    }
}
