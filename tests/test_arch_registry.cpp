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
        LlamaModel::load(g);
        FAIL("expected gguf::Error");
    } catch (const locus::gguf::Error& e) {
        const std::string msg = e.what();
        REQUIRE(msg.find("qwen2") != std::string::npos);
        REQUIRE(msg.find("llama") != std::string::npos);
        REQUIRE(msg.find("deepseek2") != std::string::npos);
    }
}
