#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/server/model_headers.hpp"

using locus::gguf::GgufFile;
using locus::gguf::TensorType;
using locus::server::is_sf_token;
using locus::server::lora_header;
using locus::server::model_file_item;
using locus::server::model_headers;
using locus::server::sf_string;

namespace {

/**
 * A GGUF with two Q4_K tensors (288 bytes) and one Q6_K tensor (210
 * bytes), as a Q4_K_M file mixes them, and the given string keys.
 *
 * @param kv Key and value pairs.
 * @returns The file's bytes.
 */
std::vector<std::byte> mixed_gguf(
    const std::vector<std::pair<std::string, std::string>>& kv) {
    GgufBuilder b;
    b.header(3, kv.size());
    for (const auto& [k, v] : kv) {
        b.kv_string(k, v);
    }
    const std::uint64_t dims[] = {256};
    b.tensor("a", dims, static_cast<std::uint32_t>(TensorType::kQ4_K), 0);
    b.tensor("b", dims, static_cast<std::uint32_t>(TensorType::kQ4_K), 160);
    b.tensor("c", dims, static_cast<std::uint32_t>(TensorType::kQ6_K), 320);
    b.pad();
    b.zeros(320 + 210);
    const auto s = b.bytes();
    return {s.begin(), s.end()};
}

/** Writes bytes to path. */
void write_file(const std::filesystem::path& path,
                const std::vector<std::byte>& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

TEST_CASE("tensor type names match ggml's", "[model_headers]") {
    CHECK(locus::gguf::tensor_type_name(TensorType::kF32) == "f32");
    CHECK(locus::gguf::tensor_type_name(TensorType::kBF16) == "bf16");
    CHECK(locus::gguf::tensor_type_name(TensorType::kQ8_0) == "q8_0");
    CHECK(locus::gguf::tensor_type_name(TensorType::kQ4_K) == "q4_k");
    CHECK(locus::gguf::tensor_type_name(TensorType::kIQ2_XXS) == "iq2_xxs");
    CHECK(locus::gguf::tensor_type_name(TensorType::kTQ2_0) == "tq2_0");
    CHECK(locus::gguf::tensor_type_name(static_cast<TensorType>(99)) ==
          "unknown");
}

TEST_CASE("the main tensor type is the one with the most bytes",
          "[model_headers]") {
    const auto bytes = mixed_gguf({});
    const GgufFile g = GgufFile::parse(bytes);
    CHECK(locus::gguf::main_tensor_type(g) == TensorType::kQ4_K);
}

TEST_CASE("structured field strings escape quotes and backslashes",
          "[model_headers]") {
    CHECK(sf_string("flux.gguf") == "\"flux.gguf\"");
    CHECK(sf_string("a\"b\\c") == "\"a\\\"b\\\\c\"");
    CHECK(sf_string("tab\there") == "\"tab?here\"");
    CHECK(sf_string("caf\xc3\xa9") == "\"caf??\"");
}

TEST_CASE("a model file item names the file and its type",
          "[model_headers]") {
    SECTION("name and version") {
        const auto bytes = mixed_gguf(
            {{"general.name", "Qwen-Image"}, {"general.version", "v1"}});
        const GgufFile g = GgufFile::parse(bytes);
        CHECK(model_file_item("qwen-image-q8_0.gguf", g) ==
              "\"qwen-image-q8_0.gguf\";name=\"Qwen-Image\";"
              "version=\"v1\";type=q4_k");
    }
    SECTION("a source revision stands in for the version") {
        const auto bytes = mixed_gguf(
            {{"general.name", "Qwen-Image"},
             {"general.source.revision",
              "75e0b4be04f60ec59a75f475837eced720f823b6"}});
        const GgufFile g = GgufFile::parse(bytes);
        CHECK(model_file_item("dit.gguf", g) ==
              "\"dit.gguf\";name=\"Qwen-Image\";version=\"75e0b4be04f6\";"
              "type=q4_k");
    }
    SECTION("no metadata: the file name and the type") {
        const auto bytes = mixed_gguf({});
        const GgufFile g = GgufFile::parse(bytes);
        CHECK(model_file_item("vae.gguf", g) == "\"vae.gguf\";type=q4_k");
    }
}

TEST_CASE("model headers list the given GGUFs in order", "[model_headers]") {
    const auto dir = std::filesystem::temp_directory_path() /
                     "locus-model-headers-test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    write_file(dir / "b-vae.gguf", mixed_gguf({{"general.name", "vae"}}));
    write_file(dir / "a-dit.gguf", mixed_gguf({{"general.name", "dit"}}));
    const std::vector<std::string> files{(dir / "a-dit.gguf").string(),
                                         (dir / "b-vae.gguf").string()};
    const auto h = model_headers("qwen-image", files);
    CHECK(h.model == "qwen-image");
    CHECK(h.files ==
          "\"a-dit.gguf\";name=\"dit\";type=q4_k, "
          "\"b-vae.gguf\";name=\"vae\";type=q4_k");
    // An architecture that is not a token leaves X-Model out; the
    // files are still listed.
    const auto bad = model_headers("llama\r\nX-Evil: 1", files);
    CHECK(bad.model.empty());
    CHECK(bad.files == h.files);
    CHECK(model_headers("", files).model.empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("structured field tokens", "[model_headers]") {
    for (const char* ok : {"llama", "qwen2vl", "t5encoder", "autoencoder_kl",
                           "qwen-image-edit", "*x", "a:b/c.d"}) {
        CHECK(is_sf_token(ok));
    }
    for (const char* no : {"", "2llama", "-x", "a b", "a\"b", "a,b", "a;b",
                           "a\r\nb", "caf\xc3\xa9"}) {
        CHECK_FALSE(is_sf_token(no));
    }
}

TEST_CASE("the LoRA header lists each name and scale", "[model_headers]") {
    CHECK(lora_header({}).empty());
    const locus::image::LoraPick one[] = {{"line-art", 0.6f}};
    CHECK(lora_header(one) == "\"line-art\";scale=0.6");
    const locus::image::LoraPick two[] = {{"a", 1.0f}, {"b", -0.25f}};
    CHECK(lora_header(two) == "\"a\";scale=1.0, \"b\";scale=-0.25");
}
