#include <cstdint>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "gguf_builder.hpp"

using locus::gguf::Error;
using locus::gguf::GgufFile;
using locus::gguf::TensorType;

namespace {

/** A minimal well-formed image: 1 metadata key, 1 f32 2x2 tensor. */
GgufBuilder valid_image() {
    GgufBuilder b;
    const std::uint64_t dims[] = {2, 2};
    b.header(1, 1)
        .kv_string("general.architecture", "llama")
        .tensor("weight", dims, 0 /* F32 */, 0)
        .pad()
        .zeros(16);
    return b;
}

}  // namespace

TEST_CASE("parses a minimal valid file", "[gguf]") {
    auto b = valid_image();
    auto g = GgufFile::parse(b.bytes());

    REQUIRE(g.version() == 3);
    REQUIRE(g.get_string("general.architecture") == "llama");
    REQUIRE(g.tensors().size() == 1);

    const auto* t = g.find_tensor("weight");
    REQUIRE(t != nullptr);
    REQUIRE(t->type == TensorType::kF32);
    REQUIRE(t->nelements() == 4);
    REQUIRE(t->nbytes == 16);
    REQUIRE(g.tensor_data(*t).size() == 16);
    REQUIRE(g.find_tensor("nope") == nullptr);
}

TEST_CASE("metadata accessors are type-safe", "[gguf]") {
    GgufBuilder b;
    const float scores[] = {1.0f, -2.5f};
    b.header(0, 3)
        .kv_u32("n", 7)
        .kv_bool("flag", true)
        .kv_f32_array("scores", scores);
    auto g = GgufFile::parse(b.bytes());

    REQUIRE(g.get_uint("n") == 7);
    REQUIRE(g.get_bool("flag") == true);
    REQUIRE(g.get_string("n") == std::nullopt);  // wrong type
    REQUIRE(g.get_uint("missing") == std::nullopt);

    const auto* arr = g.get_array("scores");
    REQUIRE(arr != nullptr);
    REQUIRE(arr->size() == 2);
    REQUIRE((*arr)[1].f == -2.5);
}

TEST_CASE("hostile images throw instead of crashing", "[gguf]") {
    SECTION("empty file") {
        REQUIRE_THROWS_AS(GgufFile::parse({}), Error);
    }
    SECTION("bad magic") {
        GgufBuilder b;
        b.u32(0xdeadbeef).u32(3).u64(0).u64(0);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("unsupported version") {
        GgufBuilder b;
        b.u32(0x46554747).u32(1).u64(0).u64(0);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("truncated header") {
        GgufBuilder b;
        b.u32(0x46554747).u32(3).u32(0);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("string length exceeding file size") {
        GgufBuilder b;
        b.header(0, 1).u64(~0ull);  // key length: 2^64-1
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("declared kv count far beyond file size") {
        GgufBuilder b;
        b.header(0, ~0ull);  // alloc-bomb attempt
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("array alloc bomb") {
        GgufBuilder b;
        b.header(0, 1).str("k").u32(9).u32(6).u64(~0ull);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("unknown value type") {
        GgufBuilder b;
        b.header(0, 1).str("k").u32(99).u32(0);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("array nesting too deep") {
        GgufBuilder b;
        b.header(0, 1).str("k").u32(9);
        for (int i = 0; i < 20; ++i) {
            b.u32(9).u64(1);  // array of arrays of ...
        }
        b.u32(6).u64(0);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("duplicate metadata key") {
        GgufBuilder b;
        b.header(0, 2).kv_u32("k", 1).kv_u32("k", 2);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("non-power-of-two alignment") {
        GgufBuilder b;
        b.header(0, 1).kv_u32("general.alignment", 3);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
}

TEST_CASE("hostile tensor tables throw", "[gguf]") {
    const std::uint64_t d22[] = {2, 2};

    SECTION("rank out of range") {
        GgufBuilder b;
        const std::uint64_t dims[] = {1, 1, 1, 1, 1};
        b.header(1, 0).tensor("t", dims, 0, 0);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("zero-sized dimension") {
        GgufBuilder b;
        const std::uint64_t dims[] = {0, 2};
        b.header(1, 0).tensor("t", dims, 0, 0);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("dimension product overflows") {
        GgufBuilder b;
        const std::uint64_t dims[] = {1ull << 40, 1ull << 40};
        b.header(1, 0).tensor("t", dims, 0, 0);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("unsupported tensor data type") {
        GgufBuilder b;
        b.header(1, 0).tensor("t", d22, 999, 0);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("row not a multiple of the quant block") {
        GgufBuilder b;
        const std::uint64_t dims[] = {33, 1};
        b.header(1, 0).tensor("t", dims, 2 /* Q4_0 */, 0);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("unaligned tensor offset") {
        GgufBuilder b;
        b.header(1, 0).tensor("t", d22, 0, 7).pad().zeros(64);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("tensor extends past end of file") {
        GgufBuilder b;
        b.header(1, 0).tensor("t", d22, 0, 0).pad().zeros(8);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("offset + size overflows") {
        GgufBuilder b;
        b.header(1, 0)
            .tensor("t", d22, 0, ~0ull & ~31ull)
            .pad()
            .zeros(64);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
    SECTION("duplicate tensor name") {
        GgufBuilder b;
        b.header(2, 0)
            .tensor("t", d22, 0, 0)
            .tensor("t", d22, 0, 0)
            .pad()
            .zeros(32);
        REQUIRE_THROWS_AS(GgufFile::parse(b.bytes()), Error);
    }
}
