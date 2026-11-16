// The image and embedding loaders' shared tensor reads (i#1045).

#include <cstdint>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/tensor_read.hpp"

namespace {

using locus::gguf::GgufFile;
using locus::image::read_floats;
using locus::image::require_tensor;

/** A GGUF with three 3-value tensors (F32, F16, BF16) and a Q8_0 one. */
std::vector<std::byte> image() {
    GgufBuilder g;
    g.header(4, 0);
    const std::uint64_t three[] = {3};
    const std::uint64_t block[] = {32};
    g.tensor("a", three, 0 /* F32 */, 0);
    g.tensor("h", three, 1 /* F16 */, 32);
    g.tensor("b", three, 30 /* BF16 */, 64);
    g.tensor("q", block, 8 /* Q8_0 */, 96);
    g.pad();
    g.f32(1.5f).f32(-2.0f).f32(0.25f).zeros(20);
    // 1.0, -2.0, 0.5 as F16; 1.0, -2.0, 0.25 as BF16.
    for (std::uint16_t h : {0x3c00, 0xc000, 0x3800}) {
        g.raw_bytes({reinterpret_cast<const std::byte*>(&h), 2});
    }
    g.zeros(26);
    for (std::uint16_t h : {0x3f80, 0xc000, 0x3e80}) {
        g.raw_bytes({reinterpret_cast<const std::byte*>(&h), 2});
    }
    g.zeros(26).zeros(34);
    auto s = g.bytes();
    return {s.begin(), s.end()};
}

}  // namespace

TEST_CASE("read_floats widens F32 F16 and BF16 and checks the count",
          "[image][tensor_read]") {
    const auto img = image();
    const GgufFile g = GgufFile::parse(img);
    CHECK(read_floats(g, "m", "a", 3) == std::vector<float>{1.5f, -2.0f,
                                                            0.25f});
    CHECK(read_floats(g, "m", "h", 3) == std::vector<float>{1.0f, -2.0f,
                                                            0.5f});
    CHECK(read_floats(g, "m", "b", 3) == std::vector<float>{1.0f, -2.0f,
                                                            0.25f});
    CHECK(require_tensor(g, "m", "q")->nelements() == 32);
    CHECK_THROWS_WITH(require_tensor(g, "m", "x"), "m: missing tensor x");
    CHECK_THROWS_WITH(read_floats(g, "m", "a", 4),
                      "m: a must be F32, F16 or BF16 with 4 values");
    CHECK_THROWS_WITH(read_floats(g, "m", "q", 32),
                      "m: q must be F32, F16 or BF16 with 32 values");
}
