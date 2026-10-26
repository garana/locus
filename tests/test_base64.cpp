#include "locus/http/base64.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"

namespace {

std::string b64(const std::string& s) {
    return locus::http::base64_encode(std::as_bytes(std::span(s)));
}

}  // namespace

TEST_CASE("base64 matches the RFC 4648 test vectors", "[http][base64]") {
    // RFC 4648 section 10: every padding case (0, 1 and 2 '=').
    REQUIRE(b64("") == "");
    REQUIRE(b64("f") == "Zg==");
    REQUIRE(b64("fo") == "Zm8=");
    REQUIRE(b64("foo") == "Zm9v");
    REQUIRE(b64("foob") == "Zm9vYg==");
    REQUIRE(b64("fooba") == "Zm9vYmE=");
    REQUIRE(b64("foobar") == "Zm9vYmFy");
}

TEST_CASE("base64 uses + and / for the high sextets", "[http][base64]") {
    // 0xFB 0xFF 0xFE: sextets 62 63 63 62 ("+//+"); 0xFF 0xFE 0xFD:
    // 63 63 59 61 ("//79").
    const std::vector<std::byte> a = {std::byte{0xFB}, std::byte{0xFF},
                                      std::byte{0xFE}};
    const std::vector<std::byte> b = {std::byte{0xFF}, std::byte{0xFE},
                                      std::byte{0xFD}};
    REQUIRE(locus::http::base64_encode(a) == "+//+");
    REQUIRE(locus::http::base64_encode(b) == "//79");
}

TEST_CASE("base64_decode inverts base64_encode", "[http][base64]") {
    using locus::http::base64_decode;
    using locus::http::base64_encode;
    const auto str = [](const std::vector<std::byte>& v) {
        return std::string(reinterpret_cast<const char*>(v.data()),
                           v.size());
    };
    for (const char* s : {"", "f", "fo", "foo", "foob", "fooba", "foobar"}) {
        INFO(s);
        REQUIRE(str(base64_decode(b64(s), 64)) == s);
    }
    REQUIRE(base64_decode("+//+", 3) ==
            std::vector<std::byte>{std::byte{0xFB}, std::byte{0xFF},
                                   std::byte{0xFE}});
    // Every length 0..300 (all three padding cases, many times).
    std::uint32_t state = 11;
    for (std::size_t n = 0; n <= 300; ++n) {
        std::vector<std::byte> v(n);
        for (auto& b : v) {
            state = state * 1664525u + 1013904223u;
            b = std::byte(state >> 24);
        }
        INFO("length " << n);
        REQUIRE(base64_decode(base64_encode(v), n) == v);
    }
}

TEST_CASE("base64_decode refuses malformed and oversized input",
          "[http][base64]") {
    using locus::http::base64_decode;
    for (const char* bad : {
             "Zg=",       // length not a multiple of 4
             "Zm9v!A==",  // a character outside the alphabet
             "Zg==Zm8=",  // padding before the end
             "Z===",      // three '='
             "====",      // only padding
             "Zm 9",      // whitespace
             "Zh==",      // nonzero bits under the padding ("f" is Zg==)
             "Zm9=",      // likewise for one '=' ("fo" is Zm8=)
         }) {
        INFO(bad);
        REQUIRE_THROWS_AS(base64_decode(bad, 64), std::invalid_argument);
    }
    // The cap applies to the decoded size, before decoding.
    REQUIRE(base64_decode("Zm9vYmFy", 6).size() == 6);
    REQUIRE_THROWS_AS(base64_decode("Zm9vYmFy", 5), std::invalid_argument);
}
