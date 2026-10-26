#include "locus/http/base64.hpp"

#include <cstddef>
#include <span>
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
