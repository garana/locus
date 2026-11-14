// The vendored files match third_party/SHA256SUMS (DESIGN.md 6), and every
// vendored file is listed there.

#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>

#include "catch_amalgamated.hpp"
#include "locus/sys/sha256.hpp"

namespace fs = std::filesystem;

namespace {

/** @returns The lowercase hex SHA-256 of `data` (tested against FIPS
 *  180-4 in test_handshake.cpp). */
std::string sha256_hex(const std::string& data) {
    return locus::sys::to_hex(locus::sys::sha256(data));
}

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

}  // namespace

TEST_CASE("vendored files match third_party/SHA256SUMS", "[third_party]") {
    const fs::path dir = fs::path(LOCUS_SOURCE_DIR) / "third_party";
    std::istringstream sums(read_file(dir / "SHA256SUMS"));
    std::map<std::string, std::string> listed;
    std::string line;
    while (std::getline(sums, line)) {
        if (line.empty()) continue;
        // "<64 hex>  <path>", as shasum -a 256 writes it.
        REQUIRE(line.size() > 66);
        REQUIRE(line.substr(64, 2) == "  ");
        listed[line.substr(66)] = line.substr(0, 64);
    }
    REQUIRE(!listed.empty());
    for (const auto& [path, want] : listed) {
        INFO(path);
        REQUIRE(fs::exists(dir / path));
        CHECK(sha256_hex(read_file(dir / path)) == want);
    }
    // Every vendored file is listed: a new one needs its hash recorded.
    for (const auto& e : fs::recursive_directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const std::string rel = fs::relative(e.path(), dir).generic_string();
        // Dot files are not vendored (a local .DS_Store, say).
        if (rel == "SHA256SUMS" || e.path().filename().string()[0] == '.') {
            continue;
        }
        INFO(rel);
        CHECK(listed.count(rel) == 1);
    }
}
