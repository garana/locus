#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/engine/disk_prefix_store.hpp"
#include "locus/engine/helper_prefix_store.hpp"

using locus::engine::DiskKey;
using locus::engine::DiskPrefixStore;
using locus::engine::HelperPrefixStore;
using locus::tok::TokenId;
namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

constexpr std::size_t kBlockBytes = 256;
constexpr std::uint64_t kCap = 1 << 20;

fs::path temp_dir(const std::string& name) {
    const fs::path d = fs::temp_directory_path() /
                       (name + "-" + std::to_string(::getpid()));
    fs::remove_all(d);
    return d;
}

/** Removes the directory when the test ends, pass or fail. */
struct DirGuard {
    fs::path dir;
    ~DirGuard() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

DiskKey key() {
    DiskKey k;
    k.model_hash = 0x1234;
    k.model_size = 1000;
    k.backend = "cpu";
    k.block_tokens = 4;
    k.kv_dim = 8;
    k.n_layers = 2;
    return k;
}

std::vector<std::vector<std::uint8_t>> blocks(std::uint32_t n,
                                              std::uint8_t seed) {
    std::vector<std::vector<std::uint8_t>> b(
        n, std::vector<std::uint8_t>(kBlockBytes));
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < kBlockBytes; ++j) {
            b[i][j] = static_cast<std::uint8_t>(seed + 31 * i + j);
        }
    }
    return b;
}

std::vector<const std::uint8_t*> srcs(
    const std::vector<std::vector<std::uint8_t>>& b) {
    std::vector<const std::uint8_t*> p;
    for (const auto& x : b) {
        p.push_back(x.data());
    }
    return p;
}

std::vector<std::uint8_t*> dsts(std::vector<std::vector<std::uint8_t>>& b) {
    std::vector<std::uint8_t*> p;
    for (auto& x : b) {
        p.push_back(x.data());
    }
    return p;
}

std::vector<char> bytes_of(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

std::vector<std::string> helper(const fs::path& dir,
                                const std::string& exit_after = "") {
    std::vector<std::string> argv = {LOCUS_PREFIX_HELPER_BIN, dir.string()};
    if (!exit_after.empty()) {
        argv.push_back("--exit-after");
        argv.push_back(exit_after);
    }
    return argv;
}

const std::vector<TokenId> kPrompt = {1, 2,  3,  4,  5,  6,  7,
                                      8, 9, 10, 11, 12, 13, 14};

}  // namespace

TEST_CASE("HelperPrefixStore stores the file store's files",
          "[disk_prefix]") {
    DirGuard g{temp_dir("locus-hps-files")};
    DirGuard g2{temp_dir("locus-hps-files-ref")};
    const auto b = blocks(3, 7);
    {
        HelperPrefixStore s(helper(g.dir), key(), kBlockBytes, kCap);
        REQUIRE(s.enabled());
        REQUIRE(s.find(kPrompt) == 0);
        REQUIRE(s.write(kPrompt, 3, srcs(b)));
        REQUIRE(s.find(kPrompt) == 3);
        // Only the 3-block prefix is stored, not its 2-block start.
        REQUIRE(s.find(std::span(kPrompt).first(11)) == 0);

        auto out = blocks(3, 0);
        REQUIRE(s.read(kPrompt, 3, dsts(out)));
        REQUIRE(out == b);
    }
    // Byte for byte what a DiskPrefixStore writes, under the same name:
    // the file store reads the helper's directory.
    DiskPrefixStore ref(g2.dir, key(), kBlockBytes, kCap);
    REQUIRE(ref.write(kPrompt, 3, srcs(b)));
    const fs::path name = ref.path_for(kPrompt, 3).filename();
    REQUIRE(fs::exists(g.dir / name));
    REQUIRE(bytes_of(g.dir / name) == bytes_of(g2.dir / name));
    DiskPrefixStore on_helper_dir(g.dir, key(), kBlockBytes, kCap);
    auto out = blocks(3, 0);
    REQUIRE(on_helper_dir.read(kPrompt, 3, dsts(out)));
    REQUIRE(out == b);

    // A new store learns the stored names from the helper's list.
    HelperPrefixStore again(helper(g.dir), key(), kBlockBytes, kCap);
    REQUIRE(again.known() == 1);
    REQUIRE(again.find(kPrompt) == 3);
    auto out2 = blocks(3, 0);
    REQUIRE(again.read(kPrompt, 3, dsts(out2)));
    REQUIRE(out2 == b);

    // Another key names other blobs.
    DiskKey other = key();
    other.kv_dim = 16;
    HelperPrefixStore o(helper(g.dir), other, kBlockBytes, kCap);
    REQUIRE(o.find(kPrompt) == 0);
}

// i#830: the helper's store may be shared or remote, so with a key the
// blob carries the same tag as a keyed file, and one signed with another
// key is a miss.
TEST_CASE("HelperPrefixStore with a key signs its blobs", "[disk_prefix]") {
    DirGuard g{temp_dir("locus-hps-mac")};
    const auto b = blocks(2, 5);
    const std::string k1(32, 'x');
    {
        HelperPrefixStore s(helper(g.dir), key(), kBlockBytes, kCap);
        s.set_mac_key(k1);
        REQUIRE(s.write(kPrompt, 2, srcs(b)));
        auto out = blocks(2, 0);
        REQUIRE(s.read(kPrompt, 2, dsts(out)));
        REQUIRE(out == b);
    }
    // Byte for byte a keyed file: a keyed file store reads it.
    DiskPrefixStore file_store(g.dir, key(), kBlockBytes, kCap);
    file_store.set_mac_key(k1);
    auto out = blocks(2, 0);
    REQUIRE(file_store.read(kPrompt, 2, dsts(out)));
    REQUIRE(out == b);
    // Another key: the same name, but the tag does not check.
    HelperPrefixStore other(helper(g.dir), key(), kBlockBytes, kCap);
    other.set_mac_key(std::string(32, 'y'));
    REQUIRE(other.find(kPrompt) == 2);
    auto out2 = blocks(2, 0);
    REQUIRE_FALSE(other.read(kPrompt, 2, dsts(out2)));
}

TEST_CASE("HelperPrefixStore: a blob that does not check is a miss",
          "[disk_prefix]") {
    DirGuard g{temp_dir("locus-hps-bad")};
    const auto b = blocks(2, 3);
    HelperPrefixStore s(helper(g.dir), key(), kBlockBytes, kCap);
    REQUIRE(s.write(kPrompt, 2, srcs(b)));
    // The file store on the same directory names the helper's file.
    const fs::path file =
        DiskPrefixStore(g.dir, key(), kBlockBytes, kCap).path_for(kPrompt, 2);
    REQUIRE(fs::exists(file));
    {
        // Flip a byte of the last block.
        std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(-1, std::ios::end);
        f.put('\x5a');
    }
    auto out = blocks(2, 0);
    REQUIRE_FALSE(s.read(kPrompt, 2, dsts(out)));
    REQUIRE(s.find(kPrompt) == 0);
    // The helper is told to drop it (a notification: no reply to wait
    // for).
    for (int i = 0; i < 200 && fs::exists(file); ++i) {
        std::this_thread::sleep_for(10ms);
    }
    REQUIRE_FALSE(fs::exists(file));
    // A blob over the cap is not sent (3 blocks: 964 bytes with the
    // header and checksums).
    HelperPrefixStore small(helper(g.dir), key(), kBlockBytes, 512);
    REQUIRE_FALSE(small.write(kPrompt, 3, srcs(blocks(3, 1))));
}

TEST_CASE("HelperPrefixStore restarts a dead helper once, then turns off",
          "[disk_prefix]") {
    DirGuard g{temp_dir("locus-hps-dead")};
    const auto b = blocks(1, 9);
    // Each helper answers list and one more request, then exits on the
    // next one (no refresh, so this test makes every request).
    HelperPrefixStore s(helper(g.dir, "2"), key(), kBlockBytes, kCap, 5s,
                        0ms);
    REQUIRE(s.write(kPrompt, 1, srcs(b)));
    // The first helper dies on this read; the second answers it.
    auto out = blocks(1, 0);
    REQUIRE(s.read(kPrompt, 1, dsts(out)));
    REQUIRE(out == b);
    REQUIRE(s.enabled());
    // The second dies too: the tier turns off and every call misses.
    REQUIRE_FALSE(s.read(kPrompt, 1, dsts(out)));
    REQUIRE_FALSE(s.enabled());
    REQUIRE(s.find(kPrompt) == 0);
    REQUIRE_FALSE(s.write(kPrompt, 1, srcs(b)));

    // A helper that cannot start turns the tier off as well.
    HelperPrefixStore none({"/nonexistent/locus-prefix-helper"}, key(),
                           kBlockBytes, kCap, 5s, 0ms);
    REQUIRE_FALSE(none.write(kPrompt, 1, srcs(b)));
    REQUIRE_FALSE(none.write(kPrompt, 1, srcs(b)));
    REQUIRE_FALSE(none.enabled());
}

TEST_CASE("HelperPrefixStore finds another store's writes after a refresh",
          "[disk_prefix]") {
    DirGuard g{temp_dir("locus-hps-shared")};
    const auto b = blocks(2, 5);
    // Two stores (two locus processes) on one shared directory.
    HelperPrefixStore reader(helper(g.dir), key(), kBlockBytes, kCap, 5s,
                             50ms);
    HelperPrefixStore writer(helper(g.dir), key(), kBlockBytes, kCap, 5s,
                             0ms);
    REQUIRE(reader.find(kPrompt) == 0);
    REQUIRE(writer.write(kPrompt, 2, srcs(b)));
    for (int i = 0; i < 200 && reader.find(kPrompt) == 0; ++i) {
        std::this_thread::sleep_for(10ms);
    }
    REQUIRE(reader.find(kPrompt) == 2);
    auto out = blocks(2, 0);
    REQUIRE(reader.read(kPrompt, 2, dsts(out)));
    REQUIRE(out == b);
}
