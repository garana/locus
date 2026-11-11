#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/engine/disk_prefix_store.hpp"

using locus::engine::DiskKey;
using locus::engine::DiskPrefixStore;
using locus::tok::TokenId;
namespace fs = std::filesystem;

namespace {

constexpr std::size_t kBlockBytes = 256;

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

/** `n` blocks of kBlockBytes, each byte a function of (seed, block,
 *  offset). */
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

/** Flips one byte of `path` at `off`. */
void flip(const fs::path& path, std::streamoff off) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekg(off);
    char c = 0;
    f.read(&c, 1);
    c = static_cast<char>(c ^ 0x40);
    f.seekp(off);
    f.write(&c, 1);
}

const std::vector<TokenId> kPrompt{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

}  // namespace

TEST_CASE("DiskPrefixStore reads back what it wrote, also after a reopen",
          "[disk_prefix]") {
    DirGuard g{temp_dir("locus-dps-roundtrip")};
    const auto in = blocks(3, 7);
    {
        DiskPrefixStore s(g.dir, key(), kBlockBytes, 1 << 20);
        REQUIRE(s.find(kPrompt) == 0);
        REQUIRE(s.write(kPrompt, 3, srcs(in)));
        REQUIRE(s.find(kPrompt) == 3);
        REQUIRE(s.used_bytes() == fs::file_size(s.path_for(kPrompt, 3)));
        auto out = blocks(3, 0);
        REQUIRE(s.read(kPrompt, 3, dsts(out)));
        REQUIRE(out == in);
    }
    // A new store on the same directory: a server restart.
    DiskPrefixStore s(g.dir, key(), kBlockBytes, 1 << 20);
    REQUIRE(s.used_bytes() == fs::file_size(s.path_for(kPrompt, 3)));
    std::vector<TokenId> longer = kPrompt;
    longer.push_back(99);
    REQUIRE(s.find(longer) == 3);
    auto out = blocks(3, 0);
    REQUIRE(s.read(longer, 3, dsts(out)));
    REQUIRE(out == in);

    SECTION("a prompt that differs in the cached blocks misses") {
        std::vector<TokenId> other = kPrompt;
        other[5] = 42;
        REQUIRE(s.find(other) == 0);
        REQUIRE_FALSE(s.read(other, 3, dsts(out)));
    }
    SECTION("a shorter prefix of the stored one is not stored") {
        REQUIRE(s.find(std::vector<TokenId>(kPrompt.begin(),
                                            kPrompt.begin() + 8)) == 0);
    }
    SECTION("writing the same prefix again keeps one file") {
        const auto before = s.used_bytes();
        REQUIRE(s.write(kPrompt, 3, srcs(in)));
        REQUIRE(s.used_bytes() == before);
    }
}

TEST_CASE("DiskPrefixStore: every key field is part of the match",
          "[disk_prefix]") {
    DirGuard g{temp_dir("locus-dps-key")};
    const auto in = blocks(2, 3);
    const DiskKey base = key();
    {
        DiskPrefixStore s(g.dir, base, kBlockBytes, 1 << 20);
        REQUIRE(s.write(kPrompt, 2, srcs(in)));
    }
    std::vector<DiskKey> others(9, base);
    others[0].model_hash ^= 1;
    others[1].model_size += 1;  // the model file changed, same name
    others[2].backend = "vulkan";
    others[3].kv_type = locus::kv::KvType::kQ8;
    others[4].kv_dim = 16;
    others[5].n_layers = 3;
    others[6].layer_base = 1;
    others[7].overrides = 5;
    others[8].block_tokens = 2;
    for (std::size_t i = 0; i < others.size(); ++i) {
        CAPTURE(i);
        DiskPrefixStore s(g.dir, others[i], kBlockBytes, 1 << 20);
        REQUIRE(s.find(kPrompt) == 0);
    }
    // block_bytes is in the key too.
    DiskPrefixStore wide(g.dir, base, 2 * kBlockBytes, 1 << 20);
    REQUIRE(wide.find(kPrompt) == 0);
}

TEST_CASE("DiskPrefixStore: a file under another file's name is a miss",
          "[disk_prefix]") {
    // A digest collision, or a file of another key under this key's
    // name: the header is compared in full, so the read misses and
    // removes the file.
    DirGuard g{temp_dir("locus-dps-collide")};
    const auto in = blocks(2, 9);
    std::vector<TokenId> other = kPrompt;
    other[0] = 77;
    fs::path stored;
    fs::path wanted;
    {
        DiskPrefixStore s(g.dir, key(), kBlockBytes, 1 << 20);
        REQUIRE(s.write(other, 2, srcs(in)));
        stored = s.path_for(other, 2);
        wanted = s.path_for(kPrompt, 2);
    }
    SECTION("other tokens") {
        fs::rename(stored, wanted);
    }
    SECTION("another key") {
        DiskKey k = key();
        k.backend = "vulkan";
        DiskPrefixStore v(g.dir, k, kBlockBytes, 1 << 20);
        fs::remove(stored);
        REQUIRE(v.write(kPrompt, 2, srcs(in)));
        fs::rename(v.path_for(kPrompt, 2), wanted);
    }
    DiskPrefixStore s(g.dir, key(), kBlockBytes, 1 << 20);
    REQUIRE(s.find(kPrompt) == 2);
    auto out = blocks(2, 0);
    REQUIRE_FALSE(s.read(kPrompt, 2, dsts(out)));
    REQUIRE_FALSE(fs::exists(wanted));
    REQUIRE(s.find(kPrompt) == 0);
    REQUIRE(s.used_bytes() == 0);
}

TEST_CASE("DiskPrefixStore: a torn or corrupted file is a miss",
          "[disk_prefix]") {
    DirGuard g{temp_dir("locus-dps-torn")};
    const auto in = blocks(3, 5);
    DiskPrefixStore s(g.dir, key(), kBlockBytes, 1 << 20);
    REQUIRE(s.write(kPrompt, 3, srcs(in)));
    const fs::path p = s.path_for(kPrompt, 3);
    const auto size = static_cast<std::streamoff>(fs::file_size(p));

    SECTION("truncated") {
        fs::resize_file(p, static_cast<std::uintmax_t>(size - 1));
    }
    SECTION("a bit flipped in the last block") {
        flip(p, size - 1);
    }
    SECTION("a bit flipped in the first block") {
        flip(p, size - 3 * static_cast<std::streamoff>(kBlockBytes));
    }
    SECTION("a bit flipped in the header") {
        flip(p, 9);
    }
    SECTION("a bit flipped in a stored block checksum") {
        // header, 3 block checksums, the header checksum, the blocks
        flip(p, size - 3 * static_cast<std::streamoff>(kBlockBytes) - 8 -
                    3 * 8);
    }
    SECTION("longer than its header says") {
        fs::resize_file(p, static_cast<std::uintmax_t>(size + 1));
    }
    SECTION("removed behind the store's back") {
        fs::remove(p);
    }
    auto out = blocks(3, 0);
    REQUIRE_FALSE(s.read(kPrompt, 3, dsts(out)));
    REQUIRE_FALSE(fs::exists(p));
    REQUIRE(s.find(kPrompt) == 0);
    REQUIRE(s.used_bytes() == 0);
}

TEST_CASE("DiskPrefixStore evicts the least recently used file to fit "
          "its cap",
          "[disk_prefix]") {
    DirGuard g{temp_dir("locus-dps-cap")};
    const auto in = blocks(2, 1);
    std::vector<TokenId> a = kPrompt;
    std::vector<TokenId> b = kPrompt;
    std::vector<TokenId> c = kPrompt;
    b[0] = 50;
    c[0] = 60;
    std::uint64_t one;
    {
        DiskPrefixStore probe(g.dir, key(), kBlockBytes, 1 << 20);
        REQUIRE(probe.write(a, 2, srcs(in)));
        one = probe.used_bytes();
        fs::remove(probe.path_for(a, 2));
    }
    // Room for two files.
    DiskPrefixStore s(g.dir, key(), kBlockBytes, 2 * one + one / 2);
    REQUIRE(s.used_bytes() == 0);
    REQUIRE(s.write(a, 2, srcs(in)));
    REQUIRE(s.write(b, 2, srcs(in)));
    auto out = blocks(2, 0);
    REQUIRE(s.read(a, 2, dsts(out)));  // a is now newer than b
    REQUIRE(s.write(c, 2, srcs(in)));
    REQUIRE(s.find(a) == 2);
    REQUIRE(s.find(b) == 0);
    REQUIRE_FALSE(fs::exists(s.path_for(b, 2)));
    REQUIRE(s.find(c) == 2);
    REQUIRE(s.used_bytes() == 2 * one);

    SECTION("writing a stored prefix again refreshes it") {
        REQUIRE(s.write(a, 2, srcs(in)));  // a is now newer than c
        std::vector<TokenId> d = kPrompt;
        d[0] = 70;
        REQUIRE(s.write(d, 2, srcs(in)));
        REQUIRE(s.find(a) == 2);
        REQUIRE(s.find(c) == 0);
        REQUIRE(s.find(d) == 2);
    }
    SECTION("the order survives a reopen through file mtimes") {
        const auto now = fs::file_time_type::clock::now();
        fs::last_write_time(s.path_for(a, 2), now - std::chrono::hours(2));
        fs::last_write_time(s.path_for(c, 2), now - std::chrono::hours(1));
        DiskPrefixStore r(g.dir, key(), kBlockBytes, 2 * one + one / 2);
        REQUIRE(r.used_bytes() == 2 * one);
        REQUIRE(r.write(b, 2, srcs(in)));
        REQUIRE(r.find(a) == 0);  // the oldest mtime went first
        REQUIRE(r.find(c) == 2);
        REQUIRE(r.find(b) == 2);
    }
    SECTION("a file larger than the cap is not written") {
        const auto big = blocks(8, 2);
        std::vector<TokenId> longp(32, 3);
        REQUIRE_FALSE(s.write(longp, 8, srcs(big)));
        REQUIRE(s.find(longp) == 0);
        REQUIRE(s.used_bytes() == 2 * one);  // nothing evicted for it
    }
}

TEST_CASE("DiskPrefixStore removes a crash's temporary files on open",
          "[disk_prefix]") {
    DirGuard g{temp_dir("locus-dps-tmp")};
    fs::create_directories(g.dir);
    const fs::path tmp = g.dir / "0123456789abcdef.lkv.tmp";
    const fs::path other = g.dir / "notes.txt";
    std::ofstream(tmp) << "half a file";
    std::ofstream(other) << "not ours";
    DiskPrefixStore s(g.dir, key(), kBlockBytes, 1 << 20);
    REQUIRE_FALSE(fs::exists(tmp));
    REQUIRE(fs::exists(other));  // files it does not own are left alone
    REQUIRE(s.used_bytes() == 0);
}

TEST_CASE("DiskPrefixStore rejects bad arguments", "[disk_prefix]") {
    DirGuard g{temp_dir("locus-dps-args")};
    DiskKey k = key();
    k.backend = std::string(33, 'x');
    REQUIRE_THROWS_AS(DiskPrefixStore(g.dir, k, kBlockBytes, 1 << 20),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(DiskPrefixStore(g.dir, key(), 0, 1 << 20),
                      std::invalid_argument);
    k = key();
    k.block_tokens = 0;
    REQUIRE_THROWS_AS(DiskPrefixStore(g.dir, k, kBlockBytes, 1 << 20),
                      std::invalid_argument);

    DiskPrefixStore s(g.dir, key(), kBlockBytes, 1 << 20);
    const auto in = blocks(3, 1);
    auto out = blocks(3, 0);
    // More blocks than the prompt covers, fewer buffers, or none.
    REQUIRE_FALSE(s.write(kPrompt, 4, srcs(blocks(4, 1))));
    const auto src = srcs(in);
    REQUIRE_FALSE(s.write(kPrompt, 3, std::span(src).first(2)));
    REQUIRE_FALSE(s.write(kPrompt, 0, srcs(in)));
    REQUIRE_FALSE(s.read(kPrompt, 4, dsts(out)));
    REQUIRE(s.used_bytes() == 0);
    REQUIRE(s.path_for(kPrompt, 4).empty());
}

TEST_CASE("DiskPrefixStore stores a prefix again after its file was "
          "removed",
          "[disk_prefix]") {
    DirGuard g{temp_dir("locus-dps-again")};
    const auto in = blocks(2, 4);
    DiskPrefixStore s(g.dir, key(), kBlockBytes, 1 << 20);
    REQUIRE(s.write(kPrompt, 2, srcs(in)));
    const auto one = s.used_bytes();
    fs::remove(s.path_for(kPrompt, 2));
    REQUIRE(s.write(kPrompt, 2, srcs(in)));
    REQUIRE(fs::exists(s.path_for(kPrompt, 2)));
    REQUIRE(s.used_bytes() == one);
    auto out = blocks(2, 0);
    REQUIRE(s.read(kPrompt, 2, dsts(out)));
    REQUIRE(out == in);
}

TEST_CASE("DiskPrefixStore: concurrent finds, reads and writes agree",
          "[disk_prefix]") {
    DirGuard g{temp_dir("locus-dps-threads")};
    DiskPrefixStore s(g.dir, key(), kBlockBytes, 1 << 20);
    constexpr int kThreads = 4;
    constexpr int kRounds = 50;
    std::atomic<int> bad{0};
    {
        std::vector<std::jthread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t] {
                // Each thread has its own prompt and also writes one
                // prompt they all share.
                std::vector<TokenId> own = kPrompt;
                own[0] = 100 + t;
                const auto mine = blocks(3, static_cast<std::uint8_t>(t));
                const auto shared = blocks(3, 200);
                auto out = blocks(3, 0);
                for (int r = 0; r < kRounds; ++r) {
                    if (!s.write(own, 3, srcs(mine)) ||
                        !s.write(kPrompt, 3, srcs(shared))) {
                        ++bad;
                    }
                    if (s.find(own) != 3 || !s.read(own, 3, dsts(out)) ||
                        out != mine) {
                        ++bad;
                    }
                    if (s.read(kPrompt, 3, dsts(out)) && out != shared) {
                        ++bad;
                    }
                }
            });
        }
    }
    REQUIRE(bad == 0);
    std::uint64_t on_disk = 0;
    std::size_t files = 0;
    for (const auto& e : fs::directory_iterator(g.dir)) {
        on_disk += e.file_size();
        ++files;
    }
    REQUIRE(files == kThreads + 1);  // no temporary file left behind
    REQUIRE(s.used_bytes() == on_disk);
}
