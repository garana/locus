#include "locus/engine/disk_prefix_store.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <tuple>

namespace locus::engine {

namespace {

namespace fs = std::filesystem;

// Files pass 2 GiB: an 8192-token f32 prefix of Llama 3 8B (256 KiB of
// KV per token) is exactly that.
static_assert(sizeof(off_t) == 8, "build with 64-bit file offsets");

constexpr char kMagic[8] = {'L', 'O', 'C', 'U', 'S', 'K', 'V', 'P'};
constexpr std::size_t kMaxBackend = 32;
constexpr const char* kSuffix = ".lkv";
constexpr const char* kTmpSuffix = ".tmp";

constexpr std::uint64_t kFnvPrime = 0x100000001b3ull;
constexpr std::uint64_t kLanePrime = 0x9e3779b97f4a7c15ull;

/** splitmix64's finalizer, as in the RAM prefix cache. */
std::uint64_t mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

/**
 * A 64-bit checksum of `n` bytes: four lanes of (h ^ word) * odd, then
 * a rotate, over 8-byte words. Each step is a bijection of the lane,
 * so a change confined to one word always changes the result; the
 * four lanes keep the multiplies independent.
 */
std::uint64_t checksum(const std::uint8_t* p, std::size_t n) {
    std::uint64_t h[4] = {n, n ^ 1, n ^ 2, n ^ 3};
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        for (int l = 0; l < 4; ++l) {
            std::uint64_t w;
            std::memcpy(&w, p + i + 8 * l, 8);
            h[l] = std::rotl((h[l] ^ w) * kLanePrime, 29);
        }
    }
    for (int l = 0; i < n; i += 8, l = (l + 1) % 4) {
        std::uint64_t w = 0;
        std::memcpy(&w, p + i, std::min<std::size_t>(8, n - i));
        h[l] = std::rotl((h[l] ^ w) * kLanePrime, 29);
    }
    return mix64(h[0] ^ mix64(h[1] ^ mix64(h[2] ^ mix64(h[3]))));
}

template <typename T>
void put(std::vector<std::uint8_t>& out, T v) {
    const std::size_t at = out.size();
    out.resize(at + sizeof v);
    std::memcpy(out.data() + at, &v, sizeof v);
}

/** pread until `n` bytes or EOF / error. @returns true on all n. */
bool pread_full(int fd, std::uint8_t* dst, std::size_t n, off_t off) {
    while (n > 0) {
        const ssize_t r = ::pread(fd, dst, n, off);
        if (r < 0 && errno == EINTR) {
            continue;
        }
        if (r <= 0) {
            return false;
        }
        dst += r;
        off += r;
        n -= static_cast<std::size_t>(r);
    }
    return true;
}

bool pwrite_full(int fd, const std::uint8_t* src, std::size_t n,
                 off_t off) {
    while (n > 0) {
        const ssize_t r = ::pwrite(fd, src, n, off);
        if (r < 0 && errno == EINTR) {
            continue;
        }
        if (r <= 0) {
            return false;
        }
        src += r;
        off += r;
        n -= static_cast<std::size_t>(r);
    }
    return true;
}

/** Closes a file descriptor on scope exit. */
struct Fd {
    int fd = -1;
    ~Fd() {
        if (fd >= 0) {
            ::close(fd);
        }
    }
};

/** @returns the digest in a "<16 hex digits>.lkv" name, or false. */
bool parse_name(const std::string& name, std::uint64_t& digest) {
    if (name.size() != 16 + std::strlen(kSuffix) ||
        name.compare(16, std::string::npos, kSuffix) != 0) {
        return false;
    }
    digest = 0;
    for (std::size_t i = 0; i < 16; ++i) {
        const char c = name[i];
        int v;
        if (c >= '0' && c <= '9') {
            v = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            v = c - 'a' + 10;
        } else {
            return false;
        }
        digest = (digest << 4) | static_cast<std::uint64_t>(v);
    }
    return true;
}

}  // namespace

fs::path ensure_private_dir(const fs::path& given) {
    if (fs::create_directories(given)) {
        fs::permissions(given, fs::perms::owner_all,
                        fs::perm_options::replace);
    }
    // Checked and used through its real path: a symlink someone else
    // owns could otherwise pass here and be repointed later.
    const fs::path dir = fs::canonical(given);
    struct stat st {};
    if (::stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        throw std::runtime_error(dir.string() + ": not a directory");
    }
    if (st.st_uid != ::geteuid()) {
        throw std::runtime_error(dir.string() +
                                 ": owned by another user; the prefix "
                                 "cache needs a directory of its own");
    }
    if ((st.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
        throw std::runtime_error(dir.string() +
                                 ": group or others can write to it; "
                                 "chmod go-w it, or use another directory");
    }
    return dir;
}

DiskPrefixStore::DiskPrefixStore(DiskKey key, std::size_t block_bytes,
                                 std::uint64_t cap_bytes, std::nullptr_t)
    : key_(std::move(key)), block_bytes_(block_bytes), cap_bytes_(cap_bytes) {
    if (key_.block_tokens == 0 || block_bytes_ == 0) {
        throw std::invalid_argument(
            "DiskPrefixStore: block_tokens and block_bytes must be > 0");
    }
    if (key_.backend.size() > kMaxBackend) {
        throw std::invalid_argument(
            "DiskPrefixStore: backend name over 32 bytes");
    }
    const std::vector<std::uint8_t> k = header({}, 0);
    key_seed_ = checksum(k.data(), k.size());
}

DiskPrefixStore::DiskPrefixStore(fs::path dir, DiskKey key,
                                 std::size_t block_bytes,
                                 std::uint64_t cap_bytes)
    : DiskPrefixStore(std::move(key), block_bytes, cap_bytes, nullptr) {
    // Files hold clients' prompt tokens: owner only, and no one else
    // may plant files (or symlinks) in the directory.
    dir_ = ensure_private_dir(dir);
    // (mtime, digest, bytes): the order across a restart is the
    // order of the files' last reads and writes.
    std::vector<std::tuple<fs::file_time_type, std::uint64_t,
                           std::uint64_t>>
        found;
    for (const fs::directory_entry& e : fs::directory_iterator(dir_)) {
        std::error_code ec;
        const std::string name = e.path().filename().string();
        if (name.size() > std::strlen(kTmpSuffix) &&
            name.ends_with(kTmpSuffix)) {
            // A write a crash cut short, or a planted symlink (remove
            // takes the link, not its target).
            fs::remove(e.path(), ec);
            continue;
        }
        if (!fs::is_regular_file(e.symlink_status(ec))) {
            continue;  // a symlink is never a cache file
        }
        std::uint64_t digest;
        if (!parse_name(name, digest)) {
            continue;
        }
        std::error_code size_ec;
        std::error_code time_ec;
        const auto size = e.file_size(size_ec);
        const auto mtime = e.last_write_time(time_ec);
        if (!size_ec && !time_ec) {
            found.emplace_back(mtime, digest, size);
        }
    }
    std::sort(found.begin(), found.end());
    for (const auto& [mtime, digest, size] : found) {
        ++clock_;
        files_[digest] = File{size, clock_, clock_};
        used_ += size;
    }
}

std::vector<std::uint8_t> DiskPrefixStore::header(
    std::span<const tok::TokenId> prompt, std::uint32_t n_blocks) const {
    const std::uint32_t n_tokens = n_blocks * key_.block_tokens;
    std::vector<std::uint8_t> h(kMagic, kMagic + sizeof kMagic);
    put(h, kFormatVersion);
    put(h, n_tokens);
    put(h, n_blocks);
    put(h, static_cast<std::uint32_t>(key_.kv_type));
    put(h, static_cast<std::uint64_t>(block_bytes_));
    put(h, key_.model_hash);
    put(h, key_.model_size);
    put(h, key_.overrides);
    put(h, static_cast<std::uint32_t>(key_.device_attention));
    put(h, static_cast<std::uint32_t>(key_.experts_on_gpu));
    put(h, key_.block_tokens);
    put(h, key_.kv_dim);
    put(h, key_.n_layers);
    put(h, key_.layer_base);
    put(h, static_cast<std::uint32_t>(key_.backend.size()));
    char backend[kMaxBackend] = {};
    std::memcpy(backend, key_.backend.data(), key_.backend.size());
    h.insert(h.end(), backend, backend + kMaxBackend);
    for (std::uint32_t i = 0; i < n_tokens; ++i) {
        put(h, prompt[i]);
    }
    return h;
}

std::uint64_t DiskPrefixStore::file_bytes(std::size_t header_bytes,
                                          std::uint32_t n_blocks) const {
    // header, one checksum per block, the header checksum, the blocks
    return header_bytes + 8ull * n_blocks + 8 +
           static_cast<std::uint64_t>(block_bytes_) * n_blocks;
}

std::vector<std::uint8_t> DiskPrefixStore::meta(
    std::span<const tok::TokenId> prompt, std::uint32_t n_blocks,
    std::span<const std::uint8_t* const> src) const {
    std::vector<std::uint8_t> m = header(prompt, n_blocks);
    for (std::uint32_t i = 0; i < n_blocks; ++i) {
        put(m, checksum(src[i], block_bytes_));
    }
    put(m, checksum(m.data(), m.size()));
    return m;
}

bool DiskPrefixStore::decode(std::span<const std::uint8_t> blob,
                             std::span<const tok::TokenId> prompt,
                             std::uint32_t n_blocks,
                             std::span<std::uint8_t* const> dst) const {
    if (n_blocks == 0 || dst.size() < n_blocks ||
        prompt.size() < std::size_t{n_blocks} * key_.block_tokens) {
        return false;
    }
    const std::vector<std::uint8_t> want = header(prompt, n_blocks);
    if (blob.size() != file_bytes(want.size(), n_blocks)) {
        return false;
    }
    const std::size_t meta = want.size() + 8ull * n_blocks + 8;
    std::uint64_t hsum;
    std::memcpy(&hsum, blob.data() + meta - 8, 8);
    if (std::memcmp(blob.data(), want.data(), want.size()) != 0 ||
        hsum != checksum(blob.data(), meta - 8)) {
        return false;
    }
    for (std::uint32_t i = 0; i < n_blocks; ++i) {
        std::uint64_t sum;
        std::memcpy(&sum, blob.data() + want.size() + 8ull * i, 8);
        const std::uint8_t* b = blob.data() + meta + block_bytes_ * i;
        if (checksum(b, block_bytes_) != sum) {
            return false;
        }
        std::memcpy(dst[i], b, block_bytes_);
    }
    return true;
}

std::vector<std::uint64_t> DiskPrefixStore::digests(
    std::span<const tok::TokenId> prompt) const {
    const std::size_t blocks = prompt.size() / key_.block_tokens;
    std::vector<std::uint64_t> d(blocks + 1);
    std::uint64_t acc = key_seed_;
    for (std::size_t j = 1; j <= blocks; ++j) {
        const std::size_t base = (j - 1) * key_.block_tokens;
        for (std::uint32_t i = 0; i < key_.block_tokens; ++i) {
            acc = (acc ^ static_cast<std::uint32_t>(prompt[base + i])) *
                  kFnvPrime;
        }
        d[j] = mix64(acc);
    }
    return d;
}

std::uint64_t DiskPrefixStore::digest_of(
    std::span<const tok::TokenId> prompt, std::uint32_t n_blocks) const {
    return digests(prompt.first(std::size_t{n_blocks} *
                                key_.block_tokens))[n_blocks];
}

fs::path DiskPrefixStore::path_of(std::uint64_t digest) const {
    char name[17];
    std::snprintf(name, sizeof name, "%016llx",
                  static_cast<unsigned long long>(digest));
    return dir_ / (std::string(name) + kSuffix);
}

fs::path DiskPrefixStore::path_for(std::span<const tok::TokenId> prompt,
                                   std::uint32_t n_blocks) const {
    if (n_blocks == 0 ||
        prompt.size() < std::size_t{n_blocks} * key_.block_tokens) {
        return {};
    }
    return path_of(digest_of(prompt, n_blocks));
}

std::uint32_t DiskPrefixStore::find(
    std::span<const tok::TokenId> prompt) const {
    const std::vector<std::uint64_t> d = digests(prompt);
    std::lock_guard lock(mu_);
    for (std::size_t j = d.size() - 1; j >= 1; --j) {
        if (files_.contains(d[j])) {
            return static_cast<std::uint32_t>(j);
        }
    }
    return 0;
}

fs::path DiskPrefixStore::forget(std::uint64_t digest, std::uint64_t gen) {
    const auto it = files_.find(digest);
    if (it == files_.end() || (gen != 0 && it->second.gen != gen)) {
        return {};  // already gone, or replaced by a newer write
    }
    used_ -= it->second.bytes;
    files_.erase(it);
    return path_of(digest);
}

void DiskPrefixStore::touch(std::uint64_t digest) {
    files_[digest].tick = ++clock_;
}

namespace {

/** Unlinks `p` (when not empty), best effort: a file that stays is
 *  only uncounted until the next open indexes it again. */
void unlink_quietly(const fs::path& p) {
    std::error_code ec;
    if (!p.empty()) {
        fs::remove(p, ec);
    }
}

/** Sets `p`'s mtime to now, best effort: it only orders a restart. */
void stamp(const fs::path& p) {
    std::error_code ec;
    fs::last_write_time(p, fs::file_time_type::clock::now(), ec);
}

}  // namespace

bool DiskPrefixStore::read(std::span<const tok::TokenId> prompt,
                           std::uint32_t n_blocks,
                           std::span<std::uint8_t* const> dst) {
    if (n_blocks == 0 || dst.size() < n_blocks ||
        prompt.size() < std::size_t{n_blocks} * key_.block_tokens) {
        return false;
    }
    const std::uint64_t digest = digest_of(prompt, n_blocks);
    const std::vector<std::uint8_t> want = header(prompt, n_blocks);
    // Open under the lock, read and verify without it: the open file
    // stays readable even if another call unlinks it meanwhile, and
    // find() never waits behind a long read.
    Fd f;
    std::uint64_t gen;
    {
        std::lock_guard lock(mu_);
        const auto it = files_.find(digest);
        if (it == files_.end()) {
            return false;
        }
        gen = it->second.gen;
        f.fd = ::open(path_of(digest).c_str(),
                      O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (f.fd < 0) {
            if (errno == ENOENT) {
                forget(digest, gen);  // removed behind the store's back
            }
            return false;  // other errors may pass: keep the file
        }
    }
    const auto mismatch = [&] {
        fs::path p;
        {
            std::lock_guard lock(mu_);
            p = forget(digest, gen);
        }
        unlink_quietly(p);
        return false;
    };
    struct stat st {};
    if (::fstat(f.fd, &st) != 0) {
        return false;
    }
    if (static_cast<std::uint64_t>(st.st_size) !=
        file_bytes(want.size(), n_blocks)) {
        return mismatch();
    }
    // The header must be byte for byte the one this key and these
    // tokens produce: that check is what makes a digest collision or
    // another key's file a miss.
    const std::size_t meta = want.size() + 8ull * n_blocks + 8;
    std::vector<std::uint8_t> got(meta);
    if (!pread_full(f.fd, got.data(), meta, 0)) {
        return false;
    }
    std::uint64_t hsum;
    std::memcpy(&hsum, got.data() + meta - 8, 8);
    if (std::memcmp(got.data(), want.data(), want.size()) != 0 ||
        hsum != checksum(got.data(), meta - 8)) {
        return mismatch();
    }
    off_t off = static_cast<off_t>(meta);
    for (std::uint32_t i = 0; i < n_blocks; ++i) {
        std::uint64_t sum;
        std::memcpy(&sum, got.data() + want.size() + 8ull * i, 8);
        if (!pread_full(f.fd, dst[i], block_bytes_, off)) {
            return false;
        }
        if (checksum(dst[i], block_bytes_) != sum) {
            return mismatch();
        }
        off += static_cast<off_t>(block_bytes_);
    }
    {
        std::lock_guard lock(mu_);
        const auto it = files_.find(digest);
        if (it == files_.end() || it->second.gen != gen) {
            return true;  // replaced meanwhile: the bytes read were good
        }
        touch(digest);
    }
    stamp(path_of(digest));
    return true;
}

bool DiskPrefixStore::write(std::span<const tok::TokenId> prompt,
                            std::uint32_t n_blocks,
                            std::span<const std::uint8_t* const> src) {
    if (n_blocks == 0 || src.size() < n_blocks ||
        prompt.size() < std::size_t{n_blocks} * key_.block_tokens) {
        return false;
    }
    const std::uint64_t digest = digest_of(prompt, n_blocks);
    const std::vector<std::uint8_t> meta =
        this->meta(prompt, n_blocks, src);
    const std::uint64_t bytes =
        meta.size() + static_cast<std::uint64_t>(block_bytes_) * n_blocks;
    if (bytes > cap_bytes_) {
        return false;
    }
    const fs::path path = path_of(digest);
    std::uint64_t seq;
    {
        std::lock_guard lock(mu_);
        if (stored(digest)) {
            touch(digest);
            seq = 0;
        } else {
            seq = ++tmp_seq_;
        }
    }
    if (seq == 0) {
        stamp(path);
        return true;
    }

    // Write a private temporary file without the lock, then take it to
    // make room and rename. O_EXCL | O_NOFOLLOW: a file or symlink
    // already at that name fails the write, never redirects it.
    const fs::path tmp =
        path.string() + "." + std::to_string(seq) + kTmpSuffix;
    bool ok;
    {
        Fd f{::open(tmp.c_str(),
                    O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                    0600)};
        ok = f.fd >= 0 && pwrite_full(f.fd, meta.data(), meta.size(), 0);
        off_t off = static_cast<off_t>(meta.size());
        for (std::uint32_t i = 0; ok && i < n_blocks; ++i) {
            ok = pwrite_full(f.fd, src[i], block_bytes_, off);
            off += static_cast<off_t>(block_bytes_);
        }
        if (f.fd >= 0) {
            ok = ::close(f.fd) == 0 && ok;
            f.fd = -1;
        }
    }
    std::error_code ec;
    if (!ok) {
        fs::remove(tmp, ec);
        return false;
    }
    // Victims leave the index under the lock and are unlinked after
    // it: unlinking a large file can take a while, and find() on the
    // engine thread takes this lock. The rename stays under it so the
    // index never names a file that is not in place yet.
    std::vector<fs::path> victims;
    bool stored_now = false;
    {
        std::lock_guard lock(mu_);
        if (stored(digest)) {  // another call stored it meanwhile
            touch(digest);
            stored_now = true;
        } else {
            // Evicting before the rename: if the rename fails the
            // evicted files are gone anyway, which costs only hits.
            while (used_ + bytes > cap_bytes_ && !files_.empty()) {
                const auto lru = std::min_element(
                    files_.begin(), files_.end(),
                    [](const auto& a, const auto& b) {
                        return a.second.tick < b.second.tick;
                    });
                victims.push_back(forget(lru->first, 0));
            }
            fs::rename(tmp, path, ec);
            ok = !ec;
            if (ok) {
                ++clock_;
                files_[digest] = File{bytes, clock_, clock_};
                used_ += bytes;
            }
        }
    }
    for (const fs::path& v : victims) {
        unlink_quietly(v);
    }
    if (stored_now || !ok) {
        unlink_quietly(tmp);
        if (stored_now) {
            stamp(path);
        }
    }
    return stored_now || ok;
}

bool DiskPrefixStore::stored(std::uint64_t digest) {
    const auto it = files_.find(digest);
    if (it == files_.end()) {
        return false;
    }
    std::error_code ec;
    if (fs::exists(path_of(digest), ec) || ec) {
        return true;
    }
    // Removed behind the store's back: forget it so it can be stored
    // again.
    used_ -= it->second.bytes;
    files_.erase(it);
    return false;
}

void DiskPrefixStore::set_device_attention(bool on) {
    std::lock_guard lock(mu_);
    key_.device_attention = on;
    // The seed names the files: the other path's files become misses
    // (and stay under the cap until LRU removes them).
    const std::vector<std::uint8_t> k = header({}, 0);
    key_seed_ = checksum(k.data(), k.size());
}

void DiskPrefixStore::set_experts_on_gpu(bool on) {
    std::lock_guard lock(mu_);
    key_.experts_on_gpu = on;
    const std::vector<std::uint8_t> k = header({}, 0);
    key_seed_ = checksum(k.data(), k.size());
}

std::uint64_t DiskPrefixStore::used_bytes() const {
    std::lock_guard lock(mu_);
    return used_;
}

}  // namespace locus::engine
