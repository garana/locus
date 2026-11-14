// Reference prefix cache helper for locus (i#655, docs/DESIGN.md).
//
// Usage: locus-prefix-helper DIR [CAP_MIB] [--exit-after N]
//
// Keeps each prefix blob as DIR/<name>.lkv (the file a DiskPrefixStore
// writes for the same name), evicting the least recently used files by
// mtime to stay under CAP_MIB (no cap when absent). It speaks the
// framed text protocol on stdin / stdout and answers one request at a
// time:
//
//   list                    -> names=<name> <name> ...
//   read name= shm= bytes=  -> ok=1 when DIR/<name>.lkv has `bytes`
//                              bytes, copied into the shm object
//   write name= shm= bytes= -> ok=1 when the shm bytes are stored
//   evict name=             (no reply) removes DIR/<name>.lkv
//
// --exit-after N exits after N requests (tests use it to kill it).
//
// REFERENCE ONLY: a real helper keeps the blobs on remote or shared
// storage. It never checks a blob: locus does.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

#include "locus/auth/helper_message.hpp"
#include "locus/engine/disk_prefix_store.hpp"

namespace {

namespace fs = std::filesystem;
using locus::auth::decode_text;
using locus::auth::encode_text;
using locus::auth::HelperDecode;
using locus::auth::HelperMessage;

void write_all(const std::string& s) {
    std::size_t off = 0;
    while (off < s.size()) {
        const ssize_t n =
            ::write(STDOUT_FILENO, s.data() + off, s.size() - off);
        if (n <= 0) {
            return;
        }
        off += static_cast<std::size_t>(n);
    }
}

/** @returns true for 16 lower-case hex digits: a name that cannot
 *  leave DIR. */
bool valid_name(const std::string* s) {
    return s != nullptr && s->size() == 16 &&
           std::all_of(s->begin(), s->end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

/** @returns true for "/locus-pfx-..." with no other slash. */
bool valid_shm(const std::string* s) {
    return s != nullptr && s->rfind("/locus-pfx-", 0) == 0 &&
           s->find('/', 1) == std::string::npos;
}

/** Maps the shm object `name` of `n` bytes; unmapped on scope exit. */
struct Map {
    void* p = MAP_FAILED;
    std::size_t n = 0;
    Map(const std::string& name, std::size_t bytes, bool writable) : n(bytes) {
        const int fd = ::shm_open(name.c_str(), writable ? O_RDWR : O_RDONLY,
                                  0);
        if (fd < 0) {
            return;
        }
        struct stat st {};
        if (::fstat(fd, &st) == 0 &&
            static_cast<std::uint64_t>(st.st_size) >= bytes && bytes > 0) {
            p = ::mmap(nullptr, bytes,
                       writable ? PROT_READ | PROT_WRITE : PROT_READ,
                       MAP_SHARED, fd, 0);
        }
        ::close(fd);
    }
    ~Map() {
        if (p != MAP_FAILED) {
            ::munmap(p, n);
        }
    }
    std::uint8_t* data() const { return static_cast<std::uint8_t*>(p); }
};

bool read_full(int fd, std::uint8_t* dst, std::size_t n) {
    while (n > 0) {
        const ssize_t r = ::read(fd, dst, n);
        if (r <= 0) {
            return false;
        }
        dst += r;
        n -= static_cast<std::size_t>(r);
    }
    return true;
}

bool write_full(int fd, const std::uint8_t* src, std::size_t n) {
    while (n > 0) {
        const ssize_t r = ::write(fd, src, n);
        if (r <= 0) {
            return false;
        }
        src += r;
        n -= static_cast<std::size_t>(r);
    }
    return true;
}

struct Store {
    fs::path dir;
    std::uint64_t cap = 0;  // 0: none

    fs::path path(const std::string& name) const {
        return dir / (name + ".lkv");
    }

    std::string list() const {
        std::string out;
        std::error_code ec;
        for (const auto& e : fs::directory_iterator(dir, ec)) {
            const std::string f = e.path().filename().string();
            if (f.size() == 20 && f.ends_with(".lkv")) {
                const std::string name = f.substr(0, 16);
                if (valid_name(&name)) {
                    out += (out.empty() ? "" : " ") + name;
                }
            }
        }
        return out;
    }

    bool read(const std::string& name, const std::string& shm,
              std::size_t bytes) const {
        const int fd =
            ::open(path(name).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) {
            return false;
        }
        struct stat st {};
        bool ok = ::fstat(fd, &st) == 0 &&
                  static_cast<std::uint64_t>(st.st_size) == bytes;
        if (ok) {
            Map m(shm, bytes, true);
            ok = m.p != MAP_FAILED && read_full(fd, m.data(), bytes);
        }
        ::close(fd);
        if (ok) {
            std::error_code ec;
            fs::last_write_time(path(name), fs::file_time_type::clock::now(),
                                ec);
        }
        return ok;
    }

    bool write(const std::string& name, const std::string& shm,
               std::size_t bytes) const {
        if (cap != 0 && bytes > cap) {
            return false;
        }
        const Map m(shm, bytes, false);
        if (m.p == MAP_FAILED) {
            return false;
        }
        // A file or symlink left at the temp name is removed (remove
        // never follows a symlink), and O_EXCL | O_NOFOLLOW makes one
        // planted after that fail the write instead of redirecting it.
        const fs::path tmp = path(name).string() + ".tmp";
        std::error_code rm_ec;
        fs::remove(tmp, rm_ec);
        const int fd = ::open(
            tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
            0600);
        if (fd < 0) {
            return false;
        }
        bool ok = write_full(fd, m.data(), bytes);
        ok = ::close(fd) == 0 && ok;
        std::error_code ec;
        if (ok) {
            fs::rename(tmp, path(name), ec);
            ok = !ec;
        }
        if (!ok) {
            fs::remove(tmp, ec);
            return false;
        }
        evict_to_cap(name);
        return true;
    }

    /** Removes the least recently used files, never `keep`, until the
     *  directory fits the cap. */
    void evict_to_cap(const std::string& keep) const {
        if (cap == 0) {
            return;
        }
        std::vector<std::tuple<fs::file_time_type, fs::path, std::uint64_t>>
            files;
        std::uint64_t used = 0;
        std::error_code ec;
        for (const auto& e : fs::directory_iterator(dir, ec)) {
            std::error_code e1, e2;
            const auto size = e.file_size(e1);
            const auto t = e.last_write_time(e2);
            if (!e1 && !e2 && e.path().extension() == ".lkv") {
                used += size;
                if (e.path().stem().string() != keep) {
                    files.emplace_back(t, e.path(), size);
                }
            }
        }
        std::sort(files.begin(), files.end());
        for (const auto& [t, p, size] : files) {
            if (used <= cap) {
                break;
            }
            fs::remove(p, ec);
            used -= size;
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    Store store;
    long exit_after = -1;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--exit-after" && i + 1 < argc) {
            exit_after = std::strtol(argv[++i], nullptr, 10);
        } else {
            pos.push_back(a);
        }
    }
    if (pos.empty() || pos.size() > 2) {
        std::fputs(
            "usage: locus-prefix-helper DIR [CAP_MIB] [--exit-after N]\n",
            stderr);
        return 2;
    }
    store.dir = pos[0];
    if (pos.size() == 2) {
        store.cap = std::strtoull(pos[1].c_str(), nullptr, 10) << 20;
    }
    std::error_code ec;
    try {
        locus::engine::ensure_private_dir(store.dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "locus-prefix-helper: %s\n", e.what());
        return 2;
    }

    std::string buf;
    char chunk[4096];
    for (;;) {
        const ssize_t n = ::read(STDIN_FILENO, chunk, sizeof chunk);
        if (n <= 0) {
            break;  // EOF: locus closed the pipe
        }
        buf.append(chunk, static_cast<std::size_t>(n));
        HelperMessage m;
        std::string err;
        HelperDecode d;
        while ((d = decode_text(buf, m, err)) == HelperDecode::kComplete) {
            if (exit_after == 0) {
                return 0;
            }
            if (exit_after > 0) {
                --exit_after;
            }
            const std::string* name = m.get("name");
            const std::string* shm = m.get("shm");
            const std::string* bytes_s = m.get("bytes");
            const std::size_t bytes =
                bytes_s ? std::strtoull(bytes_s->c_str(), nullptr, 10) : 0;
            HelperMessage r;
            r.req_id = m.req_id;
            r.type = m.type;
            if (m.type == "list") {
                r.set("names", store.list());
            } else if (m.type == "read" || m.type == "write") {
                const bool ok =
                    valid_name(name) && valid_shm(shm) && bytes > 0 &&
                    (m.type == "read" ? store.read(*name, *shm, bytes)
                                      : store.write(*name, *shm, bytes));
                r.set("ok", ok ? "1" : "0");
            } else if (m.type == "evict" && valid_name(name)) {
                fs::remove(store.path(*name), ec);
            }
            std::string wire;
            if (m.req_id != 0 && encode_text(r, wire)) {
                write_all(wire);
            }
            m = HelperMessage{};
        }
        if (d == HelperDecode::kError) {
            break;
        }
    }
    return 0;
}
