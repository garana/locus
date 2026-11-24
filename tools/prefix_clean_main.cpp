// locus-prefix-clean: removes the disk prefix cache files that no
// current server reads (i#1180). Each locus-server prints its key at
// boot ("prefix cache key <16 hex digits>"); the files of every other
// key (another model, backend or build) are removed.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "locus/engine/disk_prefix_store.hpp"

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--dry-run] --keep KEY [--keep KEY ...] DIR\n"
                 "\n"
                 "Removes from the --prefix-cache-dir DIR every cache file\n"
                 "whose key is not a KEY: the 16 hex digits each\n"
                 "locus-server prints at boot (\"prefix cache key ...\").\n"
                 "Give the key of every server that shares DIR. Files whose\n"
                 "header does not parse, other files and symlinks are left\n"
                 "alone; DIR must not be a symlink.\n"
                 "  --dry-run  list what would go, remove nothing\n",
                 argv0);
}

/** @returns whether `s` is 16 lowercase or uppercase hex digits. */
bool parse_key(const char* s, std::uint64_t& out) {
    if (std::strlen(s) != 16) {
        return false;
    }
    out = 0;
    for (int i = 0; i < 16; ++i) {
        const char c = s[i];
        int v;
        if (c >= '0' && c <= '9') {
            v = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            v = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            v = c - 'A' + 10;
        } else {
            return false;
        }
        out = (out << 4) | static_cast<std::uint64_t>(v);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    bool dry_run = false;
    std::vector<std::uint64_t> keep;
    std::string dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dry-run") {
            dry_run = true;
        } else if (a == "--keep" && i + 1 < argc) {
            std::uint64_t k;
            if (!parse_key(argv[++i], k)) {
                std::fprintf(stderr, "%s: --keep %s: not 16 hex digits\n",
                             argv[0], argv[i]);
                return 2;
            }
            keep.push_back(k);
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else if (dir.empty() && !a.starts_with("-")) {
            dir = a;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (dir.empty() || keep.empty()) {
        usage(argv[0]);
        return 2;
    }
    try {
        std::vector<std::filesystem::path> gone;
        const auto r =
            locus::engine::clean_disk_prefix(dir, keep, dry_run, &gone);
        for (const auto& p : gone) {
            std::printf("%s %s\n", dry_run ? "would remove" : "removed",
                        p.c_str());
        }
        std::printf("%s %llu files (%llu MiB), kept %llu, skipped %llu "
                    "unreadable\n",
                    dry_run ? "would remove" : "removed",
                    static_cast<unsigned long long>(r.removed),
                    static_cast<unsigned long long>(r.removed_bytes >> 20),
                    static_cast<unsigned long long>(r.kept),
                    static_cast<unsigned long long>(r.skipped));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
        return 1;
    }
    return 0;
}
