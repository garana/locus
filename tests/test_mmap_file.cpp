#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

#include "catch_amalgamated.hpp"
#include "locus/sys/mmap_file.hpp"

namespace {

const void* hook_p = nullptr;
std::size_t hook_n = 0;
bool hook_saw_live = false;

void record_unmap(const void* p, std::size_t n) {
    hook_p = p;
    hook_n = n;
    // The hook runs before munmap, while the range is still mapped.
    hook_saw_live = locus::sys::in_file_mapping(p, n);
}

}  // namespace

TEST_CASE("mapped files are registered and report their unmap",
          "[sys]") {
    using locus::sys::in_file_mapping;
    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("locus_mmap_" + std::to_string(::getpid()) + ".bin"))
            .string();
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        REQUIRE(f != nullptr);
        const std::vector<char> bytes(10000, 'x');
        REQUIRE(std::fwrite(bytes.data(), 1, bytes.size(), f) ==
                bytes.size());
        std::fclose(f);
    }
    std::vector<int> heap(64);
    REQUIRE_FALSE(in_file_mapping(heap.data(), sizeof(int)));

    const std::byte* data = nullptr;
    {
        // The previous hook (the Vulkan backend's, if an earlier test
        // built its state) goes back even when a check below throws, or
        // the rest of the process would keep ours (i#331).
        struct HookGuard {
            locus::sys::UnmapHook prev;
            ~HookGuard() { locus::sys::set_unmap_hook(prev); }
        } guard{locus::sys::set_unmap_hook(&record_unmap)};
        {
            locus::sys::MappedFile mf(path);
            data = mf.bytes().data();
            REQUIRE(in_file_mapping(data, 10000));
            REQUIRE(in_file_mapping(data + 100, 50));
            // Past the end, or straddling it, is not inside the mapping.
            REQUIRE_FALSE(in_file_mapping(data + 9990, 20));
            REQUIRE_FALSE(in_file_mapping(data + 20000, 1));
        }
        // The slot still held ours when the file unmapped.
        REQUIRE(locus::sys::set_unmap_hook(&record_unmap) == &record_unmap);
    }
    REQUIRE(hook_p == data);
    REQUIRE(hook_n == 10000);
    REQUIRE(hook_saw_live);
    REQUIRE_FALSE(in_file_mapping(data, 1));
    std::remove(path.c_str());
}
