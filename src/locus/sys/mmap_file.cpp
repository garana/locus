#include "locus/sys/mmap_file.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <mutex>
#include <system_error>
#include <utility>
#include <vector>

namespace locus::sys {

namespace {

[[noreturn]] void fail(const char* what) {
    throw std::system_error(errno, std::generic_category(), what);
}

/** Live mappings as [begin, end) (in_file_mapping). */
struct Registry {
    std::mutex mu;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> live;
};

Registry& registry() {
    static Registry r;
    return r;
}

std::atomic<UnmapHook> unmap_hook{nullptr};

}  // namespace

bool in_file_mapping(const void* p, std::size_t n) {
    const auto b = reinterpret_cast<std::uintptr_t>(p);
    Registry& r = registry();
    std::lock_guard lk(r.mu);
    return std::any_of(r.live.begin(), r.live.end(), [&](const auto& m) {
        return b >= m.first && b <= m.second && n <= m.second - b;
    });
}

UnmapHook set_unmap_hook(UnmapHook hook) {
    return unmap_hook.exchange(hook);
}

MappedFile::MappedFile(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fail("open");
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        fail("fstat");
    }
    size_ = static_cast<std::size_t>(st.st_size);
    if (size_ > 0) {
        data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd);
            fail("mmap");
        }
        const auto b = reinterpret_cast<std::uintptr_t>(data_);
        Registry& r = registry();
        std::lock_guard lk(r.mu);
        r.live.emplace_back(b, b + size_);
    }
    ::close(fd);
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        this->~MappedFile();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

MappedFile::~MappedFile() {
    if (data_ != nullptr) {
        if (auto hook = unmap_hook.load()) {
            hook(data_, size_);
        }
        {
            const auto b = reinterpret_cast<std::uintptr_t>(data_);
            Registry& r = registry();
            std::lock_guard lk(r.mu);
            std::erase(r.live, std::pair{b, b + size_});
        }
        ::munmap(data_, size_);
    }
}

void advise_willneed(const void* p, std::size_t n) {
    if (p == nullptr || n == 0) {
        return;
    }
    static const std::size_t page =
        static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const auto addr = reinterpret_cast<std::uintptr_t>(p);
    const std::uintptr_t base = addr & ~(page - 1);
    // Best-effort hint; failures (e.g. non-mmap memory) are fine.
    (void)::madvise(reinterpret_cast<void*>(base),
                    n + (addr - base), MADV_WILLNEED);
}

void advise_dontneed(const void* p, std::size_t n) {
    if (p == nullptr || n == 0) {
        return;
    }
    static const std::size_t page =
        static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const auto addr = reinterpret_cast<std::uintptr_t>(p);
    const std::uintptr_t base = (addr + page - 1) & ~(page - 1);
    const std::uintptr_t end = (addr + n) & ~(page - 1);
    if (end <= base) {
        return;  // range smaller than one full page
    }
    (void)::madvise(reinterpret_cast<void*>(base), end - base,
                    MADV_DONTNEED);
}

bool lock_resident(const void* p, std::size_t n) {
    if (p == nullptr || n == 0) {
        return true;
    }
    static const std::size_t page =
        static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const auto addr = reinterpret_cast<std::uintptr_t>(p);
    const std::uintptr_t base = addr & ~(page - 1);
    return ::mlock(reinterpret_cast<void*>(base),
                   n + (addr - base)) == 0;
}

}  // namespace locus::sys
