#include "locus/sys/mmap_file.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <system_error>
#include <utility>

namespace locus::sys {

namespace {

[[noreturn]] void fail(const char* what) {
    throw std::system_error(errno, std::generic_category(), what);
}

}  // namespace

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

}  // namespace locus::sys
