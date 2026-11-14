#include "locus/sys/child_fds.hpp"

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <string>

namespace locus::sys {

int child_fd_limit() {
    rlimit rl{};
    long n = ::sysconf(_SC_OPEN_MAX);
    if (::getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) {
        n = std::max<long>(n, static_cast<long>(rl.rlim_cur));
    }
    if (n <= 0) n = 1024;
    return static_cast<int>(std::min<long>(n, 1L << 20));
}

void close_fds_from_3(int limit) {
#if defined(__linux__) && defined(SYS_close_range)
    if (::syscall(SYS_close_range, 3U, ~0U, 0U) == 0) return;
#endif
    for (int fd = 3; fd < limit; ++fd) ::close(fd);
}

namespace {

bool executable_file(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) &&
           ::access(p.c_str(), X_OK) == 0;
}

}  // namespace

std::string find_executable(const std::string& name) {
    if (name.empty()) return {};
    if (name.find('/') != std::string::npos) {
        return executable_file(name) ? name : std::string();
    }
    const char* env = std::getenv("PATH");
    const std::string path = env != nullptr ? env : "/usr/bin:/bin";
    std::size_t start = 0;
    for (;;) {
        const std::size_t colon = path.find(':', start);
        std::string dir = path.substr(start, colon == std::string::npos
                                                 ? std::string::npos
                                                 : colon - start);
        // An empty entry is the current dir. assign(1, '.') rather than
        // = ".", which trips GCC 12's -Wrestrict false positive.
        if (dir.empty()) dir.assign(1, '.');
        std::string p = dir;
        p += '/';
        p += name;
        if (executable_file(p)) return p;
        if (colon == std::string::npos) return {};
        start = colon + 1;
    }
}

bool lift_above_std(int& fd) {
    if (fd > STDERR_FILENO) return true;
    const int up = ::fcntl(fd, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
    if (up < 0) return false;
    ::close(fd);
    fd = up;
    return true;
}

}  // namespace locus::sys
