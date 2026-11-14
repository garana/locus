#include "locus/sys/child_fds.hpp"

#include <sys/resource.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif

#include <algorithm>

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

}  // namespace locus::sys
