#include "locus/auth/helper_spawn.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <exception>

#include "locus/sys/child_fds.hpp"

namespace locus::auth {

namespace detail {

SpawnedChild::~SpawnedChild() {
    if (!armed_) {
        return;  // released into a HelperConnection; it owns them now
    }
    // Error teardown: close the retained fds and reap the child.
    // SIGKILL is uncatchable, so the blocking waitpid returns.
    ::close(read_fd_);
    ::close(write_fd_);
    ::kill(pid_, SIGKILL);
    ::waitpid(pid_, nullptr, 0);
}

std::unique_ptr<HelperConnection> SpawnedChild::release() {
    auto conn = std::make_unique<HelperConnection>(read_fd_, write_fd_,
                                                   pid_);
    armed_ = false;  // connection now owns the fds + child
    return conn;
}

}  // namespace detail

namespace {
void close_all(int a, int b, int c, int d) {
    ::close(a);
    ::close(b);
    ::close(c);
    ::close(d);
}
#if !defined(__linux__)
void set_cloexec(int fd) {
    const int f = ::fcntl(fd, F_GETFD);
    if (f != -1) {
        ::fcntl(fd, F_SETFD, f | FD_CLOEXEC);
    }
}
#endif
/**
 * pipe() with both ends close-on-exec. Linux sets the flag
 * atomically; macOS has no pipe2, so a fork+exec on another thread
 * between pipe() and fcntl() can still inherit the fds.
 */
int cloexec_pipe(int fds[2]) {
#if defined(__linux__)
    return ::pipe2(fds, O_CLOEXEC);
#else
    if (::pipe(fds) != 0) {
        return -1;
    }
    set_cloexec(fds[0]);
    set_cloexec(fds[1]);
    return 0;
#endif
}
}  // namespace

std::unique_ptr<HelperConnection> spawn_helper(
    const std::vector<std::string>& argv, std::string* err) {
    auto fail = [&](const char* what) {
        if (err != nullptr) {
            *err = std::string(what) + ": " + std::strerror(errno);
        }
        return std::unique_ptr<HelperConnection>{};
    };
    if (argv.empty()) {
        if (err != nullptr) {
            *err = "spawn_helper: empty argv";
        }
        return nullptr;
    }
    // in_pipe: parent writes -> child stdin. out_pipe: child stdout
    // -> parent reads. All four ends are close-on-exec, so no other
    // child (another helper, or one forked on another thread) inherits
    // them: an inherited out_pipe write end would keep the parent from
    // ever seeing EOF when this helper dies.
    int in_pipe[2];
    int out_pipe[2];
    if (cloexec_pipe(in_pipe) != 0) {
        return fail("pipe");
    }
    if (cloexec_pipe(out_pipe) != 0) {
        ::close(in_pipe[0]);
        ::close(in_pipe[1]);
        return fail("pipe");
    }
    // A server started with fd 0, 1 or 2 closed gets one of them here;
    // the child's dup2 onto it would then keep close-on-exec.
    if (!sys::lift_above_std(in_pipe[0]) || !sys::lift_above_std(in_pipe[1]) ||
        !sys::lift_above_std(out_pipe[0]) ||
        !sys::lift_above_std(out_pipe[1])) {
        close_all(in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1]);
        return fail("fcntl");
    }

    // Everything the child needs is made here, before fork: the child of
    // a threaded process may only make async-signal-safe calls (another
    // thread may hold the allocator's lock at the fork), so it does not
    // allocate.
    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for (const std::string& a : argv) {
        cargv.push_back(const_cast<char*>(a.c_str()));
    }
    cargv.push_back(nullptr);
    const int fd_limit = sys::child_fd_limit();
    // The PATH search too: execvp's is not async-signal-safe. A name that
    // resolves to nothing is not exec'd at all (exec'ing a bare name
    // would open it in the cwd, which execvp never did): the child exits
    // 127 and the connection reads as broken, as before.
    const std::string exe = sys::find_executable(argv[0]);

    const pid_t pid = ::fork();
    if (pid < 0) {
        close_all(in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1]);
        return fail("fork");
    }
    if (pid == 0) {
        // Child: stdin <- in_pipe read end, stdout -> out_pipe write
        // end. dup2 clears CLOEXEC on the new std fds. Then every other
        // fd goes, so the helper holds none of the server's sockets or
        // files (not every fd locus opens is close-on-exec).
        ::dup2(in_pipe[0], STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        sys::close_fds_from_3(fd_limit);
        // A signal mask survives exec: locus-server blocks SIGINT and
        // SIGTERM for its stopper thread, and the helper must get them.
        sigset_t none;
        sigemptyset(&none);
        ::sigprocmask(SIG_SETMASK, &none, nullptr);
        if (!exe.empty()) ::execv(exe.c_str(), cargv.data());
        _exit(127);  // exec failed
    }
    // Parent: keep in_pipe[1] (write to child) + out_pipe[0] (read).
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);
    // Own the child + fds via RAII across construction: the
    // HelperConnection ctor starts a reader thread that can throw
    // (thread limit / OOM). Without this the exception would escape
    // before the connection exists, leaving the forked child
    // unreaped and the two pipe fds leaked. On throw, `child`'s dtor
    // reaps and closes.
    detail::SpawnedChild child(pid, out_pipe[0] /*read*/,
                               in_pipe[1] /*write*/);
    try {
        return child.release();
    } catch (const std::exception& e) {
        if (err != nullptr) {
            *err = std::string("spawn_helper: connection: ") +
                   e.what();
        }
        return nullptr;
    }
}

}  // namespace locus::auth
