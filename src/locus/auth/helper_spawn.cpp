#include "locus/auth/helper_spawn.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <exception>

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
void set_cloexec(int fd) {
    const int f = ::fcntl(fd, F_GETFD);
    if (f != -1) {
        ::fcntl(fd, F_SETFD, f | FD_CLOEXEC);
    }
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
    // -> parent reads. (no pipe2 on macOS; set CLOEXEC by hand.)
    int in_pipe[2];
    int out_pipe[2];
    if (::pipe(in_pipe) != 0) {
        return fail("pipe");
    }
    if (::pipe(out_pipe) != 0) {
        ::close(in_pipe[0]);
        ::close(in_pipe[1]);
        return fail("pipe");
    }
    // The parent-retained ends must not leak into the next helper.
    set_cloexec(in_pipe[1]);
    set_cloexec(out_pipe[0]);

    const pid_t pid = ::fork();
    if (pid < 0) {
        close_all(in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1]);
        return fail("fork");
    }
    if (pid == 0) {
        // Child: stdin <- in_pipe read end, stdout -> out_pipe write
        // end. dup2 clears CLOEXEC on the new std fds.
        ::dup2(in_pipe[0], STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        close_all(in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1]);
        std::vector<char*> cargv;
        cargv.reserve(argv.size() + 1);
        for (const std::string& a : argv) {
            cargv.push_back(const_cast<char*>(a.c_str()));
        }
        cargv.push_back(nullptr);
        ::execvp(cargv[0], cargv.data());
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
