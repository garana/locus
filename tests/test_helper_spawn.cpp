#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <utility>

#include "catch_amalgamated.hpp"
#include "locus/auth/helper_connection.hpp"
#include "locus/auth/helper_message.hpp"
#include "locus/auth/helper_spawn.hpp"
#include "locus/sys/child_fds.hpp"

using locus::auth::HelperMessage;
using locus::auth::spawn_helper;
using locus::auth::detail::SpawnedChild;
using namespace std::chrono_literals;

TEST_CASE("spawn_helper round-trips through a real process", "[auth]") {
    // `cat` echoes stdin -> stdout verbatim, so it replays the framed
    // request (same req_id) back as its "response" -- exercising the
    // pipe wiring, exec, and req_id correlation end to end.
    std::string err;
    auto conn = spawn_helper({"cat"}, &err);
    REQUIRE(conn);

    HelperMessage q;
    q.type = "auth";
    q.set("credential", "hello");
    auto resp = conn->request(std::move(q), 2000ms);
    REQUIRE(resp.has_value());
    REQUIRE(resp->type == "auth");
    REQUIRE(*resp->get("credential") == "hello");
}

// i#625 F10: the helper gets only stdin, stdout and stderr. A pipe the
// server holds without close-on-exec must not stay open in the child:
// once the parent closes its write end, the read end sees EOF at once,
// even with the helper still running.
TEST_CASE("spawn_helper does not leak the server's fds", "[auth]") {
    int p[2];
    REQUIRE(::pipe(p) == 0);  // no FD_CLOEXEC, on purpose
    std::string err;
    auto conn = spawn_helper({"cat"}, &err);
    REQUIRE(conn);
    ::close(p[1]);
    pollfd pfd{p[0], POLLIN, 0};
    REQUIRE(::poll(&pfd, 1, 2000) == 1);  // not a timeout: no writer left
    char c;
    REQUIRE(::read(p[0], &c, 1) == 0);  // EOF
    ::close(p[0]);
    // The helper itself is still up and answering.
    HelperMessage q;
    q.type = "auth";
    q.set("credential", "still-here");
    auto resp = conn->request(std::move(q), 2000ms);
    REQUIRE(resp.has_value());
}

// i#828: with fd 0 closed (a server started with stdin closed), the
// helper's stdin pipe end was fd 0 itself; dup2(0, 0) kept its
// close-on-exec, so the helper started with no stdin and exited.
TEST_CASE("spawn_helper works with the server's stdin closed", "[auth]") {
    const int saved = ::dup(STDIN_FILENO);
    REQUIRE(saved >= 0);
    REQUIRE(::close(STDIN_FILENO) == 0);
    std::string err;
    auto conn = spawn_helper({"cat"}, &err);
    // Put stdin back before any REQUIRE can leave the test.
    const bool restored = ::dup2(saved, STDIN_FILENO) == STDIN_FILENO;
    ::close(saved);
    REQUIRE(restored);
    REQUIRE(conn);
    HelperMessage q;
    q.type = "auth";
    q.set("credential", "stdin-closed");
    auto resp = conn->request(std::move(q), 2000ms);
    REQUIRE(resp.has_value());
    REQUIRE(*resp->get("credential") == "stdin-closed");
}

TEST_CASE("spawn_helper resolves the command on PATH in the parent",
          "[auth]") {
    // execv (not execvp) runs what find_executable resolved before fork.
    CHECK(locus::sys::find_executable("cat").find('/') != std::string::npos);
    CHECK(locus::sys::find_executable("no-such-locus-cmd-xyz").empty());
    std::string err;
    auto conn = spawn_helper({"cat"}, &err);
    REQUIRE(conn);
    HelperMessage q;
    q.type = "auth";
    q.set("credential", "on-path");
    REQUIRE(conn->request(std::move(q), 2000ms).has_value());
}

// A bare name not on PATH must not run a file of that name in the
// server's working directory (execvp never looked there).
TEST_CASE("spawn_helper does not run a bare name from the cwd", "[auth]") {
    namespace fs = std::filesystem;
    const fs::path d = fs::temp_directory_path() /
                       ("locus-cwd-" + std::to_string(::getpid()));
    fs::create_directories(d);
    const fs::path script = d / "locus-cwd-helper-xyz";
    std::ofstream(script) << "#!/bin/sh\nexec cat\n";
    fs::permissions(script, fs::perms::owner_all, fs::perm_options::replace);
    const fs::path back = fs::current_path();
    fs::current_path(d);
    std::string err;
    auto conn = spawn_helper({"locus-cwd-helper-xyz"}, &err);
    fs::current_path(back);
    REQUIRE(conn);
    HelperMessage q;
    q.type = "auth";
    q.set("credential", "x");
    CHECK_FALSE(conn->request(std::move(q), 1000ms).has_value());
    conn.reset();
    fs::remove_all(d);
}

TEST_CASE("spawn_helper on a bad command reports broken", "[auth]") {
    std::string err;
    auto conn = spawn_helper({"/no/such/locus-helper-xyz"}, &err);
    REQUIRE(conn);  // the fork itself succeeds

    HelperMessage q;
    q.type = "auth";
    q.set("credential", "x");
    auto resp = conn->request(std::move(q), 1000ms);
    REQUIRE_FALSE(resp.has_value());  // exec failed -> stdout EOF
    for (int i = 0; i < 100 && !conn->broken(); ++i) {
        std::this_thread::sleep_for(2ms);
    }
    REQUIRE(conn->broken());
}

TEST_CASE("spawn_helper rejects empty argv", "[auth]") {
    std::string err;
    auto conn = spawn_helper({}, &err);
    REQUIRE_FALSE(conn);
    REQUIRE_FALSE(err.empty());
}

TEST_CASE("a helper gets SIGTERM though the server blocks it",
          "[auth]") {
    // locus-server blocks SIGINT / SIGTERM for its stopper thread, and
    // a signal mask survives fork and exec. The child must unblock
    // them: the dtor's SIGTERM ends this sleep at once when it can be
    // received, and only the SIGKILL after the grace (about a second)
    // does when it is blocked. Not a shell: dash clears the mask itself.
    sigset_t term;
    sigemptyset(&term);
    sigaddset(&term, SIGTERM);
    sigset_t old;
    REQUIRE(pthread_sigmask(SIG_BLOCK, &term, &old) == 0);
    std::string err;
    auto conn = spawn_helper({"/bin/sleep", "5"}, &err);
    REQUIRE(pthread_sigmask(SIG_SETMASK, &old, nullptr) == 0);
    REQUIRE(conn);
    const auto t0 = std::chrono::steady_clock::now();
    conn.reset();
    CHECK(std::chrono::steady_clock::now() - t0 < 500ms);
}

TEST_CASE("helper dtor never hangs on a SIGTERM-ignoring child",
          "[auth]") {
    // A helper that traps SIGTERM and never reads stdin ignores both
    // the closed-stdin EOF and the SIGTERM the dtor sends. Pre-fix
    // the dtor's blocking waitpid would hang shutdown forever; the
    // bounded grace + SIGKILL must reap it and return.
    // `exec sleep` replaces the shell so the child is a SINGLE
    // process (== child_pid_) with SIGTERM ignored (SIG_IGN survives
    // exec); no forked grandchild to orphan and leak past SIGKILL.
    std::string err;
    auto conn = spawn_helper(
        {"/bin/sh", "-c", "trap '' TERM; exec sleep 3600"}, &err);
    REQUIRE(conn);

    std::promise<void> done;
    auto fut = done.get_future();
    std::jthread t([conn = std::move(conn), &done]() mutable {
        conn.reset();  // runs ~HelperConnection: grace -> SIGKILL
        done.set_value();
    });

    // Grace is ~1s; allow a generous margin.
    const bool returned =
        fut.wait_for(5s) == std::future_status::ready;
    if (returned) {
        t.join();
    } else {
        t.detach();  // pre-fix: dtor hung; leak the worker
    }
    REQUIRE(returned);
}

TEST_CASE("SpawnedChild reaps the child and closes fds if not released",
          "[auth]") {
    // R4: models spawn_helper's error path -- a forked child + its
    // parent-side pipe fds owned by a SpawnedChild that is destroyed
    // WITHOUT release() (as when the HelperConnection ctor throws).
    // It must reap the child (no zombie) and close both fds (no leak).
    int in_pipe[2];
    int out_pipe[2];
    REQUIRE(::pipe(in_pipe) == 0);
    REQUIRE(::pipe(out_pipe) == 0);
    const pid_t pid = ::fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        // Child: wait quietly to be killed (no Catch calls here).
        ::close(in_pipe[1]);
        ::close(out_pipe[0]);
        ::pause();
        _exit(0);
    }
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);
    const int read_fd = out_pipe[0];
    const int write_fd = in_pipe[1];

    {
        SpawnedChild child(pid, read_fd, write_fd);
        // No release(): the destructor must clean everything up.
    }

    // Child reaped: a follow-up waitpid finds no such child.
    REQUIRE(::waitpid(pid, nullptr, WNOHANG) == -1);
    REQUIRE(errno == ECHILD);
    // Both fds closed: fcntl on them now fails with EBADF.
    REQUIRE(::fcntl(read_fd, F_GETFD) == -1);
    REQUIRE(::fcntl(write_fd, F_GETFD) == -1);
}
