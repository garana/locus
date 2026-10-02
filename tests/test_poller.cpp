#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/sys/poller.hpp"

using locus::sys::Poller;

namespace {
// Counts SIGALRM deliveries so the test knows how many of the burst
// actually landed. `g = g + 1` (a plain load+store, serialized within
// one thread's handler) rather than `++`, which on a volatile is
// deprecated in C++20 (-Wvolatile) and would break the warning-clean
// build.
volatile std::sig_atomic_t g_alarms = 0;
void on_alarm(int /*sig*/) { g_alarms = g_alarms + 1; }
}  // namespace

namespace {

// A pipe whose ends close on destruction; [0] read, [1] write.
struct Pipe {
    int fd[2] = {-1, -1};
    Pipe() { REQUIRE(::pipe(fd) == 0); }
    ~Pipe() {
        if (fd[0] >= 0) ::close(fd[0]);
        if (fd[1] >= 0) ::close(fd[1]);
    }
};

bool has_fd(const std::vector<Poller::Event>& evs, int fd) {
    return std::any_of(
        evs.begin(), evs.end(),
        [fd](const Poller::Event& e) { return e.fd == fd; });
}

// A connected TCP socket pair over loopback; {client, server} fds.
// Sockets (not pipes) are what a stage loop actually watches, and they
// diverge from pipes on close: see the FIN/RST cases below. @returns
// false if any step fails (the caller skips the case).
struct TcpPair {
    int client = -1;
    int server = -1;
    ~TcpPair() {
        if (client >= 0) ::close(client);
        if (server >= 0) ::close(server);
    }
    bool open() {
        const int ln = ::socket(AF_INET, SOCK_STREAM, 0);
        if (ln < 0) return false;
        int one = 1;
        ::setsockopt(ln, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = 0;  // kernel picks a free port
        if (::bind(ln, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0 ||
            ::listen(ln, 1) < 0) {
            ::close(ln);
            return false;
        }
        socklen_t al = sizeof(a);
        ::getsockname(ln, reinterpret_cast<sockaddr*>(&a), &al);
        client = ::socket(AF_INET, SOCK_STREAM, 0);
        if (client < 0 ||
            ::connect(client, reinterpret_cast<sockaddr*>(&a),
                      sizeof(a)) < 0) {
            ::close(ln);
            return false;
        }
        server = ::accept(ln, nullptr, nullptr);
        ::close(ln);
        return server >= 0;
    }
};

}  // namespace

TEST_CASE("Poller reports a readable fd, not an idle one", "[poller]") {
    Pipe a;
    Pipe b;
    Poller p;
    p.add_read(a.fd[0]);
    p.add_read(b.fd[0]);

    std::vector<Poller::Event> events;
    // Nothing written yet: a short wait times out with no ready fds.
    REQUIRE(p.wait(events, 20) == 0);
    REQUIRE(events.empty());

    // Write to b only: b's read end becomes readable, a's does not.
    const char x = 'x';
    REQUIRE(::write(b.fd[1], &x, 1) == 1);
    REQUIRE(p.wait(events, 1000) == 1);
    REQUIRE(events.size() == 1);
    REQUIRE(events[0].fd == b.fd[0]);
    REQUIRE(events[0].readable);
    REQUIRE_FALSE(events[0].hangup);
}

TEST_CASE("Poller is level-triggered: still ready until drained",
          "[poller]") {
    Pipe a;
    Poller p;
    p.add_read(a.fd[0]);
    const char two[2] = {'a', 'b'};
    REQUIRE(::write(a.fd[1], two, 2) == 2);

    std::vector<Poller::Event> events;
    REQUIRE(p.wait(events, 1000) == 1);  // readable
    char one = 0;
    REQUIRE(::read(a.fd[0], &one, 1) == 1);  // drain only one byte
    REQUIRE(p.wait(events, 1000) == 1);  // still readable (1 byte left)
    REQUIRE(::read(a.fd[0], &one, 1) == 1);  // drain the rest
    REQUIRE(p.wait(events, 20) == 0);  // now idle
}

TEST_CASE("Poller reports several ready fds at once", "[poller]") {
    Pipe a;
    Pipe b;
    Pipe c;
    Poller p;
    p.add_read(a.fd[0]);
    p.add_read(b.fd[0]);
    p.add_read(c.fd[0]);
    const char x = 'x';
    REQUIRE(::write(a.fd[1], &x, 1) == 1);
    REQUIRE(::write(c.fd[1], &x, 1) == 1);

    std::vector<Poller::Event> events;
    REQUIRE(p.wait(events, 1000) == 2);
    REQUIRE(has_fd(events, a.fd[0]));
    REQUIRE(has_fd(events, c.fd[0]));
}

TEST_CASE("Poller.remove stops reporting an fd", "[poller]") {
    Pipe a;
    Poller p;
    p.add_read(a.fd[0]);
    p.remove(a.fd[0]);
    const char x = 'x';
    REQUIRE(::write(a.fd[1], &x, 1) == 1);
    std::vector<Poller::Event> events;
    REQUIRE(p.wait(events, 20) == 0);  // removed: not reported
}

TEST_CASE("Poller self-pipe wakes a blocked wait (the signal pattern)",
          "[poller]") {
    // Models how a SIGHUP handler wakes the event loop: a byte written
    // to a registered pipe makes wait() return promptly even though it
    // was asked to block for a long time.
    Pipe wake;
    Poller p;
    p.add_read(wake.fd[0]);

    std::vector<Poller::Event> events;
    const auto t0 = std::chrono::steady_clock::now();
    const char w = 1;
    REQUIRE(::write(wake.fd[1], &w, 1) == 1);  // "signal" before waiting
    const int n = p.wait(events, 5000);  // would block 5s with no wake
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    REQUIRE(n == 1);
    REQUIRE(events[0].fd == wake.fd[0]);
    REQUIRE(ms < 2000);  // returned well before the 5s timeout
}

// The EINTR branch of wait() is the one part no other case reaches
// (they never deliver a signal), and it carries the deadline fix: a
// retry must resume with the remaining budget, not restart timeout_ms.
// Deliver a BOUNDED burst of SIGALRM to the waiting thread (handler
// without SA_RESTART, so each interrupts epoll_wait/kevent with EINTR)
// and check wait() still returns near the requested timeout. The burst
// stops, so a regression returns late rather than hanging: with the fix
// ~400 ms, without it the last signal (~250 ms) restarts a full 400 ms
// -> ~650 ms, and the < 550 ms bound separates them with ~100 ms margin
// either side.
TEST_CASE("Poller.wait honors the deadline across EINTR retries",
          "[poller]") {
    struct sigaction sa;
    sa.sa_handler = on_alarm;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // no SA_RESTART: SIGALRM interrupts the wait
    struct sigaction old_sa;
    REQUIRE(::sigaction(SIGALRM, &sa, &old_sa) == 0);
    g_alarms = 0;

    int fds[2];
    REQUIRE(::pipe(fds) == 0);  // read end never written -> never ready

    Poller p;
    p.add_read(fds[0]);

    const pthread_t waiter = ::pthread_self();
    std::thread burst([waiter] {
        for (int i = 0; i < 5; ++i) {  // fire at ~50..250 ms, then stop
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            ::pthread_kill(waiter, SIGALRM);
        }
    });

    std::vector<Poller::Event> events;
    const auto t0 = std::chrono::steady_clock::now();
    const int n = p.wait(events, 400);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    burst.join();
    ::sigaction(SIGALRM, &old_sa, nullptr);  // restore
    ::close(fds[0]);
    ::close(fds[1]);

    // Need most of the burst to have landed: with fewer signals a
    // regression's return time (last-signal + 400 ms) can itself drop
    // under 550 ms and pass vacuously. Require >= 4 of 5 so the < 550
    // bound still discriminates.
    if (g_alarms < 4) {
        SKIP("only " << static_cast<int>(g_alarms)
                     << " of 5 signals landed; inconclusive");
    }
    REQUIRE(n == 0);     // timed out: the pipe never became readable
    REQUIRE(ms >= 350);  // did not return early on the first EINTR
    REQUIRE(ms < 550);   // did not restart the 400 ms per signal
}

// A closed pipe writer is the one shape both backends agree on: read
// end reports readable+hangup, and the read returns 0 (EOF). The three
// cases after this one are the shapes where the backends DIVERGE, which
// is why the contract tells callers to trust read()==0, not the flags.
TEST_CASE("Poller: closed pipe writer -> readable+hangup, read()==0",
          "[poller]") {
    int fds[2];
    REQUIRE(::pipe(fds) == 0);
    Poller p;
    p.add_read(fds[0]);
    ::close(fds[1]);  // peer closed its write end

    std::vector<Poller::Event> events;
    REQUIRE(p.wait(events, 1000) == 1);
    REQUIRE(events[0].fd == fds[0]);
    REQUIRE(events[0].hangup);
    REQUIRE(events[0].readable);   // a read won't block; returns 0 (EOF)
    REQUIRE_FALSE(events[0].error);
    char c = 0;
    REQUIRE(::read(fds[0], &c, 1) == 0);  // EOF is the authoritative cue
    ::close(fds[0]);
}

// hangup coexists with buffered data on BOTH backends: a writer that
// wrote then closed yields hangup=1 AND a read that returns the bytes,
// not 0. So "hangup -> nothing left, drop the peer" would silently
// discard a buffered frame. This pins the drain-first rule.
TEST_CASE("Poller: hangup with bytes still buffered still reads them",
          "[poller]") {
    int fds[2];
    REQUIRE(::pipe(fds) == 0);
    const char eight[8] = "ABCDEFG";  // 7 chars + NUL = 8 bytes
    REQUIRE(::write(fds[1], eight, 8) == 8);
    ::close(fds[1]);  // close AFTER writing: data + EOF both pending
    Poller p;
    p.add_read(fds[0]);

    std::vector<Poller::Event> events;
    REQUIRE(p.wait(events, 1000) == 1);
    REQUIRE(events[0].readable);
    REQUIRE(events[0].hangup);  // hung up...
    char buf[16] = {0};
    REQUIRE(::read(fds[0], buf, sizeof(buf)) == 8);  // ...yet 8 bytes
    REQUIRE(::read(fds[0], buf, sizeof(buf)) == 0);  // THEN the EOF
    ::close(fds[0]);
}

// A clean socket FIN (peer close) is the most common real event and the
// one where the flags diverge: kqueue sets hangup (EVFILT_READ reports
// EV_EOF), epoll does not (a half-close raises only EPOLLIN). What is
// portable, and all a caller may rely on, is readable=1 with read()==0.
TEST_CASE("Poller: clean socket FIN -> readable, read()==0 (portable)",
          "[poller]") {
    TcpPair s;
    if (!s.open()) {
        SKIP("could not set up a loopback TCP pair");
    }
    Poller p;
    p.add_read(s.server);
    ::close(s.client);  // client sends FIN
    s.client = -1;

    std::vector<Poller::Event> events;
    REQUIRE(p.wait(events, 1000) == 1);
    REQUIRE(events[0].fd == s.server);
    REQUIRE(events[0].readable);    // portable: a read won't block...
    REQUIRE_FALSE(events[0].error);
    char c = 0;
    REQUIRE(::read(s.server, &c, 1) == 0);  // ...and returns 0 (gone)
#if defined(__linux__)
    REQUIRE_FALSE(events[0].hangup);  // epoll: FIN raises no EPOLLHUP
#else
    REQUIRE(events[0].hangup);        // kqueue: EV_EOF set on FIN
#endif
}

// A socket RST (abortive close) diverges the other way: epoll reports
// error=1 with readable=0, while kqueue reports readable=1 and the read
// itself fails with ECONNRESET (no error flag). Either way the peer is
// gone; the portable detection is "error flag, OR a read that returns
// <= 0 with a non-retriable errno" -- never the hangup flag alone.
TEST_CASE("Poller: socket RST -> peer gone via error flag or read()<0",
          "[poller]") {
    TcpPair s;
    if (!s.open()) {
        SKIP("could not set up a loopback TCP pair");
    }
    struct linger lg;
    lg.l_onoff = 1;
    lg.l_linger = 0;  // close() now sends RST instead of FIN
    ::setsockopt(s.client, SOL_SOCKET, SO_LINGER, &lg, sizeof(lg));
    Poller p;
    p.add_read(s.server);
    ::close(s.client);
    s.client = -1;

    std::vector<Poller::Event> events;
    REQUIRE(p.wait(events, 1000) == 1);
    REQUIRE(events[0].fd == s.server);
#if defined(__linux__)
    REQUIRE(events[0].error);         // epoll: EPOLLERR...
    REQUIRE_FALSE(events[0].readable);  // ...and not readable
#else
    REQUIRE(events[0].readable);      // kqueue: readable, no error flag
    REQUIRE_FALSE(events[0].error);
    char c = 0;
    errno = 0;
    REQUIRE(::read(s.server, &c, 1) < 0);  // the read carries the error
    REQUIRE(errno == ECONNRESET);
#endif
}
