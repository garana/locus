#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
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

// A peer closing its end must surface as hangup (and readable, since a
// read now returns 0 at EOF without blocking). This is the signal the
// multiplexed loop uses to drop a gone connection; it was folded into
// "readable" before the Event reshape, with no way to tell it apart.
TEST_CASE("Poller reports hangup when the peer closes", "[poller]") {
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
