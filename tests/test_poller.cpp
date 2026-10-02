#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/sys/poller.hpp"

using locus::sys::Poller;

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

}  // namespace

TEST_CASE("Poller reports a readable fd, not an idle one", "[poller]") {
    Pipe a;
    Pipe b;
    Poller p;
    p.add_read(a.fd[0]);
    p.add_read(b.fd[0]);

    std::vector<int> ready;
    // Nothing written yet: a short wait times out with no ready fds.
    REQUIRE(p.wait(ready, 20) == 0);
    REQUIRE(ready.empty());

    // Write to b only: b's read end becomes readable, a's does not.
    const char x = 'x';
    REQUIRE(::write(b.fd[1], &x, 1) == 1);
    REQUIRE(p.wait(ready, 1000) == 1);
    REQUIRE(ready.size() == 1);
    REQUIRE(ready[0] == b.fd[0]);
}

TEST_CASE("Poller is level-triggered: still ready until drained",
          "[poller]") {
    Pipe a;
    Poller p;
    p.add_read(a.fd[0]);
    const char two[2] = {'a', 'b'};
    REQUIRE(::write(a.fd[1], two, 2) == 2);

    std::vector<int> ready;
    REQUIRE(p.wait(ready, 1000) == 1);  // readable
    char one = 0;
    REQUIRE(::read(a.fd[0], &one, 1) == 1);  // drain only one byte
    REQUIRE(p.wait(ready, 1000) == 1);  // still readable (1 byte left)
    REQUIRE(::read(a.fd[0], &one, 1) == 1);  // drain the rest
    REQUIRE(p.wait(ready, 20) == 0);  // now idle
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

    std::vector<int> ready;
    REQUIRE(p.wait(ready, 1000) == 2);
    const bool has_a =
        std::find(ready.begin(), ready.end(), a.fd[0]) != ready.end();
    const bool has_c =
        std::find(ready.begin(), ready.end(), c.fd[0]) != ready.end();
    REQUIRE(has_a);
    REQUIRE(has_c);
}

TEST_CASE("Poller.remove stops reporting an fd", "[poller]") {
    Pipe a;
    Poller p;
    p.add_read(a.fd[0]);
    p.remove(a.fd[0]);
    const char x = 'x';
    REQUIRE(::write(a.fd[1], &x, 1) == 1);
    std::vector<int> ready;
    REQUIRE(p.wait(ready, 20) == 0);  // removed: not reported
}

TEST_CASE("Poller self-pipe wakes a blocked wait (the signal pattern)",
          "[poller]") {
    // Models how a SIGHUP handler wakes the event loop: a byte written
    // to a registered pipe makes wait() return promptly even though it
    // was asked to block for a long time.
    Pipe wake;
    Poller p;
    p.add_read(wake.fd[0]);

    std::vector<int> ready;
    const auto t0 = std::chrono::steady_clock::now();
    const char w = 1;
    REQUIRE(::write(wake.fd[1], &w, 1) == 1);  // "signal" before waiting
    const int n = p.wait(ready, 5000);  // would block 5s with no wake
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    REQUIRE(n == 1);
    REQUIRE(ready[0] == wake.fd[0]);
    REQUIRE(ms < 2000);  // returned well before the 5s timeout
}
