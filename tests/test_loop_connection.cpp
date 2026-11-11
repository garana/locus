#include <fcntl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/http/loop_connection.hpp"
#include "locus/pipeline/net.hpp"
#include "locus/sys/poller.hpp"

using locus::http::LoopConfig;
using locus::http::LoopConnection;
using locus::http::Request;
using locus::http::Response;
using Next = LoopConnection::Next;

namespace {

/** A connected pair of non-blocking stream sockets: [0] the server side
 *  LoopConnection drives, [1] the client side the test reads and
 *  writes. Closed on destruction (the test may close [1] early). */
struct Pair {
    int fd[2] = {-1, -1};
    Pair() {
        REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        for (int f : fd) {
            REQUIRE(::fcntl(f, F_SETFL, ::fcntl(f, F_GETFL) | O_NONBLOCK) ==
                    0);
        }
    }
    ~Pair() {
        for (int f : fd) {
            if (f >= 0) ::close(f);
        }
    }
    void send(const std::string& s) const {
        REQUIRE(::send(fd[1], s.data(), s.size(), 0) ==
                static_cast<ssize_t>(s.size()));
    }
    /** @returns Everything the client side can read right now. */
    std::string recv_all() const {
        std::string out;
        char buf[4096];
        for (;;) {
            const ssize_t n = ::recv(fd[1], buf, sizeof(buf), 0);
            if (n <= 0) break;
            out.append(buf, static_cast<std::size_t>(n));
        }
        return out;
    }
    void close_client() {
        ::close(fd[1]);
        fd[1] = -1;
    }
};

locus::sys::Poller::Event readable(int fd) {
    locus::sys::Poller::Event e;
    e.fd = fd;
    e.readable = true;
    return e;
}

locus::sys::Poller::Event writable(int fd) {
    locus::sys::Poller::Event e;
    e.fd = fd;
    e.writable = true;
    return e;
}

/** Answers every request with "path=<path>\n". */
void echo_path(const Request& req, Response& res) {
    res.set_content("path=" + req.path + "\n", "text/plain");
}

std::string get(const std::string& path, const std::string& extra = "") {
    return "GET " + path + " HTTP/1.1\r\nHost: x\r\n" + extra + "\r\n";
}

std::size_t count(const std::string& s, const std::string& what) {
    std::size_t n = 0;
    for (auto p = s.find(what); p != std::string::npos;
         p = s.find(what, p + what.size())) {
        ++n;
    }
    return n;
}

}  // namespace

TEST_CASE("loop connection answers a request fed one byte at a time",
          "[http][loop]") {
    Pair p;
    LoopConnection c(p.fd[0], {}, echo_path);
    const std::string req = get("/metrics");
    for (std::size_t i = 0; i + 1 < req.size(); ++i) {
        p.send(req.substr(i, 1));
        REQUIRE(c.on_event(readable(p.fd[0])) == Next::kRead);
        REQUIRE(p.recv_all().empty());  // nothing until the blank line
    }
    p.send(req.substr(req.size() - 1));
    REQUIRE(c.on_event(readable(p.fd[0])) == Next::kRead);  // keep-alive
    const std::string got = p.recv_all();
    CHECK(got.rfind("HTTP/1.1 200", 0) == 0);
    CHECK(got.find("path=/metrics\n") != std::string::npos);
}

TEST_CASE("loop connection answers two pipelined requests in order",
          "[http][loop]") {
    Pair p;
    LoopConnection c(p.fd[0], {}, echo_path);
    p.send(get("/a") + get("/b"));
    REQUIRE(c.on_event(readable(p.fd[0])) == Next::kRead);
    const std::string got = p.recv_all();
    REQUIRE(count(got, "HTTP/1.1 200") == 2);
    const auto a = got.find("path=/a\n"), b = got.find("path=/b\n");
    REQUIRE(a != std::string::npos);
    REQUIRE(b != std::string::npos);
    CHECK(a < b);
}

TEST_CASE("loop connection closes when the peer leaves mid-response",
          "[http][loop]") {
    // SIGPIPE at its default (kill), so this fails loudly if the write
    // to the gone peer is not sent with MSG_NOSIGNAL (SO_NOSIGPIPE).
    struct Default {
        void (*old)(int) = std::signal(SIGPIPE, SIG_DFL);
        ~Default() { std::signal(SIGPIPE, old); }
    } sigpipe_default;
    Pair p;
    LoopConnection c(p.fd[0], {}, [](const Request&, Response& res) {
        res.set_content(std::string(8 * 1024 * 1024, 'x'), "text/plain");
    });
    p.send(get("/big"));
    // 8 MiB does not fit the socket buffer: the write is left pending.
    REQUIRE(c.on_event(readable(p.fd[0])) == Next::kWrite);
    p.close_client();
    // The next write fails (EPIPE, no SIGPIPE) and the loop is told to
    // close; the connection never blocks on the gone peer.
    REQUIRE(c.on_event(writable(p.fd[0])) == Next::kClose);
}

TEST_CASE("loop connection gives a partial head a 408 at its deadline",
          "[http][loop]") {
    Pair p;
    LoopConfig cfg;
    cfg.request_deadline_ms = 1000;  // wide, so "not due" holds on a busy box
    LoopConnection c(p.fd[0], cfg, echo_path);
    p.send("GET /metrics HTTP/1.1\r\nHost:");
    REQUIRE(c.on_event(readable(p.fd[0])) == Next::kRead);
    // Not due yet: a timeout call changes nothing.
    REQUIRE(c.on_timeout() == Next::kRead);
    REQUIRE(p.recv_all().empty());
    std::this_thread::sleep_until(c.deadline() +
                                  std::chrono::milliseconds(5));
    REQUIRE(c.on_timeout() == Next::kClose);  // 408 written, then close
    const std::string got = p.recv_all();
    CHECK(got.rfind("HTTP/1.1 408", 0) == 0);
    // The API's JSON error envelope, as http::Server writes it.
    CHECK(got.find("Content-Type: application/json") != std::string::npos);
    CHECK(got.ends_with(locus::http::status_error_body(408)));
}

TEST_CASE("loop connection closes an idle keep-alive connection",
          "[http][loop]") {
    Pair p;
    LoopConfig cfg;
    cfg.idle_timeout_ms = 50;
    LoopConnection c(p.fd[0], cfg, echo_path);
    p.send(get("/a"));
    REQUIRE(c.on_event(readable(p.fd[0])) == Next::kRead);
    CHECK(p.recv_all().rfind("HTTP/1.1 200", 0) == 0);
    std::this_thread::sleep_until(c.deadline() +
                                  std::chrono::milliseconds(5));
    REQUIRE(c.on_timeout() == Next::kClose);
    CHECK(p.recv_all().empty());  // idle: no response, just the close
}

TEST_CASE("loop connection answers a malformed request and closes",
          "[http][loop]") {
    Pair p;
    LoopConnection c(p.fd[0], {}, echo_path);
    p.send("GET /a HTTP/1.1\nHost: x\n\n");  // bare LF
    REQUIRE(c.on_event(readable(p.fd[0])) == Next::kClose);
    CHECK(p.recv_all().rfind("HTTP/1.1 400", 0) == 0);
}

TEST_CASE("loop connection turns a throwing handler into a 500",
          "[http][loop]") {
    Pair p;
    LoopConnection c(p.fd[0], {}, [](const Request&, Response&) {
        throw std::runtime_error("boom");
    });
    p.send(get("/a"));
    REQUIRE(c.on_event(readable(p.fd[0])) == Next::kClose);
    const std::string got = p.recv_all();
    CHECK(got.rfind("HTTP/1.1 500", 0) == 0);
    CHECK(got.find("boom") == std::string::npos);  // no internals leak
    CHECK(got.find("Content-Type: application/json") != std::string::npos);
    CHECK(got.ends_with(locus::http::status_error_body(500)));
}

TEST_CASE("loop connection honors Connection: close and the request cap",
          "[http][loop]") {
    SECTION("Connection: close") {
        Pair p;
        LoopConnection c(p.fd[0], {}, echo_path);
        p.send(get("/a", "Connection: close\r\n"));
        REQUIRE(c.on_event(readable(p.fd[0])) == Next::kClose);
        CHECK(p.recv_all().find("path=/a\n") != std::string::npos);
    }
    SECTION("max_requests_per_conn") {
        Pair p;
        LoopConfig cfg;
        cfg.max_requests_per_conn = 2;
        LoopConnection c(p.fd[0], cfg, echo_path);
        p.send(get("/a") + get("/b") + get("/c"));
        REQUIRE(c.on_event(readable(p.fd[0])) == Next::kClose);
        const std::string got = p.recv_all();
        CHECK(count(got, "HTTP/1.1 200") == 2);  // /c is never answered
        CHECK(got.find("path=/c") == std::string::npos);
    }
    SECTION("a half-close still gets both pipelined answers") {
        Pair p;
        LoopConnection c(p.fd[0], {}, echo_path);
        p.send(get("/a") + get("/b"));
        REQUIRE(::shutdown(p.fd[1], SHUT_WR) == 0);
        REQUIRE(c.on_event(readable(p.fd[0])) == Next::kClose);
        const std::string got = p.recv_all();
        CHECK(got.find("path=/a\n") != std::string::npos);
        CHECK(got.find("path=/b\n") != std::string::npos);
    }
}

TEST_CASE("loop connection answers a pipelined request after a long write",
          "[http][loop]") {
    Pair p;
    LoopConnection c(p.fd[0], {}, [](const Request& req, Response& res) {
        if (req.path == "/big") {
            res.set_content(std::string(2 * 1024 * 1024, 'x'), "text/plain");
        } else {
            echo_path(req, res);
        }
    });
    p.send(get("/big") + get("/b"));
    REQUIRE(c.on_event(readable(p.fd[0])) == Next::kWrite);
    // The client drains; each writable event writes on, and once /big is
    // out the buffered /b is answered without a new read event.
    std::string got;
    Next n = Next::kWrite;
    for (int i = 0; i < 100000 && n == Next::kWrite; ++i) {
        got += p.recv_all();
        n = c.on_event(writable(p.fd[0]));
    }
    got += p.recv_all();
    REQUIRE(n == Next::kRead);
    CHECK(count(got, "HTTP/1.1 200") == 2);
    CHECK(got.find("path=/b\n") != std::string::npos);
}

TEST_CASE("loop connection answers 413 when input fills the cap unfinished",
          "[http][loop]") {
    Pair p;
    LoopConfig cfg;
    cfg.limits.max_body_bytes = 64 * 1024;
    LoopConnection c(p.fd[0], cfg, echo_path);
    // One-byte chunks: 6 raw bytes per body byte, so the framing fills
    // the input cap long before the body reaches max_body_bytes.
    std::string body;
    for (int i = 0; i < 60000; ++i) body += "1\r\nX\r\n";
    const std::string req =
        "POST /x HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n" +
        body;
    std::size_t sent = 0;
    Next n = Next::kRead;
    for (int i = 0; i < 100000 && n == Next::kRead; ++i) {
        if (sent < req.size()) {
            const ssize_t k = ::send(p.fd[1], req.data() + sent,
                                     req.size() - sent, MSG_NOSIGNAL);
            if (k > 0) sent += static_cast<std::size_t>(k);
        }
        n = c.on_event(readable(p.fd[0]));
    }
    // Not a spin on a full buffer: a 413 and a close.
    REQUIRE(n == Next::kClose);
    CHECK(p.recv_all().rfind("HTTP/1.1 413", 0) == 0);
}

TEST_CASE("loop connection closes silently when no request arrives",
          "[http][loop]") {
    Pair p;
    LoopConfig cfg;
    cfg.idle_timeout_ms = 50;
    LoopConnection c(p.fd[0], cfg, echo_path);
    std::this_thread::sleep_until(c.deadline() +
                                  std::chrono::milliseconds(5));
    REQUIRE(c.on_timeout() == Next::kClose);
    CHECK(p.recv_all().empty());  // no 408 for a client that sent nothing
}

TEST_CASE("loop connection answers HEAD without a body", "[http][loop]") {
    Pair p;
    LoopConnection c(p.fd[0], {}, echo_path);
    p.send("HEAD /a HTTP/1.1\r\nHost: x\r\n\r\n");
    REQUIRE(c.on_event(readable(p.fd[0])) == Next::kRead);
    const std::string got = p.recv_all();
    CHECK(got.rfind("HTTP/1.1 200", 0) == 0);
    CHECK(got.find("Content-Length: 8\r\n") != std::string::npos);
    CHECK(got.find("path=") == std::string::npos);
    CHECK(got.size() >= 4);
    CHECK(got.substr(got.size() - 4) == "\r\n\r\n");
}

// The shape a stage's I/O thread uses: its own Poller, a listening
// socket it accepts on, and one LoopConnection per accepted fd, with
// the interest set from each Next.
TEST_CASE("loop connection serves a client from a Poller loop",
          "[http][loop]") {
    int port = 0;
    const int lfd = locus::pipeline::listen_on("127.0.0.1", 0, &port);
    REQUIRE(lfd >= 0);
    std::string got;
    // Joined on every way out, a failed REQUIRE included.
    struct Joiner {
        std::thread t;
        ~Joiner() {
            if (t.joinable()) t.join();
        }
    } client;
    client.t = std::thread([&] {
        const int fd = locus::pipeline::connect_to("127.0.0.1", port, 2000);
        if (fd < 0) return;
        // Bounded, so a failed server side cannot hang the join.
        const timeval tv{5, 0};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        const std::string req = get("/metrics", "Connection: close\r\n");
        if (::send(fd, req.data(), req.size(), MSG_NOSIGNAL) > 0) {
            char buf[1024];
            ssize_t n = 0;
            while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) {
                got.append(buf, static_cast<std::size_t>(n));
            }
        }
        ::close(fd);
    });
    locus::sys::Poller poller;
    poller.add(lfd);
    std::map<int, std::unique_ptr<LoopConnection>> conns;
    bool served = false;
    const auto until =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    std::vector<locus::sys::Poller::Event> events;
    while (!served && std::chrono::steady_clock::now() < until) {
        poller.wait(events, 50);
        for (const auto& ev : events) {
            if (ev.fd == lfd) {
                const int fd = locus::pipeline::accept_one(lfd);
                REQUIRE(fd >= 0);
                REQUIRE(::fcntl(fd, F_SETFL,
                                ::fcntl(fd, F_GETFL) | O_NONBLOCK) == 0);
                poller.add(fd);
                conns[fd] = std::make_unique<LoopConnection>(fd, LoopConfig{},
                                                             echo_path);
                continue;
            }
            auto it = conns.find(ev.fd);
            if (it == conns.end()) continue;
            switch (it->second->on_event(ev)) {
                case Next::kRead:
                    poller.modify(ev.fd, locus::sys::Poller::Dir::kRead);
                    break;
                case Next::kWrite:
                    poller.modify(ev.fd, locus::sys::Poller::Dir::kWrite);
                    break;
                case Next::kClose:
                    poller.remove(ev.fd);  // remove before close
                    ::close(ev.fd);
                    conns.erase(it);
                    served = true;
                    break;
            }
        }
    }
    client.t.join();
    poller.remove(lfd);
    ::close(lfd);
    REQUIRE(served);
    CHECK(got.rfind("HTTP/1.1 200", 0) == 0);
    CHECK(got.find("path=/metrics\n") != std::string::npos);
}
