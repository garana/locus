#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/http/loop_server.hpp"
#include "locus/pipeline/net.hpp"

using locus::http::LoopServer;
using locus::http::Request;
using locus::http::RequestContext;
using locus::http::ServerConfig;
using locus::http::ServerResponse;
using locus::http::Sink;
using Clock = std::chrono::steady_clock;

namespace {

/** A connected client socket with a receive timeout, so a test fails
 *  instead of hanging. */
int dial(int port) {
    const int fd = locus::pipeline::connect_to("127.0.0.1", port, 2000);
    REQUIRE(fd >= 0);
    timeval tv{5, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    return fd;
}

void send_all(int fd, const std::string& s) {
    REQUIRE(::send(fd, s.data(), s.size(), MSG_NOSIGNAL) ==
            static_cast<ssize_t>(s.size()));
}

/** @returns What the server sends until it closes (or the timeout). */
std::string read_to_close(int fd) {
    std::string out;
    char buf[4096];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        out.append(buf, static_cast<std::size_t>(n));
    }
    return out;
}

std::string get(const std::string& path, bool close = true) {
    return "GET " + path + " HTTP/1.1\r\nHost: x\r\n" +
           (close ? "Connection: close\r\n" : "") + "\r\n";
}

/** A LoopServer on an ephemeral port, stopped on scope exit. */
struct Running {
    LoopServer server;
    int port;
    Running(ServerConfig cfg, locus::http::Handler h,
            locus::http::HeadHandler head = {})
        : server((cfg.host = "127.0.0.1", cfg.port = 0, std::move(cfg)),
                 std::move(h), std::move(head)),
          port(server.start()) {
        REQUIRE(port > 0);
    }
};

/** The routes the tests use. */
void routes(const Request& req, ServerResponse& res,
            const RequestContext& ctx) {
    if (req.path == "/hello") {
        res.set_content("hello\n", "text/plain");
    } else if (req.path == "/stream") {
        res.set_chunked_content_provider("text/plain", [](Sink& sink) {
            sink.write("a");
            sink.write("bb");
        });
    } else if (req.path == "/throw") {
        throw std::runtime_error("boom");
    } else if (req.path == "/stream-throw") {
        res.set_chunked_content_provider("text/plain", [](Sink& sink) {
            sink.write("part");
            throw std::runtime_error("boom");
        });
    } else if (req.path == "/slow") {
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        res.set_content("slow\n", "text/plain");
    } else if (req.path == "/until-gone") {
        // A long handler that stops when its client leaves.
        const auto until = Clock::now() + std::chrono::seconds(5);
        while (ctx.client_alive() && Clock::now() < until) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        res.status = 499;
        res.set_content(ctx.client_alive() ? "timeout\n" : "gone\n",
                        "text/plain");
    } else {
        res.status = 404;
        res.set_content("no\n", "text/plain");
    }
}

}  // namespace

TEST_CASE("loop server answers with keep-alive and pipelining",
          "[http][loop_server]") {
    Running r({}, routes);
    const int fd = dial(r.port);
    send_all(fd, get("/hello", false) + get("/nope"));
    const std::string got = read_to_close(fd);
    ::close(fd);
    CHECK(got.rfind("HTTP/1.1 200", 0) == 0);
    CHECK(got.find("hello\n") != std::string::npos);
    CHECK(got.find("HTTP/1.1 404") != std::string::npos);
    CHECK(got.find("hello\n") < got.find("HTTP/1.1 404"));
}

TEST_CASE("loop server streams a chunked response", "[http][loop_server]") {
    Running r({}, routes);
    const int fd = dial(r.port);
    send_all(fd, get("/stream"));
    const std::string got = read_to_close(fd);
    ::close(fd);
    CHECK(got.find("Transfer-Encoding: chunked") != std::string::npos);
    CHECK(got.find("\r\n\r\n1\r\na\r\n2\r\nbb\r\n0\r\n\r\n") !=
          std::string::npos);
}

TEST_CASE("loop server turns a throw into a 500 or a truncated stream",
          "[http][loop_server]") {
    ServerConfig cfg;
    cfg.metrics = std::make_shared<locus::http::HttpMetrics>(
        std::vector<std::string>{"/stream-throw"});
    const auto m = cfg.metrics;
    Running r(cfg, routes);
    int fd = dial(r.port);
    send_all(fd, get("/throw"));
    std::string got = read_to_close(fd);
    ::close(fd);
    CHECK(got.rfind("HTTP/1.1 500", 0) == 0);
    // The head was out: the part, then a close with no terminating chunk.
    fd = dial(r.port);
    send_all(fd, get("/stream-throw"));
    got = read_to_close(fd);
    ::close(fd);
    CHECK(got.rfind("HTTP/1.1 200", 0) == 0);
    CHECK(got.find("4\r\npart\r\n") != std::string::npos);
    CHECK(got.find("0\r\n\r\n") == std::string::npos);
    // i#980: counted (and logged) on its route, as on Server; the 500
    // before any head is not a stream failure.
    CHECK(m->at(m->route("/stream-throw")).stream_failures.load() == 1);
    CHECK(m->at(m->other()).stream_failures.load() == 0);
    // And the server lives.
    fd = dial(r.port);
    send_all(fd, get("/hello"));
    CHECK(read_to_close(fd).rfind("HTTP/1.1 200", 0) == 0);
    ::close(fd);
}

TEST_CASE("loop server runs the head stage before the handler",
          "[http][loop_server]") {
    std::atomic<int> handled{0};
    Running r(
        {},
        [&](const Request& req, ServerResponse& res,
            const RequestContext& ctx) {
            ++handled;
            res.set_content("id=" + ctx.identity + "\n", "text/plain");
            (void)req;
        },
        [](const Request& req, ServerResponse& res, RequestContext& ctx) {
            if (req.get_header_value("Authorization") != "Bearer k") {
                res.status = 401;
                res.set_content("denied\n", "text/plain");
                return true;
            }
            ctx.identity = "k";
            return false;
        });
    int fd = dial(r.port);
    send_all(fd, get("/x"));
    std::string got = read_to_close(fd);
    ::close(fd);
    CHECK(got.rfind("HTTP/1.1 401", 0) == 0);
    CHECK(handled.load() == 0);
    fd = dial(r.port);
    send_all(fd, "GET /x HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k\r\n"
                 "Connection: close\r\n\r\n");
    got = read_to_close(fd);
    ::close(fd);
    CHECK(got.find("id=k\n") != std::string::npos);
    CHECK(handled.load() == 1);
}

TEST_CASE("loop server: idle clients and a busy pool block no fast path",
          "[http][loop_server]") {
    ServerConfig cfg;
    cfg.worker_threads = 1;
    cfg.fast_path = [](std::string_view m, std::string_view t) {
        return m == "GET" && t == "/hello";
    };
    Running r(cfg, routes);
    // Twenty connections that send nothing hold no thread.
    std::vector<int> idle;
    for (int i = 0; i < 20; ++i) idle.push_back(dial(r.port));
    // The one job thread is busy for 1.5 s.
    std::thread slow([&] {
        const int fd = dial(r.port);
        send_all(fd, get("/slow"));
        CHECK(read_to_close(fd).find("slow\n") != std::string::npos);
        ::close(fd);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto t0 = Clock::now();
    const int fd = dial(r.port);
    send_all(fd, get("/hello"));
    CHECK(read_to_close(fd).find("hello\n") != std::string::npos);
    ::close(fd);
    CHECK(Clock::now() - t0 < std::chrono::milliseconds(1000));
    slow.join();
    for (const int f : idle) ::close(f);
}

TEST_CASE("loop server answers a client that half-closed while it waits",
          "[http][loop_server]") {
    Running r({}, routes);
    const int fd = dial(r.port);
    send_all(fd, get("/until-gone"));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ::shutdown(fd, SHUT_WR);  // as a client that gives up its request
    const std::string got = read_to_close(fd);
    ::close(fd);
    // client_alive turned false, and the handler's answer still came.
    CHECK(got.rfind("HTTP/1.1 499", 0) == 0);
    CHECK(got.find("gone\n") != std::string::npos);
}

TEST_CASE("loop server caps the connections of one address",
          "[http][loop_server]") {
    ServerConfig cfg;
    cfg.max_conns_per_ip = 2;
    Running r(cfg, routes);
    const int a = dial(r.port), b = dial(r.port);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const int c = dial(r.port);
    // The third is closed at once: EOF, no response (nothing sent, as a
    // send could fail on the closed socket).
    CHECK(read_to_close(c).empty());
    ::close(c);
    send_all(a, get("/hello"));
    CHECK(read_to_close(a).find("hello\n") != std::string::npos);
    for (const int f : {a, b}) ::close(f);
}

TEST_CASE("loop server stops with a stream producer blocked on its client",
          "[http][loop_server]") {
    std::atomic<bool> returned{false};
    ServerConfig cfg;
    Running r(cfg, [&](const Request&, ServerResponse& res,
                       const RequestContext&) {
        res.set_chunked_content_provider("text/plain", [&](Sink& sink) {
            const std::string big(64 * 1024, 'x');
            while (sink.write(big)) {
            }
            returned = true;
        });
    });
    const int fd = dial(r.port);  // never reads: the queue fills
    send_all(fd, get("/flood"));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const auto t0 = Clock::now();
    r.server.stop();
    CHECK(Clock::now() - t0 < std::chrono::seconds(3));
    CHECK(returned.load());
    ::close(fd);
}
