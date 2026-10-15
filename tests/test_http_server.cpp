#include <csignal>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <thread>

#include "catch_amalgamated.hpp"
#include "locus/http/server.hpp"
#include "locus/pipeline/net.hpp"

using locus::http::Request;
using locus::http::RequestContext;
using locus::http::Server;
using locus::http::ServerConfig;
using locus::http::ServerResponse;

namespace {

// Writing to a socket the server has closed raises SIGPIPE on Linux,
// which would kill the test runner; ignore it process-wide (once).
const int kIgnoreSigpipe = [] {
    std::signal(SIGPIPE, SIG_IGN);
    return 0;
}();

// Connects a blocking client to 127.0.0.1:port. Fails the test on error.
int client_connect(int port) {
    const int fd = locus::pipeline::connect_to("127.0.0.1", port, 2000);
    REQUIRE(fd >= 0);
    return fd;
}

void send_all(int fd, std::string_view data) {
    std::size_t off = 0;
    while (off < data.size()) {
        const ssize_t n = ::write(fd, data.data() + off, data.size() - off);
        if (n <= 0) {
            break;
        }
        off += static_cast<std::size_t>(n);
    }
}

// Reads from `fd` until the peer is quiet for `idle_ms` or closes.
// Loopback responses arrive in a burst, so a short idle gap reliably
// marks the end of a complete response.
std::string drain(int fd, int idle_ms = 300) {
    locus::pipeline::set_recv_timeout(fd, idle_ms);
    std::string out;
    char buf[8192];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n > 0) {
            out.append(buf, static_cast<std::size_t>(n));
            continue;
        }
        break;  // 0 == closed, <0 == recv timeout (quiet)
    }
    return out;
}

// Most tests do not care about the head stage, so they pass a 2-arg
// handler; the fixture wraps it into the real 3-arg Handler.
using SimpleHandler = std::function<void(const Request&, ServerResponse&)>;

// A server fixture: owns the Server and the port it bound.
struct Fixture {
    Server server;
    int port;
    explicit Fixture(SimpleHandler h, ServerConfig cfg = {})
        : server(with_defaults(cfg), wrap(std::move(h))), port(server.start()) {
        REQUIRE(port > 0);
    }
    // Full form for head-stage tests: a 3-arg Handler plus a HeadHandler.
    Fixture(locus::http::Handler h, locus::http::HeadHandler hh,
            ServerConfig cfg = {})
        : server(with_defaults(cfg), std::move(h), std::move(hh)),
          port(server.start()) {
        REQUIRE(port > 0);
    }
    static locus::http::Handler wrap(SimpleHandler h) {
        return [h = std::move(h)](const Request& req, ServerResponse& res,
                                  const locus::http::RequestContext&) {
            h(req, res);
        };
    }
    static ServerConfig with_defaults(ServerConfig cfg) {
        cfg.host = "127.0.0.1";
        cfg.port = 0;  // ephemeral
        return cfg;
    }
};

// An echo handler: 200, body "<METHOD> <path>".
void echo_handler(const Request& req, ServerResponse& res) {
    res.status = 200;
    res.set_content(req.method + " " + req.path, "text/plain");
}

}  // namespace

TEST_CASE("http server: a GET gets a 200 with the handler's body",
          "[http_server]") {
    Fixture f(echo_handler);
    const int c = client_connect(f.port);
    send_all(c, "GET /hello HTTP/1.1\r\nHost: h\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("HTTP/1.1 200 OK") != std::string::npos);
    REQUIRE(r.find("GET /hello") != std::string::npos);
    REQUIRE(r.find("Connection: keep-alive") != std::string::npos);
    ::close(c);
}

TEST_CASE("http server: two pipelined requests in one write are both "
          "answered without a second read (#11 persistent buffer)",
          "[http_server]") {
    Fixture f(echo_handler);
    const int c = client_connect(f.port);
    // Both requests in a SINGLE write; the client never writes again.
    send_all(c,
             "GET /a HTTP/1.1\r\nHost: h\r\n\r\n"
             "GET /b HTTP/1.1\r\nHost: h\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("GET /a") != std::string::npos);
    REQUIRE(r.find("GET /b") != std::string::npos);
    ::close(c);
}

TEST_CASE("http server: keep-alive serves sequential requests on one "
          "connection", "[http_server]") {
    Fixture f(echo_handler);
    const int c = client_connect(f.port);
    send_all(c, "GET /one HTTP/1.1\r\nHost: h\r\n\r\n");
    REQUIRE(drain(c).find("GET /one") != std::string::npos);
    send_all(c, "GET /two HTTP/1.1\r\nHost: h\r\n\r\n");
    REQUIRE(drain(c).find("GET /two") != std::string::npos);
    ::close(c);
}

TEST_CASE("http server: Connection: close ends the connection after one",
          "[http_server]") {
    Fixture f(echo_handler);
    const int c = client_connect(f.port);
    send_all(c, "GET /x HTTP/1.1\r\nConnection: close\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("GET /x") != std::string::npos);
    REQUIRE(r.find("Connection: close") != std::string::npos);
    // A 2nd request must get nothing: the server closed.
    send_all(c, "GET /y HTTP/1.1\r\n\r\n");
    REQUIRE(drain(c).empty());
    ::close(c);
}

TEST_CASE("http server: max_requests_per_conn closes after the cap",
          "[http_server]") {
    ServerConfig cfg;
    cfg.max_requests_per_conn = 1;
    Fixture f(echo_handler, cfg);
    const int c = client_connect(f.port);
    send_all(c, "GET /only HTTP/1.1\r\nHost: h\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("GET /only") != std::string::npos);
    REQUIRE(r.find("Connection: close") != std::string::npos);  // last one
    ::close(c);
}

TEST_CASE("http server: a parse error answers the suggested status and "
          "closes", "[http_server]") {
    Fixture f(echo_handler);
    const int c = client_connect(f.port);
    send_all(c, "GET / HTTP/2.0\r\nHost: h\r\n\r\n");  // bad version -> 505
    const std::string r = drain(c);
    REQUIRE(r.find("HTTP/1.1 505") != std::string::npos);
    REQUIRE(r.find("Connection: close") != std::string::npos);
    ::close(c);
}

TEST_CASE("http server: an oversized Content-Length is 413'd on the head, "
          "before the body is read", "[http_server]") {
    ServerConfig cfg;
    cfg.limits.max_body_bytes = 8;
    Fixture f(echo_handler, cfg);
    const int c = client_connect(f.port);
    // Head only, promising 100 bytes we never send. A 413 must come back
    // anyway -- the cap fires on the head, so the server does not wait.
    send_all(c, "POST / HTTP/1.1\r\nContent-Length: 100\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("HTTP/1.1 413") != std::string::npos);
    REQUIRE(r.find("Connection: close") != std::string::npos);
    ::close(c);
}

TEST_CASE("http server: a HEAD omits the body but keeps the headers",
          "[http_server]") {
    Fixture f(echo_handler);
    const int c = client_connect(f.port);
    send_all(c, "HEAD /h HTTP/1.1\r\nHost: h\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("HTTP/1.1 200") != std::string::npos);
    REQUIRE(r.find("Content-Length: 7") != std::string::npos);  // "HEAD /h"
    REQUIRE(r.find("HEAD /h") == std::string::npos);  // body omitted
    ::close(c);
}

TEST_CASE("http server: a chunked stream decodes end to end",
          "[http_server]") {
    Fixture f([](const Request&, ServerResponse& res) {
        res.set_chunked_content_provider(
            "text/event-stream", [](locus::http::Sink& sink) {
                REQUIRE(sink.write("a"));
                REQUIRE(sink.write("bc"));
                sink.done();
            });
    });
    const int c = client_connect(f.port);
    send_all(c, "GET /s HTTP/1.1\r\nHost: h\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("Transfer-Encoding: chunked") != std::string::npos);
    REQUIRE(r.find("1\r\na\r\n") != std::string::npos);
    REQUIRE(r.find("2\r\nbc\r\n") != std::string::npos);
    REQUIRE(r.find("0\r\n\r\n") != std::string::npos);  // terminator
    ::close(c);
}

TEST_CASE("http server: a HEAD to a streaming handler sends headers but "
          "no chunks", "[http_server]") {
    std::atomic<bool> provider_ran{false};
    Fixture f([&](const Request&, ServerResponse& res) {
        res.set_chunked_content_provider(
            "text/event-stream", [&](locus::http::Sink& sink) {
                provider_ran.store(true);
                sink.write("should-not-be-sent");
            });
    });
    const int c = client_connect(f.port);
    send_all(c, "HEAD /s HTTP/1.1\r\nHost: h\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("Transfer-Encoding: chunked") != std::string::npos);
    REQUIRE(r.find("should-not-be-sent") == std::string::npos);  // no body
    REQUIRE_FALSE(provider_ran.load());  // provider skipped entirely
    ::close(c);
}

TEST_CASE("http server: an empty chunk write is a no-op, not a premature "
          "end-of-stream", "[http_server]") {
    // encode_chunk("") is byte-identical to the terminator, so an empty
    // write must do nothing; the body after it must still arrive.
    Fixture f([](const Request&, ServerResponse& res) {
        res.set_chunked_content_provider(
            "text/plain", [](locus::http::Sink& sink) {
                REQUIRE(sink.write(""));   // no-op, must not terminate
                REQUIRE(sink.write("end"));
                sink.done();
            });
    });
    const int c = client_connect(f.port);
    send_all(c, "GET /s HTTP/1.1\r\nHost: h\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("3\r\nend\r\n") != std::string::npos);  // body survived
    // Exactly one terminator, and it is after the data, not before it.
    const std::size_t term = r.find("0\r\n\r\n");
    const std::size_t data = r.find("3\r\nend\r\n");
    REQUIRE(term != std::string::npos);
    REQUIRE(data < term);
    ::close(c);
}

TEST_CASE("http server: Expect: 100-continue gets an interim 100 before "
          "the body", "[http_server]") {
    Fixture f([](const Request& req, ServerResponse& res) {
        res.status = 200;
        res.set_content(req.body, "text/plain");
    });
    const int c = client_connect(f.port);
    // Send the head only; the server must answer 100 before we send body.
    send_all(c,
             "POST /p HTTP/1.1\r\nHost: h\r\nContent-Length: 5\r\n"
             "Expect: 100-continue\r\n\r\n");
    const std::string interim = drain(c, 300);
    REQUIRE(interim.find("HTTP/1.1 100 Continue") != std::string::npos);
    // Now send the body; the final 200 echoes it.
    send_all(c, "hello");
    const std::string fin = drain(c);
    REQUIRE(fin.find("HTTP/1.1 200") != std::string::npos);
    REQUIRE(fin.find("hello") != std::string::npos);
    ::close(c);
}

TEST_CASE("http server: an idle keep-alive connection is closed on the "
          "idle timeout", "[http_server]") {
    ServerConfig cfg;
    cfg.idle_timeout_ms = 200;
    Fixture f(echo_handler, cfg);
    const int c = client_connect(f.port);
    // Send nothing. The server must drop the idle connection; drain
    // returns on the close (empty since the server sent nothing).
    const auto t0 = std::chrono::steady_clock::now();
    const std::string r = drain(c, 1500);
    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    REQUIRE(r.empty());
    REQUIRE(dt < 1200);  // closed near the 200 ms idle bound, not hung
    ::close(c);
}

TEST_CASE("http server: a partial head trips the head deadline with 408",
          "[http_server]") {
    ServerConfig cfg;
    cfg.head_deadline_ms = 200;
    Fixture f(echo_handler, cfg);
    const int c = client_connect(f.port);
    send_all(c, "GET / HTTP/1.1\r\n");  // head never completes
    const std::string r = drain(c, 1500);
    REQUIRE(r.find("HTTP/1.1 408") != std::string::npos);
    ::close(c);
}

TEST_CASE("http server: a slow-but-steady body upload is NOT cut (rate-"
          "based, not a flat floor)", "[http_server]") {
    // The upload spans longer than head_deadline_ms (the base grace) but
    // sustains well above the minimum rate, so the token bucket keeps the
    // deadline ahead and the request completes. A flat deadline would
    // 408 this -- the regression the rate-based bound fixes.
    ServerConfig cfg;
    cfg.head_deadline_ms = 200;            // base grace
    cfg.min_ingest_bytes_per_sec = 4096;   // 4 KB/s floor
    Fixture f(
        [](const Request& req, ServerResponse& res) {
            res.status = 200;
            res.set_content(std::to_string(req.body.size()), "text/plain");
        },
        cfg);
    const int c = client_connect(f.port);
    send_all(c,
             "POST /up HTTP/1.1\r\nHost: h\r\nContent-Length: 8192\r\n\r\n");
    // 8 KB in 1 KB steps, 80 ms apart: ~640 ms total (> the 200 ms grace)
    // at ~12.5 KB/s (3x the floor).
    const std::string kb(1024, 'y');
    for (int i = 0; i < 8; ++i) {
        send_all(c, kb);
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }
    const std::string r = drain(c);
    REQUIRE(r.find("HTTP/1.1 200") != std::string::npos);
    REQUIRE(r.find("8192") != std::string::npos);  // whole body received
    ::close(c);
}

TEST_CASE("http server: a stalled body upload trips the body deadline",
          "[http_server]") {
    ServerConfig cfg;
    cfg.head_deadline_ms = 200;  // body grace
    Fixture f(echo_handler, cfg);
    const int c = client_connect(f.port);
    // Head promises a body, then the client sends nothing: no rate credit
    // is earned, so the body deadline (grace) trips with a 408.
    send_all(c, "POST /up HTTP/1.1\r\nHost: h\r\nContent-Length: 10\r\n\r\n");
    const std::string r = drain(c, 1500);
    REQUIRE(r.find("HTTP/1.1 408") != std::string::npos);
    ::close(c);
}

TEST_CASE("http server: a write blocked past the deadline drops the "
          "stream (#5)", "[http_server]") {
    std::atomic<bool> write_failed{false};
    std::atomic<int> written{0};
    ServerConfig cfg;
    cfg.write_deadline_ms = 200;
    Fixture f(
        [&](const Request&, ServerResponse& res) {
            res.set_chunked_content_provider(
                "application/octet-stream", [&](locus::http::Sink& sink) {
                    const std::string chunk(4096, 'x');
                    for (int i = 0; i < 4096; ++i) {
                        if (!sink.write(chunk)) {
                            write_failed.store(true);
                            return;  // deadline tripped: stop producing
                        }
                        written.fetch_add(1);
                    }
                });
        },
        cfg);
    const int c = client_connect(f.port);
    send_all(c, "GET /big HTTP/1.1\r\nHost: h\r\n\r\n");
    // The client never reads. The socket buffer fills, send() blocks, and
    // after the write deadline the stream is dropped.
    std::this_thread::sleep_for(std::chrono::milliseconds(900));
    REQUIRE(write_failed.load());
    REQUIRE(written.load() < 4096);  // stopped early, did not send it all
    ::close(c);
}

TEST_CASE("http server: stop() returns promptly and closes live "
          "connections", "[http_server]") {
    Fixture f(echo_handler);
    const int c = client_connect(f.port);
    send_all(c, "GET /one HTTP/1.1\r\nHost: h\r\n\r\n");
    REQUIRE(drain(c).find("GET /one") != std::string::npos);
    const auto t0 = std::chrono::steady_clock::now();
    f.server.stop();  // idle keep-alive connection must unblock promptly
    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    REQUIRE(dt < 1000);  // not waiting out the 5 s idle timeout
    REQUIRE(drain(c).empty());  // connection closed by shutdown
    ::close(c);
}

// ---- head-stage hook (inc 3b) ----

TEST_CASE("http server: a head handler reject sends its full response and "
          "closes", "[http_server]") {
    Fixture f(
        [](const Request&, ServerResponse& res, const RequestContext&) {
            res.set_content("HANDLER-RAN", "text/plain");  // must NOT run
        },
        [](const Request&, ServerResponse& res, RequestContext&) {
            res.status = 401;
            res.set_content(
                "{\"error\":{\"type\":\"authentication_error\"}}",
                "application/json");
            return true;  // reject
        });
    const int c = client_connect(f.port);
    send_all(c, "GET /x HTTP/1.1\r\nHost: h\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("HTTP/1.1 401") != std::string::npos);
    REQUIRE(r.find("authentication_error") != std::string::npos);  // body kept
    REQUIRE(r.find("Connection: close") != std::string::npos);
    REQUIRE(r.find("HANDLER-RAN") == std::string::npos);  // handler skipped
    ::close(c);
}

TEST_CASE("http server: a head handler that proceeds has its res DISCARDED",
          "[http_server]") {
    // The proceed path must not let the head stage half-fill the response.
    Fixture f(
        [](const Request&, ServerResponse& res, const RequestContext&) {
            res.set_content("REAL", "text/plain");
        },
        [](const Request&, ServerResponse& res, RequestContext&) {
            res.set_header("X-Leak", "1");
            res.set_content("LEAK", "text/plain");
            return false;  // proceed -> res discarded
        });
    const int c = client_connect(f.port);
    send_all(c, "GET /x HTTP/1.1\r\nHost: h\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("REAL") != std::string::npos);
    REQUIRE(r.find("LEAK") == std::string::npos);
    REQUIRE(r.find("X-Leak") == std::string::npos);
    ::close(c);
}

TEST_CASE("http server: the head stage threads identity to the handler",
          "[http_server]") {
    Fixture f(
        [](const Request&, ServerResponse& res, const RequestContext& ctx) {
            res.set_content("id=" + ctx.identity, "text/plain");
        },
        [](const Request&, ServerResponse&, RequestContext& ctx) {
            ctx.identity = "alice";  // set at the head stage
            return false;            // proceed
        });
    const int c = client_connect(f.port);
    send_all(c, "GET /x HTTP/1.1\r\nHost: h\r\n\r\n");
    REQUIRE(drain(c).find("id=alice") != std::string::npos);
    ::close(c);
}

TEST_CASE("http server: a head reject on a POST never reads the body",
          "[http_server]") {
    // Head promises a 1000-byte body we never send; the reject must come
    // back anyway (decided on the head), so a 401 does not cost an upload.
    Fixture f(
        [](const Request&, ServerResponse& res, const RequestContext&) {
            res.set_content("unreached", "text/plain");
        },
        [](const Request&, ServerResponse& res, RequestContext&) {
            res.status = 401;
            res.set_content("no", "text/plain");
            return true;
        });
    const int c = client_connect(f.port);
    send_all(c, "POST /x HTTP/1.1\r\nHost: h\r\nContent-Length: 1000\r\n\r\n");
    const std::string r = drain(c);
    REQUIRE(r.find("HTTP/1.1 401") != std::string::npos);
    REQUIRE(r.find("Connection: close") != std::string::npos);
    ::close(c);
}

TEST_CASE("http server: wait() blocks until stop()", "[http_server]") {
    Fixture f(echo_handler);
    std::atomic<bool> returned{false};
    std::thread waiter([&] {
        f.server.wait();
        returned.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE_FALSE(returned.load());  // still blocked while serving
    f.server.stop();
    waiter.join();
    REQUIRE(returned.load());  // wait() returned once stopped
}

// i#81: peer_half_closed is the liveness probe behind
// RequestContext::client_alive. It must report a closed peer WITHOUT
// consuming any pending request bytes, so a long-running handler can
// abandon work for a client that has gone while never corrupting a
// pipelined next request.
TEST_CASE("peer_half_closed detects a closed peer without consuming",
          "[http]") {
    int sv[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);

    // Both ends open, nothing pending: not readable -> connected.
    REQUIRE_FALSE(locus::http::peer_half_closed(sv[0]));

    // Pending inbound data (e.g. a pipelined request) is NOT a close,
    // and the peek must leave the byte in the socket buffer.
    REQUIRE(::send(sv[1], "x", 1, 0) == 1);
    REQUIRE_FALSE(locus::http::peer_half_closed(sv[0]));
    char b = 0;
    REQUIRE(::recv(sv[0], &b, 1, 0) == 1);  // still there to read
    REQUIRE(b == 'x');

    // Peer closes its end: orderly EOF -> reported closed.
    ::close(sv[1]);
    REQUIRE(locus::http::peer_half_closed(sv[0]));

    ::close(sv[0]);
}

// i#85: a fast-path route (e.g. /metrics, /health) must be served even
// when every worker is busy in a long request. With a single worker,
// one blocking request saturates the pool; pre-fix the fast GET waited
// in the kernel backlog until the slow one finished (the worker was the
// only acceptor). Post-fix the dedicated acceptor serves it inline.
TEST_CASE("http server: fast-path route answers while the pool is busy",
          "[http_server]") {
    std::atomic<bool> slow_started{false};
    std::atomic<bool> release{false};
    auto handler = [&](const Request& req, ServerResponse& res) {
        if (req.path == "/fast") {
            res.status = 200;
            res.set_content("fast", "text/plain");
            return;
        }
        // /slow occupies the lone worker until released, standing in for
        // a long generation that holds the pool.
        slow_started.store(true);
        while (!release.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        res.status = 200;
        res.set_content("slow", "text/plain");
    };
    ServerConfig cfg;
    cfg.worker_threads = 1;  // one worker: a single /slow saturates it
    cfg.fast_path = [](std::string_view m, std::string_view t) {
        return m == "GET" && t == "/fast";
    };
    Fixture f(handler, cfg);

    // Occupy the worker with /slow on its own connection.
    std::thread slow([&] {
        const int c = client_connect(f.port);
        send_all(c, "GET /slow HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n");
        drain(c, 3000);  // blocks in the handler until released
        ::close(c);
    });
    // Always release the slow handler and join, even if an assertion
    // below throws and aborts the case (otherwise the un-joined thread
    // would terminate the runner).
    struct Releaser {
        std::atomic<bool>& release;
        std::thread& t;
        ~Releaser() {
            release.store(true);
            if (t.joinable()) {
                t.join();
            }
        }
    } releaser{release, slow};

    for (int i = 0; i < 2000 && !slow_started.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(slow_started.load());  // the pool is now fully busy

    // The fast-path GET must come back promptly, not after /slow ends.
    const int c = client_connect(f.port);
    send_all(c, "GET /fast HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n");
    const std::string r = drain(c, 1000);
    ::close(c);
    REQUIRE(r.find("HTTP/1.1 200 OK") != std::string::npos);
    REQUIRE(r.find("fast") != std::string::npos);
}

// i#85 blocker: a fast-path connection must be served ONE request and
// closed, even on a keep-alive request. serve_connection is a keep-alive
// loop, so if the acceptor adopted the connection for more than one
// request, a later non-fast (inference) request on that same connection
// would run inline on the acceptor and stall ALL accept. Assert the
// acceptor response forces Connection: close and the socket is closed
// after one, so it can never be adopted for a second request.
TEST_CASE("http server: a fast-path connection serves one request and "
          "closes (no keep-alive escape)", "[http_server]") {
    ServerConfig cfg;
    cfg.fast_path = [](std::string_view m, std::string_view t) {
        return m == "GET" && t == "/fast";
    };
    Fixture f(echo_handler, cfg);
    const int c = client_connect(f.port);
    // Keep-alive request (NO Connection: close): the acceptor must still
    // answer once and close.
    send_all(c, "GET /fast HTTP/1.1\r\nHost: h\r\n\r\n");
    const std::string r1 = drain(c);
    REQUIRE(r1.find("HTTP/1.1 200 OK") != std::string::npos);
    REQUIRE(r1.find("Connection: close") != std::string::npos);
    // A second request on the SAME socket gets nothing: it was closed,
    // so the acceptor can never be carried into a later request.
    send_all(c, "GET /fast HTTP/1.1\r\nHost: h\r\n\r\n");
    REQUIRE(drain(c).empty());
    ::close(c);
}

// i#88: classification must not serialize the acceptor. Before this, the
// sole acceptor classified each connection with a BLOCKING peek bounded
// by 100 ms, so N clients that connect and send nothing cost N x 100 ms
// serially and delayed even fast-path accepts. Classification is now
// event-driven: a connection is peeked only once it has data, so a
// silent client occupies a slot and costs no wall time.
//
// Asserted as a SLOPE as well as a bound: the fast-path latency must not
// grow with the number of silent connections. That is the invariant, and
// it does not depend on how many the test happens to open.
TEST_CASE("http server: silent connections do not stall the acceptor",
          "[http_server]") {
    ServerConfig cfg;
    cfg.worker_threads = 1;
    cfg.fast_path = [](std::string_view m, std::string_view t) {
        return m == "GET" && t == "/fast";
    };
    Fixture f(echo_handler, cfg);

    // Times one fast-path GET on a fresh connection, with `n_silent`
    // connections already open that have sent nothing at all.
    auto fast_ms_with_silent = [&](int n_silent) {
        std::vector<int> silent;
        for (int i = 0; i < n_silent; ++i) {
            silent.push_back(client_connect(f.port));
        }
        // Let the acceptor see them all before we time anything.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto t0 = std::chrono::steady_clock::now();
        const int c = client_connect(f.port);
        send_all(c, "GET /fast HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n");
        const std::string r = drain(c, 3000);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        ::close(c);
        for (const int s : silent) {
            ::close(s);
        }
        REQUIRE(r.find("HTTP/1.1 200 OK") != std::string::npos);
        return ms;
    };

    const double few = fast_ms_with_silent(2);
    const double many = fast_ms_with_silent(10);

    // Pre-fix this was ~100 ms per silent connection, serially: ~200 ms
    // for 2 and ~1000 ms for 10. Post-fix both are ~1 ms. The bound is
    // generous (3x one classification window) so a loaded machine does
    // not flake it, while still being far below the ~1000 ms the old
    // blocking peek took for 10.
    REQUIRE(many < 300.0);
    // The slope is the real invariant: eight more silent connections
    // must not add anything like eight classification windows.
    REQUIRE(many - few < 100.0);
}

// i#88, the other half: event-driven classification must still classify
// a GENUINE fast-path request whose line arrives a round trip AFTER
// connect. This is why the fix is event-driven rather than a shorter
// peek window -- a non-blocking peek at accept time would see an empty
// socket, call it non-fast, and hand the scrape to the pool, which is
// the starvation the fast path exists to remove. Guard it so nobody
// "fixes" a future stall by shrinking the window to zero.
TEST_CASE("http server: fast path still matches when the request line "
          "arrives after connect", "[http_server]") {
    ServerConfig cfg;
    cfg.worker_threads = 1;
    cfg.fast_path = [](std::string_view m, std::string_view t) {
        return m == "GET" && t == "/fast";
    };
    std::atomic<bool> release{false};
    auto handler = [&](const Request& req, ServerResponse& res) {
        if (req.path == "/fast") {
            res.status = 200;
            res.set_content("fast", "text/plain");
            return;
        }
        while (!release.load()) {  // occupies the lone worker
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        res.status = 200;
        res.set_content("slow", "text/plain");
    };
    Fixture f(handler, cfg);

    // Saturate the single worker, so a misclassified /fast (handed to
    // the pool) would NOT come back and the test fails loudly.
    std::thread slow([&] {
        const int c = client_connect(f.port);
        send_all(c, "GET /slow HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n");
        drain(c, 3000);
        ::close(c);
    });
    struct Releaser {
        std::atomic<bool>& release;
        std::thread& t;
        ~Releaser() {
            release.store(true);
            if (t.joinable()) {
                t.join();
            }
        }
    } releaser{release, slow};
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Connect, stay silent past a round trip, THEN send the request line.
    const int c = client_connect(f.port);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    send_all(c, "GET /fast HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n");
    const std::string r = drain(c, 2000);
    ::close(c);
    REQUIRE(r.find("HTTP/1.1 200 OK") != std::string::npos);
    REQUIRE(r.find("fast") != std::string::npos);
}
