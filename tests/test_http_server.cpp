#include <csignal>
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
