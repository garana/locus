// The first-party HTTP client (i#388 step 1): the pure response parser
// and chunk decoder, and the blocking Client against locus's own
// http::Server and against raw sockets for the edge cases.

#include <csignal>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/http/client.hpp"
#include "locus/http/server.hpp"
#include "locus/pipeline/net.hpp"

using locus::http::BodyFraming;
using locus::http::ChunkDecoder;
using locus::http::Client;
using locus::http::ClientOptions;
using locus::http::ClientRequest;
using locus::http::ParseState;
using locus::http::ResponseHead;
using locus::http::ResponseLimits;
using locus::http::parse_response_head;

namespace {

const int kIgnoreSigpipe = [] {
    std::signal(SIGPIPE, SIG_IGN);
    return 0;
}();

ParseState head_state(std::string_view raw, ResponseHead& h,
                      bool head_req = false) {
    return parse_response_head(raw, head_req, ResponseLimits{}, h).state;
}

/** Decodes a whole chunked body fed in pieces of `step` bytes. */
std::string dechunk(std::string_view body, std::size_t step,
                    ChunkDecoder::State& st, std::string& rest) {
    ChunkDecoder d;
    std::string out;
    st = ChunkDecoder::State::kNeedMore;
    for (std::size_t i = 0; i < body.size() &&
                            st == ChunkDecoder::State::kNeedMore;
         i += step) {
        st = d.feed(body.substr(i, step),
                    [&](std::string_view s) { out.append(s); }, rest);
    }
    return out;
}

/** A locus::http::Server on an ephemeral loopback port. */
struct ServerFixture {
    locus::http::Server server;
    int port;
    explicit ServerFixture(locus::http::Handler h)
        : server(config(), std::move(h)), port(server.start()) {
        REQUIRE(port > 0);
    }
    static locus::http::ServerConfig config() {
        locus::http::ServerConfig c;
        c.host = "127.0.0.1";
        c.port = 0;
        c.worker_threads = 4;
        return c;
    }
};

/** A raw listener that answers one connection with fixed bytes. */
struct RawServer {
    int listen_fd = -1;
    int port = 0;
    std::thread th;
    RawServer(std::string reply, int hold_ms = 0) {
        listen_fd = locus::pipeline::listen_on("127.0.0.1", 0, &port);
        REQUIRE(listen_fd >= 0);
        th = std::thread([this, reply = std::move(reply), hold_ms] {
            const int c = locus::pipeline::accept_one(listen_fd, nullptr,
                                                      3000);
            if (c < 0) return;
            char buf[4096];
            // The request; its size does not matter here. GCC ignores a
            // (void) cast on warn_unused_result, so keep the results.
            [[maybe_unused]] const ssize_t got = ::read(c, buf, sizeof(buf));
            if (!reply.empty()) {
                [[maybe_unused]] const ssize_t put =
                    ::write(c, reply.data(), reply.size());
            }
            if (hold_ms > 0) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(hold_ms));
            }
            ::close(c);
        });
    }
    ~RawServer() {
        th.join();
        ::close(listen_fd);
    }
};

}  // namespace

TEST_CASE("http client: response heads and their body framing",
          "[http_client]") {
    ResponseHead h;
    REQUIRE(head_state("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello",
                       h) == ParseState::kComplete);
    CHECK(h.status == 200);
    CHECK(h.reason == "OK");
    CHECK(h.framing == BodyFraming::kLength);
    CHECK(h.content_length == 5);
    CHECK(h.keep_alive);

    REQUIRE(head_state("HTTP/1.1 200 OK\r\ntransfer-encoding: Chunked\r\n"
                       "Connection: close\r\n\r\n",
                       h) == ParseState::kComplete);
    CHECK(h.framing == BodyFraming::kChunked);
    CHECK_FALSE(h.keep_alive);

    // No length: the body runs to the close.
    REQUIRE(head_state("HTTP/1.0 200 OK\r\n\r\n", h) ==
            ParseState::kComplete);
    CHECK(h.framing == BodyFraming::kClose);
    CHECK_FALSE(h.keep_alive);

    // No body for 204, 304, 1xx and a HEAD request.
    for (const char* s : {"HTTP/1.1 204 No Content\r\n\r\n",
                          "HTTP/1.1 304 Not Modified\r\n\r\n",
                          "HTTP/1.1 100 Continue\r\n\r\n"}) {
        REQUIRE(head_state(s, h) == ParseState::kComplete);
        CHECK(h.framing == BodyFraming::kNone);
    }
    REQUIRE(head_state("HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\n", h,
                       true) == ParseState::kComplete);
    CHECK(h.framing == BodyFraming::kNone);

    // A prefix needs more bytes.
    CHECK(head_state("HTTP/1.1 200 OK\r\nContent-Le", h) ==
          ParseState::kNeedMore);
    CHECK(head_state("HTTP/1.1 503 Service Unavailable\r\nRetry-After: 2"
                     "\r\nContent-Length: 0\r\n\r\n",
                     h) == ParseState::kComplete);
    CHECK(h.get_header_value("retry-after") == "2");
}

TEST_CASE("http client: malformed or hostile heads are errors",
          "[http_client]") {
    ResponseHead h;
    for (const char* s : {
             "HTTP/1.1 200 OK\n\n",                       // bare LF
             "HTTP/2 200 OK\r\n\r\n",                     // version
             "HTTP/1.1 20 OK\r\n\r\n",                    // code
             "HTTP/1.1 200 OK\r\n folded\r\n\r\n",        // obs-fold
             "HTTP/1.1 200 OK\r\nBad Name: x\r\n\r\n",    // name
             "HTTP/1.1 200 OK\r\nX: a\x01b\r\n\r\n",      // control byte
             "HTTP/1.1 200 OK\r\nContent-Length: -1\r\n\r\n",
             "HTTP/1.1 200 OK\r\nContent-Length: 1\r\n"
             "Content-Length: 1\r\n\r\n",                 // repeated
             "HTTP/1.1 200 OK\r\nContent-Length: 99999999999999999999999"
             "\r\n\r\n",                                  // overflow
             "HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\n",
             "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
             "Content-Length: 3\r\n\r\n"}) {               // both
        INFO(s);
        CHECK(head_state(s, h) == ParseState::kError);
    }
    // A head that never ends is bounded.
    ResponseLimits small;
    small.max_header_bytes = 64;
    const std::string big =
        "HTTP/1.1 200 OK\r\nX: " + std::string(100, 'a');
    CHECK(parse_response_head(big, false, small, h).state ==
          ParseState::kError);
}

TEST_CASE("http client: chunk decoder in any feed size", "[http_client]") {
    const std::string body =
        "5\r\nhello\r\n7;ext=1\r\n, world\r\n0\r\nX-T: 1\r\n\r\nNEXT";
    for (const std::size_t step : {std::size_t{1}, std::size_t{3},
                                   std::size_t{1000}}) {
        INFO("step " << step);
        ChunkDecoder::State st;
        std::string rest;
        CHECK(dechunk(body, step, st, rest) == "hello, world");
        CHECK(st == ChunkDecoder::State::kDone);
        if (step == 1000) CHECK(rest == "NEXT");  // the next response
    }
    ChunkDecoder::State st;
    std::string rest;
    dechunk("zz\r\nab\r\n", 100, st, rest);
    CHECK(st == ChunkDecoder::State::kError);
    dechunk("2\r\nabX\r\n", 100, st, rest);
    CHECK(st == ChunkDecoder::State::kError);
    ChunkDecoder capped(4);
    CHECK(capped.feed("5\r\nhello\r\n", [](std::string_view) {}, rest) ==
          ChunkDecoder::State::kError);
}

TEST_CASE("http client: requests to locus's own server", "[http_client]") {
    std::atomic<int> hits{0};
    ServerFixture f([&](const locus::http::Request& req,
                        locus::http::ServerResponse& res,
                        const locus::http::RequestContext&) {
        ++hits;
        if (req.path == "/stream") {
            res.set_chunked_content_provider(
                "text/event-stream", [](locus::http::Sink& sink) {
                    for (int i = 0; i < 3; ++i) {
                        if (!sink.write("data: " + std::to_string(i) +
                                        "\n\n")) {
                            return;
                        }
                    }
                });
            return;
        }
        res.status = req.path == "/missing" ? 404 : 200;
        res.set_header("X-Echo", req.get_header_value("X-In"));
        res.set_content(req.method + " " + req.path + " " + req.body,
                        "text/plain");
    });
    Client c("127.0.0.1", f.port);

    SECTION("buffered GET and POST on one kept-alive connection") {
        ClientRequest get;
        get.target = "/a";
        get.headers = {{"X-In", "v1"}};
        auto r = c.send(get);
        REQUIRE(r.ok);
        CHECK(r.head.status == 200);
        CHECK(r.body == "GET /a ");
        CHECK(r.head.get_header_value("X-Echo") == "v1");

        ClientRequest post;
        post.method = "POST";
        post.target = "/b";
        post.body = "{\"x\":1}";
        r = c.send(post);
        REQUIRE(r.ok);
        CHECK(r.body == "POST /b {\"x\":1}");

        r = c.send(ClientRequest{"GET", "/missing", {}, ""});
        REQUIRE(r.ok);
        CHECK(r.head.status == 404);
        CHECK(hits == 3);
    }
    SECTION("a chunked stream arrives as it is written") {
        std::vector<std::string> parts;
        int status = 0;
        auto r = c.stream(
            ClientRequest{"GET", "/stream", {}, ""},
            [&](const ResponseHead& h) {
                status = h.status;
                return true;
            },
            [&](std::string_view d) {
                parts.emplace_back(d);
                return true;
            });
        REQUIRE(r.ok);
        CHECK(status == 200);
        CHECK(r.head.framing == BodyFraming::kChunked);
        std::string all;
        for (const auto& p : parts) all += p;
        CHECK(all == "data: 0\n\ndata: 1\n\ndata: 2\n\n");
        // The connection is reused after the stream.
        CHECK(c.send(ClientRequest{"GET", "/after", {}, ""}).body ==
              "GET /after ");
    }
    SECTION("the caller can stop a stream") {
        int seen = 0;
        auto r = c.stream(
            ClientRequest{"GET", "/stream", {}, ""}, nullptr,
            [&](std::string_view) { return ++seen < 1; });
        CHECK_FALSE(r.ok);
        CHECK(seen == 1);
        // A new connection works afterwards.
        CHECK(c.send(ClientRequest{"GET", "/x", {}, ""}).ok);
    }
    SECTION("reserved or unsafe request headers are refused") {
        CHECK_FALSE(
            c.send(ClientRequest{"GET", "/", {{"Content-Length", "3"}}, ""})
                .ok);
        CHECK_FALSE(
            c.send(ClientRequest{"GET", "/", {{"X", "a\r\nY: b"}}, ""}).ok);
        CHECK_FALSE(c.send(ClientRequest{"GET", "/a b", {}, ""}).ok);
        CHECK(hits == 0);
    }
}

TEST_CASE("http client: a server restart between requests is retried once",
          "[http_client]") {
    // The first server answers and closes the kept-alive connection
    // after its idle timeout; the client notices only on the next send.
    locus::http::ServerConfig cfg = ServerFixture::config();
    cfg.idle_timeout_ms = 100;
    locus::http::Server s(cfg, [](const locus::http::Request& req,
                                  locus::http::ServerResponse& res,
                                  const locus::http::RequestContext&) {
        res.set_content("ok " + req.path, "text/plain");
    });
    const int port = s.start();
    REQUIRE(port > 0);
    Client c("127.0.0.1", port);
    REQUIRE(c.send(ClientRequest{"GET", "/1", {}, ""}).body == "ok /1");
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    const auto r = c.send(ClientRequest{"GET", "/2", {}, ""});
    REQUIRE(r.ok);
    CHECK(r.body == "ok /2");
}

TEST_CASE("http client: close-delimited bodies, refusals and timeouts",
          "[http_client]") {
    SECTION("a body that runs to the close") {
        RawServer raw("HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\n"
                      "all of it");
        Client c("127.0.0.1", raw.port);
        const auto r = c.send(ClientRequest{});
        REQUIRE(r.ok);
        CHECK(r.body == "all of it");
    }
    SECTION("an interim 100 Continue is skipped") {
        RawServer raw("HTTP/1.1 100 Continue\r\n\r\n"
                      "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                      "Connection: close\r\n\r\nok");
        Client c("127.0.0.1", raw.port);
        const auto r = c.send(ClientRequest{});
        REQUIRE(r.ok);
        CHECK(r.head.status == 200);
        CHECK(r.body == "ok");
    }
    SECTION("a body cut short is an error") {
        RawServer raw("HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort");
        Client c("127.0.0.1", raw.port);
        const auto r = c.send(ClientRequest{});
        CHECK_FALSE(r.ok);
        CHECK(r.error.find("mid-body") != std::string::npos);
    }
    SECTION("a server that never answers times out") {
        RawServer raw("", 1500);
        ClientOptions o;
        o.read_timeout_ms = 200;
        Client c("127.0.0.1", raw.port, o);
        const auto t0 = std::chrono::steady_clock::now();
        const auto r = c.send(ClientRequest{});
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        CHECK_FALSE(r.ok);
        CHECK(r.error.find("timed out") != std::string::npos);
        CHECK(ms < 1200);
    }
    SECTION("nothing listening") {
        int port = 0;
        const int fd = locus::pipeline::listen_on("127.0.0.1", 0, &port);
        REQUIRE(fd >= 0);
        ::close(fd);  // the port is free now
        Client c("127.0.0.1", port);
        const auto r = c.send(ClientRequest{});
        CHECK_FALSE(r.ok);
        CHECK(r.error.find("connect") != std::string::npos);
    }
}
