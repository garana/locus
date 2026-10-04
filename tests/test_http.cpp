#include <string>
#include <string_view>

#include "catch_amalgamated.hpp"
#include "locus/http/http.hpp"

using locus::http::Limits;
using locus::http::parse_request;
using locus::http::ParseState;
using locus::http::Request;
using locus::http::Response;

namespace {
// Parses `buf` with default limits; returns the result + fills `out`.
locus::http::ParseResult parse(std::string_view buf, Request& out,
                               Limits limits = {}) {
    return parse_request(buf, limits, out);
}
}  // namespace

TEST_CASE("parse_request: a GET parses and consumes exactly", "[http]") {
    Request r;
    const std::string req =
        "GET /v1/models?x=1 HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Accept: application/json\r\n"
        "\r\n";
    auto res = parse(req, r);
    REQUIRE(res.state == ParseState::kComplete);
    REQUIRE(res.consumed == req.size());
    REQUIRE(r.method == "GET");
    REQUIRE(r.target == "/v1/models?x=1");
    REQUIRE(r.path == "/v1/models");
    REQUIRE(r.query == "x=1");
    REQUIRE(r.minor_version == 1);
    REQUIRE(r.keep_alive);
    REQUIRE(r.has_header("host"));              // case-insensitive
    REQUIRE(r.get_header_value("ACCEPT") == "application/json");
    REQUIRE(r.body.empty());
    REQUIRE_FALSE(r.is_head);
}

TEST_CASE("parse_request: POST with a Content-Length body", "[http]") {
    Request r;
    const std::string req =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 13\r\n"
        "\r\n"
        "{\"hello\":\"x\"}";
    auto res = parse(req, r);
    REQUIRE(res.state == ParseState::kComplete);
    REQUIRE(res.consumed == req.size());
    REQUIRE(r.method == "POST");
    REQUIRE(r.body == "{\"hello\":\"x\"}");
}

TEST_CASE("parse_request: trailing bytes (pipelined 2nd request) stay "
          "unconsumed", "[http]") {
    Request r;
    const std::string first = "GET /a HTTP/1.1\r\nHost: h\r\n\r\n";
    const std::string second = "GET /b HTTP/1.1\r\nHost: h\r\n\r\n";
    auto res = parse(first + second, r);
    REQUIRE(res.state == ParseState::kComplete);
    REQUIRE(res.consumed == first.size());  // only the first request
    REQUIRE(r.path == "/a");
}

TEST_CASE("parse_request: HEAD recognized; HTTP/1.0 defaults to close",
          "[http]") {
    Request r;
    REQUIRE(parse("HEAD /x HTTP/1.1\r\nHost: h\r\n\r\n", r).state ==
            ParseState::kComplete);
    REQUIRE(r.is_head);

    Request r2;
    REQUIRE(parse("GET /x HTTP/1.0\r\nHost: h\r\n\r\n", r2).state ==
            ParseState::kComplete);
    REQUIRE(r2.minor_version == 0);
    REQUIRE_FALSE(r2.keep_alive);  // 1.0 closes by default

    Request r3;
    REQUIRE(parse("GET /x HTTP/1.0\r\nConnection: keep-alive\r\n\r\n",
                  r3)
                .state == ParseState::kComplete);
    REQUIRE(r3.keep_alive);  // 1.0 keep-alive on request

    Request r4;
    REQUIRE(parse("GET /x HTTP/1.1\r\nConnection: close\r\n\r\n", r4)
                .state == ParseState::kComplete);
    REQUIRE_FALSE(r4.keep_alive);  // 1.1 close on request
}

TEST_CASE("parse_request: incremental byte-at-a-time reaches Complete",
          "[http]") {
    const std::string req =
        "POST /x HTTP/1.1\r\nContent-Length: 3\r\n\r\nabc";
    // Every strict prefix is kNeedMore; the whole buffer is kComplete.
    for (std::size_t n = 1; n < req.size(); ++n) {
        Request r;
        auto res = parse(std::string_view(req).substr(0, n), r);
        REQUIRE(res.state == ParseState::kNeedMore);
        REQUIRE(res.consumed == 0);
    }
    Request full;
    auto res = parse(req, full);
    REQUIRE(res.state == ParseState::kComplete);
    REQUIRE(full.body == "abc");
}

TEST_CASE("parse_request: chunked body decodes; trailers bounded",
          "[http]") {
    Request r;
    const std::string req =
        "POST /x HTTP/1.1\r\n"
        "Transfer-Encoding: chunked\r\n"
        "\r\n"
        "4\r\nWiki\r\n"
        "5\r\npedia\r\n"
        "0\r\n"
        "X-Trailer: v\r\n"
        "\r\n";
    auto res = parse(req, r);
    REQUIRE(res.state == ParseState::kComplete);
    REQUIRE(res.consumed == req.size());
    REQUIRE(r.body == "Wikipedia");
}

// ---- hardening rejections ----

TEST_CASE("parse_request rejects the hostile shapes", "[http]") {
    auto is_error = [](std::string_view req, Limits lim = {}) {
        Request r;
        return parse_request(req, lim, r).state == ParseState::kError;
    };

    SECTION("bad / unsupported version") {
        REQUIRE(is_error("GET / HTTP/2.0\r\nHost: h\r\n\r\n"));
        REQUIRE(is_error("GET / HTTP/1.2\r\nHost: h\r\n\r\n"));
        REQUIRE(is_error("GET / FTP\r\nHost: h\r\n\r\n"));
    }
    SECTION("bare LF (CRLF required)") {
        REQUIRE(is_error("GET / HTTP/1.1\nHost: h\n\n"));
        REQUIRE(is_error("GET / HTTP/1.1\r\nHost: h\n\r\n"));
    }
    SECTION("Content-Length: sign, embedded space, junk, overflow") {
        REQUIRE(is_error("POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n"));
        REQUIRE(is_error("POST / HTTP/1.1\r\nContent-Length: +5\r\n\r\n"));
        // Embedded space is junk (surrounding OWS is spec-legal, trimmed).
        REQUIRE(is_error("POST / HTTP/1.1\r\nContent-Length: 1 2\r\n\r\n"));
        REQUIRE(is_error("POST / HTTP/1.1\r\nContent-Length: 1x\r\n\r\n"));
        REQUIRE(is_error(
            "POST / HTTP/1.1\r\nContent-Length: "
            "99999999999999999999999999\r\n\r\n"));
        // Surrounding OWS trimmed -> a clean integer, accepted.
        Request ok;
        REQUIRE(parse("POST / HTTP/1.1\r\nContent-Length:  0  \r\n\r\n",
                      ok)
                    .state == ParseState::kComplete);
    }
    SECTION("duplicate Content-Length (even if equal)") {
        REQUIRE(is_error(
            "POST / HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: "
            "0\r\n\r\n"));
    }
    SECTION("Content-Length + Transfer-Encoding together (smuggling)") {
        REQUIRE(is_error(
            "POST / HTTP/1.1\r\nContent-Length: 5\r\nTransfer-Encoding: "
            "chunked\r\n\r\n0\r\n\r\n"));
    }
    SECTION("Transfer-Encoding not sole chunked") {
        REQUIRE(is_error(
            "POST / HTTP/1.1\r\nTransfer-Encoding: gzip, chunked\r\n\r\n"
            "0\r\n\r\n"));
        REQUIRE(is_error(
            "POST / HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n"));
    }
    SECTION("obs-fold continuation") {
        REQUIRE(is_error(
            "GET / HTTP/1.1\r\nX: a\r\n b\r\nHost: h\r\n\r\n"));
    }
    SECTION("bad chunk size / missing CRLF after chunk data") {
        REQUIRE(is_error(
            "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
            "zz\r\n\r\n"));  // non-hex size
        REQUIRE(is_error(
            "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
            "4\r\nWikiXX0\r\n\r\n"));  // data not followed by CRLF
    }
    SECTION("chunk extensions rejected") {
        REQUIRE(is_error(
            "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
            "4;ext=1\r\nWiki\r\n0\r\n\r\n"));
    }
    SECTION("empty / malformed header line") {
        REQUIRE(is_error("GET / HTTP/1.1\r\n: noname\r\n\r\n"));
        REQUIRE(is_error("GET / HTTP/1.1\r\nnocolon\r\n\r\n"));
    }
    SECTION("limits: request line, header count, body") {
        Limits tiny;
        tiny.max_request_line = 16;
        REQUIRE(is_error(
            "GET /this/target/is/way/too/long HTTP/1.1\r\n\r\n", tiny));
        Limits few;
        few.max_header_count = 1;
        REQUIRE(is_error("GET / HTTP/1.1\r\nA: 1\r\nB: 2\r\n\r\n", few));
        Limits small;
        small.max_body_bytes = 4;
        REQUIRE(is_error(
            "POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello", small));
    }
}

// ---- response writer ----

TEST_CASE("write_response round-trips and sets framing", "[http]") {
    Response res;
    res.status = 200;
    res.set_content("{\"ok\":true}", "application/json");
    std::string out;
    std::string err;
    REQUIRE(write_response(res, /*head=*/false, /*keep_alive=*/true, out,
                           err));
    REQUIRE(out.find("HTTP/1.1 200 OK\r\n") == 0);
    REQUIRE(out.find("Content-Type: application/json\r\n") !=
            std::string::npos);
    REQUIRE(out.find("Content-Length: 11\r\n") != std::string::npos);
    REQUIRE(out.find("Connection: keep-alive\r\n") != std::string::npos);
    REQUIRE(out.find("\r\n\r\n{\"ok\":true}") != std::string::npos);
}

TEST_CASE("write_response: HEAD omits the body but keeps Content-Length",
          "[http]") {
    Response res;
    res.set_content("abcdef", "text/plain");
    std::string out;
    std::string err;
    REQUIRE(write_response(res, /*head=*/true, /*keep_alive=*/false, out,
                           err));
    REQUIRE(out.find("Content-Length: 6\r\n") != std::string::npos);
    REQUIRE(out.find("Connection: close\r\n") != std::string::npos);
    REQUIRE(out.substr(out.size() - 4) == "\r\n\r\n");  // no body after
}

TEST_CASE("write_response rejects header injection and bad framing",
          "[http]") {
    std::string out;
    std::string err;
    {
        Response res;
        res.set_header("X-Evil", "a\r\nInjected: 1");
        REQUIRE_FALSE(write_response(res, false, true, out, err));
        REQUIRE_FALSE(err.empty());
    }
    {
        Response res;
        res.set_header("X-Nul", std::string("a\0b", 3));
        REQUIRE_FALSE(write_response(res, false, true, out, err));
    }
    {
        Response res;  // the writer owns framing
        res.set_header("Content-Length", "5");
        REQUIRE_FALSE(write_response(res, false, true, out, err));
    }
    {
        Response res;
        res.status = 299;  // not an emittable status
        REQUIRE_FALSE(write_response(res, false, true, out, err));
    }
}

TEST_CASE("chunk framing bytes are exact", "[http]") {
    REQUIRE(locus::http::encode_chunk("Wiki") == "4\r\nWiki\r\n");
    REQUIRE(locus::http::encode_chunk(std::string(16, 'x')) ==
            "10\r\n" + std::string(16, 'x') + "\r\n");
    REQUIRE(locus::http::last_chunk() == "0\r\n\r\n");
}

// Regression: the header-section limit must bound the HEADERS, not the
// whole buffer. A request delivered whole (headers + body in one read)
// has the body already buffered while headers parse; counting it as
// header bytes would reject a tiny-header request with a large but legal
// body. 100 KB body > the 64 KB default max_header_bytes, well under the
// 8 MB max_body_bytes -- must parse, not "headers exceed limit".
TEST_CASE("parse_request: a large body is not counted as header bytes",
          "[http]") {
    Request r;
    const std::string payload(100 * 1024, 'x');
    const std::string req =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Length: " + std::to_string(payload.size()) + "\r\n"
        "\r\n" + payload;
    auto res = parse(req, r);  // default Limits (64 KB headers, 8 MB body)
    REQUIRE(res.state == ParseState::kComplete);
    REQUIRE(res.consumed == req.size());
    REQUIRE(r.body.size() == payload.size());
    REQUIRE(r.body == payload);
}
