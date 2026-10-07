#include <string>
#include <string_view>

#include "catch_amalgamated.hpp"
#include "locus/http/http.hpp"

using locus::http::Limits;
using locus::http::parse_head;
using locus::http::parse_request;
using locus::http::ParseState;
using locus::http::Request;
using locus::http::Response;

namespace {
// Parses `buf` with default limits; returns the result + fills `out`.
locus::http::ParseResult parse(std::string_view buf, Request& out,
                               Limits limits = {}) {
    locus::http::ParseContext ctx;
    return parse_request(buf, limits, ctx, out);
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
        locus::http::ParseContext ctx;
        return parse_request(req, lim, ctx, r).state == ParseState::kError;
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
    SECTION("multiple Transfer-Encoding headers (two chunked lines)") {
        // Two "chunked" lines are the list chunked,chunked -- the single-
        // line list path already rejects non-sole-final-chunked, so the
        // two spellings of one message must agree.
        REQUIRE(is_error(
            "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n"
            "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"));
    }
    SECTION("control bytes in a header value (incl. NUL)") {
        // NUL would truncate a %s log line; every CTL but HTAB is barred.
        std::string nul = "GET / HTTP/1.1\r\nX: a";  // embedded NUL built
        nul.push_back('\0');                         // explicitly so the
        nul += "b\r\n\r\n";                          // view keeps its len
        REQUIRE(is_error(nul));
        REQUIRE(is_error("GET / HTTP/1.1\r\nX: a\x01y\r\n\r\n"));
        REQUIRE(is_error("GET / HTTP/1.1\r\nX: a\x1by\r\n\r\n"));  // ESC
        REQUIRE(is_error("GET / HTTP/1.1\r\nX: a\x7fy\r\n\r\n"));  // DEL
    }
    SECTION("unimplemented methods rejected (routing desync guard)") {
        REQUIRE(is_error("PUT / HTTP/1.1\r\nHost: h\r\n\r\n"));
        REQUIRE(is_error("DELETE / HTTP/1.1\r\nHost: h\r\n\r\n"));
        REQUIRE(is_error("OPTIONS / HTTP/1.1\r\nHost: h\r\n\r\n"));
        // Methods are case-sensitive: "head" is not HEAD.
        REQUIRE(is_error("head / HTTP/1.1\r\nHost: h\r\n\r\n"));
    }
}

// A header value may carry obs-text (0x80-0xFF): UTF-8 model names,
// user-agents, and echoed error strings must survive. The signed-char
// trap: plain char is SIGNED on x86-64, so a naive `c < 0x20` test
// rejects every 0x80+ byte on that platform only -- this asserts the
// bytes are ACCEPTED, which a rejection-only suite cannot catch.
TEST_CASE("parse_request: obs-text (UTF-8) header values are accepted",
          "[http]") {
    Request r;
    const std::string req =
        "GET / HTTP/1.1\r\n"
        "X-Model: caf\xc3\xa9-7b\r\n"   // "café-7b" in UTF-8
        "X-Hi: \xff\x80\xc3\r\n"          // raw high bytes + HTAB below
        "X-Tab: a\tb\r\n"
        "\r\n";
    auto res = parse(req, r);
    REQUIRE(res.state == ParseState::kComplete);
    REQUIRE(r.get_header_value("X-Model") == "caf\xc3\xa9-7b");
    REQUIRE(r.get_header_value("X-Tab") == "a\tb");  // interior HTAB kept
}

// The suggested_status lets the connection loop map the condition
// without matching error text (an existing server test pins 413).
TEST_CASE("parse_request: suggested_status distinguishes the 4xx/5xx",
          "[http]") {
    Request r;
    locus::http::ParseContext ctx;
    Limits small;
    small.max_body_bytes = 4;
    auto over = parse_request(
        "POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello", small, ctx, r);
    REQUIRE(over.state == ParseState::kError);
    REQUIRE(over.suggested_status == 413);  // over a size cap

    auto malformed = parse("GET / HTTP/1.1\r\nnocolon\r\n\r\n", r);
    REQUIRE(malformed.suggested_status == 400);  // malformed

    auto method = parse("PUT / HTTP/1.1\r\nHost: h\r\n\r\n", r);
    REQUIRE(method.suggested_status == 501);  // unimplemented method

    auto version = parse("GET / HTTP/2.0\r\nHost: h\r\n\r\n", r);
    REQUIRE(version.suggested_status == 505);  // unsupported version
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

// ---- parser invariants ----
// These hold for ANY input. A parser that re-scans from the front can
// silently break them, so they are pinned rather than left implicit.

namespace {
// Feeds `req` one byte at a time through a single ParseContext (what the
// connection loop does). Sets `prefixes_ok` false if any STRICT prefix
// was decided early (not kNeedMore/consumed==0). @returns the whole-buffer
// result, filling `out`.
locus::http::ParseResult feed_incremental(std::string_view req,
                                          Request& out, bool& prefixes_ok,
                                          Limits limits = {}) {
    locus::http::ParseContext ctx;
    prefixes_ok = true;
    for (std::size_t n = 1; n < req.size(); ++n) {
        Request tmp;
        auto r = parse_request(req.substr(0, n), limits, ctx, tmp);
        if (r.state != ParseState::kNeedMore || r.consumed != 0) {
            prefixes_ok = false;
        }
    }
    locus::http::ParseContext whole_ctx;
    return parse_request(req, limits, whole_ctx, out);
}
}  // namespace

TEST_CASE("parse_request invariant: byte-at-a-time equals whole",
          "[http]") {
    const std::string reqs[] = {
        "GET /a?b=1 HTTP/1.1\r\nHost: h\r\nAccept: */*\r\n\r\n",
        "POST /x HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello",
        "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
        "4\r\nWiki\r\n5\r\npedia\r\n0\r\nX-T: v\r\n\r\n",
        "HEAD / HTTP/1.0\r\n\r\n",
    };
    for (const auto& req : reqs) {
        Request whole;
        locus::http::ParseContext wc;
        auto w = parse_request(req, Limits{}, wc, whole);
        REQUIRE(w.state == ParseState::kComplete);

        bool prefixes_ok = false;
        Request inc;
        auto i = feed_incremental(req, inc, prefixes_ok);
        REQUIRE(prefixes_ok);               // no prefix decided early
        REQUIRE(i.state == w.state);        // same terminal state
        REQUIRE(i.consumed == w.consumed);  // same byte count
        REQUIRE(inc.method == whole.method);
        REQUIRE(inc.path == whole.path);
        REQUIRE(inc.query == whole.query);
        REQUIRE(inc.body == whole.body);
        REQUIRE(inc.keep_alive == whole.keep_alive);
        REQUIRE(inc.headers == whole.headers);
    }
}

TEST_CASE("parse_request invariant: an errored prefix stays errored",
          "[http]") {
    // Appending bytes to an error must not yield kComplete, or a
    // dribbling client bypasses every limit by error-then-recover.
    const std::string bad = "GET / HTTP/2.0\r\n";  // unsupported version
    Request r;
    REQUIRE(parse(bad, r).state == ParseState::kError);
    REQUIRE(parse(bad + "Host: h\r\n\r\n", r).state == ParseState::kError);
    REQUIRE(parse(bad + std::string(100, 'x'), r).state ==
            ParseState::kError);
}

TEST_CASE("parse_request invariant: kNeedMore is bounded", "[http]") {
    // A buffer past every limit summed must be decided, never still
    // kNeedMore (else the caller buffers without bound).
    Limits lim;
    const std::size_t sum = lim.max_request_line + lim.max_header_bytes +
                            lim.max_body_bytes + lim.max_trailer_bytes;
    Request r;
    auto res = parse(std::string(sum + 1, 'x'), r);  // no CRLF ever
    REQUIRE(res.state == ParseState::kError);
}

TEST_CASE("parse_request: a request line exactly at the cap is accepted "
          "regardless of TCP segmentation", "[http]") {
    Limits lim;
    lim.max_request_line = 40;
    const std::string target = "/" + std::string(26, 'a');   // 27 chars
    const std::string line = "GET " + target + " HTTP/1.1";  // 40 chars
    REQUIRE(line.size() == lim.max_request_line);
    const std::string req = line + "\r\nHost: h\r\n\r\n";

    Request whole;
    REQUIRE(parse(req, whole, lim).state == ParseState::kComplete);
    // The prefix is the whole line + its CR (one byte short of the LF):
    // the segmentation the old buf.size() measure wrongly rejected.
    Request r;
    REQUIRE(parse(line + "\r", r, lim).state == ParseState::kNeedMore);
}

TEST_CASE("parse_request: a header section exactly at the cap is accepted "
          "regardless of TCP segmentation", "[http]") {
    Limits lim;
    lim.max_header_bytes = 20;
    const std::string hdr = "X: " + std::string(15, 'v');  // "X: " + 15
    // the header line is 18 bytes; section measured at the blank line is
    // 18 + the line CRLF's contribution == 20 == cap.
    const std::string req = "GET / HTTP/1.1\r\n" + hdr + "\r\n\r\n";
    Request whole;
    REQUIRE(parse(req, whole, lim).state == ParseState::kComplete);
    // Prefix ending at the blank-line CR: old buf.size() measure rejected.
    Request r;
    const std::string pre = "GET / HTTP/1.1\r\n" + hdr + "\r\n\r";
    REQUIRE(parse(pre, r, lim).state == ParseState::kNeedMore);
}

TEST_CASE("parse_request: the trailer cap ignores a pipelined next "
          "request on keep-alive", "[http]") {
    // The trailer-bytes cap must measure the trailer section, not the
    // whole buffer: a large 2nd request already buffered behind the
    // chunked one must not be miscounted as trailer bytes.
    Limits lim;
    lim.max_trailer_bytes = 8;
    const std::string first =
        "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
        "4\r\nWiki\r\n0\r\n\r\n";  // zero trailers
    const std::string next =
        "GET /big HTTP/1.1\r\nX-Big: " + std::string(200, 'A') + "\r\n\r\n";
    Request r;
    auto res = parse(first + next, r, lim);
    REQUIRE(res.state == ParseState::kComplete);
    REQUIRE(res.consumed == first.size());  // only the first request
    REQUIRE(r.body == "Wiki");
}

// ---- parse_head: head-only parse for the connection server's head hook ----

TEST_CASE("parse_head: returns on the head alone, before the body",
          "[http]") {
    // A declared 1000-byte body that has NOT arrived: parse_request would
    // be kNeedMore, but parse_head completes on the head.
    Request r;
    const std::string head = "POST /x HTTP/1.1\r\nContent-Length: 1000\r\n\r\n";
    auto res = parse_head(head, Limits{}, r);
    REQUIRE(res.state == ParseState::kComplete);
    REQUIRE(res.consumed == head.size());  // through the CRLFCRLF
    REQUIRE(r.method == "POST");
    REQUIRE(r.path == "/x");
    REQUIRE(r.get_header_value("content-length") == "1000");
    REQUIRE(r.body.empty());  // body not read

    // parse_request on the same bytes still waits for the body.
    Request r2;
    locus::http::ParseContext ctx;
    REQUIRE(parse_request(head, Limits{}, ctx, r2).state ==
            ParseState::kNeedMore);
}

TEST_CASE("parse_head: a partial head is kNeedMore; a malformed head "
          "carries the suggested status", "[http]") {
    Request r;
    REQUIRE(parse_head("GET / HTTP/1.1\r\nHost: h\r\n", Limits{}, r).state ==
            ParseState::kNeedMore);  // no blank line yet
    auto bad = parse_head("GET / HTTP/2.0\r\n\r\n", Limits{}, r);
    REQUIRE(bad.state == ParseState::kError);
    REQUIRE(bad.suggested_status == 505);  // same head rules as the full parse
}

TEST_CASE("parse_head: head caps apply, max_body_bytes does not", "[http]") {
    // A huge declared body is a BODY concern; the head parses regardless,
    // so the hook can run auth/caps and decide before reading the body.
    Limits lim;
    lim.max_body_bytes = 8;
    Request r;
    const std::string head =
        "POST /x HTTP/1.1\r\nContent-Length: 100000\r\n\r\n";
    REQUIRE(parse_head(head, lim, r).state == ParseState::kComplete);
    // But a header-section cap still bites.
    lim.max_header_bytes = 4;
    Request r2;
    REQUIRE(parse_head(head, lim, r2).state == ParseState::kError);
}

TEST_CASE("parse_head agrees with parse_request on the head (one home)",
          "[http]") {
    // The head-only parse and the full parse must produce the same head,
    // since they share one implementation.
    const std::string req =
        "POST /a?b=1 HTTP/1.1\r\nHost: h\r\nX-One: y\r\n"
        "Connection: close\r\nContent-Length: 2\r\n\r\nhi";
    Request hr;
    auto h = parse_head(req, Limits{}, hr);
    REQUIRE(h.state == ParseState::kComplete);
    Request fr;
    locus::http::ParseContext ctx;
    auto f = parse_request(req, Limits{}, ctx, fr);
    REQUIRE(f.state == ParseState::kComplete);
    REQUIRE(hr.method == fr.method);
    REQUIRE(hr.target == fr.target);
    REQUIRE(hr.path == fr.path);
    REQUIRE(hr.query == fr.query);
    REQUIRE(hr.minor_version == fr.minor_version);
    REQUIRE(hr.is_head == fr.is_head);
    REQUIRE(hr.keep_alive == fr.keep_alive);  // both see Connection: close
    REQUIRE_FALSE(hr.keep_alive);
    REQUIRE(hr.headers == fr.headers);
    REQUIRE(hr.body.empty());      // head-only
    REQUIRE(fr.body == "hi");      // full parse read the body
}

// ---- i#70: chunked body resumes across feeds via ParseContext ----

namespace {
// Feeds `req` to parse_request in growing prefixes `step` bytes at a time
// through ONE ParseContext (the connection-loop pattern: the buffer grows,
// the context persists across kNeedMore). Returns the feed that completes
// (or the last feed) and fills `out` from it.
locus::http::ParseResult feed_in_steps(std::string_view req, Request& out,
                                       std::size_t step, Limits limits = {}) {
    locus::http::ParseContext ctx;
    locus::http::ParseResult r;
    for (std::size_t n = step;; n += step) {
        if (n > req.size()) {
            n = req.size();
        }
        r = parse_request(req.substr(0, n), limits, ctx, out);
        if (r.state != ParseState::kNeedMore || n == req.size()) {
            return r;
        }
    }
}
}  // namespace

TEST_CASE("parse_request: a chunked body resumes to the same result as a "
          "whole parse (i#70)", "[http]") {
    const std::string req =
        "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
        "4\r\nWiki\r\n5\r\npedia\r\n3\r\n123\r\n0\r\nX-T: v\r\n\r\n";
    Request whole;
    locus::http::ParseContext wc;
    auto w = parse_request(req, Limits{}, wc, whole);
    REQUIRE(w.state == ParseState::kComplete);
    REQUIRE(whole.body == "Wikipedia123");

    // Incremental through ONE context, at several step sizes -- the resume
    // must land on the same body, consumed count and state as the whole.
    for (std::size_t step : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                             std::size_t{7}}) {
        Request inc;
        auto r = feed_in_steps(req, inc, step);
        REQUIRE(r.state == ParseState::kComplete);
        REQUIRE(r.consumed == w.consumed);
        REQUIRE(inc.body == whole.body);
    }
}

TEST_CASE("parse_request: chunked resume over many small chunks is correct",
          "[http]") {
    // 100 one-byte chunks: every chunk boundary falls on a separate feed
    // when fed one byte at a time, exercising the resume checkpoint path.
    std::string req =
        "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n";
    std::string expect;
    for (int i = 0; i < 100; ++i) {
        const char c = static_cast<char>('a' + (i % 26));
        req += "1\r\n";
        req += c;
        req += "\r\n";
        expect += c;
    }
    req += "0\r\n\r\n";

    Request inc;
    auto r = feed_in_steps(req, inc, 1);  // one byte per feed
    REQUIRE(r.state == ParseState::kComplete);
    REQUIRE(r.consumed == req.size());
    REQUIRE(inc.body == expect);
}
