#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace locus::http {

/**
 * First-party HTTP/1.1 request parser + response writer (issue 23),
 * replacing the vendored httplib. This module is PURE: it never touches
 * a socket. The parser is fed bytes the caller has read and reports how
 * many it consumed, so the connection loop (increment 2) owns all I/O
 * and can parse the next request from the same buffer on a keep-alive
 * connection. Being socket-free is also what makes it fuzzable.
 *
 * It parses only what the API server needs and rejects the rest: a
 * malformed or hostile request is reported as an error (the caller
 * answers 4xx and closes the connection), never undefined behaviour.
 */

/** Caps on an incoming request; exceeding any is a parse error. They
 * also bound how much the parser buffers while waiting for more bytes,
 * so an attacker cannot make it hold an unbounded amount by never
 * sending the terminator. */
struct Limits {
    std::size_t max_request_line = 8 * 1024;    /**< METHOD SP target SP
                                                 *   version CRLF. */
    std::size_t max_header_line = 8 * 1024;     /**< one header line. */
    std::size_t max_header_count = 100;         /**< number of headers. */
    std::size_t max_header_bytes = 64 * 1024;   /**< all headers total. */
    std::size_t max_body_bytes = 8 * 1024 * 1024;  /**< decoded body. */
    std::size_t max_trailer_bytes = 4 * 1024;   /**< chunked trailers. */
};

/** A parsed HTTP/1.1 request. Plain data; the connection layer wraps it
 * in the handler-facing shim (increment 2). Header lookups are
 * case-insensitive per RFC 9110. */
struct Request {
    std::string method;        /**< e.g. "GET", "POST", "HEAD". */
    std::string target;        /**< raw request-target (path?query). */
    std::string path;          /**< target with any query removed. */
    std::string query;         /**< after '?', without it; "" if none. */
    int minor_version = 1;     /**< HTTP/1.<minor>: 0 or 1. */
    bool is_head = false;      /**< method == "HEAD" (no response body). */
    bool keep_alive = true;    /**< connection reuse (see the parser). */
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;

    /** @returns true if a header named `name` is present (case-
     * insensitive). */
    bool has_header(std::string_view name) const;
    /** @returns the first value of header `name`, or "" if absent
     * (case-insensitive). */
    std::string get_header_value(std::string_view name) const;
};

/** Parser outcome. kNeedMore: the buffer holds a prefix of a request,
 * feed more; nothing is consumed. kComplete: a full request is parsed
 * and `consumed` bytes of the buffer were it. kError: malformed or over
 * a limit; the caller must answer 4xx and close (never keep-alive). */
enum class ParseState { kNeedMore, kComplete, kError };

struct ParseResult {
    ParseState state = ParseState::kNeedMore;
    std::size_t consumed = 0;  /**< bytes forming the request (kComplete).
                                *   The caller erases them; any bytes
                                *   after are the next request on a
                                *   keep-alive connection. */
    std::string error;         /**< diagnostic when state == kError. */
};

/**
 * Parses ONE request from the front of `buf`. Pure: `buf` is not
 * modified; on kComplete the caller drops `result.consumed` bytes. On
 * kNeedMore the caller appends more bytes and calls again (feeding the
 * whole accumulated buffer each time). Enforces every Limit, including
 * while still kNeedMore, so a client that never sends the terminator is
 * bounded.
 *
 * Hardening (issue 23): CRLF required on every line (a bare LF is an
 * error); obs-fold continuation lines rejected; Content-Length guarded
 * against sign/whitespace/overflow and rejected if duplicated; a request
 * bearing both Content-Length and Transfer-Encoding rejected (request
 * smuggling); Transfer-Encoding matched as a case-insensitive list whose
 * sole, final member must be `chunked`; chunk sizes overflow-guarded,
 * chunk-extensions rejected, trailers bounded; only HTTP/1.0 and 1.1
 * accepted.
 *
 * @param buf The bytes read so far.
 * @param limits Size/count caps.
 * @param out Filled only when the result is kComplete.
 * @returns the parse state, bytes consumed, and an error diagnostic.
 */
ParseResult parse_request(std::string_view buf, const Limits& limits,
                          Request& out);

/** A response to serialize. `headers` must not contain Content-Length or
 * Transfer-Encoding; the writer sets framing itself. */
struct Response {
    int status = 200;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;

    /** Appends a header. */
    void set_header(std::string name, std::string value);
    /** Replaces the body and (implicitly) its Content-Length. */
    void set_content(std::string content, std::string content_type);
};

/**
 * Serializes `res` as a complete HTTP/1.1 response into `out`
 * (appended). Sets Content-Length from the body and a Connection header
 * from `keep_alive`. For a HEAD request the body is omitted but its
 * Content-Length is still reported, per RFC 9110.
 *
 * Rejects (returns false, sets `err`, writes nothing) any header name or
 * value containing CR, LF, or NUL -- the response-splitting / header-
 * injection guard, since header values can carry data derived from the
 * request. A name or status outside the writable set is likewise an
 * error rather than emitted raw.
 *
 * @returns true on success; false (with `err`) if a header is unsafe.
 */
bool write_response(const Response& res, bool head_request,
                    bool keep_alive, std::string& out, std::string& err);

/** @returns the reason phrase for a status code ("OK", "Not Found", ...),
 * or "" if the code is not one this server emits. */
std::string status_reason(int status);

/** Encodes one chunked-transfer chunk -- `<hex size>CRLF<data>CRLF` --
 * for SSE streaming (increment 2's DataSink). Empty `data` is NOT the
 * terminator; use last_chunk() for that. */
std::string encode_chunk(std::string_view data);

/** @returns the terminating chunk `0CRLFCRLF` (no trailers). */
std::string last_chunk();

}  // namespace locus::http
