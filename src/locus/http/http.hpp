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
    int suggested_status = 400; /**< status the caller should answer on
                                 *   kError: 400 malformed, 413 over a
                                 *   size cap, 501 unimplemented method,
                                 *   505 unsupported version. Lets the
                                 *   connection loop map the condition
                                 *   rather than match `error` text. */
};

/**
 * Per-connection parse state the caller threads across feeds of one
 * request. The caller holds one per connection and passes it on every
 * feed; it does NOT have to reset it -- parse_request clears the context
 * itself on any terminal result (kComplete/kError), keeping state only
 * across kNeedMore, so stale state cannot leak into the next request on
 * a keep-alive connection.
 *
 * It exists to pin the parse API before the connection loop (increment
 * 2) is written against it: the current implementation re-parses from
 * the front of the buffer on every feed (simple and correct, but
 * quadratic in the number of reads for a large chunked body, bounded by
 * max_body_bytes). Issue 70 fills this with resume state so the re-scan
 * is removed WITHOUT changing the loop's signature. Today it carries
 * nothing; pass a default-constructed value, one per connection. */
struct ParseContext {};

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
 * sole, final member must be `chunked`, and a repeated Transfer-Encoding
 * header rejected (two `chunked` lines are the list `chunked, chunked`);
 * chunk sizes overflow-guarded, chunk-extensions rejected, trailers
 * bounded; only HTTP/1.0 and 1.1 accepted; the method restricted to the
 * implemented set (GET, POST, HEAD) so a router cannot desync on an
 * unexpected verb; header values restricted to RFC 9110 field-content
 * (VCHAR, SP, HTAB, obs-text 0x80-0xFF) so a control byte -- NUL in
 * particular, which truncates a `%s` log line -- cannot pass to a later
 * consumer.
 *
 * @param buf The bytes read so far.
 * @param limits Size/count caps.
 * @param ctx Per-connection resume state; one per connection, reset to a
 *   fresh value after each kComplete (see ParseContext).
 * @param out Filled only when the result is kComplete.
 * @returns the parse state, bytes consumed, a suggested error status,
 *   and an error diagnostic.
 */
ParseResult parse_request(std::string_view buf, const Limits& limits,
                          ParseContext& ctx, Request& out);

/**
 * Parses ONLY the request HEAD (through the blank line), for the
 * connection server's head-stage hook -- so auth and caps can decide
 * before the body is read and a 401 need not cost a full upload.
 *
 * On kComplete `consumed` covers through the CRLFCRLF and `out` holds
 * method, target, path, query, minor_version, is_head, keep_alive and
 * headers; out.body is EMPTY and must not be read as meaningful (the body
 * has not arrived). kNeedMore until the blank line; kError (with a
 * suggested status) on a malformed head. Only the head caps apply
 * (max_request_line, max_header_line, max_header_count, max_header_bytes),
 * not max_body_bytes.
 *
 * This is the SAME head phase parse_request runs (one shared
 * implementation, never a second head parser), so every head hardening
 * rule -- duplicate Content-Length, the sole-chunked Transfer-Encoding
 * list and repeated-TE rejection, obs-fold rejection, the CTL-in-value
 * rule, and the CL+TE smuggling rejection -- has exactly one home.
 *
 * Unlike parse_request this takes no ParseContext, by design: the head
 * is bounded by max_header_bytes, so re-scanning it from the front on
 * every read is negligible (about that bound times the read count). The
 * context exists for the UNBOUNDED body, whose re-scan is quadratic
 * (i#70); the head never needs it.
 *
 * @param buf The bytes read so far.
 * @param limits Size/count caps (head caps only apply).
 * @param out Filled on kComplete; body left empty.
 * @returns the parse state, bytes consumed through the head, a suggested
 *   error status, and an error diagnostic.
 */
ParseResult parse_head(std::string_view buf, const Limits& limits,
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

/**
 * Serializes the HEAD of a CHUNKED (streaming) response into `out`
 * (appended): status line, the caller's headers, `Transfer-Encoding:
 * chunked`, and a Connection header from `keep_alive`. The body then
 * follows as encode_chunk() calls terminated by last_chunk(). Used for
 * SSE, where the length is not known up front.
 *
 * `res.body` is ignored (a stream has none). The same injection guard as
 * write_response applies: a CR/LF/NUL in any header name/value, or a
 * caller-set framing header (Content-Length / Transfer-Encoding /
 * Connection), is an error (returns false, sets `err`, writes nothing).
 *
 * @returns true on success; false (with `err`) if a header is unsafe.
 */
bool write_chunked_head(const Response& res, bool keep_alive,
                        std::string& out, std::string& err);

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
