#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "locus/http/http.hpp"

namespace locus::http {

/**
 * First-party HTTP/1.1 client (issue 388, the router's upstream side).
 * Like the request parser, the response parser below is PURE: fed the
 * bytes read so far, it reports how many form the response, so the
 * client owns the socket and the parser can be tested and fuzzed alone.
 */

/** Caps on a response; exceeding any is a parse error. */
struct ResponseLimits {
    std::size_t max_status_line = 8 * 1024;   /**< HTTP/1.x SP code ... */
    std::size_t max_header_line = 8 * 1024;   /**< one header line. */
    std::size_t max_header_count = 100;       /**< number of headers. */
    std::size_t max_header_bytes = 64 * 1024; /**< all headers total. */
    std::size_t max_body_bytes = 64 * 1024 * 1024; /**< decoded body. */
};

/** How a response body is delimited (RFC 9112 section 6.3). */
enum class BodyFraming {
    kNone,     /**< no body: HEAD, 1xx, 204, 304. */
    kLength,   /**< Content-Length bytes. */
    kChunked,  /**< Transfer-Encoding: chunked. */
    kClose,    /**< everything until the server closes. */
};

/** A parsed response head. Header lookups are case-insensitive. */
struct ResponseHead {
    int status = 0;
    std::string reason;
    int minor_version = 1;
    std::vector<std::pair<std::string, std::string>> headers;
    BodyFraming framing = BodyFraming::kNone;
    std::size_t content_length = 0;  /**< for kLength. */
    /** The server keeps the connection open after this response. */
    bool keep_alive = true;

    /** @returns the first value of header `name`, or "" if absent. */
    std::string get_header_value(std::string_view name) const;
};

/**
 * Parses a response HEAD (through the blank line) from the front of
 * `buf`. Pure: `buf` is not modified. kNeedMore until the blank line;
 * kComplete with `consumed` through the CRLFCRLF; kError on a malformed
 * head or one over a limit.
 *
 * Hardening, as parse_request: CRLF on every line, no obs-fold, header
 * values restricted to field-content, Content-Length guarded against
 * signs, spaces, overflow and duplicates, both Content-Length and
 * Transfer-Encoding rejected, and Transfer-Encoding accepted only as a
 * sole `chunked`.
 *
 * @param buf The bytes read so far.
 * @param head_request The request was HEAD, so no body follows.
 * @param limits Size and count caps.
 * @param out Filled on kComplete.
 */
ParseResult parse_response_head(std::string_view buf, bool head_request,
                                const ResponseLimits& limits,
                                ResponseHead& out);

/**
 * Decodes a chunked body as it arrives, so a stream (SSE) is passed on
 * chunk by chunk. Feed it the bytes after the head; it hands each
 * chunk's data to the sink and reports when the last chunk and the
 * trailers are through.
 */
class ChunkDecoder {
  public:
    explicit ChunkDecoder(std::size_t max_body_bytes = 64 * 1024 * 1024)
        : max_body_(max_body_bytes) {}

    /** Outcome of one feed. */
    enum class State { kNeedMore, kDone, kError };

    /**
     * Decodes as much of `in` as forms whole pieces (a size line, data,
     * the CRLF after it, the trailers). Bytes that do not complete a
     * piece stay in the decoder for the next feed.
     * @param in New bytes.
     * @param sink Receives each run of decoded data, in order.
     * @param[out] rest On kDone, the bytes after the body (the next
     *     response on a kept-alive connection).
     * @returns kDone after the last chunk and the trailers, kError on a
     *     malformed chunk or past max_body_bytes, else kNeedMore.
     */
    State feed(std::string_view in,
               const std::function<void(std::string_view)>& sink,
               std::string& rest);

    /** @returns a diagnostic after kError. */
    const std::string& error() const { return error_; }

  private:
    enum class Step { kSize, kData, kDataEnd, kTrailers, kDone };
    Step step_ = Step::kSize;
    std::string buf_;
    std::size_t left_ = 0;   ///< data bytes left in this chunk
    std::size_t total_ = 0;  ///< decoded bytes so far
    std::size_t max_body_;
    std::string error_;
};

/** One request to send. The client sets Host, Content-Length and
 * Connection itself. */
struct ClientRequest {
    std::string method = "GET";
    std::string target = "/";
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;
};

/** A buffered response, or how a request failed. */
struct ClientResponse {
    /** True when a whole response was read; false on a connect, write,
     * read or parse failure (see error), with status 0. */
    bool ok = false;
    std::string error;
    /** A connection to the server was made (false: the connect itself
     * failed, so the server is down or unreachable). */
    bool connected = false;
    ResponseHead head;
    std::string body;  ///< empty when streamed to a sink
};

/** Time limits, in milliseconds; 0 means no limit. */
struct ClientOptions {
    int connect_timeout_ms = 5000;
    /** Longest gap between two reads (not the whole response, which
     * for a video can take minutes). */
    int read_timeout_ms = 60000;
    /** Longest wait for the server to take more request bytes: a bound
     * on each stall, not on the whole write (a server that takes a
     * byte now and then keeps a large body going). */
    int write_timeout_ms = 60000;
    /** Longest time for one whole request, from begin() through the
     *  last body byte; 0 means none (a proxied stream can run long).
     *  A server that sends a byte now and then stays under the read
     *  timeout, not under this. */
    int total_timeout_ms = 0;
    ResponseLimits limits;
};

/**
 * A blocking HTTP/1.1 client for one host:port, reusing one kept-alive
 * connection across requests. Before a request goes on a reused
 * connection, a close by the server while it sat idle is noticed and a
 * new connection is opened; a request is sent again after that only
 * when it is idempotent (not a POST), so a reply cut short is never a
 * second run on the server. Not thread-safe: one per thread (the
 * router keeps a pool of them).
 */
class Client {
  public:
    Client(std::string host, int port, ClientOptions opt = {});
    ~Client();
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    /**
     * Sends `req` and reads the whole response into the result.
     */
    ClientResponse send(const ClientRequest& req);

    /** Sets the read timeout (ClientOptions::read_timeout_ms) for the
     *  next requests: a pooled connection serves requests of different
     *  lengths. */
    void set_read_timeout_ms(int ms) { opt_.read_timeout_ms = ms; }

    /**
     * Sends `req` and passes the body to `on_body` as it arrives (each
     * chunk of a chunked body, or each read of a Content-Length or
     * close-delimited one), after `on_head` has seen the head. Either
     * callback may return false to stop: the connection is then closed,
     * which tells the server the client went away.
     * @returns the head and ok, with an empty body.
     */
    ClientResponse stream(
        const ClientRequest& req,
        const std::function<bool(const ResponseHead&)>& on_head,
        const std::function<bool(std::string_view)>& on_body);

    /**
     * The first half of a request: sends `req` and reads the response
     * head, so a proxy can answer with the upstream status before the
     * body arrives. Follow it with read_body() (or close()) before the
     * next request.
     * @returns ok and the head, with an empty body.
     */
    ClientResponse begin(const ClientRequest& req);

    /**
     * The second half: reads the body of the response begin() started,
     * passing each piece to `on_body`, which may return false to stop
     * (the connection is then closed).
     * @param[out] err Set on failure.
     * @returns true when the whole body was read.
     */
    bool read_body(const std::function<bool(std::string_view)>& on_body,
                   std::string& err);

    /** Closes the kept-alive connection, if any. */
    void close();

  private:
    bool connect(std::string& err);
    /** @returns whether the kept-alive connection is closed (or has
     * stray bytes) on the server's side, without blocking. */
    bool stale() const;
    /** Sends all of `data`, waiting at most write_timeout_ms each time
     * the server takes no more. */
    bool write_all(std::string_view data, std::string& err);
    /** Reads into buf_; @returns bytes read, 0 at EOF, -1 on error or
     * timeout (err set). */
    long read_some(std::string& err);
    /**
     * @param ms A wait's own cap (0: none).
     * @returns the poll timeout for that wait, cut to what is left of
     *     total_timeout_ms; -1 for no limit, 0 once the total ran out.
     */
    int wait_ms(int ms) const;
    /** @returns whether total_timeout_ms ran out for this request. */
    bool expired() const;
    /** @returns whether a wait of `t` (from wait_ms(ms)) was cut by the
     *  total, so its timeout is the request's, not the wait's own. */
    bool total_bound(int ms, int t) const;

    std::string host_;
    int port_;
    ClientOptions opt_;
    int fd_ = -1;
    std::string buf_;  ///< read but not yet parsed bytes
    /** The head of the response begin() started; framing kNone once
     * its body is read. */
    ResponseHead pending_;
    bool active_ = false;
    /** When the current request's total_timeout_ms runs out. */
    std::chrono::steady_clock::time_point deadline_{};
};

}  // namespace locus::http
