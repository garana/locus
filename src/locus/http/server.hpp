#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "locus/http/http.hpp"
#include "locus/pipeline/net.hpp"

namespace locus::http {

/**
 * First-party HTTP/1.1 connection server (issue 23, increment 2),
 * replacing the vendored httplib transport. It owns sockets; the parser
 * (increment 1) owns bytes. Model (A): one thread per connection, so a
 * handler may block (the engine does) without starving others. No routes
 * and no handler changes live here -- increment 3 ports the API routes
 * onto this, increment 4 makes httplib a tests-only dependency.
 *
 * Bounds, all explicit (httplib supplied these as defaults nobody on
 * this project chose; removing it removes them, so they are re-chosen):
 * a fixed worker pool whose size IS the concurrency cap (excess
 * connections wait unread in the kernel's accept backlog, so they hold
 * no userspace buffer -- httplib's shape), a per-connection request cap,
 * a head-receipt deadline, an idle-keep-alive timeout, and a per-write
 * deadline. There is deliberately NO whole-connection timeout: a long
 * streaming response is legitimate as long as each write makes progress
 * (see ServerConfig).
 *
 * Worst-case buffered request memory is worker_threads x
 * limits.max_body_bytes (only a serving worker holds a body; queued
 * connections sit in the kernel). Budget it against the same RAM the
 * KV-cache guard reasons about.
 *
 * The cost of the pool shape: these bounds stop STALLED clients, not
 * SLOW ones. A client that keeps making progress just above the minimum
 * rates (ingest and per-write) holds its worker for as long as it keeps
 * progressing, so the real bound on concurrent slow clients is
 * worker_threads itself -- worker_threads slow-but-legal connections can
 * occupy every worker and make others wait in the backlog. httplib had
 * the same exposure with the same pool shape; it is inherent to model
 * (A) and not removable with per-request bounds (an event loop would
 * change it).
 */

/**
 * Streaming output handed to a chunked response provider (SSE). One
 * write() per event; this class owns the chunk framing, the handler does
 * not. Backpressure is the return value: a slow or gone client trips the
 * write deadline and write() returns false, at which point the provider
 * must stop producing.
 */
class Sink {
 public:
    virtual ~Sink() = default;
    /** Sends `data` as one chunk. @returns false if the client tripped
     * the write deadline, closed, or the server is shutting down; the
     * provider must then return. */
    virtual bool write(std::string_view data) = 0;
    /** Sends the terminating chunk. Idempotent; the loop also calls it
     * once after the provider returns, so a handler need not. */
    virtual void done() = 0;
};

/** The provider a handler installs to stream a response body. */
using StreamProvider = std::function<void(Sink&)>;

/**
 * A response the handler fills: EITHER a buffered body (set_content) or a
 * chunked stream (set_chunked_content_provider), not both. The writer
 * owns framing; do not set Content-Length / Transfer-Encoding /
 * Connection here (doing so is a 500, like the parser's writer).
 */
struct ServerResponse {
    int status = 200;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;
    StreamProvider provider;  /**< set => stream instead of body. */

    /** Appends a header. */
    void set_header(std::string name, std::string value);
    /** Sets a buffered body and its Content-Type. */
    void set_content(std::string content, std::string content_type);
    /** Streams the body: emits `content_type`, then `provider` writes
     * chunks via the Sink. */
    void set_chunked_content_provider(std::string content_type,
                                      StreamProvider provider);
};

/** Fills `res` from `req`, on the connection's worker thread. */
using Handler = std::function<void(const Request& req, ServerResponse& res)>;

/** Server configuration; every bound is explicit (see the class note). */
struct ServerConfig {
    std::string host = "0.0.0.0";   /**< "" / "0.0.0.0" == wildcard. */
    int port = 8080;                /**< 0 == OS-assigned ephemeral. */
    /** Worker threads; this IS the concurrency cap. 0 == auto
     * (max(8, ncpu - 1), capped). Worst-case buffered request memory is
     * this x limits.max_body_bytes -- budget it against host RAM. */
    int worker_threads = 0;
    /** Kernel accept backlog: connections beyond worker_threads wait
     * here, unread, holding no userspace buffer; beyond this the kernel
     * refuses the connection (no 503, no parse, no read). */
    int listen_backlog = 128;
    int max_requests_per_conn = 100;  /**< then close (keep-alive). */
    /** Flat wall deadline for receiving the request HEAD (first byte to
     * the blank line), as a deadline NOT an inter-byte timer (a client
     * dribbling 1 B / N s must still trip it). The head has no legitimate
     * size, so a flat bound is right here. It also serves as the body
     * phase's base grace (see min_ingest_bytes_per_sec). */
    int head_deadline_ms = 10000;
    /** Minimum sustained body upload rate. Once the head is in, the body
     * deadline is head_deadline_ms of grace plus credit earned at this
     * rate, clamped so a burst cannot bank time: a client holding above
     * this rate is never cut, one below it (a dribbler, a staller) trips.
     * A flat body deadline would instead impose max_body_bytes / deadline
     * as a FLOOR rate and fail a slow-but-honest upload. 0 disables the
     * body-rate check. Works for chunked too (no declared length needed).
     * A streaming RESPONSE is never bounded here -- that is the per-write
     * deadline's job. */
    int min_ingest_bytes_per_sec = 32 * 1024;
    int idle_timeout_ms = 5000;     /**< idle keep-alive connection. */
    int write_deadline_ms = 10000;  /**< a single BLOCKED write; not a
                                     *   produce-every-N timer (prefill
                                     *   may exceed it legitimately). */
    Limits limits;                  /**< parser caps. */
    std::vector<locus::pipeline::Cidr> allow;  /**< empty == allow all. */
};

/**
 * Owns a listening socket and a background accept loop. Thread-safe to
 * stop() from any thread; the destructor stops. Not copyable.
 */
class Server {
 public:
    Server(ServerConfig cfg, Handler handler);
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    /** Binds host:port and starts accepting on a background thread.
     * @returns the bound port (> 0), or -1 on bind/listen failure. */
    int start();
    /** Stops accepting, wakes and drains every worker promptly (a self-
     * pipe watched by each poll), closes the listen socket, and returns
     * once all workers have exited. Safe to call more than once. */
    void stop();

 private:
    void worker_loop();
    void serve_connection(int fd, const std::string& peer_ip);
    /** Runs the handler and writes its response (buffered or streamed).
     * @returns true iff the connection may be kept alive for the next
     * request (write succeeded AND `keep`). */
    bool handle_request(int fd, const Request& req, bool keep);
    /** Writes a minimal status-only response (the reason as a text body).
     * @returns true iff the write succeeded. */
    bool write_status_response(int fd, bool head_request, int status,
                               bool keep);

    ServerConfig cfg_;
    Handler handler_;
    int listen_fd_ = -1;
    int bound_port_ = -1;
    int wake_r_ = -1;         /**< self-pipe read end (shutdown signal). */
    int wake_w_ = -1;         /**< self-pipe write end. */
    std::vector<std::thread> workers_;
    std::atomic<bool> stopping_{false};
    std::atomic<bool> started_{false};
};

}  // namespace locus::http
