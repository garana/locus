#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
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

/**
 * Per-request state the head stage produces for the handler: whatever a
 * HeadHandler decides on the head that the Handler needs later (today,
 * the authenticated identity for the audit event). One per request.
 */
struct RequestContext {
    std::string identity;  /**< set by the head handler; read by Handler. */
    /**
     * Client-liveness probe the server installs before dispatch:
     * returns true while the peer is still connected, false once it
     * has half-closed. A long-running Handler can poll this to abandon
     * work for a client that has gone away. Empty for callers that do
     * not install one (treat as always-connected).
     *
     * Limitation: TCP gives the server no way to distinguish an
     * orderly half-close (a client that shut its write side but still
     * waits for the response) from an abandoned connection -- both
     * arrive as FIN. This probe reports BOTH as gone, so a client must
     * not half-close its request before reading the response or its
     * work may be cancelled. A handler acting on this should emit a
     * diagnosable status/body (not a default empty 200) in case the
     * detection was wrong.
     *
     * Deliberately NOT used by the streaming (SSE) response path: a
     * streamed handler has stronger evidence than this probe -- the Sink
     * write itself. A write that fails is proof the client is gone, so
     * the stream cancels on that and need not poll client_alive; and a
     * client that merely half-closed still has its read side open to
     * receive the stream, so cancelling it on a FIN would be wrong. The
     * consequence (a design choice, not an oversight): the same
     * half-close yields a 499 on a non-streaming request but a full
     * stream on a streaming one. Do not "fix" this by adding a liveness
     * check to the stream; the write-failure signal is the correct one
     * there.
     */
    std::function<bool()> client_alive;
};

/**
 * @returns true when the peer on `fd` has half-closed (orderly EOF) or
 * the socket is in error, detected WITHOUT consuming any pending
 * request bytes (0-timeout poll + MSG_PEEK). Pending inbound data or a
 * would-block both read as "still connected", so a pipelined next
 * request never trips it. Basis for RequestContext::client_alive.
 */
bool peer_half_closed(int fd);

/**
 * Runs once the request HEAD is buffered, BEFORE the body is read and
 * before any 100-continue -- so auth and policy decide on the head and a
 * rejected request (e.g. a 401) never buffers its body.
 *
 * Return true to REJECT: fill `res` with the full response (status AND
 * body, so an error keeps its JSON shape); the server writes it and
 * closes. Return false to PROCEED: `res` is DISCARDED (do NOT stage
 * response state here -- anything that must reach the response goes
 * through `ctx`), the body is read, and the Handler runs with the same
 * `ctx`. `req` is a head-only parse: its body is empty.
 */
using HeadHandler = std::function<bool(const Request& head,
                                       ServerResponse& res,
                                       RequestContext& ctx)>;

/** Fills `res` from `req` (and the head stage's `ctx`), on the
 * connection's worker thread. */
using Handler = std::function<void(const Request& req, ServerResponse& res,
                                   const RequestContext& ctx)>;

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
    /** Routes served on the fast-path acceptor thread rather than the
     * inference worker pool, so cheap control endpoints (/health,
     * /metrics) are not starved when every worker is busy in a long
     * generation. Matched on the request line, peeked without consuming;
     * each fast-path connection is served exactly ONE request and closed,
     * so a keep-alive connection can never carry a later (inference)
     * request into the acceptor. Empty (default) = no fast path; every
     * request goes to the pool. The predicate must be cheap and must not
     * block.
     *
     * Scope: this removes starvation by a busy worker POOL. A fast route
     * can still queue behind the acceptor's own per-connection peek -- a
     * client that connects and sends nothing holds the single acceptor
     * for up to the peek window. Making classification event-driven so
     * one slow client cannot block the acceptor is tracked in i#88. */
    std::function<bool(std::string_view method, std::string_view target)>
        fast_path;
};

/**
 * Owns a listening socket and a background accept loop. Thread-safe to
 * stop() from any thread; the destructor stops. Not copyable.
 */
class Server {
 public:
    /** @param head_handler optional; runs on the head before the body
     * (auth/policy). Null = no head stage. */
    Server(ServerConfig cfg, Handler handler, HeadHandler head_handler = {});
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
    /** Blocks until stop() is called (from any thread). Lets a caller
     * turn the background accept loop into a serve-forever call. Returns
     * immediately if already stopping. */
    void wait();

 private:
    void worker_loop();
    /** Sole acceptor: accepts connections, and for a fast_path route
     * (peeked without consuming) serves it inline here so it never
     * waits on the inference workers; every other connection is pushed
     * to the handoff queue for the worker pool. */
    void acceptor_loop();
    /** Accepts one connection and either hands it straight to the pool
     * (no fast_path configured, or no room to classify) or adds it to
     * the classification poll set. Never blocks on the new peer. */
    void accept_one_conn();
    /** Classifies an unclassified connection the poll set has reported
     * readable: a matching fast_path route is served inline (one
     * request, then closed); anything else goes to the pool. */
    void classify_ready(int fd);
    /** Hands every unclassified connection whose classification
     * deadline has passed to the pool, which is where a client that has
     * not sent a request line belongs (its head deadline rules there). */
    void expire_unclassified();
    /** Drops `fd` from the unclassified list. The readiness set is a
     * pollfd array rebuilt from that list each iteration, so there is
     * no persistent registration to unwind and no stale-registration or
     * recycled-fd-number hazard. */
    void unwatch_unclassified(int fd);
    /** Pushes one accepted connection onto the worker handoff queue and
     * wakes a worker. @returns false when the queue is at its cap, in
     * which case the caller must close `fd` (shed, never accumulate). */
    bool enqueue_handoff(int fd, const std::string& peer_ip);
    /** Peeks (MSG_PEEK, never consumes) the request line on `fd` and
     * extracts method + target. The caller must only call this for an
     * fd already reported readable: there is no poll inside, because a
     * blocking per-connection poll here serialized the sole acceptor
     * (i#88). @returns true if a full request line was read; false
     * (treat as non-fast) on a short/garbled head or a read error. */
    bool peek_request_line(int fd, std::string& method, std::string& target);
    /** @param single_request when true, serve at most ONE request and
     * close (forces keep-alive off). The acceptor uses it for fast-path
     * connections so an adopted keep-alive connection can never carry a
     * later inference request into the acceptor thread. */
    void serve_connection(int fd, const std::string& peer_ip,
                          bool single_request = false);
    /** Runs the handler and writes its response (buffered or streamed).
     * @returns true iff the connection may be kept alive for the next
     * request (write succeeded AND `keep`). */
    bool handle_request(int fd, const Request& req, const RequestContext& ctx,
                        bool keep);
    /** Serializes a buffered ServerResponse (status + headers + body) and
     * writes it; a HEAD omits the body. On a framing/unsafe header it
     * sends a 500 and returns false. @returns true iff the write
     * succeeded. */
    bool write_buffered_response(int fd, const ServerResponse& res,
                                 bool head_request, bool keep);
    /** Writes a minimal status-only response (the reason as a text body).
     * @returns true iff the write succeeded. */
    bool write_status_response(int fd, bool head_request, int status,
                               bool keep);

    ServerConfig cfg_;
    Handler handler_;
    HeadHandler head_handler_;  /**< optional; null = no head stage. */
    int listen_fd_ = -1;
    int bound_port_ = -1;
    int wake_r_ = -1;         /**< self-pipe read end (shutdown signal). */
    int wake_w_ = -1;         /**< self-pipe write end. */
    std::vector<std::thread> workers_;
    std::thread acceptor_;            /**< sole acceptor + fast-path. */
    /** One accepted connection awaiting a pool worker: its fd and the
     * peer IP (carried so pooled connections see the real client for
     * any future peer-IP audit, not an empty string). */
    struct Pending {
        int fd;
        std::string peer_ip;
    };
    std::mutex handoff_mu_;           /**< guards handoff_. */
    std::condition_variable handoff_cv_; /**< acceptor -> workers. */
    std::deque<Pending> handoff_;     /**< accepted conns for the pool. */
    /**
     * One accepted connection not yet classified fast vs pool. The
     * acceptor watches it for readability alongside the listen and wake
     * fds and peeks it only once it HAS data, so a client that connects
     * and sends nothing costs the acceptor nothing but a slot (i#88).
     * `deadline` bounds that wait: past it the connection goes to the
     * pool, the same decision the old blocking peek made on timeout,
     * without holding the thread for it.
     */
    struct Unclassified {
        int fd;
        std::string peer_ip;
        std::chrono::steady_clock::time_point deadline;
    };
    /** Acceptor-thread only (no lock): the acceptor is the sole owner,
     * and stop() drains this only after joining it. The readiness set
     * passed to poll() is rebuilt from this each iteration (listen fd,
     * wake fd, then these), so it holds no kernel-side registration. */
    std::vector<Unclassified> unclassified_;
    std::atomic<bool> stopping_{false};
    std::atomic<bool> started_{false};
    std::mutex wait_mu_;              /**< guards the wait() handoff. */
    std::condition_variable wait_cv_; /**< signalled by stop(). */
};

}  // namespace locus::http
