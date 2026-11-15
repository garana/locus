#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "locus/http/http.hpp"
#include "locus/sys/poller.hpp"

namespace locus::http {

/**
 * The connections whose replies have news for the loop (i#990), so a
 * wake visits only those: a reply pushes its connection's id, and the
 * first push after a take() writes one byte to the loop's wake fd.
 * Thread-safe; ids are never reused, so a closed connection's late
 * push names no other client.
 */
class WakeList {
  public:
    /** @param wake_fd The loop's wake pipe (non-blocking write end). */
    explicit WakeList(int wake_fd) : wake_fd_(wake_fd) {}

    /** Marks connection `id` as having news and wakes the loop. */
    void push(std::uint64_t id);
    /** @returns The ids pushed since the last take, each once (any
     *  order). Read the wake fd first: a push after that read is
     *  either in this take or wakes the loop again. */
    std::vector<std::uint64_t> take();

  private:
    const int wake_fd_;
    std::mutex mu_;
    std::vector<std::uint64_t> ids_;
};

/** Bounds of one LoopConnection; the same meaning as the matching
 * ServerConfig fields. */
struct LoopConfig {
    Limits limits;                    /**< parser caps. */
    int max_requests_per_conn = 100;  /**< then close (keep-alive). */
    /** Flat deadline for a whole request, head and body, from its first
     * byte; a miss is a 408 and a close. A flat bound suits the small
     * GETs a loop serves; http::Server's body-rate credit is not needed
     * for them. */
    int request_deadline_ms = 10000;
    /** A connection with no request under way (before the first, or
     * between two) closes silently after this. */
    int idle_timeout_ms = 5000;
    /** No write progress for this long closes the connection. */
    int write_deadline_ms = 10000;
    /** Deferred replies (LoopReply) write a byte here when they have
     *  something for the loop, which then calls on_wake(); -1 for a
     *  loop that only answers at once. Non-blocking, owned by the
     *  loop, open for as long as any reply can be written. */
    int wake_fd = -1;
    /** When set, replies wake the loop through this instead (pushing
     *  wake_id), so the loop knows which connection to look at. */
    std::shared_ptr<WakeList> wake_list;
    std::uint64_t wake_id = 0;
    /** Most bytes of a stream queued and not yet on the socket; a
     *  producer waits while it is over this (backpressure). */
    std::size_t max_stream_buffer = 1 << 20;
};

/**
 * Runs on the loop thread for each request: fills `res` from `req`. It
 * must not block (rendering /metrics from atomics is fine; an engine
 * call is not). An exception becomes a 500.
 */
using LoopHandler = std::function<void(const Request& req, Response& res)>;

/**
 * The answer to one request, given later and from any thread (i#930):
 * either one buffered response (send), or a chunked stream
 * (begin_stream, write..., end). Each call wakes the loop through
 * LoopConfig::wake_fd; the connection then writes what it holds. Once
 * the client is gone (a closed or failed connection, or EOF while the
 * answer is pending) the reply is cancelled: its calls do nothing and
 * write() returns false, so the producer stops.
 */
class LoopReply {
  public:
    /** Wakes the loop through `wake_list` (pushing wake_id) when it is
     *  set, else through wake_fd. */
    LoopReply(int wake_fd, std::size_t max_buffer, int write_deadline_ms,
              std::shared_ptr<WakeList> wake_list = nullptr,
              std::uint64_t wake_id = 0);

    /** Answers with a buffered response. Ignored after any other call. */
    void send(Response res);
    /**
     * Starts a chunked response with head's status and headers (its body
     * is ignored).
     * @returns false when cancelled or already answered.
     */
    bool begin_stream(Response head);
    /**
     * Queues one chunk of the stream. While more than max_buffer bytes
     * wait for the socket it blocks, for at most the write deadline.
     * @returns false when cancelled, not streaming, or the client did not
     *     take the bytes in time (the reply is then cancelled).
     */
    bool write(std::string_view data);
    /** Ends the stream (idempotent). */
    void end();
    /**
     * Ends the stream as a failure (the producer threw after the head
     * went out): what is queued is written, then the connection closes
     * without the terminating chunk, so the client sees a truncated
     * transfer, as http::Server does. A pending reply gets a 500.
     */
    void fail();
    /** @returns Whether the connection is gone (closed, a failed write):
     *  the reply's calls do nothing. */
    bool cancelled() const;
    /**
     * @returns Whether the client has left: cancelled, or it half-closed
     * while the answer was pending (TCP cannot tell an orderly half-close
     * from a gone client, as http::Server's liveness probe notes). The
     * answer may still be written; a long handler should stop.
     */
    bool client_gone() const;
    /** @returns Whether write() would queue at once: the stream's queue
     *  is under its bound (i#932, for a producer that must not block). */
    bool has_room() const;
    /**
     * Sets a callback the loop runs when the producer has something to
     * look at (i#1005): the queue was taken (room again), the reply was
     * cancelled, or the client half-closed. It runs with the reply's
     * mutex held, so it must not call into this reply; after
     * set_notify({}) returns it runs no more.
     */
    void set_notify(std::function<void()> f);

  private:
    friend class LoopConnection;
    enum class State { kPending, kBuffered, kStreaming, kEnded, kFailed };
    /** Wakes the loop. Call with mu_ released. */
    void wake() const;
    /** Marks the reply cancelled and wakes any waiting producer. */
    void cancel();

    const int wake_fd_;
    const std::shared_ptr<WakeList> wake_list_;
    const std::uint64_t wake_id_;
    const std::size_t max_buffer_;
    const int write_deadline_ms_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    State state_ = State::kPending;
    bool cancelled_ = false;
    bool peer_closed_ = false;  /**< EOF while pending (client_gone). */
    std::function<void()> notify_;  /**< set_notify; called under mu_. */
    bool head_taken_ = false;  /**< the loop wrote the stream's head. */
    Response res_;             /**< the buffered response, or the head. */
    std::string chunks_;       /**< encoded chunks the loop has not taken. */
};

/**
 * Runs on the loop thread for each request, as LoopHandler, but answers
 * through `reply`, now or later from another thread. It must not block;
 * an exception before it answers becomes a 500.
 */
using AsyncLoopHandler =
    std::function<void(const Request& req, std::shared_ptr<LoopReply> reply)>;

/**
 * One HTTP/1.1 connection driven by an event loop it does not own
 * (sys::Poller), for a server that wants HTTP on its own I/O thread
 * (a pipeline stage's /metrics, issue 22). The loop accepts, owns the
 * fd's poller registration and closes the fd; this object reads,
 * parses, calls the handler and writes, never blocking.
 *
 * The loop calls on_event() for every Poller event on the fd and
 * on_timeout() once deadline() has passed, and sets the fd's interest
 * from the returned Next: kRead or kWrite, or remove and close it on
 * kClose. read() and write() are the authority on a peer that left;
 * the hangup and error hints only make it look sooner.
 *
 * A LoopHandler answers at once with a buffered response; an
 * AsyncLoopHandler answers through a LoopReply, later and possibly as a
 * chunked stream, and the loop calls on_wake() when LoopConfig::wake_fd
 * is readable. No 100-continue. Keep-alive with pipelining: requests
 * already in the buffer are answered in order, one response in flight
 * at a time, and a peer that half-closes still gets an answer to every
 * request it sent before (for a deferred one, EOF while it is pending
 * also marks the client gone, LoopReply::client_gone, as http::Server's
 * liveness probe reads FIN, so a long handler stops; its answer is still
 * written). Input is read at most 64 KiB per event; a request that cannot
 * finish within the input cap (its caps, with room for chunk framing)
 * is a 413.
 */
class LoopConnection {
  public:
    /** What the loop should watch the fd for next. */
    /** kIdle (async replies only): watch nothing until on_wake(), the
     *  client having half-closed while its answer is pending. */
    enum class Next { kRead, kWrite, kClose, kIdle };
    using Clock = std::chrono::steady_clock;

    /**
     * @param fd An accepted socket, already non-blocking; not owned.
     * @param cfg Bounds.
     * @param handler Called once per complete request.
     */
    LoopConnection(int fd, LoopConfig cfg, LoopHandler handler);
    /** As above, with answers through a LoopReply (i#930). */
    LoopConnection(int fd, LoopConfig cfg, AsyncLoopHandler handler);
    /** Cancels a reply still pending, so its producer stops. */
    ~LoopConnection();

    LoopConnection(const LoopConnection&) = delete;
    LoopConnection& operator=(const LoopConnection&) = delete;

    /** @returns The fd this connection serves. */
    int fd() const { return fd_; }

    /**
     * Handles one Poller event for fd(): reads what is readable, answers
     * complete requests, writes what is writable. Never blocks.
     * @returns What to watch next, or kClose.
     */
    Next on_event(const sys::Poller::Event& ev);

    /** @returns When on_timeout() is due: the request, idle or write
     *  deadline of the current state. */
    Clock::time_point deadline() const;

    /**
     * Handles a passed deadline(): a request that began but did not
     * finish in time gets a 408 (then a close once written); an idle or
     * stalled connection closes without a response.
     * @returns What to watch next, or kClose.
     */
    Next on_timeout();

    /**
     * Takes what a pending LoopReply has (the response, the stream's
     * head and chunks, its end) and writes it. The loop calls it after
     * draining LoopConfig::wake_fd (for each connection: one wake can
     * cover many replies). Never blocks.
     * @returns What to watch next, or kClose.
     */
    Next on_wake();

  private:
    /** Moves a pending reply's output into out_ (only when out_ is
     *  empty, so at most one buffer's worth waits for the socket). */
    void take_reply();
    /** Writes out_, then takes and writes more of the reply, and
     *  answers the next request once it is done. @returns false on a
     *  hard write error. */
    bool pump();
    /** Reads until the socket would block. @returns false once the peer
     *  has gone (EOF or a hard error). */
    bool fill();
    /** Answers the next complete request in the buffer, if any. */
    void answer();
    /** Queues a status-only response (reason + "\n") and a close. */
    void queue_status(int status, bool head_request);
    /** Writes until done or the socket would block. @returns false on a
     *  hard write error. */
    bool flush();
    /** @returns The interest for the current state. */
    Next next() const;

    int fd_;
    LoopConfig cfg_;
    LoopHandler handler_;
    AsyncLoopHandler async_;
    /** The request being answered through a reply, and how. */
    std::shared_ptr<LoopReply> pending_;
    bool pending_keep_ = false;
    bool pending_head_ = false;
    /** The reply was cancelled under way (a producer that timed out
     *  on a slow client): close, as the stream cannot finish. */
    bool abort_ = false;
    std::string in_;
    ParseContext ctx_;
    std::string out_;
    std::size_t out_off_ = 0;
    bool close_after_ = false;  /**< close once out_ is written. */
    bool peer_gone_ = false;    /**< read saw EOF: no more requests. */
    int served_ = 0;
    Clock::time_point request_start_;  /**< first byte (or accept). */
    Clock::time_point last_progress_;  /**< last read or write. */
};

}  // namespace locus::http
