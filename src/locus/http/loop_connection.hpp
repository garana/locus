#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <string>

#include "locus/http/http.hpp"
#include "locus/sys/poller.hpp"

namespace locus::http {

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
};

/**
 * Runs on the loop thread for each request: fills `res` from `req`. It
 * must not block (rendering /metrics from atomics is fine; an engine
 * call is not). An exception becomes a 500.
 */
using LoopHandler = std::function<void(const Request& req, Response& res)>;

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
 * Buffered responses only (no streaming, no 100-continue); keep-alive
 * with pipelining: requests already in the buffer are answered in
 * order, one response in flight at a time, and a peer that half-closes
 * still gets an answer to every request it sent before. Input is read
 * at most 64 KiB per event; a request that cannot finish within the
 * input cap (its caps, with room for chunk framing) is a 413.
 */
class LoopConnection {
  public:
    /** What the loop should watch the fd for next. */
    enum class Next { kRead, kWrite, kClose };
    using Clock = std::chrono::steady_clock;

    /**
     * @param fd An accepted socket, already non-blocking; not owned.
     * @param cfg Bounds.
     * @param handler Called once per complete request.
     */
    LoopConnection(int fd, LoopConfig cfg, LoopHandler handler);

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

  private:
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
