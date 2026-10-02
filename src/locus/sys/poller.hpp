#pragma once

#include <unordered_map>
#include <vector>

namespace locus::sys {

/**
 * A readiness multiplexer over epoll (Linux) and kqueue (BSD/macOS),
 * the primitive the event-loop serving model is built on (DESIGN.md
 * i#24): one loop watches many fds for read-readiness instead of a
 * thread or process per connection. The backend is chosen at compile
 * time; the API is identical on both.
 *
 * Level-triggered: a ready fd keeps being reported by wait() until the
 * caller acts on it (drains a readable fd, or writes down a writable
 * one), so a caller that handles only part of what is ready does not
 * lose the rest.
 *
 * An fd is watched for read, write, or both (Dir). A listening socket
 * reports readable when a connection is pending, so accept() will not
 * block; a pipe/socket reports readable when there are bytes or the peer
 * closed (so a drained read returns 0/EOF); a socket reports writable
 * when a write will not block (including a non-blocking connect that has
 * completed -- the caller checks getsockopt(SO_ERROR) to tell success
 * from failure).
 */
class Poller {
  public:
    /**
     * What an fd is watched for. A bit mask: kReadWrite == kRead |
     * kWrite. Read-readiness and write-readiness are independent, so a
     * fd watched kReadWrite can be reported readable, writable, or both
     * in one wait().
     */
    enum class Dir : unsigned {
        kRead = 1,
        kWrite = 2,
        kReadWrite = 3,
    };

    /**
     * One ready fd and what happened to it.
     *
     * - `readable`: a read will not block. It may return bytes, 0 at
     *   EOF, or -1 with an error (e.g. ECONNRESET).
     * - `writable`: a write will not block. It may succeed, or fail with
     *   an error (e.g. a non-blocking connect that failed, or a peer that
     *   closed its read side -- EPIPE/ECONNRESET). As with readable, the
     *   write call is the authority on success, not this flag; a caller
     *   that used a non-blocking connect checks getsockopt(SO_ERROR) on
     *   the first writable event to tell a completed connect from a
     *   failed one.
     * - `hangup`: an advisory hint that the peer closed or half-closed.
     *   It is NOT a reliable drop trigger: on epoll a clean socket FIN
     *   raises no hangup (only `readable`), while on kqueue the same FIN
     *   does set it; and on BOTH backends `hangup` can be set while
     *   unread bytes are still buffered. Never drop a peer on `hangup`
     *   alone, and never use it to decide when to unregister an fd.
     * - `error`: the fd is unusable; `readable` and `writable` are both
     *   false. The caller MUST remove or close it or wait() keeps
     *   reporting it. This is effectively epoll-only: on kqueue a socket
     *   error such as a RST or a failed connect surfaces as `hangup`
     *   (EV_EOF) with the next read/write returning -1, not as `error`.
     *
     * Portable rule for the serving loop: the read()/write() call is the
     * authority -- a read()==0 or a read()/write()==-1 with a
     * non-retriable errno is the only cross-backend "peer gone" signal.
     * Treat `hangup` and `error` as hints, not the authority. (epoll
     * EPOLLIN/EPOLLOUT/EPOLLHUP/EPOLLERR and kqueue EVFILT_READ /
     * EVFILT_WRITE data / EV_EOF / EV_ERROR map onto these fields.)
     */
    struct Event {
        int fd = -1;
        bool readable = false;
        bool writable = false;
        bool hangup = false;
        bool error = false;
    };

    /** @throws std::runtime_error if the OS poll fd cannot be created. */
    Poller();
    ~Poller();
    Poller(const Poller&) = delete;
    Poller& operator=(const Poller&) = delete;

    /**
     * Registers a not-yet-registered `fd` to be watched for `dir`. To
     * change the direction of an already-registered fd use modify(), not
     * a second add(): re-adding a registered fd throws (both backends).
     * @throws std::runtime_error if `fd` is already registered or on an
     *     OS registration failure.
     */
    void add(int fd, Dir dir = Dir::kRead);

    /**
     * Changes an already-registered `fd` to be watched for exactly
     * `dir`, narrowing or widening it (e.g. kReadWrite -> kRead once a
     * pending write has drained). Only the directions that actually
     * change are touched, so an unchanged side keeps any readiness it
     * already had. @throws std::runtime_error if `fd` is not registered
     *     or on an OS failure.
     */
    void modify(int fd, Dir dir);

    /** Unregisters `fd` (all directions); harmless if it was never added
     * or already closed. */
    void remove(int fd);

    /**
     * Waits up to `timeout_ms` (-1 = forever) for registered fds to
     * become ready, retrying internally on EINTR. Clears `events` and
     * appends one Event per ready fd (its readable / writable / hangup /
     * error fields set; a fd ready for both read and write is reported
     * once with both flags, never twice).
     *
     * @returns the number of ready fds (0 on timeout), or -1 on a poll
     *     error (errno set).
     */
    int wait(std::vector<Event>& events, int timeout_ms);

  private:
    int poll_fd_ = -1;  // epoll fd (Linux) or kqueue fd (BSD/macOS)
    // Current direction of each registered fd. Lets add()/modify()
    // enforce their contracts identically on both backends, and lets the
    // kqueue modify() touch only the filters that actually change --
    // re-issuing EV_ADD on an unchanged filter resets its pending state,
    // which would drop an already-readable fd. Single-threaded, like the
    // rest of the class (one event loop owns a Poller).
    std::unordered_map<int, Dir> dir_;
};

}  // namespace locus::sys
