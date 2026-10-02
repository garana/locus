#pragma once

#include <vector>

namespace locus::sys {

/**
 * A readiness multiplexer over epoll (Linux) and kqueue (BSD/macOS),
 * the primitive the event-loop serving model is built on (DESIGN.md
 * i#24): one loop watches many fds for read-readiness instead of a
 * thread or process per connection. The backend is chosen at compile
 * time; the API is identical on both.
 *
 * Level-triggered: a readable fd keeps being reported by wait() until
 * the caller drains it, so a caller that reads only part of what is
 * ready does not lose the rest. Read-readiness only for now (the fds
 * this serves -- listeners, a self-pipe, stage sockets -- are all read
 * sides); write-readiness is a later addition when it is needed.
 *
 * A listening socket reports readable when a connection is pending, so
 * accept() will not block; a pipe/socket reports readable when there
 * are bytes or the peer closed (so a drained read returns 0/EOF).
 */
class Poller {
  public:
    /**
     * One ready fd and what happened to it.
     *
     * - `readable`: a read will not block. It may return bytes, 0 at
     *   EOF, or -1 with an error (e.g. ECONNRESET).
     * - `hangup`: an advisory hint that the peer closed or half-closed.
     *   It is NOT a reliable drop trigger: on epoll a clean socket FIN
     *   raises no hangup (only `readable`), while on kqueue the same FIN
     *   does set it; and on BOTH backends `hangup` can be set while
     *   unread bytes are still buffered. Never drop a peer on `hangup`
     *   alone, and never use it to decide when to unregister an fd.
     * - `error`: the fd is unusable and `readable` is false; the caller
     *   MUST remove or close it or wait() keeps reporting it. This is
     *   effectively epoll-only: on kqueue a socket error such as a RST
     *   surfaces as `readable` with the next read returning -1, not as
     *   `error`.
     *
     * Portable rule for the serving loop: a `read()` returning 0 (clean
     * EOF) or -1 with a non-retriable errno is the only cross-backend
     * "peer gone" signal, so always drain to such a read before dropping
     * a peer; treat `hangup` and `error` as hints, not the authority.
     * (epoll EPOLLIN/EPOLLHUP/EPOLLERR and kqueue EVFILT_READ data /
     * EV_EOF / EV_ERROR map onto these fields.)
     */
    struct Event {
        int fd = -1;
        bool readable = false;
        bool hangup = false;
        bool error = false;
    };

    /** @throws std::runtime_error if the OS poll fd cannot be created. */
    Poller();
    ~Poller();
    Poller(const Poller&) = delete;
    Poller& operator=(const Poller&) = delete;

    /** Registers `fd` for read-readiness (no-op if already registered).
     * @throws std::runtime_error on an OS registration failure. */
    void add_read(int fd);

    /** Unregisters `fd`; harmless if it was never added or already
     * closed. */
    void remove(int fd);

    /**
     * Waits up to `timeout_ms` (-1 = forever) for registered fds to
     * become ready, retrying internally on EINTR. Clears `events` and
     * appends one Event per ready fd (readable / hangup / error; see
     * Event).
     *
     * @returns the number of ready fds (0 on timeout), or -1 on a poll
     *     error (errno set).
     */
    int wait(std::vector<Event>& events, int timeout_ms);

  private:
    int poll_fd_ = -1;  // epoll fd (Linux) or kqueue fd (BSD/macOS)
};

}  // namespace locus::sys
