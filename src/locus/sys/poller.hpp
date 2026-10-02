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
     * One ready fd and what happened to it. `readable` means a read
     * will not block (it may return data or 0 at EOF), so a caller
     * always reads on it; `hangup` means the peer closed its end and
     * `error` an error condition. A hung-up fd is reported with both
     * `readable` and `hangup` set, so the read() that returns 0 is the
     * authoritative "peer gone" signal on both backends; the flags are
     * the explicit hint that lets a caller drop a connection without a
     * zero-length read first. (epoll EPOLLIN/EPOLLHUP/EPOLLERR map to
     * these; kqueue EVFILT_READ data / EV_EOF / EV_ERROR map to them.)
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
