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
     * become readable, retrying internally on EINTR. Clears `ready` and
     * appends the readable fds (including error/hangup, which also read
     * as readable).
     *
     * @returns the number of ready fds (0 on timeout), or -1 on a poll
     *     error (errno set).
     */
    int wait(std::vector<int>& ready, int timeout_ms);

  private:
    int poll_fd_ = -1;  // epoll fd (Linux) or kqueue fd (BSD/macOS)
};

}  // namespace locus::sys
