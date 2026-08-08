#include "locus/auth/helper_connection.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <ctime>
#include <mutex>

namespace locus::auth {

namespace {

/** Reaping a helper child: poll waitpid(WNOHANG) for up to
 * kReapGraceTicks * kReapTickNs (1s) after SIGTERM before escalating
 * to SIGKILL, so a helper that ignores EOF and SIGTERM cannot hang
 * shutdown. */
constexpr int kReapGraceTicks = 100;
constexpr long kReapTickNs = 10L * 1000 * 1000;  // 10ms
/**
 * Writing to a helper pipe/socket whose peer has closed raises
 * SIGPIPE, which by default kills the process. Per-fd suppression
 * (MSG_NOSIGNAL / SO_NOSIGPIPE) does not cover pipes, so a process
 * that writes to child pipes must ignore SIGPIPE and handle EPIPE on
 * the write() return instead. Done once, idempotently.
 */
void ignore_sigpipe_once() {
    static std::once_flag once;
    std::call_once(once, [] { std::signal(SIGPIPE, SIG_IGN); });
}

/** Puts an fd in non-blocking mode so reads can drain fully and
 * writes never block indefinitely. Best-effort. */
void set_nonblocking(int fd) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags >= 0) {
        ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }
}

/** Cap on a fire-and-forget notify() write: bounds how long an event
 * push can block on a stalled helper before we give up. */
constexpr std::chrono::milliseconds kNotifyWriteTimeout{2000};
}  // namespace

HelperConnection::HelperConnection(int read_fd, int write_fd)
    : HelperConnection(read_fd, write_fd, -1) {}

HelperConnection::HelperConnection(int read_fd, int write_fd,
                                   pid_t child_pid)
    : read_fd_(read_fd),
      write_fd_(write_fd),
      child_pid_(child_pid) {
    ignore_sigpipe_once();
    // Non-blocking both directions BEFORE the reader starts: the
    // reader drains every buffered (correlated) reply before honoring
    // a hangup, and write_all polls with a deadline so a stalled
    // helper cannot wedge callers under write_mu_.
    set_nonblocking(read_fd_);
    if (write_fd_ != read_fd_) {
        set_nonblocking(write_fd_);
    }
    reader_ = std::thread([this] { reader_loop(); });
}

HelperConnection::~HelperConnection() {
    stop_.store(true);
    // Closing the write fd signals EOF to the helper; the reader
    // wakes from poll() on its own timeout and observes stop_.
    if (reader_.joinable()) {
        reader_.join();
    }
    ::close(write_fd_);
    if (read_fd_ != write_fd_) {
        ::close(read_fd_);
    }
    // Reap a spawned child. Closed stdin gives it EOF and SIGTERM
    // asks it to quit, but a buggy helper may ignore BOTH; a plain
    // blocking waitpid would then hang shutdown forever. Give it a
    // bounded grace period, then SIGKILL (uncatchable) and reap.
    if (child_pid_ > 0) {
        ::kill(child_pid_, SIGTERM);
        bool reaped = false;
        for (int i = 0; i < kReapGraceTicks; ++i) {
            const pid_t w = ::waitpid(child_pid_, nullptr, WNOHANG);
            if (w == child_pid_ || (w < 0 && errno == ECHILD)) {
                reaped = true;
                break;
            }
            const timespec tick{0, kReapTickNs};
            ::nanosleep(&tick, nullptr);
        }
        if (!reaped) {
            ::kill(child_pid_, SIGKILL);
            ::waitpid(child_pid_, nullptr, 0);
        }
    }
}

bool HelperConnection::write_all(const std::string& bytes,
                                 std::chrono::milliseconds timeout) {
    std::lock_guard<std::mutex> lk(write_mu_);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::size_t off = 0;
    while (off < bytes.size()) {
        const ssize_t n =
            ::write(write_fd_, bytes.data() + off, bytes.size() - off);
        if (n > 0) {
            off += static_cast<std::size_t>(n);
            continue;
        }
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            // Pipe full: the helper is not draining. Wait (bounded)
            // for it to make room instead of blocking forever under
            // write_mu_; on timeout fail closed.
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                fail_all();  // wake any other in-flight waiters too
                return false;
            }
            const auto ms = std::chrono::duration_cast<
                std::chrono::milliseconds>(deadline - now)
                                .count();
            pollfd pfd{write_fd_, POLLOUT, 0};
            const int pr = ::poll(&pfd, 1, static_cast<int>(ms));
            if (pr <= 0 ||
                (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                fail_all();  // timeout or peer gone; wake waiters
                return false;
            }
            continue;
        }
        fail_all();  // EPIPE or other fatal write error; wake waiters
        return false;
    }
    return true;
}

void HelperConnection::fail_all() {
    broken_.store(true);
    std::lock_guard<std::mutex> lk(mu_);
    cv_.notify_all();  // waiters re-check broken_ and give up
}

void HelperConnection::reader_loop() {
    std::string buf;
    char chunk[4096];
    while (!stop_.load()) {
        pollfd pfd{read_fd_, POLLIN, 0};
        const int pr = ::poll(&pfd, 1, 100);  // 100ms tick
        if (pr < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail_all();
            return;
        }
        if (pr == 0) {
            continue;  // timeout: re-check stop_
        }

        // Drain ALL currently-readable bytes -- decoding correlated
        // replies as we go -- BEFORE honoring a hangup or error. A
        // POLLHUP can arrive together with the final replies still in
        // the pipe buffer; failing without draining would discard
        // valid, correlated responses and surface them as spurious
        // auth failures. read_fd_ is non-blocking, so the loop ends
        // at EAGAIN.
        bool eof = false;
        bool ioerr = false;
        if ((pfd.revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
            for (;;) {
                const ssize_t n =
                    ::read(read_fd_, chunk, sizeof chunk);
                if (n > 0) {
                    buf.append(chunk, static_cast<std::size_t>(n));
                    HelperMessage msg;
                    std::string err;
                    HelperDecode d;
                    while ((d = decode_text(buf, msg, err)) ==
                           HelperDecode::kComplete) {
                        std::lock_guard<std::mutex> lk(mu_);
                        auto it = pending_.find(msg.req_id);
                        if (it != pending_.end()) {
                            it->second->resp = std::move(msg);
                            it->second->done = true;
                            cv_.notify_all();
                        }
                        // Unknown/0 req_id (event ack) is ignored.
                        msg = HelperMessage{};
                    }
                    if (d == HelperDecode::kError) {
                        fail_all();
                        return;
                    }
                    continue;
                }
                if (n == 0) {
                    eof = true;  // peer closed stdout; fully drained
                    break;
                }
                if (errno == EINTR) {
                    continue;
                }
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    break;  // no more readable data right now
                }
                ioerr = true;  // real read error
                break;
            }
        }

        if (eof || ioerr ||
            (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
            fail_all();  // drained above; now honor hangup/error
            return;
        }
    }
    fail_all();  // stop_ requested: wake any stragglers
}

std::optional<HelperMessage> HelperConnection::request(
    HelperMessage req, std::chrono::milliseconds timeout) {
    if (broken_.load()) {
        return std::nullopt;
    }
    const std::uint64_t id = next_id_.fetch_add(1);
    req.req_id = id;
    std::string wire;
    if (!encode_text(req, wire)) {
        return std::nullopt;
    }
    auto slot = std::make_shared<Slot>();
    {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.emplace(id, slot);
    }
    if (!write_all(wire, timeout)) {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.erase(id);
        return std::nullopt;
    }
    std::unique_lock<std::mutex> lk(mu_);
    const bool ok = cv_.wait_for(lk, timeout, [&] {
        return slot->done || broken_.load();
    });
    pending_.erase(id);
    if (!ok || !slot->done) {
        return std::nullopt;  // timed out or connection broke
    }
    return std::move(slot->resp);
}

bool HelperConnection::notify(HelperMessage msg) {
    msg.req_id = 0;  // fire-and-forget: helper sends no reply
    std::string wire;
    if (!encode_text(msg, wire)) {
        return false;
    }
    return write_all(wire, kNotifyWriteTimeout);
}

}  // namespace locus::auth
