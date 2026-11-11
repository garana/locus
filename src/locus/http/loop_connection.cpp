#include "locus/http/loop_connection.hpp"

#include <sys/socket.h>

#include <cerrno>
#include <utility>

namespace locus::http {

namespace {

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;  // SO_NOSIGPIPE is set on the fd instead
#endif

/** Bytes read per event: input is answered as it arrives, so a
 *  pipelined flood costs one read's worth, not a whole request cap. */
constexpr std::size_t kReadBudget = 64 * 1024;

/** @returns The most unanswered input a connection holds: one request
 *  at its caps, with room for chunk framing (which can outweigh the
 *  decoded body), plus the start of the next request. */
std::size_t input_cap(const Limits& l) {
    return l.max_request_line + l.max_header_bytes +
           2 * l.max_body_bytes + l.max_trailer_bytes + 64 * 1024;
}

}  // namespace

LoopConnection::LoopConnection(int fd, LoopConfig cfg, LoopHandler handler)
    : fd_(fd), cfg_(std::move(cfg)), handler_(std::move(handler)) {
#if !defined(MSG_NOSIGNAL) && defined(SO_NOSIGPIPE)
    const int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    request_start_ = last_progress_ = Clock::now();
}

bool LoopConnection::fill() {
    const bool was_empty = in_.empty();
    const std::size_t cap = input_cap(cfg_.limits);
    std::size_t got = 0;
    char buf[16 * 1024];
    while (got < kReadBudget && in_.size() < cap) {
        const ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
        if (n > 0) {
            in_.append(buf, static_cast<std::size_t>(n));
            got += static_cast<std::size_t>(n);
            last_progress_ = Clock::now();
            if (was_empty && got == static_cast<std::size_t>(n)) {
                request_start_ = last_progress_;  // a request began
            }
            continue;
        }
        if (n == 0) return false;  // EOF: the peer will send no more
        if (errno == EINTR) continue;
        return errno == EAGAIN || errno == EWOULDBLOCK;
    }
    return true;  // the rest waits for the next readable event
}

void LoopConnection::queue_status(int status, bool head_request) {
    Response r;
    r.status = status;
    r.set_content(status_reason(status) + "\n", "text/plain");
    std::string err;
    write_response(r, head_request, false, out_, err);
    close_after_ = true;
}

void LoopConnection::answer() {
    if (in_.empty() || close_after_) return;
    Request req;
    const ParseResult pr = parse_request(in_, cfg_.limits, ctx_, req);
    if (pr.state == ParseState::kNeedMore) {
        // Full and still incomplete: no more input can finish it, and
        // waiting would spin on a socket that stays readable.
        if (in_.size() >= input_cap(cfg_.limits)) {
            in_.clear();
            queue_status(413, false);
        }
        return;
    }
    if (pr.state == ParseState::kError) {
        in_.clear();
        queue_status(pr.suggested_status, false);
        return;
    }
    in_.erase(0, pr.consumed);
    ++served_;
    Response res;
    try {
        handler_(req, res);
    } catch (...) {
        queue_status(500, req.is_head);
        return;
    }
    // A peer that half-closed still gets every request it sent before
    // its EOF; next() closes once none is left.
    const bool keep =
        req.keep_alive && served_ < cfg_.max_requests_per_conn;
    std::string err;
    if (!write_response(res, req.is_head, keep, out_, err)) {
        out_.clear();  // an unsafe header: nothing of it was written
        queue_status(500, req.is_head);
        return;
    }
    if (!keep) close_after_ = true;
    request_start_ = Clock::now();  // a pipelined next request starts now
}

bool LoopConnection::flush() {
    while (out_off_ < out_.size()) {
        const ssize_t n = ::send(fd_, out_.data() + out_off_,
                                 out_.size() - out_off_, kSendFlags);
        if (n > 0) {
            out_off_ += static_cast<std::size_t>(n);
            last_progress_ = Clock::now();
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return true;
        return false;  // EPIPE, ECONNRESET: the peer is gone
    }
    out_.clear();
    out_off_ = 0;
    return true;
}

LoopConnection::Next LoopConnection::next() const {
    if (!out_.empty()) return Next::kWrite;
    if (close_after_ || peer_gone_) return Next::kClose;
    return Next::kRead;
}

LoopConnection::Next LoopConnection::on_event(const sys::Poller::Event& ev) {
    if (ev.error) return Next::kClose;
    if (ev.readable && out_.empty() && !peer_gone_) {
        if (!fill()) peer_gone_ = true;
    }
    if (ev.writable && !out_.empty() && !flush()) return Next::kClose;
    // Answer what is buffered, one response in flight at a time, and
    // write each at once: the socket is usually writable already.
    while (out_.empty() && !close_after_) {
        answer();
        if (out_.empty()) break;  // nothing complete yet
        if (!flush()) return Next::kClose;
    }
    return next();
}

LoopConnection::Clock::time_point LoopConnection::deadline() const {
    using std::chrono::milliseconds;
    if (!out_.empty()) {
        return last_progress_ + milliseconds(cfg_.write_deadline_ms);
    }
    if (in_.empty()) {  // idle: before the first request or between two
        return last_progress_ + milliseconds(cfg_.idle_timeout_ms);
    }
    return request_start_ + milliseconds(cfg_.request_deadline_ms);
}

LoopConnection::Next LoopConnection::on_timeout() {
    if (Clock::now() < deadline()) return next();  // not due yet
    if (!out_.empty()) return Next::kClose;        // a stalled write
    if (in_.empty()) return Next::kClose;          // idle: no response
    // A request that began did not finish in time.
    in_.clear();
    queue_status(408, false);
    if (!flush()) return Next::kClose;
    return next();
}

}  // namespace locus::http
