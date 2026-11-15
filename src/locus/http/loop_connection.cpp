#include "locus/http/loop_connection.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <utility>

namespace locus::http {

namespace {

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;  // SO_NOSIGPIPE, set on the fd, covers it
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

LoopReply::LoopReply(int wake_fd, std::size_t max_buffer,
                     int write_deadline_ms)
    : wake_fd_(wake_fd),
      max_buffer_(max_buffer == 0 ? 1 : max_buffer),
      write_deadline_ms_(write_deadline_ms) {}

void LoopReply::wake() const {
    if (wake_fd_ < 0) return;
    const char b = 1;
    // EAGAIN: the pipe is full, so a wake is already pending.
    while (::write(wake_fd_, &b, 1) < 0 && errno == EINTR) {
    }
}

void LoopReply::send(Response res) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (cancelled_ || state_ != State::kPending) return;
        res_ = std::move(res);
        state_ = State::kBuffered;
    }
    wake();
}

bool LoopReply::begin_stream(Response head) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (cancelled_ || state_ != State::kPending) return false;
        res_ = std::move(head);
        res_.body.clear();
        state_ = State::kStreaming;
    }
    wake();
    return true;
}

bool LoopReply::write(std::string_view data) {
    {
        std::unique_lock<std::mutex> lk(mu_);
        if (cancelled_ || state_ != State::kStreaming) return false;
        // An empty chunk would read as the stream's end.
        if (data.empty()) return true;
        const auto until = std::chrono::steady_clock::now() +
                           std::chrono::milliseconds(write_deadline_ms_);
        if (!cv_.wait_until(lk, until, [this] {
                return cancelled_ || chunks_.size() < max_buffer_;
            })) {
            cancelled_ = true;  // the client stopped reading
            lk.unlock();
            wake();  // the loop closes the connection
            return false;
        }
        if (cancelled_) return false;
        chunks_ += encode_chunk(data);
    }
    wake();
    return true;
}

void LoopReply::end() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        // Cancelled: the loop may have dropped the connection and
        // closed (or reused) wake_fd, so write nothing to it.
        if (cancelled_ || state_ != State::kStreaming) return;
        state_ = State::kEnded;
    }
    wake();
}

bool LoopReply::cancelled() const {
    std::lock_guard<std::mutex> lk(mu_);
    return cancelled_;
}

void LoopReply::cancel() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        cancelled_ = true;
    }
    cv_.notify_all();
}

LoopConnection::LoopConnection(int fd, LoopConfig cfg, LoopHandler handler)
    : fd_(fd), cfg_(std::move(cfg)), handler_(std::move(handler)) {
#ifdef SO_NOSIGPIPE
    // As http::Server does: set wherever it exists (macOS defines both
    // it and MSG_NOSIGNAL), so a send without the flag is covered too.
    const int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    request_start_ = last_progress_ = Clock::now();
}

LoopConnection::LoopConnection(int fd, LoopConfig cfg,
                               AsyncLoopHandler handler)
    : LoopConnection(fd, std::move(cfg), LoopHandler{}) {
    async_ = std::move(handler);
}

LoopConnection::~LoopConnection() {
    if (pending_) pending_->cancel();
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
    r.set_content(status_error_body(status), "application/json");
    std::string err;
    write_response(r, head_request, false, out_, err);
    close_after_ = true;
}

void LoopConnection::answer() {
    if (in_.empty() || close_after_ || pending_) return;
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
    // A peer that half-closed still gets every request it sent before
    // its EOF; next() closes once none is left.
    const bool keep =
        req.keep_alive && served_ < cfg_.max_requests_per_conn;
    if (async_) {
        auto reply = std::make_shared<LoopReply>(
            cfg_.wake_fd, cfg_.max_stream_buffer, cfg_.write_deadline_ms);
        pending_ = reply;
        pending_keep_ = keep;
        pending_head_ = req.is_head;
        try {
            async_(req, reply);
        } catch (...) {
            bool answered;
            {
                std::lock_guard<std::mutex> lk(reply->mu_);
                answered = reply->state_ != LoopReply::State::kPending;
                reply->cancelled_ = reply->cancelled_ || !answered;
            }
            if (!answered) {
                pending_.reset();
                queue_status(500, req.is_head);
                return;
            }
        }
        take_reply();  // the handler may have answered already
        return;
    }
    Response res;
    try {
        handler_(req, res);
    } catch (...) {
        queue_status(500, req.is_head);
        return;
    }
    std::string err;
    if (!write_response(res, req.is_head, keep, out_, err)) {
        out_.clear();  // an unsafe header: nothing of it was written
        queue_status(500, req.is_head);
        return;
    }
    if (!keep) close_after_ = true;
    request_start_ = Clock::now();  // a pipelined next request starts now
}

void LoopConnection::take_reply() {
    if (!pending_ || !out_.empty()) return;
    LoopReply& r = *pending_;
    bool done = false;
    bool taken = false;
    {
        std::lock_guard<std::mutex> lk(r.mu_);
        if (r.cancelled_) {
            abort_ = true;  // the producer gave up on a slow client
            return;
        }
        std::string err;
        switch (r.state_) {
            case LoopReply::State::kPending:
                return;
            case LoopReply::State::kBuffered:
                if (!write_response(r.res_, pending_head_, pending_keep_,
                                    out_, err)) {
                    out_.clear();  // an unsafe header: nothing written
                    queue_status(500, pending_head_);
                }
                done = true;
                break;
            case LoopReply::State::kStreaming:
            case LoopReply::State::kEnded:
                if (!r.head_taken_) {
                    if (!write_chunked_head(r.res_, pending_keep_, out_,
                                            err)) {
                        out_.clear();
                        r.cancelled_ = true;
                        taken = true;  // wake the producer: it stops
                        queue_status(500, pending_head_);
                        done = true;
                        break;
                    }
                    r.head_taken_ = true;
                }
                // A HEAD request gets the head alone.
                if (!pending_head_) out_ += r.chunks_;
                if (!r.chunks_.empty()) {
                    r.chunks_.clear();
                    taken = true;
                }
                if (r.state_ == LoopReply::State::kEnded) {
                    if (!pending_head_) out_ += last_chunk();
                    done = true;
                }
                break;
        }
    }
    if (taken) r.cv_.notify_all();
    if (done) {
        if (!pending_keep_) close_after_ = true;
        pending_.reset();
        request_start_ = Clock::now();  // a pipelined next request
    }
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

bool LoopConnection::pump() {
    for (;;) {
        if (!out_.empty() && !flush()) return false;
        if (!out_.empty()) return true;  // the socket is full: wait
        if (pending_) {
            take_reply();
            if (abort_ || out_.empty()) return true;  // nothing new yet
            continue;
        }
        if (close_after_) return true;
        // Answer what is buffered, one response in flight at a time,
        // and write each at once: the socket is usually writable.
        answer();
        if (out_.empty() && !pending_) return true;  // nothing complete
    }
}

LoopConnection::Next LoopConnection::next() const {
    if (abort_) return Next::kClose;
    if (!out_.empty()) return Next::kWrite;
    if (pending_) return Next::kRead;  // the reply comes by on_wake()
    if (close_after_ || peer_gone_) return Next::kClose;
    return Next::kRead;
}

LoopConnection::Next LoopConnection::on_event(const sys::Poller::Event& ev) {
    if (ev.error) return Next::kClose;
    if (ev.readable && !peer_gone_ && (out_.empty() || pending_)) {
        if (!fill()) peer_gone_ = true;
    }
    if (pending_) {
        // EOF while the answer is pending: the client is gone (as
        // http::Server's probe reads it), so its producer stops. Input
        // at the cap can no longer be read, and a level-triggered
        // readable socket would spin: close that client too.
        if (peer_gone_ || in_.size() >= input_cap(cfg_.limits)) {
            return Next::kClose;  // the destructor cancels the reply
        }
    }
    if (!pump()) return Next::kClose;
    return next();
}

LoopConnection::Next LoopConnection::on_wake() {
    if (!pump()) return Next::kClose;
    return next();
}

LoopConnection::Clock::time_point LoopConnection::deadline() const {
    using std::chrono::milliseconds;
    if (!out_.empty()) {
        return last_progress_ + milliseconds(cfg_.write_deadline_ms);
    }
    if (pending_) {
        // The handler takes the time it takes; a stalled write is the
        // only bound, and it starts once there are bytes to write.
        return Clock::now() + std::chrono::hours(1);
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
