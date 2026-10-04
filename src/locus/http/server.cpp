#include "locus/http/server.hpp"

#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>

#include "locus/pipeline/net.hpp"

namespace locus::http {

namespace {

// send() flag that suppresses SIGPIPE on a write to a closed peer, where
// the platform offers it (Linux). On macOS/BSD the same is achieved with
// SO_NOSIGPIPE on the socket (set in serve_connection), so 0 is correct.
#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif

void make_nonblocking(int fd) {
    const int fl = ::fcntl(fd, F_GETFL, 0);
    if (fl >= 0) {
        ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    }
}

char lower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (lower(a[i]) != lower(b[i])) {
            return false;
        }
    }
    return true;
}

std::string_view trim_ows(std::string_view s) {
    std::size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) {
        --e;
    }
    return s.substr(b, e - b);
}

/**
 * Writes all of `data`, bouncing on EAGAIN via poll(POLLOUT). The
 * deadline bounds a single BLOCKED write (one poll wait), NOT the total
 * time: a write that keeps making progress is never cut off, which is
 * what lets a long SSE stream and a slow prefill through. A write that
 * cannot make progress for `deadline_ms`, a shutdown, or a dead peer all
 * return false. @returns true iff every byte was sent.
 */
bool write_all_deadline(int fd, std::string_view data, int wake_r,
                        int deadline_ms,
                        const std::atomic<bool>& stopping) {
    std::size_t off = 0;
    while (off < data.size()) {
        if (stopping.load()) {
            return false;
        }
        const ssize_t n =
            ::send(fd, data.data() + off, data.size() - off, kSendFlags);
        if (n > 0) {
            off += static_cast<std::size_t>(n);
            continue;
        }
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                pollfd pfds[2] = {{fd, POLLOUT, 0}, {wake_r, POLLIN, 0}};
                const int pr = ::poll(pfds, 2, deadline_ms);
                if (pr < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    return false;
                }
                if (pr == 0) {
                    return false;  // blocked past the deadline: drop
                }
                if ((pfds[1].revents & POLLIN) != 0) {
                    return false;  // shutdown
                }
                continue;  // writable: retry the send
            }
            return false;  // EPIPE / ECONNRESET / ...: peer gone
        }
        // n == 0 is not expected from send(); treat as no progress.
        return false;
    }
    return true;
}

// Scans the request head (bytes up to, not including, the blank line) for
// `Expect: 100-continue` (case-insensitive, OWS-tolerant). Used to decide
// whether to send an interim 100 before reading the body.
bool head_requests_continue(std::string_view head) {
    std::size_t p = head.find("\r\n");  // skip the request line
    if (p == std::string_view::npos) {
        return false;
    }
    p += 2;
    while (p < head.size()) {
        std::size_t e = head.find("\r\n", p);
        if (e == std::string_view::npos) {
            e = head.size();
        }
        const std::string_view line = head.substr(p, e - p);
        const std::size_t c = line.find(':');
        if (c != std::string_view::npos) {
            if (iequals(line.substr(0, c), "expect") &&
                iequals(trim_ows(line.substr(c + 1)), "100-continue")) {
                return true;
            }
        }
        p = (e == head.size()) ? e : e + 2;
    }
    return false;
}

long ms_until(std::chrono::steady_clock::time_point tp) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               tp - std::chrono::steady_clock::now())
        .count();
}

/** Streaming Sink backed by a connection fd: each write() is one chunk. */
class ServerSink : public Sink {
 public:
    ServerSink(int fd, int wake_r, int deadline_ms,
               const std::atomic<bool>& stopping)
        : fd_(fd),
          wake_r_(wake_r),
          deadline_ms_(deadline_ms),
          stopping_(stopping) {}

    bool write(std::string_view data) override {
        if (done_ || !ok_) {
            return false;
        }
        if (data.empty()) {
            return true;  // encode_chunk("") is "0\r\n\r\n" -- the
                          // terminator; an empty write must be a no-op,
                          // not an accidental end-of-stream.
        }
        if (stopping_.load()) {
            ok_ = false;
            return false;
        }
        if (!write_all_deadline(fd_, encode_chunk(data), wake_r_,
                                deadline_ms_, stopping_)) {
            ok_ = false;
            return false;
        }
        return true;
    }

    void done() override {
        if (done_) {
            return;
        }
        done_ = true;
        if (ok_ && !write_all_deadline(fd_, last_chunk(), wake_r_,
                                       deadline_ms_, stopping_)) {
            ok_ = false;
        }
    }

    bool ok() const { return ok_; }

 private:
    int fd_;
    int wake_r_;
    int deadline_ms_;
    const std::atomic<bool>& stopping_;
    bool ok_ = true;
    bool done_ = false;
};

}  // namespace

void ServerResponse::set_header(std::string name, std::string value) {
    headers.emplace_back(std::move(name), std::move(value));
}

void ServerResponse::set_content(std::string content,
                                 std::string content_type) {
    body = std::move(content);
    set_header("Content-Type", std::move(content_type));
}

void ServerResponse::set_chunked_content_provider(std::string content_type,
                                                  StreamProvider p) {
    set_header("Content-Type", std::move(content_type));
    provider = std::move(p);
}

Server::Server(ServerConfig cfg, Handler handler)
    : cfg_(std::move(cfg)), handler_(std::move(handler)) {}

Server::~Server() { stop(); }

int Server::start() {
    if (started_.exchange(true)) {
        return bound_port_;  // idempotent
    }
    int fds[2];
    if (::pipe(fds) != 0) {
        return -1;
    }
    wake_r_ = fds[0];
    wake_w_ = fds[1];
    make_nonblocking(wake_r_);
    make_nonblocking(wake_w_);

    listen_fd_ = locus::pipeline::listen_on(cfg_.host, cfg_.port, &bound_port_);
    if (listen_fd_ < 0) {
        return -1;
    }
    make_nonblocking(listen_fd_);
    // Re-issue listen() to set our backlog (listen_on uses a small fixed
    // one): connections beyond the worker pool queue here, and beyond the
    // backlog the kernel refuses them -- no userspace buffer, no 503.
    if (cfg_.listen_backlog > 0) {
        ::listen(listen_fd_, cfg_.listen_backlog);
    }

    int n = cfg_.worker_threads;
    if (n <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        n = static_cast<int>(hc > 1 ? hc - 1 : 1);
        if (n < 8) {
            n = 8;
        }
        if (n > 64) {
            n = 64;  // cap the auto size; bodies cost n x max_body_bytes
        }
    }
    workers_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
    return bound_port_;
}

void Server::stop() {
    if (stopping_.exchange(true)) {
        return;  // once
    }
    if (wake_w_ >= 0) {
        const char b = 1;
        ssize_t w = ::write(wake_w_, &b, 1);  // level-triggered wake-all
        (void)w;
    }
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    workers_.clear();
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (wake_r_ >= 0) {
        ::close(wake_r_);
        wake_r_ = -1;
    }
    if (wake_w_ >= 0) {
        ::close(wake_w_);
        wake_w_ = -1;
    }
}

void Server::worker_loop() {
    while (!stopping_.load()) {
        pollfd pfds[2] = {{listen_fd_, POLLIN, 0}, {wake_r_, POLLIN, 0}};
        const int pr = ::poll(pfds, 2, -1);
        if (pr < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if ((pfds[1].revents & POLLIN) != 0) {
            break;  // shutdown
        }
        if ((pfds[0].revents & POLLIN) == 0) {
            continue;
        }
        std::string peer_ip;
        const int fd = locus::pipeline::accept_one(listen_fd_, &peer_ip, 0);
        if (fd < 0) {
            continue;  // another worker took it, or a transient error
        }
        if (!locus::pipeline::ip_allowed(peer_ip, cfg_.allow)) {
            ::close(fd);
            continue;
        }
        serve_connection(fd, peer_ip);
        ::close(fd);
    }
}

void Server::serve_connection(int fd, const std::string& peer_ip) {
    (void)peer_ip;  // allowlist already enforced by worker_loop
    make_nonblocking(fd);
#ifdef SO_NOSIGPIPE
    int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#endif
    int nodelay = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    std::string buf;
    ParseContext ctx;
    int served = 0;
    bool req_in_flight = false;  // bytes of the current request have arrived
    bool continue_done = false;  // 100-continue decided for this request
    std::chrono::steady_clock::time_point head_deadline;

    for (;;) {
        if (stopping_.load()) {
            return;
        }
        Request req;
        const ParseResult pr = parse_request(buf, cfg_.limits, ctx, req);
        if (pr.state == ParseState::kComplete) {
            buf.erase(0, pr.consumed);  // #11: consume EXACTLY the request
            req_in_flight = false;
            continue_done = false;
            ++served;
            const bool last = served >= cfg_.max_requests_per_conn;
            const bool keep = req.keep_alive && !last && !stopping_.load();
            if (!handle_request(fd, req, keep) || !keep) {
                return;  // write failed, or keep-alive declined: close
            }
            continue;  // parse the next (possibly pipelined) request
        }
        if (pr.state == ParseState::kError) {
            write_status_response(fd, req.is_head, pr.suggested_status, false);
            return;  // #1: any parse error closes, never keep-alive
        }

        // kNeedMore: offer 100-continue if the head is in and asks for it.
        if (!continue_done) {
            const std::size_t he = buf.find("\r\n\r\n");
            if (he != std::string_view::npos) {
                continue_done = true;  // head complete: decision is made
                if (head_requests_continue(std::string_view(buf).substr(0,
                                                                        he))) {
                    static constexpr std::string_view k100 =
                        "HTTP/1.1 100 Continue\r\n\r\n";
                    if (!write_all_deadline(fd, k100, wake_r_,
                                            cfg_.write_deadline_ms,
                                            stopping_)) {
                        return;  // client gone
                    }
                }
            }
        }

        int timeout_ms;
        if (buf.empty()) {
            timeout_ms = cfg_.idle_timeout_ms;  // between requests
        } else {
            if (!req_in_flight) {
                req_in_flight = true;
                head_deadline = std::chrono::steady_clock::now() +
                                std::chrono::milliseconds(cfg_.head_deadline_ms);
            }
            const long rem = ms_until(head_deadline);
            if (rem <= 0) {
                write_status_response(fd, false, 408, false);
                return;
            }
            timeout_ms = static_cast<int>(rem);
        }

        pollfd pfds[2] = {{fd, POLLIN, 0}, {wake_r_, POLLIN, 0}};
        const int prd = ::poll(pfds, 2, timeout_ms);
        if (prd < 0) {
            if (errno == EINTR) {
                continue;
            }
            return;
        }
        if (prd == 0) {
            if (buf.empty()) {
                return;  // idle keep-alive timeout: close silently
            }
            write_status_response(fd, false, 408, false);
            return;  // head-receipt deadline tripped
        }
        if ((pfds[1].revents & POLLIN) != 0) {
            return;  // shutdown
        }
        if ((pfds[0].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
            char tmp[16384];
            const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
            if (n == 0) {
                return;  // peer closed
            }
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK ||
                    errno == EINTR) {
                    continue;
                }
                return;  // read error
            }
            buf.append(tmp, static_cast<std::size_t>(n));
        }
    }
}

bool Server::handle_request(int fd, const Request& req, bool keep) {
    ServerResponse res;
    handler_(req, res);

    if (res.provider) {
        Response head;
        head.status = res.status;
        head.headers = res.headers;
        std::string out, werr;
        if (!write_chunked_head(head, keep, out, werr)) {
            // Handler set a framing/unsafe header: answer 500 and close.
            return write_status_response(fd, req.is_head, 500, false), false;
        }
        if (!write_all_deadline(fd, out, wake_r_, cfg_.write_deadline_ms,
                                stopping_)) {
            return false;
        }
        ServerSink sink(fd, wake_r_, cfg_.write_deadline_ms, stopping_);
        res.provider(sink);
        sink.done();  // terminating chunk (idempotent if the handler did)
        return sink.ok() && keep;
    }

    Response r;
    r.status = res.status;
    r.headers = res.headers;
    r.body = res.body;
    std::string out, werr;
    if (!write_response(r, req.is_head, keep, out, werr)) {
        return write_status_response(fd, req.is_head, 500, false), false;
    }
    return write_all_deadline(fd, out, wake_r_, cfg_.write_deadline_ms,
                              stopping_) &&
           keep;
}

bool Server::write_status_response(int fd, bool head_request, int status,
                                   bool keep) {
    Response r;
    r.status = status;
    r.set_content(status_reason(status) + "\n", "text/plain");
    std::string out, werr;
    if (!write_response(r, head_request, keep, out, werr)) {
        return false;
    }
    return write_all_deadline(fd, out, wake_r_, cfg_.write_deadline_ms,
                              stopping_);
}

}  // namespace locus::http
