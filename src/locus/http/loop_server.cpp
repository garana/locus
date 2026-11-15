#include "locus/http/loop_server.hpp"

#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <exception>
#include <utility>

#include "locus/pipeline/net.hpp"

namespace locus::http {

namespace {

using Clock = std::chrono::steady_clock;

void make_nonblocking(int fd) {
    const int fl = ::fcntl(fd, F_GETFL, 0);
    if (fl >= 0) {
        ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    }
}

/** A stream's Sink over a LoopReply: write blocks at the reply's queue
 *  bound, and fails once the client is gone. */
class ReplySink : public Sink {
 public:
    explicit ReplySink(std::shared_ptr<LoopReply> r) : r_(std::move(r)) {}
    bool write(std::string_view data) override {
        if (!ok_) return false;
        ok_ = r_->write(data);
        if (ok_) bytes_ += data.size();
        return ok_;
    }
    void done() override { r_->end(); }
    std::uint64_t bytes() const { return bytes_; }

 private:
    std::shared_ptr<LoopReply> r_;
    bool ok_ = true;
    std::uint64_t bytes_ = 0;
};

/** @returns res as the parser's Response (status, headers, body). */
Response to_response(ServerResponse& res) {
    Response r;
    r.status = res.status;
    r.headers = std::move(res.headers);
    r.body = std::move(res.body);
    return r;
}

}  // namespace

LoopServer::LoopServer(ServerConfig cfg, Handler handler,
                       HeadHandler head_handler)
    : cfg_(std::move(cfg)),
      handler_(std::move(handler)),
      head_handler_(std::move(head_handler)) {}

LoopServer::~LoopServer() { stop(); }

int LoopServer::start() {
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
    if (!locus::pipeline::attach_allow_filter(listen_fd_, cfg_.allow)) {
        std::fprintf(stderr,
                     "http server: --allow is not in the kernel (filter "
                     "refused); it applies at accept\n");
    }
    if (cfg_.listen_backlog > 0) {
        ::listen(listen_fd_, cfg_.listen_backlog);
    }
    // The pool bounds the requests being computed, as Server's workers
    // bound its connections.
    int n = cfg_.worker_threads;
    if (n <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        n = std::clamp(static_cast<int>(hc > 1 ? hc - 1 : 1), 8, 64);
    }
    workers_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
    io_ = std::thread([this] { io_loop(); });
    return bound_port_;
}

void LoopServer::wait() {
    std::unique_lock<std::mutex> lk(wait_mu_);
    wait_cv_.wait(lk, [this] { return stopping_.load(); });
}

void LoopServer::stop() {
    if (stopping_.exchange(true)) {
        return;  // once
    }
    {
        std::lock_guard<std::mutex> lk(wait_mu_);
    }
    wait_cv_.notify_all();
    if (wake_w_ >= 0) {
        const char b = 1;
        const ssize_t w = ::write(wake_w_, &b, 1);
        (void)w;
    }
    // The I/O thread closes every connection as it leaves, which cancels
    // their replies, so a producer blocked on one returns.
    if (io_.joinable()) {
        io_.join();
    }
    {
        std::lock_guard<std::mutex> lk(jobs_mu_);
        jobs_.clear();  // their clients are gone
    }
    jobs_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    // Last: no reply can write to the wake pipe now.
    for (int* fd : {&listen_fd_, &wake_r_, &wake_w_}) {
        if (*fd >= 0) {
            ::close(*fd);
            *fd = -1;
        }
    }
}

void LoopServer::io_loop() {
    sys::Poller poller;
    poller.add(listen_fd_);
    poller.add(wake_r_);
    std::vector<sys::Poller::Event> events;
    std::vector<int> fds;
    while (!stopping_.load()) {
        // Sleep to the nearest connection deadline (at most a second).
        const auto now = Clock::now();
        auto timeout = std::chrono::milliseconds(1000);
        for (const auto& [fd, c] : conns_) {
            const auto d = c.conn->deadline();
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(d - now);
            timeout = std::max(std::chrono::milliseconds(0),
                               std::min(timeout, left));
        }
        if (poller.wait(events, static_cast<int>(timeout.count())) < 0) {
            continue;
        }
        bool woke = false;
        for (const auto& ev : events) {
            if (ev.fd == listen_fd_) {
                accept_ready(poller);
            } else if (ev.fd == wake_r_) {
                woke = true;
            } else if (const auto it = conns_.find(ev.fd);
                       it != conns_.end()) {
                apply(poller, ev.fd, it->second.conn->on_event(ev));
            }
        }
        if (woke) {
            // Read before taking the replies: a wake after this read is
            // another event, so none is lost. One wake covers them all.
            char buf[256];
            while (::read(wake_r_, buf, sizeof(buf)) > 0) {
            }
            if (stopping_.load()) {
                break;
            }
            fds.clear();
            for (const auto& [fd, c] : conns_) fds.push_back(fd);
            for (const int fd : fds) {
                if (const auto it = conns_.find(fd); it != conns_.end()) {
                    apply(poller, fd, it->second.conn->on_wake());
                }
            }
        }
        const auto later = Clock::now();
        fds.clear();
        for (const auto& [fd, c] : conns_) {
            if (c.conn->deadline() <= later) fds.push_back(fd);
        }
        for (const int fd : fds) {
            if (const auto it = conns_.find(fd); it != conns_.end()) {
                apply(poller, fd, it->second.conn->on_timeout());
            }
        }
    }
    fds.clear();
    for (const auto& [fd, c] : conns_) fds.push_back(fd);
    for (const int fd : fds) {
        close_conn(poller, fd);
    }
}

void LoopServer::accept_ready(sys::Poller& poller) {
    for (;;) {
        std::string peer_ip;
        const int fd = locus::pipeline::accept_one(listen_fd_, &peer_ip, 0);
        if (fd < 0) {
            return;  // none left (EAGAIN), or transient
        }
        if (!locus::pipeline::ip_allowed(peer_ip, cfg_.allow)) {
            ::close(fd);
            continue;
        }
        // i#625 F21: a peer at its cap is closed at once.
        if (cfg_.max_conns_per_ip > 0 &&
            per_ip_[peer_ip] >= cfg_.max_conns_per_ip) {
            ::close(fd);
            continue;
        }
        ++per_ip_[peer_ip];
        make_nonblocking(fd);
        const int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        LoopConfig lc;
        lc.limits = cfg_.limits;
        lc.max_requests_per_conn = cfg_.max_requests_per_conn;
        // A flat bound for head and body: the head deadline plus the
        // largest body at the minimum ingest rate (Server credits the
        // rate as the body arrives; the loop keeps one deadline).
        const long long rate =
            cfg_.min_ingest_bytes_per_sec > 0 ? cfg_.min_ingest_bytes_per_sec
                                              : 32 * 1024;
        lc.request_deadline_ms = static_cast<int>(std::min<long long>(
            cfg_.head_deadline_ms +
                static_cast<long long>(cfg_.limits.max_body_bytes) * 1000 /
                    rate,
            24LL * 3600 * 1000));
        lc.idle_timeout_ms = cfg_.idle_timeout_ms;
        lc.write_deadline_ms = cfg_.write_deadline_ms;
        lc.wake_fd = wake_w_;
        auto conn = std::make_unique<LoopConnection>(
            fd, lc,
            AsyncLoopHandler(
                [this](const Request& req, std::shared_ptr<LoopReply> r) {
                    dispatch(req, std::move(r));
                }));
        poller.add(fd);
        conns_.emplace(fd, Conn{std::move(conn), peer_ip});
    }
}

void LoopServer::apply(sys::Poller& poller, int fd,
                       LoopConnection::Next next) {
    const auto it = conns_.find(fd);
    if (it == conns_.end()) {
        return;
    }
    Conn& c = it->second;
    const auto watch = [&](sys::Poller::Dir d) {
        if (c.idle) {
            poller.add(fd, d);
            c.idle = false;
        } else {
            poller.modify(fd, d);
        }
    };
    switch (next) {
        case LoopConnection::Next::kRead:
            watch(sys::Poller::Dir::kRead);
            return;
        case LoopConnection::Next::kWrite:
            watch(sys::Poller::Dir::kWrite);
            return;
        case LoopConnection::Next::kIdle:
            // A half-closed client's socket stays readable: watch nothing
            // until its reply wakes the loop.
            if (!c.idle) {
                poller.remove(fd);
                c.idle = true;
            }
            return;
        case LoopConnection::Next::kClose:
            close_conn(poller, fd);
            return;
    }
}

void LoopServer::close_conn(sys::Poller& poller, int fd) {
    const auto it = conns_.find(fd);
    if (it == conns_.end()) {
        return;
    }
    const std::string ip = it->second.peer_ip;
    conns_.erase(it);  // cancels a pending reply
    poller.remove(fd);
    ::close(fd);
    const auto p = per_ip_.find(ip);
    if (p != per_ip_.end() && --p->second == 0) {
        per_ip_.erase(p);
    }
}

void LoopServer::dispatch(const Request& req,
                          std::shared_ptr<LoopReply> reply) {
    // The cheap control routes answer on the I/O thread, so they do not
    // queue behind the pool (as Server's acceptor fast path).
    if (cfg_.fast_path && cfg_.fast_path(req.method, req.target)) {
        serve(req, reply);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(jobs_mu_);
        jobs_.push_back(Job{req, std::move(reply)});
    }
    jobs_cv_.notify_one();
}

void LoopServer::worker_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(jobs_mu_);
            jobs_cv_.wait(lk, [this] {
                return stopping_.load() || !jobs_.empty();
            });
            if (jobs_.empty()) {
                return;  // stopping
            }
            job = std::move(jobs_.front());
            jobs_.pop_front();
        }
        if (!job.reply->cancelled()) {
            serve(job.req, job.reply);
        }
    }
}

void LoopServer::serve(const Request& req,
                       const std::shared_ptr<LoopReply>& reply) {
    const auto start = Clock::now();
    HttpMetrics* m = cfg_.metrics.get();
    if (m != nullptr) m->in_flight.fetch_add(1, std::memory_order_relaxed);
    struct Done {
        HttpMetrics* m;
        ~Done() {
            if (m != nullptr) {
                m->in_flight.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    } done{m};
    const std::uint64_t req_bytes = req.body.size();
    RequestContext ctx;
    ctx.client_alive = [reply] { return !reply->client_gone(); };
    if (head_handler_) {
        ServerResponse hres;
        bool reject = false;
        try {
            reject = head_handler_(req, hres, ctx);
        } catch (...) {
            reply->fail();  // pending: a 500 (i#243)
            record(req.path, 500, req_bytes, 0, start);
            return;
        }
        if (reject) {
            const int status = hres.status;
            const std::uint64_t n = req.is_head ? 0 : hres.body.size();
            reply->send(to_response(hres));
            record(req.path, status, req_bytes, n, start);
            return;
        }
    }
    ServerResponse res;
    try {
        handler_(req, res, ctx);
    } catch (...) {
        // Last resort (i#243): a 500, and the server lives.
        reply->fail();
        record(req.path, 500, req_bytes, 0, start);
        return;
    }
    if (res.provider) {
        Response head;
        head.status = res.status;
        head.headers = std::move(res.headers);
        if (!reply->begin_stream(std::move(head))) {
            record(req.path, res.status, req_bytes, 0, start);
            return;
        }
        if (req.is_head) {
            // RFC 9110: the head a GET would get, no body (do not run the
            // provider).
            reply->end();
            record(req.path, res.status, req_bytes, 0, start);
            return;
        }
        ReplySink sink(reply);
        // The head is out: a truncated transfer, logged and counted, as
        // Server does (i#980).
        const auto failed = [&](const char* why) {
            std::fprintf(stderr,
                         "locus-server: %s %s: the stream failed after its "
                         "head: %s\n",
                         req.method.c_str(), req.path.c_str(), why);
            if (m != nullptr) {
                m->stream_failed(m->route(req.path));
            }
            reply->fail();
            record(req.path, res.status, req_bytes, sink.bytes(), start);
        };
        try {
            res.provider(sink);
        } catch (const std::exception& e) {
            failed(e.what());
            return;
        } catch (...) {
            failed("unknown exception");
            return;
        }
        reply->end();
        record(req.path, res.status, req_bytes, sink.bytes(), start);
        return;
    }
    const int status = res.status;
    const std::uint64_t n = req.is_head ? 0 : res.body.size();
    reply->send(to_response(res));
    record(req.path, status, req_bytes, n, start);
}

void LoopServer::record(std::string_view path, int status,
                        std::uint64_t req_bytes, std::uint64_t resp_bytes,
                        Clock::time_point start) {
    HttpMetrics* m = cfg_.metrics.get();
    if (m == nullptr) return;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now() - start)
                        .count();
    m->record(m->route(path), status, req_bytes, resp_bytes,
              ns > 0 ? static_cast<std::uint64_t>(ns) : 0);
}

}  // namespace locus::http
