#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <unordered_map>
#include <vector>

#include "locus/http/loop_connection.hpp"
#include "locus/http/server.hpp"

namespace locus::http {

/**
 * The event-loop HTTP server (i#931): one I/O thread owns every socket
 * (sys::Poller and a LoopConnection per client), and a job pool runs the
 * handlers. An idle keep-alive connection or a slow reader then costs a
 * buffer, not a thread; the pool bounds only the requests being
 * computed. Same ServerConfig, Handler and HeadHandler as Server:
 *
 * - accept: the `allow` list and `max_conns_per_ip`, as Server;
 * - a `fast_path` request runs its head stage and handler on the I/O
 *   thread, so it answers while every job is busy; it must answer at
 *   once and never stream, as a wait there stalls every connection
 *   (locus-server keeps only /health on it);
 * - any other request goes to the pool: the head stage (on the whole
 *   request: LoopConnection has read the body, within limits), then the
 *   handler with a RequestContext whose client_alive is the reply's
 *   cancel flag. A buffered response goes out as is; a streamed one
 *   through a Sink whose write blocks at the reply's queue bound (the
 *   same backpressure as Server's blocking writes); a provider that
 *   throws ends the transfer truncated, as on Server;
 * - every answered request is recorded in `metrics`;
 * - head and body share one flat deadline (head_deadline_ms plus the
 *   largest body at min_ingest_bytes_per_sec), where Server credits the
 *   rate as the body arrives.
 *
 * Shutdown (stop): the I/O thread stops, every connection closes (which
 * cancels its pending reply, so producers stop), the pool drains, and
 * only then does the wake pipe close: no reply can write to it after.
 */
class LoopServer : public Service {
 public:
    LoopServer(ServerConfig cfg, Handler handler,
               HeadHandler head_handler = {});
    ~LoopServer() override;
    LoopServer(const LoopServer&) = delete;
    LoopServer& operator=(const LoopServer&) = delete;

    int start() override;
    void stop() override;
    void wait() override;

    /** @returns How many times the I/O thread has looked at one
     *  connection for a wake or a deadline (tests: the work per chunk
     *  must not grow with the connection count, i#990). */
    std::uint64_t visits() const { return visits_.load(); }

 private:
    /** One request for the pool. */
    struct Job {
        Request req;
        std::shared_ptr<LoopReply> reply;
    };
    using Clock = std::chrono::steady_clock;
    struct Conn {
        std::unique_ptr<LoopConnection> conn;
        std::string peer_ip;
        std::uint64_t id = 0;  ///< never reused (WakeList, timers_)
        bool idle = false;  ///< out of the poller (Next::kIdle)
        /** The time of its live timers_ entry (max: none). */
        Clock::time_point queued = Clock::time_point::max();
    };
    /** A deadline to look at: (when, connection id). Entries go stale
     *  when the deadline moves; a pop checks against Conn::queued. */
    using Timer = std::pair<Clock::time_point, std::uint64_t>;

    void io_loop();
    void accept_ready(sys::Poller& poller);
    /** Applies a LoopConnection's Next to fd (and closes it on kClose). */
    void apply(sys::Poller& poller, int fd, LoopConnection::Next next);
    void close_conn(sys::Poller& poller, int fd);
    /** Queues c's deadline when it moved earlier than its live entry (a
     *  later move is found when the entry pops). */
    void arm(Conn& c);
    /** @returns The poll timeout: to the nearest live deadline, at most a
     *  second (drops stale entries off the top). */
    int next_timeout_ms();
    /** Calls on_timeout on each connection whose deadline has passed. */
    void run_timers(sys::Poller& poller);
    /** LoopConnection's handler: a fast path inline, else a pool job. */
    void dispatch(const Request& req, std::shared_ptr<LoopReply> reply);
    void worker_loop();
    /** The head stage, the handler and the reply for one request. */
    void serve(const Request& req, const std::shared_ptr<LoopReply>& reply);
    void record(std::string_view path, int status, std::uint64_t req_bytes,
                std::uint64_t resp_bytes,
                std::chrono::steady_clock::time_point start);

    ServerConfig cfg_;
    Handler handler_;
    HeadHandler head_handler_;
    int listen_fd_ = -1;
    int bound_port_ = -1;
    int wake_r_ = -1;  ///< replies and stop() write here; the loop reads
    int wake_w_ = -1;
    std::atomic<bool> started_{false};
    std::atomic<bool> stopping_{false};
    std::thread io_;
    std::vector<std::thread> workers_;

    /** Replies push their connection's id here (i#990). */
    std::shared_ptr<WakeList> wake_list_;

    // I/O thread only.
    std::unordered_map<int, Conn> conns_;
    std::unordered_map<std::uint64_t, int> fd_of_;  ///< Conn::id -> fd
    std::uint64_t next_id_ = 1;
    std::priority_queue<Timer, std::vector<Timer>, std::greater<Timer>>
        timers_;
    std::unordered_map<std::string, std::size_t> per_ip_;
    std::atomic<std::uint64_t> visits_{0};

    std::mutex jobs_mu_;
    std::condition_variable jobs_cv_;
    std::deque<Job> jobs_;

    std::mutex wait_mu_;
    std::condition_variable wait_cv_;
};

}  // namespace locus::http
