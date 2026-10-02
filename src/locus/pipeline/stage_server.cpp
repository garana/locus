#include "locus/pipeline/stage_server.hpp"

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "locus/pipeline/message.hpp"
#include "locus/sys/poller.hpp"

namespace locus::pipeline {

namespace {

// Accepts one allowed input connection on listen_fd, logging (and
// dropping) peers outside `allow`. @returns the fd, or -1 if accept
// fails.
int accept_allowed(int listen_fd, const std::vector<Cidr>& allow) {
    for (;;) {
        std::string peer;
        const int fd = accept_one(listen_fd, &peer);
        if (fd < 0) {
            return -1;
        }
        if (!ip_allowed(peer, allow)) {
            std::fprintf(stderr,
                         "serve_stage: refused connection from %s "
                         "(not in --allow)\n",
                         peer.c_str());
            ::close(fd);
            continue;
        }
        return fd;
    }
}

// Connects to one live downstream from `pool`, round-robin starting at
// *cursor so successive sessions spread across the replicas. Each sweep
// tries every host once, in order, and uses the first that accepts (an
// established connection is the health check, so a dead replica is
// skipped). A sweep that finds none live counts as one failed attempt;
// per the reconnect policy it then waits reconnect_wait_ms and sweeps
// again (no backoff). On success *cursor is left just past the chosen
// host. @returns the fd, or -1 if it gives up (reconnect_attempts
// sweeps). Precondition: `pool` is non-empty (the `% n` below divides
// by pool.size()); serve_stage guarantees this before calling.
int connect_pool(const std::vector<HostPort>& pool, std::size_t* cursor,
                 const StageConn& conn) {
    const std::size_t n = pool.size();
    assert(n > 0 && "connect_pool requires a non-empty pool");
    for (int sweep = 1;; ++sweep) {
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t idx = (*cursor + k) % n;
            const HostPort& hp = pool[idx];
            const int fd =
                connect_to(hp.host, hp.port, conn.connect_timeout_ms);
            if (fd >= 0) {
                *cursor = (idx + 1) % n;  // next session starts here
                return fd;
            }
            if (n > 1) {
                std::fprintf(stderr,
                             "serve_stage: downstream %s:%d unreachable, "
                             "trying next in pool\n",
                             hp.host.c_str(), hp.port);
            }
        }
        if (conn.reconnect_attempts > 0 &&
            sweep >= conn.reconnect_attempts) {
            std::fprintf(stderr,
                         "serve_stage: no live downstream in a pool of "
                         "%zu after %d sweep(s)\n",
                         n, sweep);
            return -1;
        }
        std::fprintf(stderr,
                     "serve_stage: no live downstream, retrying in "
                     "%d ms\n",
                     conn.reconnect_wait_ms);
        std::this_thread::sleep_for(
            std::chrono::milliseconds(conn.reconnect_wait_ms));
    }
}

// Puts `fd` in non-blocking mode so the event loop can drain it with
// recv() without a slow or partial-frame peer stalling every other
// session. @returns false if the fcntl calls fail.
bool set_nonblocking(int fd) {
    const int fl = ::fcntl(fd, F_GETFL, 0);
    if (fl < 0) {
        return false;
    }
    return ::fcntl(fd, F_SETFL, fl | O_NONBLOCK) == 0;
}

// One live session the mux loop multiplexes: an accepted upstream fd,
// the downstream fd dialed for it, this session's own KV sequence, and
// a buffer holding the bytes of a not-yet-complete input frame (so a
// frame split across reads is reassembled, not lost).
struct MuxSession {
    int in_fd = -1;
    int out_fd = -1;
    kv::PagedKvCache::Seq seq;
    std::string inbuf;
};

}  // namespace

bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<Cidr>& allow,
                 const std::vector<HostPort>& downstreams,
                 const StageConn& conn, const StageReload& reload) {
    if (downstreams.empty()) {
        std::fprintf(stderr, "serve_stage: empty downstream pool\n");
        ::close(listen_fd);
        return false;
    }
    // Live, mutable copies so a hot reload (SIGHUP) can refresh the
    // allowlist, pool and policy between sessions without disturbing the
    // caller's originals.
    std::vector<Cidr> allow_live = allow;
    std::vector<HostPort> pool_live = downstreams;
    StageConn conn_live = conn;

    int served = 0;
    bool all_clean = true;   // any session ended abnormally?
    std::size_t cursor = 0;  // round-robin position in the pool

    // Event loop: watch the listener (and the reload wake pipe, if any)
    // for readiness rather than blocking in accept(), the foundation of
    // the multi-worker serving model (DESIGN.md i#24).
    sys::Poller poller;
    poller.add(listen_fd);
    if (reload.wake_fd >= 0) {
        poller.add(reload.wake_fd);
    }
    std::vector<sys::Poller::Event> events;

    for (;;) {
        // Wait for the next peer. A SIGHUP handler sets reload.flag and
        // writes reload.wake_fd; the poll returns and the reload is
        // applied before this session starts. Because the written byte
        // stays readable (level-triggered), a SIGHUP delivered in the
        // check-then-wait window is still pending and wakes the poll, so
        // an idle stage reloads promptly -- no lost-wakeup race. A reload
        // that arrives mid-session lands here, at the next boundary.
        for (;;) {
            if (reload.flag != nullptr && *reload.flag != 0 &&
                reload.apply) {
                *reload.flag = 0;  // clear first: a signal during apply
                                   // re-fires and is caught next loop
                reload.apply(allow_live, pool_live, conn_live);
            }
            const int pr = poller.wait(events, -1);
            if (pr < 0) {
                ::close(listen_fd);
                return false;  // poll error
            }
            bool listen_ready = false;
            for (const auto& ev : events) {
                if (ev.fd == reload.wake_fd) {
                    char buf[64];  // drain the (non-blocking) wake pipe
                    while (::read(reload.wake_fd, buf, sizeof(buf)) > 0) {
                    }
                } else if (ev.fd == listen_fd) {
                    listen_ready = true;
                }
            }
            // Apply a reload the wakeup signaled before accepting, so the
            // new session uses the fresh config.
            if (reload.flag != nullptr && *reload.flag != 0 &&
                reload.apply) {
                continue;
            }
            if (listen_ready) {
                break;  // a peer is pending, accept will not block
            }
            // Woke only for a reload with no peer: loop to re-wait.
        }

        // Automatic reconnect: each session re-accepts a fresh input
        // and re-dials the downstream, so a dropped peer (detected by
        // keepalive, or a clean EOF) is recovered by serving the next
        // connection rather than exiting. Per the terminal-kTimeout
        // contract, a session that ends always tears its fds down (in
        // stage.run) and we re-accept -- never re-read a dead fd.
        const int in_fd = accept_allowed(listen_fd, allow_live);
        if (in_fd < 0) {
            ::close(listen_fd);
            return false;  // accept failed (listener broken)
        }
        set_keepalive(in_fd, conn_live.keepalive_idle_s,
                      conn_live.keepalive_intvl_s,
                      conn_live.keepalive_count);
        if (conn_live.recv_timeout_ms > 0) {
            set_recv_timeout(in_fd, conn_live.recv_timeout_ms);
        }

        const int out_fd = connect_pool(pool_live, &cursor, conn_live);
        if (out_fd < 0) {
            ::close(in_fd);
            ::close(listen_fd);
            return false;  // no live downstream within the budget
        }
        set_keepalive(out_fd, conn_live.keepalive_idle_s,
                      conn_live.keepalive_intvl_s,
                      conn_live.keepalive_count);

        stage.reset();  // fresh KV state for this session's sequence
        if (!stage.run(in_fd, out_fd)) {  // closes in_fd and out_fd
            all_clean = false;
            std::fprintf(stderr,
                         "serve_stage: session ended abnormally; "
                         "re-accepting\n");
        }
        ++served;
        if (conn_live.serve_sessions > 0 &&
            served >= conn_live.serve_sessions) {
            ::close(listen_fd);
            return all_clean;  // false if any session ended abnormally
        }
    }
}

bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<Cidr>& allow,
                 const std::string& downstream_host,
                 int downstream_port, const StageConn& conn) {
    return serve_stage(stage, listen_fd, allow,
                       {{downstream_host, downstream_port}}, conn);
}

bool serve_stage_mux(PipelineStage& stage, int listen_fd,
                     const std::vector<Cidr>& allow,
                     const std::vector<HostPort>& downstreams,
                     const StageConn& conn, const StageReload& reload) {
    if (downstreams.empty()) {
        std::fprintf(stderr, "serve_stage_mux: empty downstream pool\n");
        ::close(listen_fd);
        return false;
    }
    // Live, mutable copies so a hot reload (SIGHUP) can refresh the
    // allowlist, pool and policy for future accepts without disturbing
    // the caller's originals or any session already running.
    std::vector<Cidr> allow_live = allow;
    std::vector<HostPort> pool_live = downstreams;
    StageConn conn_live = conn;

    std::size_t cursor = 0;  // round-robin position in the pool
    int completed = 0;       // sessions that have ended
    bool all_clean = true;   // any session ended abnormally?

    sys::Poller poller;
    poller.add(listen_fd);
    if (reload.wake_fd >= 0) {
        poller.add(reload.wake_fd);
    }
    // Keyed by input fd (the fd the poller reports). Each entry owns its
    // own downstream fd and KV sequence, so sessions are independent.
    std::unordered_map<int, MuxSession> sessions;

    // Ends one session: unregister its input fd, free its KV, close both
    // fds. `clean` records whether it ended at a frame boundary (for the
    // return value and the serve_sessions count).
    auto drop = [&](int fd, bool clean) {
        auto it = sessions.find(fd);
        if (it == sessions.end()) {
            return;
        }
        poller.remove(fd);
        stage.reset(it->second.seq);
        ::close(it->second.in_fd);
        ::close(it->second.out_fd);
        sessions.erase(it);
        if (!clean) {
            all_clean = false;
        }
        ++completed;
    };

    // Tears down everything and returns `ok`; used on both the normal
    // serve_sessions stop and a fatal error.
    auto shutdown = [&](bool ok) {
        for (auto& [fd, s] : sessions) {
            (void)fd;
            stage.reset(s.seq);
            ::close(s.in_fd);
            ::close(s.out_fd);
        }
        sessions.clear();
        ::close(listen_fd);
        return ok;
    };

    std::vector<sys::Poller::Event> events;
    for (;;) {
        const int pr = poller.wait(events, -1);
        if (pr < 0) {
            return shutdown(false);  // poll error
        }

        bool listen_ready = false;
        for (const auto& ev : events) {
            if (reload.wake_fd >= 0 && ev.fd == reload.wake_fd) {
                char buf[64];  // drain the (non-blocking) wake pipe
                while (::read(reload.wake_fd, buf, sizeof(buf)) > 0) {
                }
                continue;
            }
            if (ev.fd == listen_fd) {
                listen_ready = true;  // accept after the session fds
                continue;
            }
            // A session input fd. It may have been dropped earlier in
            // this same batch; a reused fd number is only handed out by
            // the accept below (which runs after this loop), so a
            // missing entry just means "already gone" -- skip it.
            auto it = sessions.find(ev.fd);
            if (it == sessions.end()) {
                continue;
            }
            MuxSession& s = it->second;

            // Drain every currently-available byte into the frame
            // buffer. read()==0 (clean EOF) or -1 with a non-retriable
            // errno is the only portable "peer gone" signal (see the
            // sys::Poller contract); the ev.hangup/ev.error flags are
            // hints we deliberately do not act on alone, and any
            // buffered bytes are decoded below before the drop.
            bool gone = false;
            bool read_error = false;
            for (;;) {
                char buf[4096];
                const ssize_t n = ::recv(s.in_fd, buf, sizeof(buf), 0);
                if (n > 0) {
                    s.inbuf.append(buf, static_cast<std::size_t>(n));
                    continue;
                }
                if (n == 0) {
                    gone = true;  // clean EOF
                    break;
                }
                if (errno == EINTR) {
                    continue;
                }
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    break;  // drained all that was ready
                }
                gone = true;
                read_error = true;  // e.g. ECONNRESET
                break;
            }

            // Process every complete frame now buffered, in order.
            bool dropped = false;
            for (;;) {
                Message in;
                std::string err;
                const Decode d = decode(s.inbuf, in, err);
                if (d == Decode::kIncomplete) {
                    break;  // wait for more bytes
                }
                if (d == Decode::kError) {
                    drop(s.in_fd, false);
                    dropped = true;
                    break;
                }
                Message out;
                try {
                    out = stage.step(s.seq, in);
                } catch (const std::exception&) {
                    drop(s.in_fd, false);
                    dropped = true;
                    break;
                }
                // Blocking write of a small per-token frame; a stalled
                // downstream can briefly block the loop. Write-readiness
                // in the poller is a later increment (i#24).
                if (!write_message(s.out_fd, out)) {
                    drop(s.in_fd, false);
                    dropped = true;
                    break;
                }
            }

            if (!dropped && gone) {
                // Clean only if the peer closed at a frame boundary
                // (nothing half-read) and the socket itself did not
                // error -- mirrors read_message's EOF-vs-truncation rule.
                const bool clean = !read_error && s.inbuf.empty();
                drop(s.in_fd, clean);
            }

            if (conn_live.serve_sessions > 0 &&
                completed >= conn_live.serve_sessions) {
                return shutdown(all_clean);
            }
        }

        // Apply a pending reload (set by the SIGHUP handler and signaled
        // on the wake pipe) before accepting, so a new session uses the
        // fresh allowlist/pool/policy. Sessions already running keep the
        // policy they started with.
        if (reload.flag != nullptr && *reload.flag != 0 && reload.apply) {
            *reload.flag = 0;
            reload.apply(allow_live, pool_live, conn_live);
        }

        // Accept at most one new session per wakeup (the listener is
        // level-triggered, so a backlog re-fires on the next wait).
        // Each session dials its own downstream and gets its own KV
        // sequence, so concurrent sessions never share state.
        if (listen_ready) {
            const int in_fd = accept_allowed(listen_fd, allow_live);
            if (in_fd < 0) {
                return shutdown(false);  // listener broken
            }
            set_keepalive(in_fd, conn_live.keepalive_idle_s,
                          conn_live.keepalive_intvl_s,
                          conn_live.keepalive_count);
            const int out_fd =
                connect_pool(pool_live, &cursor, conn_live);
            if (out_fd < 0) {
                ::close(in_fd);
                return shutdown(false);  // no live downstream in budget
            }
            set_keepalive(out_fd, conn_live.keepalive_idle_s,
                          conn_live.keepalive_intvl_s,
                          conn_live.keepalive_count);
            if (!set_nonblocking(in_fd)) {
                ::close(in_fd);
                ::close(out_fd);
                return shutdown(false);
            }
            MuxSession s;
            s.in_fd = in_fd;
            s.out_fd = out_fd;
            poller.add(in_fd);
            sessions.emplace(in_fd, std::move(s));
        }
    }
}

}  // namespace locus::pipeline
