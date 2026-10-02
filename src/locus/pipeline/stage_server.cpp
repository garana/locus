#include "locus/pipeline/stage_server.hpp"

#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

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
    poller.add_read(listen_fd);
    if (reload.wake_fd >= 0) {
        poller.add_read(reload.wake_fd);
    }
    std::vector<int> ready;

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
            const int pr = poller.wait(ready, -1);
            if (pr < 0) {
                ::close(listen_fd);
                return false;  // poll error
            }
            bool listen_ready = false;
            for (const int fd : ready) {
                if (fd == reload.wake_fd) {
                    char buf[64];  // drain the (non-blocking) wake pipe
                    while (::read(reload.wake_fd, buf, sizeof(buf)) > 0) {
                    }
                } else if (fd == listen_fd) {
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

}  // namespace locus::pipeline
