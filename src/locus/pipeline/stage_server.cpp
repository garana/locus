#include "locus/pipeline/stage_server.hpp"

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "locus/pipeline/message.hpp"
#include "locus/pipeline/resolver.hpp"
#include "locus/pipeline/stage_executor.hpp"
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
            // With a resolver, turn a hostname into a cached IP first so
            // the dial does not pay a getaddrinfo (its DNS round-trip)
            // each time. An empty result (unresolved or a primed-but-
            // not-yet-filled name) is treated like an unreachable
            // replica: skip it and sweep on. Without a resolver,
            // connect_to resolves the host itself, as before.
            std::string target = hp.host;
            if (conn.resolver != nullptr) {
                target = conn.resolver->resolve(hp.host);
                if (target.empty()) {
                    if (n > 1) {
                        std::fprintf(stderr,
                                     "serve_stage: downstream %s "
                                     "unresolved, trying next in pool\n",
                                     hp.host.c_str());
                    }
                    continue;
                }
            }
            const int fd =
                connect_to(target, hp.port, conn.connect_timeout_ms);
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

// Flags for send() on the downstream: MSG_NOSIGNAL on Linux so a write
// to a closed peer returns EPIPE instead of raising SIGPIPE. macOS has
// no MSG_NOSIGNAL but connect_to sets SO_NOSIGPIPE on the fd, so 0 is
// safe there.
#if defined(MSG_NOSIGNAL)
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif

// One live session the mux loop multiplexes: an accepted upstream fd,
// the downstream fd dialed for it, this session's own KV sequence, a
// buffer holding the bytes of a not-yet-complete INPUT frame (so a frame
// split across reads is reassembled), and a buffer of OUTPUT bytes not
// yet accepted by the downstream (so a stalled downstream parks its
// backlog here instead of blocking the whole loop). `out_watched` is
// true while out_fd is registered for write-readiness -- only while
// outbuf has unsent bytes.
struct MuxSession {
    int in_fd = -1;
    int out_fd = -1;
    kv::PagedKvCache::Seq seq;
    std::string inbuf;
    std::string outbuf;
    bool out_watched = false;
    // Async-step state (executor seam, i#24 increment 1). A session has
    // at most one step in flight on the executor at a time (the loop
    // submits the next frame only on the previous completion), which is
    // what keeps ordering free. Teardown is deferred so the executor is
    // never holding this session's seq when it is freed: on a drop the
    // session is marked `cancelled` and kept alive (fds open, entry in
    // the map) until its KV release completes.
    bool in_flight = false;   // a step is queued/running on the executor
    bool cancelled = false;   // dropped; being torn down
    bool releasing = false;   // a release has been submitted
    bool eof = false;         // in_fd hit EOF/error; finish when idle
    bool read_error = false;  // that EOF was a socket error (unclean)
    bool clean_end = true;    // recorded cleanliness for the release
    // Which executor this session is pinned to (index into `executors`).
    // Fixed at accept for KV locality; its steps and release route here.
    std::size_t exec = 0;
};

// A pipe whose ends close on destruction. The executor's completion
// wake pipe is held in one of these, declared BEFORE the executor so it
// destructs AFTER it: the executor's destructor writes a wake byte per
// queued release while draining, so the write fd must still be open (and
// its number not yet reusable by another thread -- e.g. the Resolver's
// DNS thread opening sockets) until the executor is gone.
struct WakePipe {
    int r = -1;
    int w = -1;
    ~WakePipe() {
        if (r >= 0) ::close(r);
        if (w >= 0) ::close(w);
    }
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
                     const StageConn& conn, const StageReload& reload,
                     std::size_t max_batch) {
    std::vector<PipelineStage*> one{&stage};
    return serve_stage_mux(one, listen_fd, allow, downstreams, conn,
                           reload, max_batch);
}

bool serve_stage_mux(const std::vector<PipelineStage*>& stages,
                     int listen_fd, const std::vector<Cidr>& allow,
                     const std::vector<HostPort>& downstreams,
                     const StageConn& conn, const StageReload& reload,
                     std::size_t max_batch) {
    if (stages.empty()) {
        std::fprintf(stderr, "serve_stage_mux: no executors\n");
        ::close(listen_fd);
        return false;
    }
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
    // Keyed by input fd (the fd the poller reports). unique_ptr so a
    // session's address -- and the seq* the executor holds across an
    // in-flight step -- stays stable as the map rehashes. Declared
    // BEFORE the executor so it is destroyed AFTER it (reverse order),
    // keeping seqs alive while the executor's destructor drains any
    // queued releases.
    std::unordered_map<int, std::unique_ptr<MuxSession>> sessions;
    // Reverse index out_fd -> in_fd for write-readiness events.
    std::unordered_map<int, int> out_index;

    // Self-pipe carrying the executor's completion wakeups into the
    // poller (same shape as the SIGHUP wake). Non-blocking write end, per
    // the StageExecutor contract.
    // Declared before the executor so its fds outlive the executor's
    // destructor (see WakePipe); closed automatically, never in the
    // function body.
    WakePipe exec_wake;
    {
        int fds[2];
        if (::pipe(fds) != 0) {
            ::close(listen_fd);
            return false;
        }
        exec_wake.r = fds[0];
        exec_wake.w = fds[1];
    }
    set_nonblocking(exec_wake.r);
    set_nonblocking(exec_wake.w);
    poller.add(exec_wake.r);
    // One executor per stage, each a continuous-batching worker on its
    // own stage (own KV cache + workspace). They share this one wake
    // pipe (any completion wakes the loop, which drains them all). Owns
    // ALL stage mutation; the loop never calls step()/reset() itself.
    // Declared AFTER exec_wake so they destruct BEFORE it -- an
    // executor's destructor drains its release backlog and writes wake
    // bytes, which needs the pipe fd still open (and declared after
    // `sessions`, so the seqs it touches then are still alive).
    std::vector<std::unique_ptr<StageExecutor>> executors;
    executors.reserve(stages.size());
    for (PipelineStage* st : stages) {
        executors.push_back(
            std::make_unique<StageExecutor>(*st, exec_wake.w, max_batch));
    }
    // Live session count per executor, for least-loaded assignment.
    std::vector<std::size_t> exec_load(stages.size(), 0);

    // Pushes as much of `s.outbuf` to the downstream as it will take
    // without blocking, keeping frame order. Starts watching out_fd for
    // write-readiness when bytes remain and stops once drained.
    // @returns 0 fully flushed, 1 partial (bytes still pending), -1 the
    //     downstream errored and the session must be dropped.
    auto flush_out = [&](MuxSession& s) -> int {
        while (!s.outbuf.empty()) {
            const ssize_t n = ::send(s.out_fd, s.outbuf.data(),
                                     s.outbuf.size(), kSendFlags);
            if (n > 0) {
                s.outbuf.erase(0, static_cast<std::size_t>(n));
                continue;
            }
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                if (!s.out_watched) {
                    poller.add(s.out_fd, sys::Poller::Dir::kWrite);
                    s.out_watched = true;
                }
                return 1;  // backpressured; resume on the writable event
            }
            return -1;  // EPIPE / ECONNRESET etc: downstream is gone
        }
        if (s.out_watched) {
            poller.remove(s.out_fd);  // nothing left to send
            s.out_watched = false;
        }
        return 0;
    };

    // Frees a torn-down session once its KV release has completed on the
    // executor (so the executor is no longer touching its seq): drop its
    // fds and map entries and count it. Called only from a release
    // completion.
    auto free_session = [&](int fd) {
        auto it = sessions.find(fd);
        if (it == sessions.end()) {
            return;
        }
        MuxSession& s = *it->second;
        poller.remove(s.in_fd);
        if (s.out_watched) {
            poller.remove(s.out_fd);
        }
        out_index.erase(s.out_fd);
        ::close(s.in_fd);
        ::close(s.out_fd);
        if (!s.clean_end) {
            all_clean = false;
        }
        --exec_load[s.exec];  // this executor has one fewer live session
        ++completed;
        sessions.erase(it);
    };

    // Begins tearing a session down (EOF, error, decode/step/write
    // failure). Never frees it here -- the executor may hold its seq.
    // Stops watching its fds, records cleanliness, and routes the KV
    // release through the executor; if a step is in flight the release is
    // deferred to that step's completion (the no-release-under-a-running-
    // step rule). free_session runs when the release completes.
    auto begin_drop = [&](MuxSession& s, bool clean) {
        if (s.cancelled) {
            return;  // already tearing down
        }
        s.cancelled = true;
        s.clean_end = clean;
        poller.remove(s.in_fd);
        if (s.out_watched) {
            poller.remove(s.out_fd);
            s.out_watched = false;
        }
        if (!s.in_flight && !s.releasing) {
            executors[s.exec]->submit_release(s.in_fd, &s.seq);
            s.releasing = true;
        }
    };

    // Submits this session's next buffered frame to the executor if it
    // is idle -- one step in flight per session, which keeps ordering
    // free. A malformed frame tears the session down.
    auto pump = [&](MuxSession& s) {
        if (s.in_flight || s.cancelled) {
            return;
        }
        Message in;
        std::string err;
        const Decode d = decode(s.inbuf, in, err);
        if (d == Decode::kIncomplete) {
            return;
        }
        if (d == Decode::kError) {
            std::fprintf(stderr,
                         "serve_stage_mux: session dropped: "
                         "malformed frame: %s\n",
                         err.c_str());
            begin_drop(s, false);
            return;
        }
        executors[s.exec]->submit_step(s.in_fd, &s.seq, std::move(in));
        s.in_flight = true;
    };

    // After recv or a step completion: if the peer is gone and the
    // session is idle (no step in flight, no complete frame left -- pump
    // would have taken it), tear it down. Clean iff it closed at a frame
    // boundary (nothing half-read) and the socket did not error.
    auto maybe_finish = [&](MuxSession& s) {
        if (s.eof && !s.in_flight && !s.cancelled) {
            begin_drop(s, !s.read_error && s.inbuf.empty());
        }
    };

    bool result = true;
    std::vector<sys::Poller::Event> events;
    std::vector<StageExecutor::Completion> comps;
    for (;;) {
        const int pr = poller.wait(events, -1);
        if (pr < 0) {
            result = false;
            break;  // poll error
        }

        bool listen_ready = false;
        for (const auto& ev : events) {
            if (reload.wake_fd >= 0 && ev.fd == reload.wake_fd) {
                char buf[64];  // drain the (non-blocking) wake pipe
                while (::read(reload.wake_fd, buf, sizeof(buf)) > 0) {
                }
                continue;
            }
            if (ev.fd == exec_wake.r) {
                // The executor finished one or more jobs. Drain the wake
                // pipe and ALL completions (level-triggered: one wake may
                // cover several), then act on each.
                char buf[64];
                while (::read(exec_wake.r, buf, sizeof(buf)) > 0) {
                }
                comps.clear();
                for (auto& ex : executors) {
                    ex->drain(comps);  // one wake may cover any executor
                }
                for (auto& c : comps) {
                    auto cit = sessions.find(c.session);
                    if (cit == sessions.end()) {
                        continue;  // already freed (should not happen)
                    }
                    if (c.is_release) {
                        free_session(c.session);  // erases the session
                        continue;
                    }
                    MuxSession& s = *cit->second;
                    s.in_flight = false;
                    if (s.cancelled) {
                        // Dropped mid-step: the step is done with the seq,
                        // so release it now and discard the output.
                        if (!s.releasing) {
                            executors[s.exec]->submit_release(s.in_fd,
                                                              &s.seq);
                            s.releasing = true;
                        }
                        continue;
                    }
                    if (!c.ok) {
                        std::fprintf(stderr,
                                     "serve_stage_mux: session dropped: "
                                     "%s\n",
                                     c.err.c_str());
                        begin_drop(s, false);
                        continue;
                    }
                    // Send the output downstream, then submit the next
                    // buffered frame (or finish if the peer already left).
                    encode(c.out, s.outbuf);
                    if (flush_out(s) < 0) {
                        std::fprintf(stderr,
                                     "serve_stage_mux: session dropped: "
                                     "downstream write failed\n");
                        begin_drop(s, false);
                        continue;
                    }
                    pump(s);
                    maybe_finish(s);
                }
                continue;
            }
            if (ev.fd == listen_fd) {
                listen_ready = true;  // accept after the session fds
                continue;
            }
            // A downstream (out) fd becomes writable: resume the parked
            // flush for its session.
            if (const auto oit = out_index.find(ev.fd);
                oit != out_index.end()) {
                const auto sit = sessions.find(oit->second);
                if (sit != sessions.end() &&
                    flush_out(*sit->second) < 0) {
                    std::fprintf(stderr,
                                 "serve_stage_mux: session dropped: "
                                 "downstream write failed\n");
                    begin_drop(*sit->second, false);
                }
                continue;
            }
            // A session input fd. A missing entry means already gone.
            auto it = sessions.find(ev.fd);
            if (it == sessions.end()) {
                continue;
            }
            MuxSession& s = *it->second;
            if (s.cancelled) {
                continue;  // torn down; ignore a late event
            }

            // Drain every available byte into the frame buffer. read()==0
            // (clean EOF) or -1 with a non-retriable errno is the only
            // portable "peer gone" signal (sys::Poller contract); the
            // flags are hints and buffered bytes are processed first.
            for (;;) {
                char buf[4096];
                const ssize_t n = ::recv(s.in_fd, buf, sizeof(buf), 0);
                if (n > 0) {
                    s.inbuf.append(buf, static_cast<std::size_t>(n));
                    continue;
                }
                if (n == 0) {
                    s.eof = true;  // clean EOF
                    break;
                }
                if (errno == EINTR) {
                    continue;
                }
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    break;  // drained all that was ready
                }
                s.eof = true;
                s.read_error = true;  // e.g. ECONNRESET
                break;
            }
            if (s.eof) {
                // Nothing more to read; stop the level-triggered EOF from
                // re-firing while a step is still in flight.
                poller.remove(s.in_fd);
            }
            // Submit the next frame if idle, then (once idle with the
            // peer gone) tear down. The drop is honored only AFTER the
            // buffered frame is submitted, so a client that sends its last
            // frame and half-closes still gets its answer.
            pump(s);
            maybe_finish(s);
        }

        // Stop once enough sessions have ended (freed, i.e. their KV
        // released). Checked after the whole batch so a drop reached from
        // any event -- input, downstream flush, or a completion -- counts.
        if (conn_live.serve_sessions > 0 &&
            completed >= conn_live.serve_sessions) {
            result = all_clean;
            break;
        }

        // Apply a pending reload before accepting, so a new session uses
        // the fresh allowlist/pool/policy. Running sessions keep theirs.
        if (reload.flag != nullptr && *reload.flag != 0 && reload.apply) {
            *reload.flag = 0;
            reload.apply(allow_live, pool_live, conn_live);
        }

        // Accept at most one new session per wakeup (level-triggered, so
        // a backlog re-fires). Each session dials its own downstream and
        // gets its own KV sequence.
        if (listen_ready) {
            const int in_fd = accept_allowed(listen_fd, allow_live);
            if (in_fd < 0) {
                result = false;
                break;  // listener broken
            }
            set_keepalive(in_fd, conn_live.keepalive_idle_s,
                          conn_live.keepalive_intvl_s,
                          conn_live.keepalive_count);
            const int out_fd =
                connect_pool(pool_live, &cursor, conn_live);
            if (out_fd < 0) {
                ::close(in_fd);
                result = false;
                break;  // no live downstream in budget
            }
            set_keepalive(out_fd, conn_live.keepalive_idle_s,
                          conn_live.keepalive_intvl_s,
                          conn_live.keepalive_count);
            // Both ends non-blocking: in_fd so recv() drains without
            // blocking, out_fd so a stalled downstream parks its backlog
            // in outbuf (see flush_out) instead of blocking the loop.
            if (!set_nonblocking(in_fd) || !set_nonblocking(out_fd)) {
                ::close(in_fd);
                ::close(out_fd);
                result = false;
                break;
            }
            auto s = std::make_unique<MuxSession>();
            s->in_fd = in_fd;
            s->out_fd = out_fd;
            // Pin to the least-loaded executor (fewest live sessions;
            // ties to the lowest index) so load spreads across the CPU
            // lanes and the session's KV stays in that executor's cache.
            std::size_t best = 0;
            for (std::size_t i = 1; i < exec_load.size(); ++i) {
                if (exec_load[i] < exec_load[best]) {
                    best = i;
                }
            }
            s->exec = best;
            ++exec_load[best];
            poller.add(in_fd);  // out_fd watched only while backpressured
            out_index[out_fd] = in_fd;
            sessions.emplace(in_fd, std::move(s));
        }
    }

    // Shutdown. The executor still owns every live seq, so route each
    // remaining session's KV release through it (FIFO after any in-flight
    // step) rather than touching the stage here. Close the fds now but
    // leave the MuxSession objects alive: the executor's destructor --
    // which runs when this function returns, BEFORE `sessions` is
    // destroyed, since it is declared after it -- drains those queued
    // releases, resetting the seqs; then `sessions` frees the objects.
    for (auto& [fd, sp] : sessions) {
        (void)fd;
        if (!sp->releasing) {
            executors[sp->exec]->submit_release(sp->in_fd, &sp->seq);
            sp->releasing = true;
        }
        ::close(sp->in_fd);
        ::close(sp->out_fd);
    }
    ::close(listen_fd);
    // exec_wake closes itself (WakePipe dtor), AFTER the executor's
    // destructor has drained its release backlog -- so its write fd
    // stays valid (and its number unreusable) throughout that drain.
    return result;
}

}  // namespace locus::pipeline
