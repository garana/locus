#pragma once

#include <csignal>
#include <functional>
#include <string>
#include <vector>

#include "locus/pipeline/net.hpp"
#include "locus/pipeline/stage.hpp"

namespace locus::pipeline {

class Resolver;  // cached hostname resolver (resolver.hpp)

/** A downstream endpoint (host + port) in a stage's downstream pool. */
struct HostPort {
    std::string host;
    int port = 0;
};

/** Timeouts / reconnect / keepalive policy for serve_stage. */
struct StageConn {
    int connect_timeout_ms = 0;    /**< 0 = OS default (blocking). */
    int reconnect_wait_ms = 1000;  /**< Fixed wait between downstream
                                    *   connect attempts (no backoff). */
    int reconnect_attempts = 0;    /**< 0 = retry until connected. */
    int recv_timeout_ms = 0;       /**< 0 = block; else the input read
                                    *   times out (kTimeout). */
    // TCP keepalive on the input + downstream sockets (seconds), so a
    // silently-dropped peer is detected in ~idle + intvl*count seconds.
    int keepalive_idle_s = 5;      /**< <= 0 disables keepalive. */
    int keepalive_intvl_s = 2;
    int keepalive_count = 3;
    int serve_sessions = 0;        /**< 0 = serve forever; else stop
                                    *   after N sessions (for tests). */
    /**
     * Optional cached hostname resolver (issue 40). When set, a
     * downstream whose host is a name is resolved to an IP through this
     * cache before connecting, so a per-dial getaddrinfo (and its
     * blocking DNS round-trip on the serve loop) is avoided on the
     * common path. nullptr (the default) keeps the old behavior:
     * connect_to resolves the host itself each dial. Not owned here --
     * the caller (stage_main) owns it and outlives serve_stage*.
     */
    Resolver* resolver = nullptr;
};

/**
 * Hot-reload hook for serve_stage (i#19 SIGHUP reload). serve_stage
 * waits for the next peer in an event loop (sys::Poller) that also
 * watches `wake_fd`; a SIGHUP handler sets `flag` and writes a byte to
 * the wake pipe, which makes the poll return at once. serve_stage then
 * clears `flag` and calls `apply` to refresh the live allowlist,
 * downstream pool and connection policy BEFORE accepting the next
 * session. Reloaded values take effect from that session on; a signal
 * arriving mid-session is applied at the next session boundary.
 *
 * The wake pipe closes the check-then-poll race a bare flag would have:
 * because the written byte stays readable (level-triggered), a SIGHUP
 * delivered between the flag check and the poll is still pending and
 * wakes it, so an idle stage reloads promptly rather than waiting for
 * the next peer.
 *
 * `flag` must be the object the handler sets, and `wake_fd` the read
 * end of the pipe it writes (both async-signal-safe: a write to a
 * volatile sig_atomic_t and a write() of one byte). serve_stage reads
 * and clears `flag` and drains `wake_fd`. Default-constructed (flag
 * null, wake_fd -1) means no reload: serve_stage keeps its original
 * behavior, waiting on the listener alone.
 */
struct StageReload {
    volatile std::sig_atomic_t* flag = nullptr;
    int wake_fd = -1;  /**< self-pipe read end the handler writes. */
    std::function<void(std::vector<Cidr>&, std::vector<HostPort>&,
                       StageConn&)>
        apply = nullptr;
};

/**
 * Runs one pipeline stage as a server (the core of the locus-stage
 * CLI). In a loop it: accepts one input connection on `listen_fd`
 * (rejecting peers not in `allow`; an empty allow means allow all),
 * applies keepalive and `conn.recv_timeout_ms` to that input, connects
 * to a downstream (with `conn.connect_timeout_ms`, retrying on a fixed
 * `conn.reconnect_wait_ms` since a stage may start before its
 * downstream), applies keepalive to the downstream, resets the stage,
 * and runs its read -> step -> write loop until the session ends. It
 * then re-accepts the next session -- the automatic reconnect -- so a
 * dropped peer is recovered by serving the next connection rather than
 * exiting.
 *
 * `downstreams` is a pool of one or more interchangeable downstream
 * replicas (each holds the same next layer range). There is no load
 * balancer: successive sessions round-robin across the pool, and each
 * connect sweeps the whole pool in order, using the first replica that
 * accepts. An established connection is itself the health check, so a
 * dead replica is simply skipped and a live one is used; a sweep that
 * finds none live counts as one failed attempt (then wait + retry per
 * the reconnect policy). A single-element pool is the plain one
 * downstream case.
 *
 * The pool buys failover and sequential spreading, not load balancing.
 * A successful connect only proves the replica's listener is up, not
 * that the replica is free: the kernel completes the handshake into the
 * listen backlog even while the replica is busy in a session, so a
 * sweep cannot tell a busy replica from an idle one and just takes the
 * first that accepts. (In a linear chain only one session is ever in
 * flight, so there is nothing to contend with; contention appears only
 * once two upstreams share a pool.) There is also no memory of dead
 * replicas: each session re-probes from the cursor, so a recovered
 * replica is picked up automatically, but a filtered (blackholed)
 * replica listed before a live one can cost up to connect_timeout_ms
 * per session.
 *
 * conn.serve_sessions bounds the loop: 0 serves forever (a long-lived
 * stage; then this returns only on a fatal error), N stops after N
 * sessions (used by tests). The listen fd stays open across sessions
 * and is closed only when this function returns. CPU/CUDA only.
 *
 * `reload` (optional) hot-reloads the allowlist, pool and policy between
 * sessions on a signal; see StageReload. serve_stage works on mutable
 * copies of `allow`/`downstreams`/`conn`, so the caller's originals are
 * untouched.
 *
 * @returns With serve_sessions > 0: true if all N sessions ended
 *     cleanly, false if any ended abnormally. Either way false on a
 *     fatal accept failure or no live downstream within
 *     reconnect_attempts sweeps. With serve_sessions == 0 it returns
 *     only on such a fatal error (false).
 */
bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<Cidr>& allow,
                 const std::vector<HostPort>& downstreams,
                 const StageConn& conn = {},
                 const StageReload& reload = {});

/** serve_stage with a single downstream (a one-element pool). */
bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<Cidr>& allow,
                 const std::string& downstream_host,
                 int downstream_port, const StageConn& conn = {});

/**
 * Concurrent (event-loop) serving model for one stage, the multiplexed
 * counterpart to serve_stage (DESIGN.md i#24). Where serve_stage runs
 * one session at a time -- accept, serve to EOF, re-accept -- this
 * watches the listener and every active input connection together in a
 * sys::Poller and interleaves them: one slow or idle peer never blocks
 * the others, so N upstreams are served at once over a single thread.
 *
 * Each accepted connection becomes an independent session: it dials its
 * own downstream from `downstreams` (round-robin, same pool semantics as
 * serve_stage), gets its own KV sequence in the stage's shared cache
 * (PipelineStage::step(seq, in)), and buffers its own partial input and
 * unsent output frames. Both sockets are non-blocking: input is drained
 * with recv() and reassembled with the codec's decode() (a frame split
 * across reads is not lost), and each output frame is queued and pushed
 * with send() as far as the downstream will take it. A stalled
 * downstream parks its backlog in the session's out buffer and the loop
 * watches that fd for write-readiness (sys::Poller kWrite), resuming the
 * flush on the writable event, so one slow downstream no longer blocks
 * the other sessions. Frames append in order, so ordering holds across
 * backpressure.
 *
 * Dropping a peer follows the sys::Poller contract exactly: a session is
 * ended only after a read drains to 0 (clean EOF) or -1 with a
 * non-retriable errno, never on a bare hangup/error flag, and any bytes
 * already buffered are decoded and processed before the drop (so a
 * hangup that arrives with a final frame still delivers it). A session
 * that ends with a half-read frame or a socket error is counted as
 * unclean. An upstream EOF tears the session down including any unsent
 * output: the request is abandoned, so the in-flight downstream frame is
 * moot and the downstream sees its own EOF and drops in turn. Ending a
 * session frees its KV at once.
 *
 * Known limit this increment (follow-up under i#24): the connect to a
 * new session's downstream is still a blocking dial at accept time (a
 * slow/unreachable downstream can delay accepting the next session).
 *
 * `conn.serve_sessions` bounds the loop for tests: 0 serves forever
 * (returns only on a fatal error), N stops once N sessions have ended.
 * `reload` (optional) applies between wakeups to the allowlist, pool and
 * policy used for FUTURE accepts; running sessions are untouched.
 *
 * @returns With serve_sessions > 0: true if all N sessions ended
 *     cleanly, false if any ended abnormally or on a fatal accept / no
 *     live downstream / poll error. With serve_sessions == 0: returns
 *     only on such a fatal error (false). CPU/CUDA only.
 */
bool serve_stage_mux(PipelineStage& stage, int listen_fd,
                     const std::vector<Cidr>& allow,
                     const std::vector<HostPort>& downstreams,
                     const StageConn& conn = {},
                     const StageReload& reload = {},
                     std::size_t max_batch = 16);

/**
 * Multiple-executor form (i#24 increment 3, DESIGN.md "per-device
 * batching executors"): serves sessions across one CPU executor per
 * entry in `stages`. Each executor owns its own stage -- its own KV
 * cache and workspace over the shared model -- and runs a continuous-
 * batching worker thread, so the box's cores are split into
 * stages.size() batching lanes instead of one.
 *
 * A new session is assigned to the least-loaded executor (fewest live
 * sessions) and pinned to it for its whole life, so its KV sequence
 * stays resident in that executor's cache. All executors share one
 * completion wake pipe; the loop drains every executor per wake and
 * routes each session's steps and its KV release to its own executor.
 * Everything else (the I/O loop, backpressure, drop/cancel rules,
 * reload) is identical to the single-stage form, which forwards here
 * with one stage.
 *
 * @param stages One stage per executor; must be non-empty and each must
 *     outlive the call. A single-element vector is exactly the
 *     single-stage behaviour.
 * @param max_batch Per-executor cap on the coalesced batch width
 *     (StageExecutor); bounds the transient activation memory and the
 *     first-queued session's wait.
 * @returns Same contract as the single-stage form. CPU/CUDA only.
 */
bool serve_stage_mux(const std::vector<PipelineStage*>& stages,
                     int listen_fd, const std::vector<Cidr>& allow,
                     const std::vector<HostPort>& downstreams,
                     const StageConn& conn = {},
                     const StageReload& reload = {},
                     std::size_t max_batch = 16);

}  // namespace locus::pipeline
