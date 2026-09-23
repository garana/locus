#pragma once

#include <string>
#include <vector>

#include "locus/pipeline/net.hpp"
#include "locus/pipeline/stage.hpp"

namespace locus::pipeline {

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
 * @returns With serve_sessions > 0: true if all N sessions ended
 *     cleanly, false if any ended abnormally. Either way false on a
 *     fatal accept failure or no live downstream within
 *     reconnect_attempts sweeps. With serve_sessions == 0 it returns
 *     only on such a fatal error (false).
 */
bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<Cidr>& allow,
                 const std::vector<HostPort>& downstreams,
                 const StageConn& conn = {});

/** serve_stage with a single downstream (a one-element pool). */
bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<Cidr>& allow,
                 const std::string& downstream_host,
                 int downstream_port, const StageConn& conn = {});

}  // namespace locus::pipeline
