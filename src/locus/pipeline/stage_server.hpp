#pragma once

#include <string>
#include <vector>

#include "locus/pipeline/net.hpp"
#include "locus/pipeline/stage.hpp"

namespace locus::pipeline {

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
 * to downstream_host:downstream_port (with `conn.connect_timeout_ms`,
 * retrying on a fixed `conn.reconnect_wait_ms` since a stage may start
 * before its downstream), applies keepalive to the downstream, resets
 * the stage, and runs its read -> step -> write loop until the session
 * ends. It then re-accepts the next session -- the automatic
 * reconnect -- so a dropped peer is recovered by serving the next
 * connection rather than exiting.
 *
 * conn.serve_sessions bounds the loop: 0 serves forever (a long-lived
 * stage; then this returns only on a fatal error), N stops after N
 * sessions (used by tests). The listen fd stays open across sessions
 * and is closed only when this function returns. The established
 * connection is itself the health signal; there is no separate probe.
 * CPU/CUDA only.
 *
 * @returns With serve_sessions > 0: true if all N sessions ended
 *     cleanly, false if any ended abnormally. Either way false on a
 *     fatal accept failure or the downstream being unreachable within
 *     reconnect_attempts. With serve_sessions == 0 it returns only on
 *     such a fatal error (false).
 */
bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<Cidr>& allow,
                 const std::string& downstream_host,
                 int downstream_port, const StageConn& conn = {});

}  // namespace locus::pipeline
