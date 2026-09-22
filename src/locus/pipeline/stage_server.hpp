#pragma once

#include <string>
#include <vector>

#include "locus/pipeline/net.hpp"
#include "locus/pipeline/stage.hpp"

namespace locus::pipeline {

/** Timeouts / reconnect policy for serve_stage (all milliseconds). */
struct StageConn {
    int connect_timeout_ms = 0;    /**< 0 = OS default (blocking). */
    int reconnect_wait_ms = 1000;  /**< Fixed wait between downstream
                                    *   connect attempts (no backoff). */
    int reconnect_attempts = 0;    /**< 0 = retry until connected. */
    int recv_timeout_ms = 0;       /**< 0 = block; else the input read
                                    *   times out (kTimeout). */
};

/**
 * Runs one pipeline stage as a server (the core of the locus-stage
 * CLI): accept one input connection on `listen_fd`, rejecting peers
 * whose address is not in `allow` (an empty allow means allow all) and
 * retrying until an allowed peer connects; apply `conn.recv_timeout_ms`
 * to that input; then connect to downstream_host:downstream_port
 * (with `conn.connect_timeout_ms`, retrying on a fixed
 * `conn.reconnect_wait_ms` since a stage may start before its
 * downstream) and run the stage's read -> step -> write loop until EOF.
 *
 * The established connection is itself the health signal; there is no
 * separate probe. The listen fd is closed once a peer is accepted.
 * CPU/CUDA only.
 *
 * @returns true on a clean EOF shutdown; false on accept/connect
 *     failure (after reconnect_attempts) or a stage error/timeout.
 */
bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<Cidr>& allow,
                 const std::string& downstream_host,
                 int downstream_port, const StageConn& conn = {});

}  // namespace locus::pipeline
