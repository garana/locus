#pragma once

#include <string>
#include <vector>

#include "locus/pipeline/net.hpp"
#include "locus/pipeline/stage.hpp"

namespace locus::pipeline {

/**
 * Runs one pipeline stage as a server (the core of the locus-stage
 * CLI): accept one input connection on `listen_fd`, rejecting peers
 * whose address is not in `allow` (an empty allow means allow all) and
 * retrying until an allowed peer connects; then connect to
 * downstream_host:downstream_port for output and run the stage's
 * read -> step -> write loop until EOF.
 *
 * The listen fd is closed once a peer is accepted (one input link per
 * stage). CPU/CUDA only (forward_layers does not run on Vulkan).
 *
 * @returns true on a clean EOF shutdown; false on accept/connect
 *     failure or a stage error.
 */
bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<CidrV4>& allow,
                 const std::string& downstream_host,
                 int downstream_port);

}  // namespace locus::pipeline
