#pragma once

#include <memory>
#include <string>
#include <vector>

#include "locus/auth/helper_connection.hpp"

namespace locus::auth {

/**
 * Launches `argv` as a child process with its stdin and stdout wired
 * to pipes, and returns a HelperConnection speaking the framed
 * protocol over them (locus writes requests/events to the child's
 * stdin, reads responses from its stdout). The returned connection
 * reaps the child on destruction.
 *
 * @param argv Command + arguments (argv[0] is looked up on PATH via
 *     execvp); must be non-empty.
 * @param[out] err Set to a message on failure.
 * @returns The connection, or nullptr on fork/pipe failure. Note: a
 *     bad command still returns a connection (the fork succeeds); the
 *     failed exec closes the child's stdout, so the connection soon
 *     reports broken() and requests time out / deny.
 */
std::unique_ptr<HelperConnection> spawn_helper(
    const std::vector<std::string>& argv, std::string* err);

}  // namespace locus::auth
