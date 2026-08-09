#pragma once

#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>

#include "locus/auth/helper_connection.hpp"

namespace locus::auth {

namespace detail {

/**
 * RAII owner of a freshly forked helper child and its two parent-side
 * pipe fds, for the window between fork() and handing them to a
 * HelperConnection. If destroyed WITHOUT release() -- e.g. the
 * HelperConnection constructor throws, since its reader thread can
 * fail to start -- it closes the fds and reaps the child (SIGKILL +
 * waitpid) so neither an fd nor a zombie leaks on that error path.
 * Exposed for testing; not part of the public spawn API.
 */
class SpawnedChild {
  public:
    SpawnedChild(pid_t pid, int read_fd, int write_fd)
        : pid_(pid), read_fd_(read_fd), write_fd_(write_fd) {}
    SpawnedChild(const SpawnedChild&) = delete;
    SpawnedChild& operator=(const SpawnedChild&) = delete;
    ~SpawnedChild();

    /** Constructs the HelperConnection (which takes over the fds +
     * child) and disarms cleanup only once it is successfully built.
     * @returns The owning connection. Throws if construction throws;
     *     the child + fds are then cleaned up by this object's dtor. */
    std::unique_ptr<HelperConnection> release();

  private:
    pid_t pid_;
    int read_fd_;
    int write_fd_;
    bool armed_ = true;
};

}  // namespace detail

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
