#pragma once

#include <string>

namespace locus::sys {

/**
 * One past the highest fd a forked child may have to close. Worked out
 * in the parent, before fork: sysconf and getrlimit are not
 * async-signal-safe, so the child must not call them.
 *
 * @returns The larger of _SC_OPEN_MAX and the soft RLIMIT_NOFILE (1024
 *     when neither is known), at most 2^20.
 */
int child_fd_limit();

/**
 * Closes every fd from 3 up to `limit` (child_fd_limit's value), so a
 * forked child does not hold the parent's sockets, pipes or files.
 * Async-signal-safe: call it in the child between fork and exec.
 */
void close_fds_from_3(int limit);

/**
 * The executable a command name runs: a name with a '/' as it is, else
 * the first match on PATH (/usr/bin:/bin when PATH is unset), as a
 * shell finds it. Call it in the parent: a child after fork may only
 * execv the result (execvp's PATH search is not async-signal-safe).
 *
 * @returns The path, or empty when nothing executable is found.
 */
std::string find_executable(const std::string& name);

/**
 * Moves `fd` above the std fds (0, 1, 2) when it is one of them, with
 * close-on-exec set. A server started with fd 0, 1 or 2 closed can get
 * one of those from pipe(); a child's dup2 onto the same number is then
 * a no-op that keeps close-on-exec, and the std fd is gone after exec.
 *
 * @returns false, with `fd` unchanged, when the move fails.
 */
bool lift_above_std(int& fd);

}  // namespace locus::sys
