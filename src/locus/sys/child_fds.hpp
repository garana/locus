#pragma once

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

}  // namespace locus::sys
