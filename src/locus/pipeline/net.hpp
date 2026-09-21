#pragma once

#include <string>

namespace locus::pipeline {

/**
 * TCP connection setup for pipeline-parallel inference across hosts
 * (multi-server). Thin blocking-socket helpers over getaddrinfo; the
 * returned fds carry pipeline::Message frames via write_message /
 * read_message. IPv4/IPv6 via getaddrinfo; TCP_NODELAY is set so
 * per-token activations are not held by Nagle, and SO_NOSIGPIPE is set
 * where available so a closed peer does not raise SIGPIPE.
 */

/**
 * Opens a listening TCP socket on host:port. port 0 asks the OS for an
 * ephemeral port, written back to *out_port (may be null). SO_REUSEADDR
 * is set. Blocking accept via accept_one.
 *
 * @returns The listen fd (>= 0), or -1 on failure (bind/listen error).
 */
int listen_on(const std::string& host, int port, int* out_port);

/**
 * Accepts one connection on a listen fd (blocking).
 * @returns The connection fd (>= 0), or -1 on failure.
 */
int accept_one(int listen_fd);

/**
 * Connects to host:port (blocking).
 * @returns The connection fd (>= 0), or -1 on failure.
 */
int connect_to(const std::string& host, int port);

}  // namespace locus::pipeline
