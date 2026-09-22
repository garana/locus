#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace locus::pipeline {

/**
 * An IPv4 CIDR range for an incoming-connection allowlist, e.g.
 * "10.0.0.0/8" or a bare "192.168.1.4" (treated as /32). Used to
 * restrict which peers a pipeline stage accepts input from.
 */
struct CidrV4 {
    std::uint32_t network = 0;  /**< Host-order base address, masked. */
    std::uint32_t mask = 0;     /**< Host-order netmask. */

    /** Parses "a.b.c.d/n" (0 <= n <= 32) or a bare "a.b.c.d" (=/32).
     * @returns nullopt on a malformed string. */
    static std::optional<CidrV4> parse(const std::string& s);
    /** @returns true if `ipv4` (dotted quad) falls in this range. */
    bool contains(const std::string& ipv4) const;
};

/** @returns true if `ipv4` matches any range in `allow`, or if `allow`
 * is empty (an empty allowlist means allow all). */
bool ip_allowed(const std::string& ipv4,
                const std::vector<CidrV4>& allow);

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
 * @param peer_ip If non-null, receives the peer's numeric address.
 * @returns The connection fd (>= 0), or -1 on failure.
 */
int accept_one(int listen_fd, std::string* peer_ip = nullptr);

/**
 * Connects to host:port (blocking).
 * @returns The connection fd (>= 0), or -1 on failure.
 */
int connect_to(const std::string& host, int port);

}  // namespace locus::pipeline
