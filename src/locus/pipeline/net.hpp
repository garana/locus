#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace locus::pipeline {

/** Address family of a parsed address or CIDR. */
enum class AddrFamily { kV4, kV6 };

/**
 * A CIDR range for an incoming-connection allowlist, IPv4 or IPv6:
 * "10.0.0.0/8", "192.168.1.4" (bare == /32), "2001:db8::/32", or
 * "::1" (bare == /128). Restricts which peers a pipeline stage
 * accepts input from. Address bytes are stored big-endian; an IPv4
 * range uses the first four.
 */
struct Cidr {
    AddrFamily family = AddrFamily::kV4;
    std::array<std::uint8_t, 16> net{};  /**< Masked network bytes. */
    std::uint8_t bits = 0;               /**< Prefix length. */

    /** Parses "addr" or "addr/n" (v4 n<=32, v6 n<=128; bare = full
     * length). @returns nullopt on a malformed string. */
    static std::optional<Cidr> parse(const std::string& s);
    /** @returns true if numeric address `ip` is in this range; false
     * on a family mismatch or a malformed `ip`. */
    bool contains(const std::string& ip) const;
};

/** @returns true if `ip` matches any range in `allow`, or if `allow`
 * is empty (an empty allowlist means allow all). */
bool ip_allowed(const std::string& ip, const std::vector<Cidr>& allow);

/**
 * Splits a "host:port" endpoint string (as a CLI takes) into its parts.
 * Accepts a bracketed IPv6 literal ("[::1]:8000", brackets stripped),
 * "host:port", and ":port" (empty host == wildcard, for a bind). A bare
 * IPv6 literal is ambiguous with the port colon, so brackets are the
 * way to give one. The port must be in [0, 65535].
 *
 * @param[out] host Receives the host part (may be empty for wildcard).
 * @param[out] port Receives the parsed port.
 * @returns false on a malformed string (host and port left unspecified).
 */
bool parse_hostport(const std::string& s, std::string& host, int& port);

/**
 * TCP connection setup for pipeline-parallel inference across hosts
 * (multi-server). Thin blocking-socket helpers over getaddrinfo; the
 * returned fds carry pipeline::Message frames via write_message /
 * read_message. TCP_NODELAY is set so per-token activations are not
 * held by Nagle, and SO_NOSIGPIPE (where available) so a closed peer
 * does not raise SIGPIPE.
 */

/**
 * Opens a listening TCP socket on host:port. port 0 asks the OS for
 * an ephemeral port, written back to *out_port (may be null).
 * SO_REUSEADDR is set. For a wildcard/dual-stack bind an IPv6 socket
 * is preferred with IPV6_V6ONLY disabled, so behavior does not depend
 * on the host's getaddrinfo ordering and one socket accepts both
 * IPv6 and IPv4 (the latter as v4-mapped, normalized by accept_one).
 *
 * @returns The listen fd (>= 0), or -1 on failure.
 */
int listen_on(const std::string& host, int port, int* out_port);

/**
 * Accepts one connection on a listen fd. An IPv4-mapped IPv6 peer
 * (::ffff:a.b.c.d, seen on a dual-stack socket) is normalized to its
 * dotted-quad form so an IPv4 allowlist rule still matches an IPv4
 * client.
 *
 * With timeout_ms > 0 the wait for a peer is bounded (poll on the listen
 * fd, retrying EINTR against a shared deadline) so a peer that never
 * arrives -- e.g. a downstream chain that never wires up -- fails fast
 * instead of blocking forever; timeout_ms == 0 blocks indefinitely.
 *
 * @param peer_ip If non-null, receives the peer's numeric address.
 * @param timeout_ms 0 blocks; otherwise give up after this many ms.
 * @returns The connection fd (>= 0), or -1 on failure/timeout.
 */
int accept_one(int listen_fd, std::string* peer_ip = nullptr,
               int timeout_ms = 0);

/**
 * Connects to host:port. With timeout_ms > 0 the connect uses a
 * non-blocking socket and gives up after the timeout (so a dead or
 * unreachable downstream fails fast instead of hanging on the OS
 * default); timeout_ms == 0 is a blocking connect.
 *
 * @returns The connection fd (>= 0), or -1 on failure/timeout.
 */
int connect_to(const std::string& host, int port, int timeout_ms = 0);

/**
 * Sets a receive timeout on `fd` (SO_RCVTIMEO): a blocking read that
 * waits longer than `ms` fails with EAGAIN, which read_message reports
 * as kTimeout. ms == 0 clears the timeout (block indefinitely).
 */
void set_recv_timeout(int fd, int ms);

/**
 * Enables and tunes TCP keepalive on `fd` so a silently-dropped peer
 * (crash, cable pull, NAT timeout) is detected in a bounded time
 * instead of a half-open connection hanging forever. idle_s <= 0
 * disables keepalive.
 *
 * Detection takes roughly idle_s + intvl_s * count seconds. Per-OS
 * knobs: Linux TCP_KEEPIDLE/TCP_KEEPINTVL/TCP_KEEPCNT; macOS/BSD
 * TCP_KEEPALIVE (idle) plus TCP_KEEPINTVL/TCP_KEEPCNT where available.
 *
 * @param idle_s  Idle seconds before the first probe (<= 0 disables).
 * @param intvl_s Seconds between probes.
 * @param count   Unacked probes before the connection is dropped.
 */
void set_keepalive(int fd, int idle_s, int intvl_s, int count);

}  // namespace locus::pipeline
