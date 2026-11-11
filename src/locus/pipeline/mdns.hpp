#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "locus/pipeline/resolver.hpp"

namespace locus::pipeline {

/** Where and how long an mDNS / DNS-SD browse asks (i#524). */
struct MdnsBrowse {
    /** The mDNS group; tests point it at a loopback responder. */
    std::string group = "224.0.0.251";
    int port = 5353;
    /** How long answers are collected after the query. */
    int timeout_ms = 500;
};

/**
 * @param service A DNS-SD service type, for example "_locus._tcp".
 * @returns A DNS query message: one PTR question for SERVICE.local.
 * @throws std::invalid_argument on a bad name (an empty label, a label
 *     over 63 bytes, or a name over 255).
 */
std::vector<std::uint8_t> mdns_query(const std::string& service);

/**
 * Reads one mDNS response: the PTR records of SERVICE.local name the
 * instances, each instance's SRV record its target and port, and the
 * target's A (else AAAA) record its address. Records with TTL 0
 * (goodbyes) are ignored, as is any record after a malformed one.
 * Bounded: at most 9000 bytes, 256 records, 16 compression jumps, each
 * pointing back.
 *
 * @returns One target per instance that has an SRV record: host is the
 *     target's address when an A or AAAA record gives it, else the
 *     target name (without the trailing dot); priority and weight as
 *     the SRV record says.
 */
std::vector<SrvTarget> mdns_parse(std::span<const std::uint8_t> msg,
                                  const std::string& service);

/** One IPv4 interface network, host byte order. */
struct Ipv4Net {
    std::uint32_t addr = 0, mask = 0;
};

/**
 * @param a An IPv4 address, host byte order.
 * @param nets The host's interface networks (local_ipv4_nets()); one
 *     with a zero mask is ignored, so it cannot make every address local.
 * @returns Whether a is loopback (127/8) or on one of nets.
 */
bool mdns_local_source(std::uint32_t a, std::span<const Ipv4Net> nets);

/** @returns This host's IPv4 interface networks (getifaddrs). */
std::vector<Ipv4Net> local_ipv4_nets();

/**
 * Browses for SERVICE: sends mdns_query to opt.group:opt.port from an
 * ephemeral UDP port (so responders answer by unicast, RFC 6762 6.7;
 * the router joins no group), collects the answers for opt.timeout_ms
 * and parses each with mdns_parse.
 *
 * Only replies from opt.port (responders answer one-shot queries from
 * 5353, RFC 6762 6.7) and from a local source (mdns_local_source) are
 * read, so a host off the local networks cannot plant instances.
 *
 * @returns The targets found, one per host:port; empty on no answer or
 *     a socket error.
 */
std::vector<SrvTarget> mdns_browse(const std::string& service,
                                   const MdnsBrowse& opt = {});

}  // namespace locus::pipeline
