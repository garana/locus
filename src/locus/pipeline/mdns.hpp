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

/**
 * Browses for SERVICE: sends mdns_query to opt.group:opt.port from an
 * ephemeral UDP port (so responders answer by unicast, RFC 6762 6.7;
 * the router joins no group), collects the answers for opt.timeout_ms
 * and parses each with mdns_parse.
 *
 * @returns The targets found, one per host:port; empty on no answer or
 *     a socket error.
 */
std::vector<SrvTarget> mdns_browse(const std::string& service,
                                   const MdnsBrowse& opt = {});

}  // namespace locus::pipeline
