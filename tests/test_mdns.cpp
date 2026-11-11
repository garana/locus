#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/pipeline/mdns.hpp"

using locus::pipeline::mdns_browse;
using locus::pipeline::mdns_parse;
using locus::pipeline::mdns_query;
using locus::pipeline::MdnsBrowse;
using Bytes = std::vector<std::uint8_t>;

namespace {

void u16(Bytes& b, std::uint16_t v) {
    b.push_back(static_cast<std::uint8_t>(v >> 8));
    b.push_back(static_cast<std::uint8_t>(v & 0xFF));
}
void u32(Bytes& b, std::uint32_t v) {
    u16(b, static_cast<std::uint16_t>(v >> 16));
    u16(b, static_cast<std::uint16_t>(v & 0xFFFF));
}
/** A name, uncompressed. */
void name(Bytes& b, const std::string& n) {
    std::size_t s = 0;
    while (s < n.size()) {
        const std::size_t d = std::min(n.find('.', s), n.size());
        b.push_back(static_cast<std::uint8_t>(d - s));
        b.insert(b.end(), n.begin() + static_cast<long>(s),
                 n.begin() + static_cast<long>(d));
        s = d + 1;
    }
    b.push_back(0);
}
/** A pointer to offset `at`. */
void ptr(Bytes& b, std::size_t at) {
    u16(b, static_cast<std::uint16_t>(0xC000 | at));
}

/**
 * A DNS-SD answer for one instance "NAME._locus._tcp.local" on
 * "host-NAME.local" at `ip`:`port`, as a responder sends it: the PTR in
 * the answers, the SRV and A records in the additionals, the instance
 * name compressed against the PTR owner. ttl 0 makes it a goodbye.
 */
Bytes answer(const std::string& inst, const std::string& ip, int port,
             std::uint32_t ttl = 120) {
    Bytes b;
    u16(b, 0);
    u16(b, 0x8400);  // a response, authoritative
    u16(b, 0);       // no question
    u16(b, 1);       // one answer
    u16(b, 0);
    u16(b, 2);  // SRV and A
    const std::size_t owner = b.size();
    name(b, "_locus._tcp.local");
    u16(b, 12);  // PTR
    u16(b, 1);
    u32(b, ttl);
    Bytes rd;
    rd.push_back(static_cast<std::uint8_t>(inst.size()));
    rd.insert(rd.end(), inst.begin(), inst.end());
    const std::size_t inst_at = b.size() + 2;  // after rdlength
    u16(rd, static_cast<std::uint16_t>(0xC000 | owner));
    u16(b, static_cast<std::uint16_t>(rd.size()));
    b.insert(b.end(), rd.begin(), rd.end());
    // SRV, owner = the instance name (a pointer to the PTR's rdata).
    ptr(b, inst_at);
    u16(b, 33);
    u16(b, 0x8001);  // IN with the cache-flush bit
    u32(b, ttl);
    Bytes srv;
    u16(srv, 0);
    u16(srv, 0);
    u16(srv, static_cast<std::uint16_t>(port));
    const std::size_t host_at = b.size() + 2 + srv.size();
    name(srv, "host-" + inst + ".local");
    u16(b, static_cast<std::uint16_t>(srv.size()));
    b.insert(b.end(), srv.begin(), srv.end());
    // A, owner = the target (a pointer into the SRV's rdata).
    ptr(b, host_at);
    u16(b, 1);
    u16(b, 0x8001);
    u32(b, ttl);
    u16(b, 4);
    in_addr a{};
    inet_pton(AF_INET, ip.c_str(), &a);
    const auto* p = reinterpret_cast<const std::uint8_t*>(&a.s_addr);
    b.insert(b.end(), p, p + 4);
    return b;
}

}  // namespace

TEST_CASE("mdns query is one PTR question for SERVICE.local", "[mdns]") {
    const Bytes q = mdns_query("_locus._tcp");
    Bytes want;
    u16(want, 0);
    u16(want, 0);
    u16(want, 1);
    u16(want, 0);
    u16(want, 0);
    u16(want, 0);
    name(want, "_locus._tcp.local");
    u16(want, 12);
    u16(want, 1);
    REQUIRE(q == want);
    REQUIRE_THROWS_AS(mdns_query("a..b"), std::invalid_argument);
    REQUIRE_THROWS_AS(mdns_query(""), std::invalid_argument);
    REQUIRE_THROWS_AS(mdns_query(std::string(64, 'x')),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(mdns_query(std::string(260, 'x')),
                      std::invalid_argument);
}

TEST_CASE("mdns parse reads PTR, SRV and A through compressed names",
          "[mdns]") {
    const Bytes b = answer("box1", "192.168.7.21", 8081);
    const auto t = mdns_parse(b, "_locus._tcp");
    REQUIRE(t.size() == 1);
    CHECK(t[0].host == "192.168.7.21");
    CHECK(t[0].port == 8081);
    // Another service's answer, a goodbye and a query give nothing.
    CHECK(mdns_parse(b, "_other._tcp").empty());
    CHECK(mdns_parse(answer("box1", "192.168.7.21", 8081, 0),
                     "_locus._tcp")
              .empty());
    Bytes q = b;
    q[2] = 0;  // clear QR: a query
    CHECK(mdns_parse(q, "_locus._tcp").empty());
    // Service names compare without case.
    CHECK(mdns_parse(b, "_LOCUS._TCP").size() == 1);
}

TEST_CASE("mdns parse survives malformed and hostile packets", "[mdns]") {
    const Bytes good = answer("box1", "10.0.0.1", 8080);
    // Every truncation: no crash, and at most the one answer.
    for (std::size_t n = 0; n < good.size(); ++n) {
        const auto t = mdns_parse(
            std::span<const std::uint8_t>(good.data(), n), "_locus._tcp");
        CHECK(t.size() <= 1);
    }
    // A pointer to itself, and one pointing forward: rejected.
    Bytes loop;
    u16(loop, 0);
    u16(loop, 0x8400);
    u16(loop, 0);
    u16(loop, 1);
    u16(loop, 0);
    u16(loop, 0);
    ptr(loop, 12);
    u16(loop, 12);
    u16(loop, 1);
    u32(loop, 120);
    u16(loop, 2);
    ptr(loop, 40);
    CHECK(mdns_parse(loop, "_locus._tcp").empty());
    // Random bytes behind a response header: no crash.
    std::mt19937 rng(524);
    for (int i = 0; i < 2000; ++i) {
        Bytes r(12 + rng() % 300);
        for (auto& c : r) c = static_cast<std::uint8_t>(rng());
        r[2] |= 0x80;
        (void)mdns_parse(r, "_locus._tcp");
    }
    // Flipping each byte of a good answer: no crash.
    for (std::size_t i = 0; i < good.size(); ++i) {
        Bytes f = good;
        f[i] ^= 0xFF;
        (void)mdns_parse(f, "_locus._tcp");
    }
}

// i#537: a reply is read only from loopback or an interface's network;
// an interface with a zero mask makes nothing local.
TEST_CASE("mdns local source is loopback or an interface network",
          "[mdns]") {
    using locus::pipeline::Ipv4Net;
    using locus::pipeline::mdns_local_source;
    const auto ip = [](unsigned a, unsigned b, unsigned c, unsigned d) {
        return std::uint32_t{a << 24 | b << 16 | c << 8 | d};
    };
    const std::vector<Ipv4Net> nets = {
        {ip(192, 168, 1, 20), ip(255, 255, 255, 0)}, {ip(10, 0, 0, 5), 0}};
    CHECK(mdns_local_source(ip(127, 0, 0, 1), nets));
    CHECK(mdns_local_source(ip(127, 9, 9, 9), {}));
    CHECK(mdns_local_source(ip(192, 168, 1, 77), nets));
    CHECK_FALSE(mdns_local_source(ip(192, 168, 2, 77), nets));
    CHECK_FALSE(mdns_local_source(ip(10, 0, 0, 6), nets));  // zero mask
    CHECK_FALSE(mdns_local_source(ip(8, 8, 8, 8), nets));
    // This host has at least one IPv4 interface (loopback).
    CHECK_FALSE(locus::pipeline::local_ipv4_nets().empty());
}

TEST_CASE("mdns browse collects answers from a loopback responder",
          "[mdns]") {
    // A responder on 127.0.0.1 and a test port, never the real group:
    // it answers the browse by unicast, as one-shot queries are.
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0);
    socklen_t len = sizeof(a);
    REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) == 0);
    const int port = ntohs(a.sin_port);
    // A second loopback socket that forges an answer from another port:
    // the browse must drop it (i#537).
    const int forger = ::socket(AF_INET, SOCK_DGRAM, 0);
    REQUIRE(forger >= 0);
    sockaddr_in fa{};
    fa.sin_family = AF_INET;
    fa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::bind(forger, reinterpret_cast<sockaddr*>(&fa), sizeof(fa)) ==
            0);
    std::atomic<int> queries{0};
    std::thread responder([&] {
        Bytes buf(1500);
        pollfd p{fd, POLLIN, 0};
        if (::poll(&p, 1, 3000) <= 0) return;
        sockaddr_in from{};
        socklen_t fl = sizeof(from);
        const ssize_t n = ::recvfrom(fd, buf.data(), buf.size(), 0,
                                     reinterpret_cast<sockaddr*>(&from), &fl);
        if (n <= 0) return;
        buf.resize(static_cast<std::size_t>(n));
        if (buf == mdns_query("_locus._tcp")) ++queries;
        const auto forged = answer("c", "127.0.0.1", 18303);
        ::sendto(forger, forged.data(), forged.size(), 0,
                 reinterpret_cast<sockaddr*>(&from), fl);
        // Two responders' answers, and one repeated.
        for (const auto& r : {answer("a", "127.0.0.1", 18301),
                              answer("b", "127.0.0.1", 18302),
                              answer("a", "127.0.0.1", 18301)}) {
            ::sendto(fd, r.data(), r.size(), 0,
                     reinterpret_cast<sockaddr*>(&from), fl);
        }
    });
    MdnsBrowse opt;
    opt.group = "127.0.0.1";
    opt.port = port;
    opt.timeout_ms = 400;
    const auto t = mdns_browse("_locus._tcp", opt);
    responder.join();
    ::close(fd);
    ::close(forger);
    CHECK(queries == 1);
    REQUIRE(t.size() == 2);  // not the forged "c" from another port
    CHECK(t[0].port + t[1].port == 18301 + 18302);
    // No responder: empty, within the timeout.
    opt.port = 9;  // discard; nothing answers on loopback
    opt.timeout_ms = 100;
    CHECK(mdns_browse("_locus._tcp", opt).empty());
}
