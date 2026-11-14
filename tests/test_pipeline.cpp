#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "catch_amalgamated.hpp"
#include "locus/pipeline/message.hpp"
#include "locus/pipeline/net.hpp"

using locus::pipeline::Decode;
using locus::pipeline::Message;
using locus::pipeline::MsgType;
using locus::pipeline::ReadResult;

namespace {

Message activation(std::uint64_t id, std::uint32_t pos,
                   std::size_t n) {
    std::vector<float> h(n);
    for (std::size_t i = 0; i < n; ++i) {
        h[i] = static_cast<float>(i) * 0.5f - 3.0f;
    }
    return locus::pipeline::make_activation(id, pos, std::move(h));
}

}  // namespace

TEST_CASE("pipeline message codec round-trips", "[pipeline]") {
    SECTION("activation vector") {
        const auto a = activation(42, 7, 64);
        std::string buf;
        locus::pipeline::encode(a, buf);
        Message out;
        std::string err;
        REQUIRE(locus::pipeline::decode(buf, out, err) ==
                Decode::kComplete);
        REQUIRE(buf.empty());
        REQUIRE(out.type == MsgType::kActivation);
        REQUIRE(out.request_id == a.request_id);
        REQUIRE(out.position == a.position);
        REQUIRE(out.data == a.data);  // bitwise-identical floats
    }

    SECTION("token message (no floats)") {
        const auto t = locus::pipeline::make_token(5, 9, 12345);
        std::string buf;
        locus::pipeline::encode(t, buf);
        Message out;
        std::string err;
        REQUIRE(locus::pipeline::decode(buf, out, err) ==
                Decode::kComplete);
        REQUIRE(out.type == MsgType::kToken);
        REQUIRE(out.token == 12345);
        REQUIRE(out.request_id == 5);
        REQUIRE(out.position == 9);
        REQUIRE(out.data.empty());
    }

    SECTION("logits message") {
        const auto l = locus::pipeline::make_logits(
            1, 0, std::vector<float>{0.1f, -2.0f, 3.5f});
        std::string buf;
        locus::pipeline::encode(l, buf);
        Message out;
        std::string err;
        REQUIRE(locus::pipeline::decode(buf, out, err) ==
                Decode::kComplete);
        REQUIRE(out.type == MsgType::kLogits);
        REQUIRE(out.data.size() == 3);
        REQUIRE(out.data[1] == -2.0f);
    }

    SECTION("session-end message") {
        const auto e = locus::pipeline::make_session_end(42);
        std::string buf;
        locus::pipeline::encode(e, buf);
        Message out;
        std::string err;
        REQUIRE(locus::pipeline::decode(buf, out, err) ==
                Decode::kComplete);
        REQUIRE(out.type == MsgType::kSessionEnd);
        REQUIRE(out.request_id == 42);
        REQUIRE(out.data.empty());
    }

    SECTION("expert call and reply carry layer, expert id and floats") {
        const std::vector<float> xb = {0.5f, -1.25f, 3.0f, -0.0f};
        const std::vector<float> d = {1e-30f, -7.5f, 2.0f, 4.0f};
        for (const auto& m :
             {locus::pipeline::make_expert_call(77, 23, 59, xb),
              locus::pipeline::make_expert_out(77, 23, 59, d)}) {
            std::string buf;
            locus::pipeline::encode(m, buf);
            Message out;
            std::string err;
            REQUIRE(locus::pipeline::decode(buf, out, err) ==
                    Decode::kComplete);
            REQUIRE(buf.empty());
            REQUIRE(out.type == m.type);
            REQUIRE(out.request_id == 77);
            REQUIRE(out.position == 23);  // layer
            REQUIRE(out.token == 59);     // expert id
            REQUIRE(out.data.size() == m.data.size());
            // Floats travel raw: the same bits, -0.0 included.
            REQUIRE(std::memcmp(out.data.data(), m.data.data(),
                                m.data.size() * sizeof(float)) == 0);
        }
        REQUIRE(locus::pipeline::make_expert_call(1, 0, 0, xb).type ==
                MsgType::kExpertCall);
        REQUIRE(locus::pipeline::make_expert_out(1, 0, 0, d).type ==
                MsgType::kExpertOut);
    }

    SECTION("an unknown message type is a malformed frame") {
        auto m = locus::pipeline::make_expert_out(5, 1, 2, {1.0f});
        std::string buf;
        locus::pipeline::encode(m, buf);
        buf[4] = 7;  // the u16 type after the u32 frame_len: no kind 7
        buf[5] = 0;
        Message out;
        std::string err;
        REQUIRE(locus::pipeline::decode(buf, out, err) == Decode::kError);
    }

    SECTION("two frames decode in sequence (pipelining)") {
        const auto a = activation(1, 2, 3);
        const auto b = locus::pipeline::make_token(9, 8, 7);
        std::string buf;
        locus::pipeline::encode(a, buf);
        locus::pipeline::encode(b, buf);
        Message o1, o2;
        std::string err;
        REQUIRE(locus::pipeline::decode(buf, o1, err) ==
                Decode::kComplete);
        REQUIRE(locus::pipeline::decode(buf, o2, err) ==
                Decode::kComplete);
        REQUIRE(buf.empty());
        REQUIRE(o1.data == a.data);
        REQUIRE(o2.type == MsgType::kToken);
        REQUIRE(o2.token == 7);
    }

    SECTION("partial buffer is incomplete, not error") {
        const auto a = activation(1, 1, 10);
        std::string buf;
        locus::pipeline::encode(a, buf);
        Message out;
        std::string err;
        std::string part = buf.substr(0, buf.size() - 4);
        const std::size_t before = part.size();
        REQUIRE(locus::pipeline::decode(part, out, err) ==
                Decode::kIncomplete);
        REQUIRE(part.size() == before);
        std::string tiny = buf.substr(0, 3);
        REQUIRE(locus::pipeline::decode(tiny, out, err) ==
                Decode::kIncomplete);
    }

    SECTION("a too-small frame length is an error") {
        std::string buf;
        buf.push_back(3);
        buf.push_back(0);
        buf.push_back(0);
        buf.push_back(0);
        Message out;
        std::string err;
        REQUIRE(locus::pipeline::decode(buf, out, err) ==
                Decode::kError);
        REQUIRE_FALSE(err.empty());
    }
}

TEST_CASE("pipeline framed transport over a socketpair",
          "[pipeline]") {
    int fds[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);

    const std::vector<Message> sent = {
        activation(1, 0, 64), locus::pipeline::make_token(2, 1, 99),
        activation(3, 2, 128)};

    std::atomic<bool> wok{true};
    std::jthread writer([&] {
        for (const auto& m : sent) {
            if (!locus::pipeline::write_message(fds[1], m)) {
                wok = false;
            }
        }
        ::close(fds[1]);
    });

    for (const auto& m : sent) {
        Message out;
        REQUIRE(locus::pipeline::read_message(fds[0], out) ==
                ReadResult::kOk);
        REQUIRE(out.type == m.type);
        REQUIRE(out.request_id == m.request_id);
        REQUIRE(out.token == m.token);
        REQUIRE(out.data == m.data);
    }
    Message out;
    REQUIRE(locus::pipeline::read_message(fds[0], out) ==
            ReadResult::kEof);

    writer.join();
    REQUIRE(wok.load());
    ::close(fds[0]);
}

TEST_CASE("pipeline read_message reports EOF and truncation",
          "[pipeline]") {
    Message out;

    SECTION("clean EOF on an empty closed socket") {
        int fds[2];
        REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
        ::close(fds[1]);
        REQUIRE(locus::pipeline::read_message(fds[0], out) ==
                ReadResult::kEof);
        ::close(fds[0]);
    }

    SECTION("truncated length prefix is an error") {
        int fds[2];
        REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
        const char two[2] = {0, 0};
        REQUIRE(::send(fds[1], two, 2, 0) == 2);
        ::close(fds[1]);
        REQUIRE(locus::pipeline::read_message(fds[0], out) ==
                ReadResult::kError);
        ::close(fds[0]);
    }

    SECTION("length prefix but truncated payload is an error") {
        int fds[2];
        REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
        const auto a = activation(1, 0, 10);
        std::string frame;
        locus::pipeline::encode(a, frame);
        REQUIRE(::send(fds[1], frame.data(), 6, 0) == 6);
        ::close(fds[1]);
        REQUIRE(locus::pipeline::read_message(fds[0], out) ==
                ReadResult::kError);
        ::close(fds[0]);
    }
}

TEST_CASE("pipeline transport over loopback TCP", "[pipeline]") {
    int port = 0;
    const int lfd = locus::pipeline::listen_on("127.0.0.1", 0, &port);
    REQUIRE(lfd >= 0);
    REQUIRE(port > 0);

    // A large activation forces TCP to segment the frame across several
    // reads, exercising read_message's read-until-complete loop.
    const std::vector<Message> sent = {
        activation(1, 0, 8192),
        locus::pipeline::make_token(2, 1, 7),
        locus::pipeline::make_logits(3, 2,
                                     std::vector<float>(2048, 1.5f))};

    std::atomic<bool> wok{true};
    std::jthread client([&] {
        const int c = locus::pipeline::connect_to("127.0.0.1", port);
        if (c < 0) {
            wok = false;
            return;
        }
        for (const auto& m : sent) {
            if (!locus::pipeline::write_message(c, m)) {
                wok = false;
            }
        }
        ::close(c);
    });

    const int s = locus::pipeline::accept_one(lfd);
    REQUIRE(s >= 0);
    for (const auto& m : sent) {
        Message out;
        REQUIRE(locus::pipeline::read_message(s, out) ==
                ReadResult::kOk);
        REQUIRE(out.type == m.type);
        REQUIRE(out.request_id == m.request_id);
        REQUIRE(out.token == m.token);
        REQUIRE(out.data == m.data);
    }
    Message out;
    REQUIRE(locus::pipeline::read_message(s, out) == ReadResult::kEof);

    client.join();
    REQUIRE(wok.load());
    ::close(s);
    ::close(lfd);
}

TEST_CASE("pipeline CIDR incoming allowlist (v4 and v6)",
          "[pipeline]") {
    using locus::pipeline::Cidr;
    using locus::pipeline::ip_allowed;

    SECTION("IPv4") {
        const auto lo = Cidr::parse("127.0.0.0/8");
        REQUIRE(lo.has_value());
        REQUIRE(lo->contains("127.0.0.1"));
        REQUIRE(lo->contains("127.5.5.5"));
        REQUIRE_FALSE(lo->contains("10.0.0.1"));

        const auto host = Cidr::parse("192.168.1.4");  // bare == /32
        REQUIRE(host.has_value());
        REQUIRE(host->contains("192.168.1.4"));
        REQUIRE_FALSE(host->contains("192.168.1.5"));

        const auto any = Cidr::parse("0.0.0.0/0");
        REQUIRE(any.has_value());
        REQUIRE(any->contains("8.8.8.8"));
    }

    SECTION("IPv6") {
        const auto doc = Cidr::parse("2001:db8::/32");
        REQUIRE(doc.has_value());
        REQUIRE(doc->contains("2001:db8:1234::1"));
        REQUIRE_FALSE(doc->contains("2001:dead::1"));

        const auto lo6 = Cidr::parse("::1");  // bare == /128
        REQUIRE(lo6.has_value());
        REQUIRE(lo6->contains("::1"));
        REQUIRE_FALSE(lo6->contains("::2"));

        // Family mismatch never matches.
        REQUIRE_FALSE(doc->contains("10.0.0.1"));
        REQUIRE_FALSE(Cidr::parse("10.0.0.0/8")->contains("::1"));
    }

    SECTION("malformed") {
        REQUIRE_FALSE(Cidr::parse("not-an-ip").has_value());
        REQUIRE_FALSE(Cidr::parse("10.0.0.0/33").has_value());
        REQUIRE_FALSE(Cidr::parse("2001:db8::/129").has_value());
        // Out-of-range prefix must be rejected on the parsed long, not
        // silently narrowed into range (2^32 + 8 must not become /8).
        REQUIRE_FALSE(Cidr::parse("10.0.0.0/4294967304").has_value());
        REQUIRE_FALSE(Cidr::parse("2001:db8::/4294967360").has_value());
    }

    SECTION("ip_allowed: empty = allow all; else only matches") {
        REQUIRE(ip_allowed("1.2.3.4", {}));
        REQUIRE(ip_allowed("::1", {}));
        const std::vector<Cidr> allow{*Cidr::parse("10.0.0.0/8"),
                                      *Cidr::parse("::1/128")};
        REQUIRE(ip_allowed("10.9.9.9", allow));
        REQUIRE(ip_allowed("::1", allow));
        REQUIRE_FALSE(ip_allowed("127.0.0.1", allow));
        REQUIRE_FALSE(ip_allowed("::2", allow));
    }
}

// Regression: a wildcard bind on a dual-stack host takes IPv4 clients
// as v4-mapped IPv6; accept_one must normalize them to a dotted quad so
// a v4 allowlist rule matches. Also covers the "::1" bind path.
TEST_CASE("pipeline wildcard/::1 listen + allowlist", "[pipeline]") {
    using locus::pipeline::Cidr;

    SECTION("wildcard bind, IPv4 client normalized + allowed") {
        int port = 0;
        const int lfd = locus::pipeline::listen_on("", 0, &port);
        REQUIRE(lfd >= 0);
        REQUIRE(port > 0);
        std::atomic<bool> ok{false};
        std::jthread client([&] {
            const int c = locus::pipeline::connect_to("127.0.0.1", port);
            if (c >= 0) {
                ok = locus::pipeline::write_message(
                    c, locus::pipeline::make_token(1, 0, 5));
                ::close(c);
            }
        });
        std::string peer;
        const int s = locus::pipeline::accept_one(lfd, &peer);
        REQUIRE(s >= 0);
        REQUIRE(peer == "127.0.0.1");  // v4-mapped normalized to quad
        const std::vector<Cidr> allow{*Cidr::parse("127.0.0.0/8")};
        REQUIRE(locus::pipeline::ip_allowed(peer, allow));
        Message m;
        REQUIRE(locus::pipeline::read_message(s, m) == ReadResult::kOk);
        REQUIRE(m.token == 5);
        client.join();
        REQUIRE(ok.load());
        ::close(s);
        ::close(lfd);
    }

    SECTION("IPv6 loopback bind, v6 client allowed") {
        int port = 0;
        const int lfd = locus::pipeline::listen_on("::1", 0, &port);
        if (lfd < 0) {
            SKIP("no IPv6 loopback on this host");
        }
        REQUIRE(port > 0);
        std::atomic<bool> ok{false};
        std::jthread client([&] {
            const int c = locus::pipeline::connect_to("::1", port);
            if (c >= 0) {
                ok = locus::pipeline::write_message(
                    c, locus::pipeline::make_token(1, 0, 9));
                ::close(c);
            }
        });
        std::string peer;
        const int s = locus::pipeline::accept_one(lfd, &peer);
        REQUIRE(s >= 0);
        REQUIRE(peer == "::1");
        const std::vector<Cidr> allow{*Cidr::parse("::1/128")};
        REQUIRE(locus::pipeline::ip_allowed(peer, allow));
        Message m;
        REQUIRE(locus::pipeline::read_message(s, m) == ReadResult::kOk);
        REQUIRE(m.token == 9);
        client.join();
        REQUIRE(ok.load());
        ::close(s);
        ::close(lfd);
    }
}

// i#851: the allowlist as a kernel filter on the listener. A refused
// SYN is dropped, so the connect times out instead of completing.
TEST_CASE("attach_allow_filter drops connects from outside the ranges",
          "[pipeline]") {
#if !defined(__linux__)
    SKIP("the socket filter is Linux only");
#else
    using locus::pipeline::Cidr;
    const auto ranges = [](std::initializer_list<const char*> s) {
        std::vector<Cidr> out;
        for (const char* c : s) {
            out.push_back(*Cidr::parse(c));
        }
        return out;
    };
    // @returns whether a connect to host:port completes within 300 ms.
    const auto connects = [](const char* host, int port) {
        const int c = locus::pipeline::connect_to(host, port, 300);
        if (c >= 0) {
            ::close(c);
        }
        return c >= 0;
    };
    int port = 0;
    const int lfd = locus::pipeline::listen_on("", 0, &port);  // dual-stack
    REQUIRE(lfd >= 0);
    const bool v6 = connects("::1", port);  // before any filter

    if (!locus::pipeline::attach_allow_filter(lfd,
                                              ranges({"10.0.0.0/8"}))) {
        // Some hosts refuse socket filters on TCP sockets (a cgroup
        // setsockopt hook, say); the check at accept is all there is.
        const int e = errno;
        ::close(lfd);
        SKIP("this host refuses a socket filter on a TCP socket: "
             << std::strerror(e));
    }
    CHECK_FALSE(connects("127.0.0.1", port));
    REQUIRE(locus::pipeline::attach_allow_filter(lfd,
                                                 ranges({"127.0.0.1"})));
    CHECK(connects("127.0.0.1", port));
    // A prefix that ends inside a byte: 126.0.0.0/7 holds 127.x.
    REQUIRE(locus::pipeline::attach_allow_filter(lfd,
                                                 ranges({"126.0.0.0/7"})));
    CHECK(connects("127.0.0.1", port));
    // IPv4 packets meet only the IPv4 ranges.
    REQUIRE(locus::pipeline::attach_allow_filter(lfd, ranges({"::/0"})));
    CHECK_FALSE(connects("127.0.0.1", port));
    if (v6) {
        CHECK(connects("::1", port));
        REQUIRE(locus::pipeline::attach_allow_filter(
            lfd, ranges({"fe80::/10", "0.0.0.0/0"})));
        CHECK_FALSE(connects("::1", port));
        CHECK(connects("127.0.0.1", port));
        // ::/127 holds ::1 (the last word decides); ::2/127 does not.
        REQUIRE(locus::pipeline::attach_allow_filter(lfd,
                                                     ranges({"::/127"})));
        CHECK(connects("::1", port));
        REQUIRE(locus::pipeline::attach_allow_filter(lfd,
                                                     ranges({"::2/127"})));
        CHECK_FALSE(connects("::1", port));
    } else {
        WARN("no IPv6 loopback: the IPv6 half did not run");
    }
    // An empty allowlist removes the filter.
    REQUIRE(locus::pipeline::attach_allow_filter(lfd, {}));
    CHECK(connects("127.0.0.1", port));
    // Past the kernel's program size: not attached, the old one stays.
    std::vector<Cidr> many;
    for (int i = 0; i < 400; ++i) {
        many.push_back(*Cidr::parse("2001:db8::" + std::to_string(i)));
    }
    CHECK_FALSE(locus::pipeline::attach_allow_filter(lfd, many));
    CHECK(connects("127.0.0.1", port));
    ::close(lfd);
#endif
}

// The same program on a UDP socket, which every Linux host lets a user
// filter: a datagram from outside the ranges never arrives. This checks
// the program itself where TCP sockets refuse filters.
TEST_CASE("attach_allow_filter's program passes and drops datagrams",
          "[pipeline]") {
#if !defined(__linux__)
    SKIP("the socket filter is Linux only");
#else
    using locus::pipeline::Cidr;
    const auto ranges = [](std::initializer_list<const char*> s) {
        std::vector<Cidr> out;
        for (const char* c : s) {
            out.push_back(*Cidr::parse(c));
        }
        return out;
    };
    // A dual-stack UDP socket on the wildcard, and its port.
    const int rx = ::socket(AF_INET6, SOCK_DGRAM, 0);
    REQUIRE(rx >= 0);
    int v6only = 0;
    ::setsockopt(rx, IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof(v6only));
    sockaddr_in6 any{};
    any.sin6_family = AF_INET6;
    any.sin6_addr = in6addr_any;
    REQUIRE(::bind(rx, reinterpret_cast<sockaddr*>(&any), sizeof(any)) == 0);
    socklen_t len = sizeof(any);
    REQUIRE(::getsockname(rx, reinterpret_cast<sockaddr*>(&any), &len) == 0);
    const std::uint16_t port = ntohs(any.sin6_port);
    // @returns whether a datagram sent from `host` reaches rx in 200 ms.
    const auto arrives = [&](const char* host) {
        addrinfo hints{};
        hints.ai_socktype = SOCK_DGRAM;
        hints.ai_flags = AI_NUMERICHOST;
        addrinfo* res = nullptr;
        if (::getaddrinfo(host, std::to_string(port).c_str(), &hints,
                          &res) != 0) {
            return false;
        }
        const int tx = ::socket(res->ai_family, SOCK_DGRAM, 0);
        const bool sent =
            tx >= 0 && ::sendto(tx, "x", 1, 0, res->ai_addr,
                                res->ai_addrlen) == 1;
        ::freeaddrinfo(res);
        if (tx >= 0) {
            ::close(tx);
        }
        pollfd p{rx, POLLIN, 0};
        char b = 0;
        return sent && ::poll(&p, 1, 200) == 1 &&
               ::recv(rx, &b, 1, 0) == 1;
    };
    const bool v6 = arrives("::1");
    REQUIRE(arrives("127.0.0.1"));

    const auto check = [&](std::initializer_list<const char*> allow,
                           bool want4, bool want6) {
        INFO(*allow.begin());
        REQUIRE(locus::pipeline::attach_allow_filter(rx, ranges(allow)));
        CHECK(arrives("127.0.0.1") == want4);
        if (v6) {
            CHECK(arrives("::1") == want6);
        }
    };
    check({"10.0.0.0/8"}, false, false);
    check({"127.0.0.1"}, true, false);
    check({"126.0.0.0/7"}, true, false);   // ends inside a byte
    check({"128.0.0.0/1"}, false, false);  // 127 has the top bit clear
    check({"0.0.0.0/0"}, true, false);
    check({"::1"}, false, true);
    check({"::/127"}, false, true);        // the last word decides
    check({"::2/127"}, false, false);
    check({"::/0"}, false, true);
    check({"fe80::/10", "10.0.0.0/8", "127.0.0.0/8"}, true, false);
    REQUIRE(locus::pipeline::attach_allow_filter(rx, {}));  // detach
    CHECK(arrives("127.0.0.1"));
    if (!v6) {
        WARN("no IPv6 loopback: the IPv6 half did not run");
    }
    ::close(rx);
#endif
}

TEST_CASE("pipeline connect_to fails fast on a dead port",
          "[pipeline]") {
    // Reserve then release a port so nothing is listening there.
    int port = 0;
    const int l = locus::pipeline::listen_on("127.0.0.1", 0, &port);
    REQUIRE(l >= 0);
    ::close(l);
    // The non-blocking connect must return failure quickly (refused),
    // not hang on the multi-second OS default.
    const auto t0 = std::chrono::steady_clock::now();
    const int c = locus::pipeline::connect_to("127.0.0.1", port, 500);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    REQUIRE(c < 0);
    REQUIRE(ms < 2000);
    if (c >= 0) {
        ::close(c);
    }
}

TEST_CASE("pipeline recv timeout yields kTimeout", "[pipeline]") {
    int fds[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    locus::pipeline::set_recv_timeout(fds[0], 100);  // 100 ms
    Message m;
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = locus::pipeline::read_message(fds[0], m);  // no data
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    REQUIRE(r == ReadResult::kTimeout);
    REQUIRE(ms >= 80);    // waited about the timeout
    REQUIRE(ms < 2000);
    ::close(fds[0]);
    ::close(fds[1]);
}

TEST_CASE("pipeline set_keepalive toggles SO_KEEPALIVE", "[pipeline]") {
    int port = 0;
    const int l = locus::pipeline::listen_on("127.0.0.1", 0, &port);
    REQUIRE(l >= 0);
    const int c = locus::pipeline::connect_to("127.0.0.1", port);
    REQUIRE(c >= 0);
    const int s = locus::pipeline::accept_one(l);
    REQUIRE(s >= 0);

    int on = -1;
    socklen_t sl = sizeof(on);
    locus::pipeline::set_keepalive(c, 5, 2, 3);  // enable
    REQUIRE(::getsockopt(c, SOL_SOCKET, SO_KEEPALIVE, &on, &sl) == 0);
    REQUIRE(on != 0);  // enabled (macOS reads back the flag bit, not 1)
    locus::pipeline::set_keepalive(c, 0, 2, 3);  // idle<=0 disables
    REQUIRE(::getsockopt(c, SOL_SOCKET, SO_KEEPALIVE, &on, &sl) == 0);
    REQUIRE(on == 0);

    ::close(c);
    ::close(s);
    ::close(l);
}
