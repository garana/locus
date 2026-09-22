#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

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
    std::thread writer([&] {
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
    std::thread client([&] {
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
        std::thread client([&] {
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
        std::thread client([&] {
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
