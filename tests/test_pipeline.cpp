#include <atomic>
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
