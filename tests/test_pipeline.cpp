#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

#include "catch_amalgamated.hpp"
#include "locus/pipeline/message.hpp"

using locus::pipeline::Activation;
using locus::pipeline::Decode;
using locus::pipeline::ReadResult;

namespace {

Activation sample(std::uint64_t id, std::uint32_t pos, std::size_t n) {
    Activation a;
    a.request_id = id;
    a.position = pos;
    a.hidden.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        a.hidden[i] = static_cast<float>(i) * 0.5f - 3.0f;
    }
    return a;
}

}  // namespace

TEST_CASE("pipeline activation codec round-trips", "[pipeline]") {
    SECTION("typical vector") {
        const auto a = sample(42, 7, 64);
        std::string buf;
        locus::pipeline::encode(a, buf);
        Activation out;
        std::string err;
        REQUIRE(locus::pipeline::decode(buf, out, err) ==
                Decode::kComplete);
        REQUIRE(buf.empty());  // fully consumed
        REQUIRE(out.request_id == a.request_id);
        REQUIRE(out.position == a.position);
        REQUIRE(out.hidden == a.hidden);  // bitwise-identical floats
    }

    SECTION("empty hidden vector") {
        const auto a = sample(1, 0, 0);
        std::string buf;
        locus::pipeline::encode(a, buf);
        Activation out;
        std::string err;
        REQUIRE(locus::pipeline::decode(buf, out, err) ==
                Decode::kComplete);
        REQUIRE(out.hidden.empty());
        REQUIRE(out.request_id == 1);
    }

    SECTION("two frames decode in sequence (pipelining)") {
        const auto a = sample(1, 2, 3);
        const auto b = sample(9, 8, 5);
        std::string buf;
        locus::pipeline::encode(a, buf);
        locus::pipeline::encode(b, buf);
        Activation o1, o2;
        std::string err;
        REQUIRE(locus::pipeline::decode(buf, o1, err) ==
                Decode::kComplete);
        REQUIRE(locus::pipeline::decode(buf, o2, err) ==
                Decode::kComplete);
        REQUIRE(buf.empty());
        REQUIRE(o1.hidden == a.hidden);
        REQUIRE(o2.request_id == 9);
        REQUIRE(o2.hidden == b.hidden);
    }

    SECTION("partial buffer is incomplete, not error") {
        const auto a = sample(1, 1, 10);
        std::string buf;
        locus::pipeline::encode(a, buf);
        Activation out;
        std::string err;
        // Missing the trailing payload bytes.
        std::string part = buf.substr(0, buf.size() - 4);
        const std::size_t before = part.size();
        REQUIRE(locus::pipeline::decode(part, out, err) ==
                Decode::kIncomplete);
        REQUIRE(part.size() == before);  // left untouched
        // Only part of the length prefix present.
        std::string tiny = buf.substr(0, 3);
        REQUIRE(locus::pipeline::decode(tiny, out, err) ==
                Decode::kIncomplete);
    }

    SECTION("a too-small frame length is an error") {
        std::string buf;
        buf.push_back(3);  // frame_len = 3, below the header size
        buf.push_back(0);
        buf.push_back(0);
        buf.push_back(0);
        Activation out;
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

    const std::vector<Activation> sent = {
        sample(1, 0, 64), sample(2, 1, 64), sample(3, 2, 128)};

    std::atomic<bool> wok{true};
    std::thread writer([&] {
        for (const auto& a : sent) {
            if (!locus::pipeline::write_message(fds[1], a)) {
                wok = false;
            }
        }
        ::close(fds[1]);  // clean EOF after the last frame
    });

    for (const auto& a : sent) {
        Activation out;
        REQUIRE(locus::pipeline::read_message(fds[0], out) ==
                ReadResult::kOk);
        REQUIRE(out.request_id == a.request_id);
        REQUIRE(out.position == a.position);
        REQUIRE(out.hidden == a.hidden);
    }
    Activation out;
    REQUIRE(locus::pipeline::read_message(fds[0], out) ==
            ReadResult::kEof);

    writer.join();
    REQUIRE(wok.load());
    ::close(fds[0]);
}

TEST_CASE("pipeline read_message reports EOF and truncation",
          "[pipeline]") {
    Activation out;

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
        const auto a = sample(1, 0, 10);
        std::string frame;
        locus::pipeline::encode(a, frame);
        // Send the 4-byte length plus only 2 payload bytes, then close.
        REQUIRE(::send(fds[1], frame.data(), 6, 0) == 6);
        ::close(fds[1]);
        REQUIRE(locus::pipeline::read_message(fds[0], out) ==
                ReadResult::kError);
        ::close(fds[0]);
    }
}
