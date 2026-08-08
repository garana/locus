#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/auth/helper_connection.hpp"
#include "locus/auth/helper_message.hpp"

using locus::auth::decode_text;
using locus::auth::encode_text;
using locus::auth::HelperConnection;
using locus::auth::HelperDecode;
using locus::auth::HelperMessage;
using namespace std::chrono_literals;

namespace {

/** Reads bytes from fd until one message decodes. Returns false on
 * EOF/error (used by the test-side "helper"). */
bool read_one(int fd, std::string& buf, HelperMessage& out) {
    std::string err;
    for (;;) {
        HelperMessage m;
        if (decode_text(buf, m, err) == HelperDecode::kComplete) {
            out = std::move(m);
            return true;
        }
        char c[512];
        const ssize_t n = ::read(fd, c, sizeof c);
        if (n <= 0) {
            return false;
        }
        buf.append(c, static_cast<std::size_t>(n));
    }
}

void write_msg(int fd, const HelperMessage& m) {
    std::string wire;
    REQUIRE(encode_text(m, wire));
    REQUIRE(::write(fd, wire.data(), wire.size()) ==
            static_cast<ssize_t>(wire.size()));
}

HelperMessage reply_to(const HelperMessage& req,
                       const std::string& result) {
    HelperMessage r;
    r.req_id = req.req_id;  // echo the correlation id
    r.type = "ack";
    r.set("result", result);
    return r;
}

}  // namespace

TEST_CASE("helper connection round-trips a request", "[auth]") {
    int fds[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    HelperConnection conn(fds[0], fds[0]);

    std::thread helper([&] {
        std::string buf;
        HelperMessage req;
        if (read_one(fds[1], buf, req)) {
            write_msg(fds[1], reply_to(req, *req.get("credential")));
        }
    });

    HelperMessage q;
    q.type = "auth";
    q.set("credential", "sk-1");
    auto resp = conn.request(std::move(q), 2000ms);
    REQUIRE(resp.has_value());
    REQUIRE(resp->type == "ack");
    REQUIRE(*resp->get("result") == "sk-1");
    helper.join();
}

TEST_CASE("helper connection correlates out-of-order replies",
          "[auth]") {
    int fds[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    HelperConnection conn(fds[0], fds[0]);

    // The helper reads BOTH requests, then answers the second one
    // first -- the client must still route each reply to its sender.
    std::thread helper([&] {
        std::string buf;
        HelperMessage a, b;
        REQUIRE(read_one(fds[1], buf, a));
        REQUIRE(read_one(fds[1], buf, b));
        write_msg(fds[1], reply_to(b, *b.get("credential")));
        write_msg(fds[1], reply_to(a, *a.get("credential")));
    });

    std::optional<HelperMessage> ra, rb;
    std::thread ta([&] {
        HelperMessage q;
        q.type = "auth";
        q.set("credential", "AAA");
        ra = conn.request(std::move(q), 2000ms);
    });
    std::thread tb([&] {
        HelperMessage q;
        q.type = "auth";
        q.set("credential", "BBB");
        rb = conn.request(std::move(q), 2000ms);
    });
    ta.join();
    tb.join();
    helper.join();

    REQUIRE(ra.has_value());
    REQUIRE(rb.has_value());
    REQUIRE(*ra->get("result") == "AAA");  // each got its own reply
    REQUIRE(*rb->get("result") == "BBB");
}

TEST_CASE("helper connection notify is fire-and-forget", "[auth]") {
    int fds[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    HelperConnection conn(fds[0], fds[0]);

    std::string got;
    std::thread helper([&] {
        std::string buf;
        HelperMessage m;
        if (read_one(fds[1], buf, m)) {
            got = m.type + ":" + *m.get("event");
            // no reply
        }
    });

    HelperMessage ev;
    ev.type = "request";
    ev.set("event", "create");
    REQUIRE(conn.notify(std::move(ev)));
    helper.join();
    REQUIRE(got == "request:create");
}

TEST_CASE("helper connection times out with no reply", "[auth]") {
    int fds[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    HelperConnection conn(fds[0], fds[0]);
    // Drain the request but never reply.
    std::thread helper([&] {
        std::string buf;
        HelperMessage m;
        read_one(fds[1], buf, m);
    });
    HelperMessage q;
    q.type = "auth";
    q.set("credential", "x");
    auto resp = conn.request(std::move(q), 100ms);
    REQUIRE_FALSE(resp.has_value());  // timed out
    helper.join();
}

TEST_CASE("helper connection breaks on helper EOF", "[auth]") {
    int fds[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    HelperConnection conn(fds[0], fds[0]);
    ::close(fds[1]);  // helper end gone -> EOF on the client's read
    HelperMessage q;
    q.type = "auth";
    q.set("credential", "x");
    auto resp = conn.request(std::move(q), 2000ms);
    REQUIRE_FALSE(resp.has_value());
    REQUIRE(conn.broken());
}

TEST_CASE("reader drains a burst of replies before honoring EOF",
          "[auth]") {
    // R3: many replies (> one 4096 read chunk) queued in a single
    // burst, immediately followed by the helper closing its end, so
    // POLLIN and POLLHUP arrive together. The reader must decode
    // EVERY buffered, correlated reply before honoring the hangup --
    // pre-fix it read one chunk then fail_all()'d, dropping the rest
    // as spurious auth failures.
    int fds[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    HelperConnection conn(fds[0], fds[0]);

    constexpr int kN = 8;
    const std::string pad(1024, 'x');  // fat replies -> > 4096 total

    std::thread helper([&] {
        std::string buf;
        std::vector<HelperMessage> reqs;
        for (int i = 0; i < kN; ++i) {
            HelperMessage r;
            REQUIRE(read_one(fds[1], buf, r));
            reqs.push_back(std::move(r));
        }
        // Burst all replies, then hang up with them still buffered.
        for (const auto& r : reqs) {
            write_msg(fds[1], reply_to(r, pad));
        }
        ::close(fds[1]);
    });

    std::vector<std::optional<HelperMessage>> res(kN);
    std::vector<std::thread> ts;
    for (int i = 0; i < kN; ++i) {
        ts.emplace_back([&, i] {
            HelperMessage q;
            q.type = "auth";
            q.set("credential", "c" + std::to_string(i));
            res[i] = conn.request(std::move(q), 2000ms);
        });
    }
    for (auto& t : ts) {
        t.join();
    }
    helper.join();

    int ok = 0;
    for (const auto& r : res) {
        if (r.has_value() && *r->get("result") == pad) {
            ++ok;
        }
    }
    REQUIRE(ok == kN);  // every reply delivered, none dropped on HUP
}

TEST_CASE("write_all fails closed when the helper stops draining",
          "[auth]") {
    // S2: a helper that never reads its stdin fills the pipe; a
    // blocking write would wedge every caller under write_mu_. The
    // bounded, non-blocking write must give up at the deadline and
    // fail closed instead of hanging.
    int fds[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    // Shrink the buffers so a modest payload overflows them.
    int sz = 2048;
    ::setsockopt(fds[0], SOL_SOCKET, SO_SNDBUF, &sz, sizeof sz);
    ::setsockopt(fds[1], SOL_SOCKET, SO_RCVBUF, &sz, sizeof sz);
    HelperConnection conn(fds[0], fds[0]);
    // fds[1] is never read: the pipe fills and stays full.

    HelperMessage q;
    q.type = "auth";
    q.set("credential", std::string(256 * 1024, 'x'));  // >> buffer

    const auto t0 = std::chrono::steady_clock::now();
    auto resp = conn.request(std::move(q), 300ms);
    const auto elapsed = std::chrono::steady_clock::now() - t0;

    REQUIRE_FALSE(resp.has_value());  // fail closed, not a hang
    REQUIRE(conn.broken());
    REQUIRE(elapsed < 3s);  // bounded by the deadline, not forever
    ::close(fds[1]);
}
