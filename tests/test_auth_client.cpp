#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/auth/auth_client.hpp"
#include "locus/auth/helper_connection.hpp"
#include "locus/auth/helper_message.hpp"
#include "locus/auth/helper_spawn.hpp"
#include "locus/auth/token_cache.hpp"

using locus::auth::AuthClient;
using locus::auth::AuthConfig;
using locus::auth::decode_text;
using locus::auth::encode_text;
using locus::auth::HelperConnection;
using locus::auth::HelperDecode;
using locus::auth::HelperMessage;
using locus::auth::ManualClock;
using namespace std::chrono_literals;

namespace {

/** A scripted auth helper on the far end of a socketpair: counts
 * requests per credential, optionally delays or stays silent. */
class FakeHelper {
  public:
    explicit FakeHelper(int fd) : fd_(fd), th_([this] { loop(); }) {}
    ~FakeHelper() {
        ::shutdown(fd_, SHUT_RDWR);  // unblock the reader
        if (th_.joinable()) {
            th_.join();
        }
        ::close(fd_);
    }
    int count(const std::string& cred) {
        std::lock_guard<std::mutex> lk(mu_);
        return counts_[cred];
    }
    std::chrono::milliseconds delay{0};
    std::set<std::string> silent;   // creds to never answer
    std::atomic<int> events{0};     // fire-and-forget notifies seen

  private:
    void loop() {
        std::string buf, err;
        for (;;) {
            HelperMessage m;
            while (decode_text(buf, m, err) != HelperDecode::kComplete) {
                char c[512];
                const ssize_t n = ::read(fd_, c, sizeof c);
                if (n <= 0) {
                    return;
                }
                buf.append(c, static_cast<std::size_t>(n));
            }
            if (m.type == "request") {  // lifecycle event push
                events.fetch_add(1);
                continue;               // no reply
            }
            const std::string cred =
                m.get("credential") ? *m.get("credential") : "";
            {
                std::lock_guard<std::mutex> lk(mu_);
                counts_[cred]++;
            }
            if (silent.count(cred) != 0) {
                continue;  // force a client timeout
            }
            if (delay.count() > 0) {
                std::this_thread::sleep_for(delay);
            }
            HelperMessage r;
            r.req_id = m.req_id;
            r.type = "auth";
            if (cred == "bad") {
                r.set("ok", "0");
            } else {
                r.set("ok", "1");
                r.set("identity", "user-" + cred);
                r.set("not_after", "4000000000000000000");
            }
            std::string wire;
            if (encode_text(r, wire)) {
                // GCC flags an ignored write() (warn_unused_result).
                [[maybe_unused]] const ssize_t wn =
                    ::write(fd_, wire.data(), wire.size());
            }
        }
    }

    int fd_;
    std::mutex mu_;
    std::map<std::string, int> counts_;
    std::thread th_;
};

/** Builds an AuthClient with one socketpair-backed helper. */
struct Fixture {
    ManualClock clock;
    std::unique_ptr<AuthClient> client;
    std::unique_ptr<FakeHelper> helper;

    explicit Fixture(AuthConfig cfg = {}) {
        int fds[2];
        REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
        std::vector<std::unique_ptr<HelperConnection>> conns;
        conns.push_back(
            std::make_unique<HelperConnection>(fds[0], fds[0]));
        helper = std::make_unique<FakeHelper>(fds[1]);
        client = std::make_unique<AuthClient>(std::move(conns), clock,
                                              cfg);
    }
    ~Fixture() { client.reset(); helper.reset(); }  // client first
};

}  // namespace

TEST_CASE("auth resolves allow/deny and caches", "[auth]") {
    Fixture f;
    auto ok = f.client->resolve("alice");
    REQUIRE(ok.ok);
    REQUIRE(ok.identity == "user-alice");
    auto no = f.client->resolve("bad");
    REQUIRE_FALSE(no.ok);

    // Repeats are served from cache -- the helper is asked once each.
    f.client->resolve("alice");
    f.client->resolve("alice");
    f.client->resolve("bad");
    REQUIRE(f.helper->count("alice") == 1);
    REQUIRE(f.helper->count("bad") == 1);
}

TEST_CASE("auth coalesces concurrent resolves of one credential",
          "[auth]") {
    Fixture f;
    f.helper->delay = 60ms;  // widen the in-flight window
    std::vector<std::thread> ts;
    std::atomic<int> ok{0};
    for (int i = 0; i < 8; ++i) {
        ts.emplace_back([&] {
            if (f.client->resolve("carol").ok) {
                ok.fetch_add(1);
            }
        });
    }
    for (auto& t : ts) {
        t.join();
    }
    REQUIRE(ok.load() == 8);            // all 8 resolved
    REQUIRE(f.helper->count("carol") == 1);  // via ONE helper call
}

TEST_CASE("auth fails closed and does not cache a timeout", "[auth]") {
    AuthConfig cfg;
    cfg.request_timeout = 80ms;
    Fixture f(cfg);
    f.helper->silent.insert("ghost");  // helper never answers
    auto r1 = f.client->resolve("ghost");
    REQUIRE_FALSE(r1.ok);  // fail closed
    auto r2 = f.client->resolve("ghost");
    REQUIRE_FALSE(r2.ok);
    // Not cached: each attempt re-queried the helper.
    REQUIRE(f.helper->count("ghost") == 2);
}

TEST_CASE("auth resolves through the reference helper binary",
          "[auth]") {
    // End to end: spawn the real locus-auth-helper with an allow-list
    // and resolve against it through AuthClient.
    ManualClock clock;
    std::string err;
    auto conn =
        locus::auth::spawn_helper({LOCUS_AUTH_HELPER_BIN, "sk-ok"},
                                  &err);
    REQUIRE(conn);
    std::vector<std::unique_ptr<HelperConnection>> conns;
    conns.push_back(std::move(conn));
    AuthClient client(std::move(conns), clock, AuthConfig{});

    auto good = client.resolve("sk-ok");
    REQUIRE(good.ok);
    REQUIRE(good.identity == "user:sk-ok");
    auto bad = client.resolve("sk-nope");
    REQUIRE_FALSE(bad.ok);
}

TEST_CASE("auth pushes fire-and-forget lifecycle events", "[auth]") {
    Fixture f;
    HelperMessage ev;
    ev.type = "request";
    ev.set("event", "create");
    ev.set("kind", "chat");
    f.client->report_event(std::move(ev));
    // Give the helper a moment to read it.
    for (int i = 0; i < 50 && f.helper->events.load() == 0; ++i) {
        std::this_thread::sleep_for(2ms);
    }
    REQUIRE(f.helper->events.load() == 1);
}
