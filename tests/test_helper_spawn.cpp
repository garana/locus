#include <chrono>
#include <future>
#include <string>
#include <thread>
#include <utility>

#include "catch_amalgamated.hpp"
#include "locus/auth/helper_connection.hpp"
#include "locus/auth/helper_message.hpp"
#include "locus/auth/helper_spawn.hpp"

using locus::auth::HelperMessage;
using locus::auth::spawn_helper;
using namespace std::chrono_literals;

TEST_CASE("spawn_helper round-trips through a real process", "[auth]") {
    // `cat` echoes stdin -> stdout verbatim, so it replays the framed
    // request (same req_id) back as its "response" -- exercising the
    // pipe wiring, exec, and req_id correlation end to end.
    std::string err;
    auto conn = spawn_helper({"cat"}, &err);
    REQUIRE(conn);

    HelperMessage q;
    q.type = "auth";
    q.set("credential", "hello");
    auto resp = conn->request(std::move(q), 2000ms);
    REQUIRE(resp.has_value());
    REQUIRE(resp->type == "auth");
    REQUIRE(*resp->get("credential") == "hello");
}

TEST_CASE("spawn_helper on a bad command reports broken", "[auth]") {
    std::string err;
    auto conn = spawn_helper({"/no/such/locus-helper-xyz"}, &err);
    REQUIRE(conn);  // the fork itself succeeds

    HelperMessage q;
    q.type = "auth";
    q.set("credential", "x");
    auto resp = conn->request(std::move(q), 1000ms);
    REQUIRE_FALSE(resp.has_value());  // exec failed -> stdout EOF
    for (int i = 0; i < 100 && !conn->broken(); ++i) {
        std::this_thread::sleep_for(2ms);
    }
    REQUIRE(conn->broken());
}

TEST_CASE("spawn_helper rejects empty argv", "[auth]") {
    std::string err;
    auto conn = spawn_helper({}, &err);
    REQUIRE_FALSE(conn);
    REQUIRE_FALSE(err.empty());
}

TEST_CASE("helper dtor never hangs on a SIGTERM-ignoring child",
          "[auth]") {
    // A helper that traps SIGTERM and never reads stdin ignores both
    // the closed-stdin EOF and the SIGTERM the dtor sends. Pre-fix
    // the dtor's blocking waitpid would hang shutdown forever; the
    // bounded grace + SIGKILL must reap it and return.
    // `exec sleep` replaces the shell so the child is a SINGLE
    // process (== child_pid_) with SIGTERM ignored (SIG_IGN survives
    // exec); no forked grandchild to orphan and leak past SIGKILL.
    std::string err;
    auto conn = spawn_helper(
        {"/bin/sh", "-c", "trap '' TERM; exec sleep 3600"}, &err);
    REQUIRE(conn);

    std::promise<void> done;
    auto fut = done.get_future();
    std::thread t([conn = std::move(conn), &done]() mutable {
        conn.reset();  // runs ~HelperConnection: grace -> SIGKILL
        done.set_value();
    });

    // Grace is ~1s; allow a generous margin.
    const bool returned =
        fut.wait_for(5s) == std::future_status::ready;
    if (returned) {
        t.join();
    } else {
        t.detach();  // pre-fix: dtor hung; leak the worker
    }
    REQUIRE(returned);
}
