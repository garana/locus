#include <chrono>
#include <string>
#include <thread>

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
