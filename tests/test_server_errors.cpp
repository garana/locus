#include "locus/server/errors.hpp"

#include <new>
#include <stdexcept>
#include <string>

#include "catch_amalgamated.hpp"
#include "json.hpp"

using locus::server::http_status_for;
using nlohmann::json;

namespace {

/** The status for whatever `f` throws. */
template <typename F>
int status_of(F f) {
    try {
        f();
    } catch (const std::exception& e) {
        return http_status_for(e);
    }
    FAIL("no exception thrown");
    return 0;
}

}  // namespace

TEST_CASE("handler errors map to 400 for the client, 500 for the server",
          "[server][errors]") {
    // Client errors: the request checks, a malformed body, a field of
    // the wrong type.
    REQUIRE(status_of([] { throw std::invalid_argument("n must be 1"); }) ==
            400);
    REQUIRE(status_of([] { (void)json::parse("{\"prompt\": "); }) == 400);
    REQUIRE(status_of([] { (void)json(1).get<std::string>(); }) == 400);
    REQUIRE(status_of([] { (void)json::object().at("prompt"); }) == 400);
    // Server-side failures.
    REQUIRE(status_of([] {
                throw std::runtime_error("janus: KV cache too small");
            }) == 500);
    REQUIRE(status_of([] { throw std::bad_alloc(); }) == 500);
    REQUIRE(status_of([] { throw std::out_of_range("index"); }) == 500);
}
