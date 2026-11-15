#include "locus/server/request_checks.hpp"

#include <stdexcept>
#include <string>

#include "catch_amalgamated.hpp"
#include "json.hpp"

using namespace locus::server;
using nlohmann::json;

namespace {

/** @returns What `f` throws as std::invalid_argument, or "" if nothing. */
template <typename F>
std::string message_of(F f) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return {};
}

}  // namespace

// i#1030: the output limit the generation endpoints share.
TEST_CASE("read_count clamps a positive limit and refuses the rest",
          "[server][request]") {
    // Parsed from text, as a request body is: a positive number is then
    // unsigned (a C++ literal 5 would be a signed one).
    const auto body = [](const char* s) { return json::parse(s); };
    CHECK(read_count(json::object(), "max_tokens", 100, 7) == 7);
    CHECK(read_count(body(R"({"max_tokens": 5})"), "max_tokens", 100, 7) ==
          5);
    CHECK(read_count(body(R"({"max_tokens": 500})"), "max_tokens", 100, 7) ==
          100);
    for (const char* bad : {"0", "-3", "1.5", "\"5\"", "null"}) {
        INFO(bad);
        const json b = body((std::string(R"({"max_tokens": )") + bad + "}")
                                .c_str());
        CHECK(message_of([&] { read_count(b, "max_tokens", 100, 7); }) ==
              "max_tokens must be a positive integer");
    }
}

// i#1030: the image and video endpoints' common checks. An optional
// field stays empty when absent: Qwen-Image-Edit runs true CFG when its
// negative_prompt is set, even to "".
TEST_CASE("the image request checks give one message per error",
          "[server][request]") {
    CHECK(body_string(json::object(), "size").empty());
    CHECK(body_string(json{{"size", "512x512"}}, "size") == "512x512");
    CHECK(message_of([] { body_string(json{{"size", 512}}, "size"); }) ==
          "size must be a string");

    CHECK_FALSE(body_optional_string(json::object(), "negative_prompt"));
    const auto empty =
        body_optional_string(json{{"negative_prompt", ""}}, "negative_prompt");
    REQUIRE(empty.has_value());
    CHECK(empty->empty());
    CHECK(message_of([] {
              body_optional_string(json{{"negative_prompt", 1}},
                                   "negative_prompt");
          }) == "negative_prompt must be a string");

    CHECK(require_prompt(json{{"prompt", "a cat"}}) == "a cat");
    CHECK(message_of([] { require_prompt(json::object()); }) ==
          "prompt must be a non-empty string");
    CHECK(message_of([] { require_prompt(json{{"prompt", ""}}); }) ==
          "prompt must be a non-empty string");
    CHECK(message_of([] { require_prompt(json{{"prompt", 3}}); }) ==
          "prompt must be a string");

    CHECK(message_of([] { require_one_image(json::object()); }).empty());
    CHECK(message_of([] { require_one_image(json{{"n", 1}}); }).empty());
    CHECK(message_of([] { require_one_image(json{{"n", 2}}); }) ==
          "n must be 1");
    CHECK(message_of([] { require_one_image(json{{"n", "1"}}); }) ==
          "n must be 1");

    CHECK(message_of([] { require_b64_json(json::object()); }).empty());
    CHECK(message_of([] {
              require_b64_json(json{{"response_format", "b64_json"}});
          }).empty());
    CHECK(message_of([] {
              require_b64_json(json{{"response_format", "url"}});
          }) == "response_format must be b64_json");
    CHECK(message_of([] {
              require_b64_json(json{{"response_format", 1}});
          }) == "response_format must be a string");
}
