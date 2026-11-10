#include <stdexcept>
#include <string>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/server/image_lora.hpp"

using locus::server::parse_lora_fields;
using nlohmann::json;

namespace {

/** @returns The message parse_lora_fields throws for body, or "". */
std::string error_of(const json& body) {
    try {
        (void)parse_lora_fields(body);
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "";
}

}  // namespace

// i#372: the "lora" and "lora_scale" fields of a FLUX image request.
TEST_CASE("image requests name their LoRAs and strengths", "[server][lora]") {
    REQUIRE(parse_lora_fields(json{{"prompt", "x"}}).empty());

    auto p = parse_lora_fields(json{{"lora", "style"}});
    REQUIRE(p.size() == 1);
    CHECK(p[0].name == "style");
    CHECK(p[0].strength == 1.0f);

    p = parse_lora_fields(json{{"lora", "style"}, {"lora_scale", 0.6}});
    CHECK(p[0].strength == Catch::Approx(0.6f));

    p = parse_lora_fields(
        json{{"lora", {"a", "b"}}, {"lora_scale", {0.5, -1.0}}});
    REQUIRE(p.size() == 2);
    CHECK(p[0].name == "a");
    CHECK(p[0].strength == 0.5f);
    CHECK(p[1].name == "b");
    CHECK(p[1].strength == -1.0f);

    // One number applies to every LoRA.
    p = parse_lora_fields(json{{"lora", {"a", "b"}}, {"lora_scale", 2}});
    CHECK(p[0].strength == 2.0f);
    CHECK(p[1].strength == 2.0f);
}

TEST_CASE("image request LoRA fields are checked", "[server][lora]") {
    CHECK(error_of(json{{"lora_scale", 1.0}}) == "lora_scale needs lora");
    CHECK(error_of(json{{"lora", ""}}).find("non-empty") !=
          std::string::npos);
    CHECK(error_of(json{{"lora", 3}}).find("non-empty") != std::string::npos);
    CHECK(error_of(json{{"lora", json::array()}}).find("1 to 8") !=
          std::string::npos);
    CHECK(error_of(json{{"lora", {"a", "b", "c", "d", "e", "f", "g", "h",
                                  "i"}}})
              .find("1 to 8") != std::string::npos);
    CHECK(error_of(json{{"lora", {"a", "a"}}}).find("twice") !=
          std::string::npos);
    CHECK(error_of(json{{"lora", "a"}, {"lora_scale", "x"}})
              .find("must be a number") != std::string::npos);
    CHECK(error_of(json{{"lora", "a"}, {"lora_scale", 4.5}})
              .find("-4 to 4") != std::string::npos);
    CHECK(error_of(json{{"lora", "a"}, {"lora_scale", {1.0}}})
              .find("one number per lora") != std::string::npos);
    CHECK(error_of(json{{"lora", {"a", "b"}}, {"lora_scale", {1.0}}})
              .find("one number per lora") != std::string::npos);
}
