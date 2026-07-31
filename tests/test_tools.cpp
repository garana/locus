#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/server/tools.hpp"

using locus::server::detect_tool_call;
using locus::server::parse_tools;
using locus::server::render_tools_system;
using nlohmann::json;

TEST_CASE("parse_tools reads OpenAI and Anthropic shapes",
          "[tools]") {
    SECTION("openai function shape") {
        json body{
            {"tools",
             json::array(
                 {{{"type", "function"},
                   {"function",
                    {{"name", "get_weather"},
                     {"description", "current weather"},
                     {"parameters",
                      {{"type", "object"}}}}}}})}};
        auto tools = parse_tools(body, /*anthropic=*/false);
        REQUIRE(tools.size() == 1);
        REQUIRE(tools[0].name == "get_weather");
        REQUIRE(tools[0].description == "current weather");
        REQUIRE(tools[0].schema["type"] == "object");
    }
    SECTION("anthropic shape") {
        json body{{"tools",
                   json::array({{{"name", "get_weather"},
                                 {"description", "w"},
                                 {"input_schema",
                                  {{"type", "object"}}}}})}};
        auto tools = parse_tools(body, /*anthropic=*/true);
        REQUIRE(tools.size() == 1);
        REQUIRE(tools[0].name == "get_weather");
        REQUIRE(tools[0].schema["type"] == "object");
    }
    SECTION("absent tools -> empty") {
        REQUIRE(parse_tools(json::object(), false).empty());
    }
}

TEST_CASE("render_tools_system", "[tools]") {
    REQUIRE(render_tools_system({}).empty());
    auto tools = parse_tools(
        json{{"tools",
              json::array({{{"type", "function"},
                            {"function",
                             {{"name", "ping"}}}}})}},
        false);
    const std::string s = render_tools_system(tools);
    REQUIRE(s.find("ping") != std::string::npos);
    REQUIRE(s.find("\"arguments\"") != std::string::npos);
}

TEST_CASE("detect_tool_call", "[tools]") {
    auto tools = parse_tools(
        json{{"tools",
              json::array({{{"type", "function"},
                            {"function",
                             {{"name", "get_weather"}}}}})}},
        false);

    SECTION("bare {name, arguments}") {
        auto tc = detect_tool_call(
            R"({"name":"get_weather","arguments":{"city":"Paris"}})",
            tools);
        REQUIRE(tc.has_value());
        REQUIRE(tc->name == "get_weather");
        REQUIRE(tc->arguments["city"] == "Paris");
    }
    SECTION("wrapped {tool_call:{...}}") {
        auto tc = detect_tool_call(
            R"({"tool_call":{"name":"get_weather","arguments":{}}})",
            tools);
        REQUIRE(tc.has_value());
        REQUIRE(tc->name == "get_weather");
        REQUIRE(tc->arguments.is_object());
    }
    SECTION("surrounding prose is tolerated") {
        auto tc = detect_tool_call(
            "Sure. {\"name\":\"get_weather\",\"arguments\":"
            "{\"city\":\"X\"}} done",
            tools);
        REQUIRE(tc.has_value());
        REQUIRE(tc->arguments["city"] == "X");
    }
    SECTION("plain text -> no call") {
        REQUIRE_FALSE(
            detect_tool_call("Hello there!", tools).has_value());
    }
    SECTION("unknown tool name -> no call") {
        REQUIRE_FALSE(
            detect_tool_call(R"({"name":"nope","arguments":{}})",
                             tools)
                .has_value());
    }
    SECTION("no tools offered -> no call") {
        REQUIRE_FALSE(
            detect_tool_call(R"({"name":"get_weather"})", {})
                .has_value());
    }
}
