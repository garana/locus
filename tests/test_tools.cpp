#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/server/tools.hpp"

using locus::server::detect_tool_call;
using locus::server::parse_tools;
using locus::server::render_tool_call;
using locus::server::render_tool_result;
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

TEST_CASE("render_tool_call and render_tool_result", "[tools]") {
    // OpenAI sends arguments as a JSON string; it renders as an object.
    REQUIRE(render_tool_call("f", "{\"a\":1}") ==
            "{\"arguments\":{\"a\":1},\"name\":\"f\"}");
    REQUIRE(render_tool_call("f", json{{"a", 1}}) ==
            render_tool_call("f", "{\"a\":1}"));
    // A string that is not JSON stays a string; none is {}.
    REQUIRE(render_tool_call("f", "oops") ==
            "{\"arguments\":\"oops\",\"name\":\"f\"}");
    REQUIRE(render_tool_call("f", json()) ==
            "{\"arguments\":{},\"name\":\"f\"}");
    REQUIRE(render_tool_result("f", "9") == "Tool result (f):\n9");
    REQUIRE(render_tool_result("", "9") == "Tool result:\n9");
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
