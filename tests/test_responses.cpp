#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/server/responses.hpp"

using namespace locus::server;
using nlohmann::json;

namespace {

/** The `type` of each SSE event in `sse`, in order. */
std::vector<std::string> event_types(const std::string& sse) {
    std::vector<std::string> out;
    std::size_t at = 0;
    const std::string dp = "data: ";
    std::uint64_t seq = 0;
    while ((at = sse.find(dp, at)) != std::string::npos) {
        at += dp.size();
        const std::size_t end = sse.find("\n\n", at);
        const json j = json::parse(sse.substr(at, end - at));
        REQUIRE(j["sequence_number"] == seq++);
        out.push_back(j["type"].get<std::string>());
        at = end;
    }
    return out;
}

/** The data JSON of the first SSE event of type `type`. */
json event_of(const std::string& sse, const std::string& type) {
    const std::string tag = "event: " + type + "\ndata: ";
    const std::size_t at = sse.find(tag);
    REQUIRE(at != std::string::npos);
    const std::size_t b = at + tag.size();
    return json::parse(sse.substr(b, sse.find("\n\n", b) - b));
}

}  // namespace

TEST_CASE("responses request: input string and instructions",
          "[responses]") {
    const auto r = parse_responses_request(
        {{"instructions", "Be brief."}, {"input", "hi"}});
    REQUIRE(r.messages.size() == 2);
    CHECK(r.messages[0].role == "system");
    CHECK(r.messages[0].content == "Be brief.");
    CHECK(r.messages[1].role == "user");
    CHECK(r.messages[1].content == "hi");
    CHECK(r.tools.empty());
    CHECK_FALSE(r.max_output_tokens);
    CHECK_FALSE(r.stream);
}

TEST_CASE("responses request: Codex-shaped items and tools",
          "[responses]") {
    // The shape OpenAI Codex sends: developer messages for the system
    // prompt, input_text parts, flat function tools, a namespace of
    // tools, a web_search tool, and earlier tool calls with outputs.
    const json body{
        {"model", "m"},
        {"stream", true},
        {"store", false},
        {"tools",
         json::array(
             {{{"type", "function"},
               {"name", "exec_command"},
               {"description", "Runs a command."},
               {"parameters", {{"type", "object"}}},
               {"strict", false}},
              {{"type", "namespace"},
               {"name", "agents"},
               {"description", "sub-agents"},
               {"tools",
                json::array({{{"type", "function"},
                              {"name", "spawn"},
                              {"parameters", {{"type", "object"}}}}})}},
              {{"type", "web_search"}}})},
        {"input",
         json::array(
             {{{"type", "message"},
               {"role", "developer"},
               {"content",
                json::array({{{"type", "input_text"}, {"text", "rules"}}})}},
              {{"type", "message"},
               {"role", "user"},
               {"content",
                json::array({{{"type", "input_text"}, {"text", "a"}},
                             {{"type", "input_text"}, {"text", "b"}}})}},
              {{"type", "reasoning"}, {"summary", json::array()}},
              {{"type", "function_call"},
               {"call_id", "c1"},
               {"name", "exec_command"},
               {"arguments", R"({"cmd":"ls"})"}},
              {{"type", "function_call_output"},
               {"call_id", "c1"},
               {"output", "file.txt"}},
              {{"type", "function_call"},
               {"call_id", "c2"},
               {"name", "spawn"},
               {"namespace", "agents"},
               {"arguments", "{}"}},
              {{"type", "function_call_output"},
               {"call_id", "c2"},
               {"output",
                json::array({{{"type", "input_text"}, {"text", "ok"}}})}},
              {{"type", "message"},
               {"role", "assistant"},
               {"content",
                json::array(
                    {{{"type", "output_text"}, {"text", "done"}}})}}})}};
    const auto r = parse_responses_request(body);
    CHECK(r.stream);

    REQUIRE(r.tools.size() == 2);
    CHECK(r.tools[0].name == "exec_command");
    CHECK(r.tools[0].description == "Runs a command.");
    CHECK(r.tools[1].name == "agents.spawn");
    REQUIRE(r.namespaced.count("agents.spawn") == 1);
    CHECK(r.namespaced.at("agents.spawn").first == "agents");
    CHECK(r.namespaced.at("agents.spawn").second == "spawn");

    REQUIRE(r.messages.size() == 7);
    // One leading system message: the tool instruction, then the
    // developer text.
    const std::string& sys = r.messages[0].content;
    CHECK(r.messages[0].role == "system");
    CHECK(sys.find("exec_command") != std::string::npos);
    CHECK(sys.find("Tool result") != std::string::npos);
    CHECK(sys.size() > 6);
    CHECK(sys.substr(sys.size() - 6) == "\nrules");
    CHECK(r.messages[1].role == "user");
    CHECK(r.messages[1].content == "a\nb");
    // The reasoning item adds nothing; the call is the model's JSON.
    CHECK(r.messages[2].role == "assistant");
    CHECK(json::parse(r.messages[2].content) ==
          json{{"name", "exec_command"}, {"arguments", {{"cmd", "ls"}}}});
    CHECK(r.messages[3].role == "user");
    CHECK(r.messages[3].content == "Tool result (exec_command):\nfile.txt");
    CHECK(json::parse(r.messages[4].content)["name"] == "agents.spawn");
    CHECK(r.messages[5].content == "Tool result (agents.spawn):\nok");
    CHECK(r.messages[6].role == "assistant");
    CHECK(r.messages[6].content == "done");
}

TEST_CASE("responses request: developer items join one system message",
          "[responses]") {
    // Codex also sends developer items later in the history; they must
    // not become a second system turn (llama2 keeps only the last).
    const auto r = parse_responses_request(
        {{"instructions", "first"},
         {"input",
          json::array({{{"role", "user"}, {"content", "q1"}},
                       {{"role", "developer"}, {"content", "later"}},
                       {{"role", "system"}, {"content", "too"}},
                       {{"role", "user"}, {"content", "q2"}}})}});
    REQUIRE(r.messages.size() == 3);
    CHECK(r.messages[0].role == "system");
    CHECK(r.messages[0].content == "first\nlater\ntoo");
    CHECK(r.messages[1].content == "q1");
    CHECK(r.messages[2].content == "q2");
}

TEST_CASE("responses request: an image in a tool's output is a placeholder",
          "[responses]") {
    const auto r = parse_responses_request(
        {{"input",
          json::array(
              {{{"type", "function_call"},
                {"call_id", "c"},
                {"name", "view_image"},
                {"arguments", "{}"}},
               {{"type", "function_call_output"},
                {"call_id", "c"},
                {"output",
                 json::array({{{"type", "input_text"}, {"text", "shot:"}},
                              {{"type", "input_image"},
                               {"image_url", "data:image/png;base64,AA"}}})}}})}});
    REQUIRE(r.messages.size() == 2);
    CHECK(r.messages[1].content ==
          "Tool result (view_image):\nshot:\n[image omitted]");
}

TEST_CASE("responses request: tool_choice none drops the tools",
          "[responses]") {
    const auto r = parse_responses_request(
        {{"input", "hi"},
         {"tool_choice", "none"},
         {"tools",
          json::array({{{"type", "function"}, {"name", "f"}}})}});
    CHECK(r.tools.empty());
    REQUIRE(r.messages.size() == 1);
    CHECK(r.messages[0].role == "user");
}

TEST_CASE("responses request: rejects what locus cannot serve",
          "[responses]") {
    using Catch::Matchers::ContainsSubstring;
    CHECK_THROWS_WITH(
        parse_responses_request({{"input", "hi"},
                                 {"previous_response_id", "resp_1"}}),
        ContainsSubstring("previous_response_id"));
    CHECK_THROWS_AS(parse_responses_request({{"model", "m"}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(parse_responses_request({{"input", 3}}),
                    std::invalid_argument);
    CHECK_THROWS_WITH(
        parse_responses_request(
            {{"input",
              json::array({{{"role", "user"},
                            {"content",
                             json::array({{{"type", "input_image"},
                                           {"image_url", "x"}}})}}})}}),
        ContainsSubstring("input_image"));
    CHECK_THROWS_AS(parse_responses_request(
                        {{"input", "hi"}, {"max_output_tokens", 0}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        parse_responses_request(
            {{"input",
              json::array({{{"type", "message"}, {"role", "tool"},
                            {"content", "x"}}})}}),
        std::invalid_argument);
    // A null previous_response_id is the same as none.
    CHECK_NOTHROW(parse_responses_request(
        {{"input", "hi"}, {"previous_response_id", nullptr}}));
}

TEST_CASE("responses output: message and function_call items",
          "[responses]") {
    const auto rq = parse_responses_request(
        {{"input", "hi"},
         {"tools",
          json::array({{{"type", "namespace"},
                        {"name", "ns"},
                        {"tools",
                         json::array({{{"type", "function"},
                                       {"name", "g"}}})}}})}});
    ResponsesResult r;
    r.id = "resp_7";
    r.model = "m";
    r.text = "hello";
    r.input_tokens = 10;
    r.cached_tokens = 4;
    r.output_tokens = 3;

    const json msg = responses_output_item(r, rq);
    CHECK(msg["type"] == "message");
    CHECK(msg["role"] == "assistant");
    CHECK(msg["content"][0] ==
          json{{"type", "output_text"},
               {"text", "hello"},
               {"annotations", json::array()}});

    r.call = ToolCall{"ns.g", {{"x", 1}}};
    const json fc = responses_output_item(r, rq);
    CHECK(fc["type"] == "function_call");
    CHECK(fc["name"] == "g");
    CHECK(fc["namespace"] == "ns");
    CHECK(fc["call_id"] == "call_resp_7");
    CHECK(json::parse(fc["arguments"].get<std::string>()) ==
          json{{"x", 1}});

    const json o = responses_object(r, json::array({fc}), "completed");
    CHECK(o["object"] == "response");
    CHECK(o["status"] == "completed");
    CHECK(o["incomplete_details"].is_null());
    CHECK(o["usage"]["input_tokens"] == 10);
    CHECK(o["usage"]["input_tokens_details"]["cached_tokens"] == 4);
    CHECK(o["usage"]["output_tokens"] == 3);
    CHECK(o["usage"]["total_tokens"] == 13);
    const json inc = responses_object(r, json::array(), "incomplete");
    CHECK(inc["incomplete_details"]["reason"] == "max_output_tokens");
    CHECK_FALSE(responses_object(r, json::array(), "in_progress")
                    .contains("usage"));
}

TEST_CASE("responses SSE: event order for text and for a tool call",
          "[responses]") {
    const auto rq = parse_responses_request(
        {{"input", "hi"},
         {"tools", json::array({{{"type", "function"}, {"name", "f"}}})}});
    ResponsesResult r;
    r.id = "resp_1";
    r.model = "m";

    SECTION("live text") {
        ResponsesSse sse;
        r.text = "ab";
        // One call per statement, as the server writes them (the
        // operands of `+` may run in either order).
        std::string s = sse.start(r);
        s += sse.message_open(r);
        s += sse.text_delta(r, "a");
        s += sse.text_delta(r, "b");
        s += sse.finish(r, rq, true);
        CHECK(event_types(s) ==
              std::vector<std::string>{
                  "response.created", "response.in_progress",
                  "response.output_item.added",
                  "response.content_part.added",
                  "response.output_text.delta",
                  "response.output_text.delta",
                  "response.output_text.done",
                  "response.content_part.done",
                  "response.output_item.done", "response.completed"});
        CHECK(event_of(s, "response.output_text.done")["text"] == "ab");
        const json done = event_of(s, "response.completed");
        CHECK(done["response"]["id"] == "resp_1");
        CHECK(done["response"]["output"][0]["content"][0]["text"] == "ab");
    }
    SECTION("buffered text that hit the cap") {
        ResponsesSse sse;
        r.text = "ab";
        r.incomplete = true;
        std::string s = sse.start(r);
        s += sse.finish(r, rq, false);
        const auto types = event_types(s);
        CHECK(types.back() == "response.incomplete");
        CHECK(std::count(types.begin(), types.end(),
                         "response.output_text.delta") == 1);
        CHECK(event_of(s, "response.output_text.delta")["delta"] == "ab");
    }
    SECTION("function call") {
        ResponsesSse sse;
        r.call = ToolCall{"f", {{"k", "v"}}};
        std::string s = sse.start(r);
        s += sse.finish(r, rq, false);
        CHECK(event_types(s) ==
              std::vector<std::string>{
                  "response.created", "response.in_progress",
                  "response.output_item.added",
                  "response.function_call_arguments.delta",
                  "response.function_call_arguments.done",
                  "response.output_item.done", "response.completed"});
        const json added = event_of(s, "response.output_item.added");
        CHECK(added["item"]["status"] == "in_progress");
        CHECK(added["item"]["arguments"] == "");
        const json item = event_of(s, "response.output_item.done")["item"];
        CHECK(item["type"] == "function_call");
        CHECK(item["name"] == "f");
        CHECK(json::parse(item["arguments"].get<std::string>()) ==
              json{{"k", "v"}});
    }
}
