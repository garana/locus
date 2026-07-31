#include "locus/server/tools.hpp"

namespace locus::server {

using nlohmann::json;

std::vector<ToolSpec> parse_tools(const json& body, bool anthropic) {
    std::vector<ToolSpec> out;
    if (!body.contains("tools") || !body["tools"].is_array()) {
        return out;
    }
    for (const auto& t : body["tools"]) {
        ToolSpec spec;
        if (anthropic) {
            spec.name = t.value("name", "");
            spec.description = t.value("description", "");
            spec.schema = t.value("input_schema", json::object());
        } else {
            if (t.value("type", std::string("function")) !=
                "function") {
                continue;
            }
            const json f = t.value("function", json::object());
            spec.name = f.value("name", "");
            spec.description = f.value("description", "");
            spec.schema = f.value("parameters", json::object());
        }
        if (!spec.name.empty()) {
            out.push_back(std::move(spec));
        }
    }
    return out;
}

std::string render_tools_system(const std::vector<ToolSpec>& tools) {
    if (tools.empty()) {
        return {};
    }
    json arr = json::array();
    for (const auto& t : tools) {
        arr.push_back({{"name", t.name},
                       {"description", t.description},
                       {"parameters", t.schema}});
    }
    return "You have access to these tools (JSON):\n" +
           arr.dump(2) +
           "\n\nWhen a tool is needed, reply with ONLY a JSON "
           "object and nothing else:\n"
           "{\"name\": \"<tool_name>\", \"arguments\": { ... }}\n"
           "Otherwise, answer the user normally in plain text.";
}

std::optional<ToolCall> detect_tool_call(
    const std::string& text, const std::vector<ToolSpec>& tools) {
    if (tools.empty()) {
        return std::nullopt;
    }
    const std::size_t b = text.find('{');
    const std::size_t e = text.rfind('}');
    if (b == std::string::npos || e == std::string::npos ||
        e < b) {
        return std::nullopt;
    }
    json j;
    try {
        j = json::parse(text.substr(b, e - b + 1));
    } catch (const std::exception&) {
        return std::nullopt;
    }
    if (j.is_object() && j.contains("tool_call") &&
        j["tool_call"].is_object()) {
        j = j["tool_call"];
    }
    if (!j.is_object() || !j.contains("name") ||
        !j["name"].is_string()) {
        return std::nullopt;
    }
    ToolCall call;
    call.name = j["name"].get<std::string>();
    bool known = false;
    for (const auto& t : tools) {
        if (t.name == call.name) {
            known = true;
            break;
        }
    }
    if (!known) {
        return std::nullopt;
    }
    call.arguments = (j.contains("arguments") &&
                      j["arguments"].is_object())
                         ? j["arguments"]
                         : json::object();
    return call;
}

}  // namespace locus::server
