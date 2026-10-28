#include "locus/server/responses.hpp"

#include <stdexcept>

namespace locus::server {

using nlohmann::json;

namespace {

constexpr auto kLossy = json::error_handler_t::replace;

std::string dump(const json& j) { return j.dump(-1, ' ', false, kLossy); }

/** Tells the model how tool results come back (see the item parser). */
constexpr const char* kToolResultNote =
    "\nA tool's output comes back as a user message that starts with "
    "\"Tool result\".";

/**
 * The text of a message item's content: a string, or parts whose
 * input_text / output_text / text entries are joined with newlines.
 */
std::string item_text(const json& content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (!content.is_array()) {
        throw std::invalid_argument(
            "content must be a string or an array of parts");
    }
    std::string out;
    for (const auto& p : content) {
        const std::string type = p.is_object() ? p.value("type", "") : "";
        if (type == "input_image" || type == "input_file") {
            throw std::invalid_argument(type +
                                        " is not supported on /v1/responses");
        }
        if (type != "input_text" && type != "output_text" &&
            type != "text") {
            continue;  // refusals and other parts carry no prompt text
        }
        if (!p.contains("text") || !p["text"].is_string()) {
            throw std::invalid_argument("text part has no text");
        }
        if (!out.empty()) out += "\n";
        out += p["text"].get<std::string>();
    }
    return out;
}

/** Adds one function tool entry (flat or inside a namespace). */
void add_function(ResponsesRequest& out, const json& t,
                  const std::string& ns) {
    ToolSpec spec;
    spec.name = t.value("name", "");
    if (spec.name.empty()) {
        return;
    }
    spec.description = t.value("description", "");
    spec.schema = t.value("parameters", json::object());
    if (!ns.empty()) {
        const std::string flat = ns + "." + spec.name;
        out.namespaced[flat] = {ns, spec.name};
        spec.name = flat;
    }
    out.tools.push_back(std::move(spec));
}

/** The flat name for a call item's name and optional namespace. */
std::string flat_name(const json& item) {
    std::string name = item.value("name", "");
    if (item.contains("namespace") && item["namespace"].is_string() &&
        !item["namespace"].get<std::string>().empty()) {
        name = item["namespace"].get<std::string>() + "." + name;
    }
    return name;
}

}  // namespace

ResponsesRequest parse_responses_request(const json& body) {
    if (!body.is_object()) {
        throw std::invalid_argument("body must be a JSON object");
    }
    if (body.contains("previous_response_id") &&
        !body["previous_response_id"].is_null()) {
        throw std::invalid_argument(
            "previous_response_id is not supported: locus stores no "
            "responses, so send the whole conversation as input");
    }
    if (!body.contains("input")) {
        throw std::invalid_argument("input is missing");
    }
    ResponsesRequest out;

    const std::string choice =
        body.contains("tool_choice") && body["tool_choice"].is_string()
            ? body["tool_choice"].get<std::string>()
            : "auto";
    if (choice != "none" && body.contains("tools") &&
        body["tools"].is_array()) {
        for (const auto& t : body["tools"]) {
            if (!t.is_object()) continue;
            const std::string type = t.value("type", "");
            if (type == "function") {
                add_function(out, t, "");
            } else if (type == "namespace" && t.contains("tools") &&
                       t["tools"].is_array()) {
                const std::string ns = t.value("name", "");
                for (const auto& f : t["tools"]) {
                    if (f.is_object() && f.value("type", "") == "function") {
                        add_function(out, f, ns);
                    }
                }
            }
        }
    }
    if (!out.tools.empty()) {
        out.messages.push_back(
            {"system", render_tools_system(out.tools) + kToolResultNote});
    }
    if (body.contains("instructions") && body["instructions"].is_string() &&
        !body["instructions"].get<std::string>().empty()) {
        out.messages.push_back(
            {"system", body["instructions"].get<std::string>()});
    }

    const json& input = body["input"];
    if (input.is_string()) {
        out.messages.push_back({"user", input.get<std::string>()});
    } else if (input.is_array()) {
        std::map<std::string, std::string> call_names;  // call_id -> name
        for (const auto& item : input) {
            if (!item.is_object()) {
                throw std::invalid_argument("input items must be objects");
            }
            const std::string type =
                item.value("type", item.contains("role") ? "message" : "");
            if (type == "message") {
                std::string role = item.value("role", "user");
                if (role == "developer") role = "system";
                if (role != "system" && role != "user" &&
                    role != "assistant") {
                    throw std::invalid_argument("unsupported message role: " +
                                                role);
                }
                if (!item.contains("content")) {
                    throw std::invalid_argument("message content is missing");
                }
                out.messages.push_back({role, item_text(item["content"])});
            } else if (type == "function_call") {
                const std::string name = flat_name(item);
                call_names[item.value("call_id", "")] = name;
                json args = json::object();
                if (item.contains("arguments") &&
                    item["arguments"].is_string()) {
                    try {
                        args = json::parse(
                            item["arguments"].get<std::string>());
                    } catch (const std::exception&) {
                        args = item["arguments"];  // keep it as sent
                    }
                }
                out.messages.push_back(
                    {"assistant",
                     dump(json{{"name", name}, {"arguments", args}})});
            } else if (type == "function_call_output") {
                const auto it = call_names.find(item.value("call_id", ""));
                std::string text = "Tool result";
                if (it != call_names.end()) {
                    text += " (" + it->second + ")";
                }
                text += ":\n";
                if (item.contains("output")) {
                    text += item_text(item["output"]);
                }
                out.messages.push_back({"user", std::move(text)});
            }
            // Other item types (reasoning, web_search_call, ...) are
            // history from other servers; nothing of them goes in.
        }
    } else {
        throw std::invalid_argument("input must be a string or an array");
    }

    if (body.contains("max_output_tokens") &&
        !body["max_output_tokens"].is_null()) {
        const auto& v = body["max_output_tokens"];
        if (!v.is_number_unsigned() || v.get<std::uint64_t>() == 0) {
            throw std::invalid_argument(
                "max_output_tokens must be a positive integer");
        }
        out.max_output_tokens = v.get<std::uint64_t>();
    }
    if (body.contains("stream")) {
        if (!body["stream"].is_boolean()) {
            throw std::invalid_argument("stream must be a boolean");
        }
        out.stream = body["stream"].get<bool>();
    }
    return out;
}

json responses_output_item(const ResponsesResult& r,
                           const ResponsesRequest& req) {
    if (r.call) {
        json item{{"type", "function_call"},
                  {"id", "fc_" + r.id},
                  {"call_id", "call_" + r.id},
                  {"name", r.call->name},
                  {"arguments", dump(r.call->arguments)},
                  {"status", "completed"}};
        if (const auto it = req.namespaced.find(r.call->name);
            it != req.namespaced.end()) {
            item["namespace"] = it->second.first;
            item["name"] = it->second.second;
        }
        return item;
    }
    return json{{"type", "message"},
                {"id", "msg_" + r.id},
                {"status", "completed"},
                {"role", "assistant"},
                {"content", json::array({json{{"type", "output_text"},
                                              {"text", r.text},
                                              {"annotations",
                                               json::array()}}})}};
}

json responses_object(const ResponsesResult& r, const json& output,
                      const std::string& status) {
    json o{{"id", r.id},
           {"object", "response"},
           {"created_at", r.created_at},
           {"status", status},
           {"model", r.model},
           {"output", output},
           {"error", nullptr},
           {"incomplete_details", nullptr},
           {"parallel_tool_calls", false},
           {"store", false}};
    if (status == "incomplete") {
        o["incomplete_details"] = {{"reason", "max_output_tokens"}};
    }
    if (status != "in_progress") {
        o["usage"] = {
            {"input_tokens", r.input_tokens},
            {"input_tokens_details", {{"cached_tokens", r.cached_tokens}}},
            {"output_tokens", r.output_tokens},
            {"output_tokens_details", {{"reasoning_tokens", 0}}},
            {"total_tokens", r.input_tokens + r.output_tokens}};
    }
    return o;
}

std::string ResponsesSse::event(json ev) {
    const std::string type = ev.at("type").get<std::string>();
    ev["sequence_number"] = seq_++;
    return "event: " + type + "\ndata: " + dump(ev) + "\n\n";
}

std::string ResponsesSse::start(const ResponsesResult& r) {
    const json resp = responses_object(r, json::array(), "in_progress");
    return event({{"type", "response.created"}, {"response", resp}}) +
           event({{"type", "response.in_progress"}, {"response", resp}});
}

std::string ResponsesSse::message_open(const ResponsesResult& r) {
    return event({{"type", "response.output_item.added"},
                  {"output_index", 0},
                  {"item",
                   {{"type", "message"},
                    {"id", "msg_" + r.id},
                    {"status", "in_progress"},
                    {"role", "assistant"},
                    {"content", json::array()}}}}) +
           event({{"type", "response.content_part.added"},
                  {"item_id", "msg_" + r.id},
                  {"output_index", 0},
                  {"content_index", 0},
                  {"part",
                   {{"type", "output_text"},
                    {"text", ""},
                    {"annotations", json::array()}}}});
}

std::string ResponsesSse::text_delta(const ResponsesResult& r,
                                     const std::string& delta) {
    return event({{"type", "response.output_text.delta"},
                  {"item_id", "msg_" + r.id},
                  {"output_index", 0},
                  {"content_index", 0},
                  {"delta", delta}});
}

std::string ResponsesSse::finish(const ResponsesResult& r,
                                 const ResponsesRequest& req,
                                 bool message_opened) {
    const json item = responses_output_item(r, req);
    std::string out;
    if (r.call) {
        json added = item;
        added["status"] = "in_progress";
        added["arguments"] = "";
        out += event({{"type", "response.output_item.added"},
                      {"output_index", 0},
                      {"item", added}});
        out += event({{"type", "response.function_call_arguments.delta"},
                      {"item_id", item["id"]},
                      {"output_index", 0},
                      {"delta", item["arguments"]}});
        out += event({{"type", "response.function_call_arguments.done"},
                      {"item_id", item["id"]},
                      {"output_index", 0},
                      {"arguments", item["arguments"]}});
    } else {
        if (!message_opened) {
            out += message_open(r);
            if (!r.text.empty()) {
                out += text_delta(r, r.text);
            }
        }
        const json part = item["content"][0];
        out += event({{"type", "response.output_text.done"},
                      {"item_id", item["id"]},
                      {"output_index", 0},
                      {"content_index", 0},
                      {"text", r.text}});
        out += event({{"type", "response.content_part.done"},
                      {"item_id", item["id"]},
                      {"output_index", 0},
                      {"content_index", 0},
                      {"part", part}});
    }
    out += event({{"type", "response.output_item.done"},
                  {"output_index", 0},
                  {"item", item}});
    const std::string status = r.incomplete ? "incomplete" : "completed";
    out += event({{"type", "response." + status},
                  {"response",
                   responses_object(r, json::array({item}), status)}});
    return out;
}

}  // namespace locus::server
