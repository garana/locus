#pragma once

#include <optional>
#include <string>
#include <vector>

#include "json.hpp"

namespace locus::server {

/**
 * Tool-calling support for the HTTP front end (output side only:
 * locus expresses the model's intent to call a tool; it never runs
 * the tool -- execution is the client/MCP-host's job).
 *
 * The convention is model-agnostic: the tool list is rendered into a
 * system instruction, and the model is asked to reply with ONLY a
 * JSON object {"name": ..., "arguments": {...}} to call one. The
 * completed output is then parsed back into a structured tool call.
 * Works in non-streaming mode; a model that follows the instruction
 * is required (quality depends on the model).
 */

/** A tool the caller offered (name + JSON-schema parameters). */
struct ToolSpec {
    std::string name;
    std::string description;
    nlohmann::json schema;  // JSON Schema for the arguments
};

/** A tool call parsed out of the model's output. */
struct ToolCall {
    std::string name;
    nlohmann::json arguments;  // object
};

/**
 * Parses the request's `tools` array. OpenAI shape is
 * {type:"function", function:{name,description,parameters}};
 * Anthropic shape is {name,description,input_schema}. Unknown or
 * nameless entries are skipped. @returns the tool specs (possibly
 * empty).
 */
std::vector<ToolSpec> parse_tools(const nlohmann::json& body,
                                  bool anthropic);

/**
 * Renders `tools` into a system-message instruction describing each
 * tool and the JSON reply convention. Empty when `tools` is empty.
 */
std::string render_tools_system(const std::vector<ToolSpec>& tools);

/**
 * Detects a tool call in generated `text`: extracts the first
 * balanced {...}, parses it, accepts {name,arguments} or
 * {tool_call:{...}}, and requires `name` to match one of `tools`.
 * @returns the call, or nullopt when the output is plain text.
 */
std::optional<ToolCall> detect_tool_call(
    const std::string& text, const std::vector<ToolSpec>& tools);

}  // namespace locus::server
