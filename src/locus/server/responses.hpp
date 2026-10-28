#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"
#include "locus/chat/template.hpp"
#include "locus/server/tools.hpp"

namespace locus::server {

/**
 * The OpenAI Responses API (POST /v1/responses), stateless: the
 * client sends the whole history each turn (as OpenAI Codex does), and
 * locus stores nothing, so `previous_response_id` is refused. This
 * header holds the request parser and the response / SSE builders; the
 * server runs the generation between them.
 */

/** A /v1/responses request, parsed into chat messages and tools. */
struct ResponsesRequest {
    /** The conversation: `instructions` and the tool instruction as
     * system messages first, then each input item in order. */
    std::vector<chat::Message> messages;
    /** The function tools the model may call, under flat names. */
    std::vector<ToolSpec> tools;
    /** Flat name -> {namespace, name} for the tools of a `namespace`
     * entry, so a call goes back out under its own name. */
    std::map<std::string, std::pair<std::string, std::string>> namespaced;
    /** `max_output_tokens`, when the request set it. */
    std::optional<std::uint64_t> max_output_tokens;
    bool stream = false;
};

/**
 * Parses a /v1/responses body. `input` is a string (one user turn) or
 * a list of items: `message` items (roles user, assistant, and system
 * or developer, which become system messages in place), with string
 * content or `input_text` / `output_text` parts; `function_call`
 * items, rendered as the assistant's JSON tool call; and
 * `function_call_output` items, rendered as a user turn that starts
 * with "Tool result". Other item types (reasoning, web search calls)
 * are history locus did not produce and are skipped. Function tools
 * come flat ({type:"function", name, parameters}) or grouped in a
 * {type:"namespace", name, tools} entry, whose tools get the flat name
 * "<namespace>.<name>"; other tool types are skipped. tool_choice
 * "none" drops the tools.
 *
 * @throws std::invalid_argument on a malformed body, image input, or a
 *     previous_response_id.
 */
ResponsesRequest parse_responses_request(const nlohmann::json& body);

/** What one generation produced, for the response builders. */
struct ResponsesResult {
    std::string id;           ///< "resp_<n>"; items derive their ids
    std::string model;
    std::int64_t created_at = 0;  ///< unix seconds
    std::string text;         ///< the reply, when it is not a call
    std::optional<ToolCall> call;  ///< the tool call, flat name
    bool incomplete = false;  ///< stopped at the output token cap
    std::uint32_t input_tokens = 0;
    std::uint32_t cached_tokens = 0;  ///< subset of input_tokens
    std::uint32_t output_tokens = 0;
};

/**
 * @returns The output item the result maps to: a `function_call`
 *     (with `namespace` for a namespaced tool) when `r.call` is set,
 *     else an assistant `message` with one `output_text` part.
 */
nlohmann::json responses_output_item(const ResponsesResult& r,
                                     const ResponsesRequest& req);

/**
 * @returns The `response` object: status "completed", or "incomplete"
 *     with reason max_output_tokens; `output` as given (empty while
 *     the response is in progress), and `usage` once it is done.
 */
nlohmann::json responses_object(const ResponsesResult& r,
                                const nlohmann::json& output,
                                const std::string& status);

/**
 * Writes Responses SSE events ("event: <type>" plus a JSON data line
 * that also carries `type` and an increasing `sequence_number`).
 */
class ResponsesSse {
  public:
    /** @returns One event for `ev`, which must hold `type`. */
    std::string event(nlohmann::json ev);
    /** @returns response.created and response.in_progress. */
    std::string start(const ResponsesResult& r);
    /**
     * @returns The opening events of an assistant message item
     *     (output_item.added, content_part.added), for live text.
     */
    std::string message_open(const ResponsesResult& r);
    /** @returns One response.output_text.delta of the open message. */
    std::string text_delta(const ResponsesResult& r,
                           const std::string& delta);
    /**
     * @returns The closing events: the message's text.done,
     *     content_part.done and output_item.done when `message_opened`,
     *     or the whole item (a function call with its arguments delta,
     *     or a message with one text delta) when not; then
     *     response.completed or response.incomplete.
     */
    std::string finish(const ResponsesResult& r,
                       const ResponsesRequest& req, bool message_opened);

  private:
    std::uint64_t seq_ = 0;
};

}  // namespace locus::server
