#pragma once

#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>

#include "json.hpp"

namespace locus::http {

/**
 * The deepest nesting of objects and arrays a JSON text may have. Request
 * bodies and upstream replies reach code that recurses once per level
 * (dump, copy, the JSON-schema and tool converters), so the depth is
 * bounded when the text is parsed, far below any thread's stack.
 */
inline constexpr int kMaxJsonDepth = 64;

/**
 * Parses JSON text as Json::parse does, refusing nesting deeper than
 * kMaxJsonDepth. The parser itself is iterative; the bound protects what
 * runs on the result.
 *
 * @throws std::invalid_argument for too deep a text;
 *     Json::parse_error for malformed text.
 */
template <class Json = nlohmann::json>
Json parse_json(std::string_view text) {
    return Json::parse(
        text, [](int depth, typename Json::parse_event_t, Json&) {
            if (depth > kMaxJsonDepth) {
                throw std::invalid_argument(
                    "JSON nested deeper than " +
                    std::to_string(kMaxJsonDepth) + " levels");
            }
            return true;
        });
}

/**
 * As parse_json, but a malformed or too deep text gives a discarded
 * value (Json::is_discarded()) instead of an exception, as Json::parse
 * with allow_exceptions false does.
 */
template <class Json = nlohmann::json>
Json parse_json_or_discarded(std::string_view text) {
    try {
        return parse_json<Json>(text);
    } catch (const std::exception&) {
        return Json(Json::value_t::discarded);
    }
}

}  // namespace locus::http
