#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "json.hpp"

/**
 * Request checks the server's endpoints share (i#1030): the output limit
 * of the generation endpoints and the image and video endpoints' common
 * fields, with one message per error.
 */
namespace locus::server {

/**
 * A positive integer limit of a request body, clamped to cap.
 *
 * @returns body[key], at most cap, or dflt when the body has no key.
 * @throws std::invalid_argument when the value is not a positive integer
 *     (null included).
 */
inline std::uint32_t read_count(const nlohmann::json& body, const char* key,
                                std::uint32_t cap, std::uint32_t dflt) {
    if (!body.contains(key)) return dflt;
    const auto& v = body[key];
    if (!v.is_number_unsigned() || v.get<std::uint64_t>() == 0) {
        throw std::invalid_argument(std::string(key) +
                                    " must be a positive integer");
    }
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(v.get<std::uint64_t>(), cap));
}

/** @returns body[k], or "" when absent.
 *  @throws std::invalid_argument when it is not a string. */
inline std::string body_string(const nlohmann::json& body, const char* k) {
    if (!body.contains(k)) return {};
    if (!body[k].is_string()) {
        throw std::invalid_argument(std::string(k) + " must be a string");
    }
    return body[k].get<std::string>();
}

/** @returns body[k], or nullopt when absent (an optional field, where
 *  an empty string means something else).
 *  @throws std::invalid_argument when it is not a string. */
inline std::optional<std::string> body_optional_string(
    const nlohmann::json& body, const char* k) {
    if (!body.contains(k)) return std::nullopt;
    return body_string(body, k);
}

/** @returns The body's prompt.
 *  @throws std::invalid_argument when it is missing, not a string or
 *      empty. */
inline std::string require_prompt(const nlohmann::json& body) {
    std::string p = body_string(body, "prompt");
    if (p.empty()) {
        throw std::invalid_argument("prompt must be a non-empty string");
    }
    return p;
}

/** @throws std::invalid_argument when the body asks for n images, n
 *      other than 1 (the pipelines make one). */
inline void require_one_image(const nlohmann::json& body) {
    if (body.contains("n") && (!body["n"].is_number_integer() ||
                               body["n"].get<std::int64_t>() != 1)) {
        throw std::invalid_argument("n must be 1");
    }
}

/** @throws std::invalid_argument when the body asks for a
 *      response_format other than b64_json (the one the endpoints
 *      give). */
inline void require_b64_json(const nlohmann::json& body) {
    if (body.contains("response_format") &&
        body_string(body, "response_format") != "b64_json") {
        throw std::invalid_argument("response_format must be b64_json");
    }
}

}  // namespace locus::server
