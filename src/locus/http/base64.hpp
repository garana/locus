#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace locus::http {

/**
 * Standard base64 (RFC 4648 section 4: A-Z a-z 0-9 + /, '=' padding,
 * no line breaks), as JSON fields like OpenAI's b64_json carry it.
 *
 * @param data The bytes to encode.
 * @returns 4 * ceil(n / 3) characters.
 */
std::string base64_encode(std::span<const std::byte> data);

}  // namespace locus::http
