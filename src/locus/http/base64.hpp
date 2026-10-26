#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace locus::http {

/**
 * Standard base64 (RFC 4648 section 4: A-Z a-z 0-9 + /, '=' padding,
 * no line breaks), as JSON fields like OpenAI's b64_json carry it.
 *
 * @param data The bytes to encode.
 * @returns 4 * ceil(n / 3) characters.
 */
std::string base64_encode(std::span<const std::byte> data);

/**
 * Strict standard base64 decode, for request fields such as an image's
 * data URL. Accepts exactly what base64_encode writes: the RFC 4648
 * alphabet, length a multiple of 4, '=' only as the last one or two
 * characters, and zero unused bits in the last group. No whitespace.
 *
 * @param text The encoded characters.
 * @param max_bytes Refuse input that would decode to more than this.
 * @returns The decoded bytes.
 * @throws std::invalid_argument on any malformed or oversized input.
 */
std::vector<std::byte> base64_decode(std::string_view text,
                                     std::size_t max_bytes);

}  // namespace locus::http
