#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace locus::tok {

/**
 * Decodes one UTF-8 codepoint at s[i] and advances i. A malformed or
 * truncated sequence yields its first byte as the codepoint.
 */
std::uint32_t next_cp(std::string_view s, std::size_t& i);

/** Appends codepoint cp to out as UTF-8 (1 to 4 bytes). */
void append_utf8(std::string& out, std::uint32_t cp);

/** @returns Whether cp has the Unicode White_Space property. */
bool is_white(std::uint32_t cp);

/**
 * @returns Whether cp is in the Unicode general category N* (Nd, Nl,
 *     No), as Python's unicodedata (Unicode 17.0) lists it.
 */
bool is_numeric(std::uint32_t cp);

}  // namespace locus::tok
