#pragma once

#include <cstdint>

namespace locus::tok {

/**
 * Codepoint category bits, as llama.cpp's `unicode_cpt_flags` defines
 * them; the tables are llama.cpp's (third_party/llama/unicode-data.cpp),
 * so the tokenizers that use these match llama.cpp's choices exactly.
 */
namespace cpt {
constexpr std::uint16_t kUndefined = 0x0001;
constexpr std::uint16_t kNumber = 0x0002;       ///< \p{N}
constexpr std::uint16_t kLetter = 0x0004;       ///< \p{L}
constexpr std::uint16_t kSeparator = 0x0008;    ///< \p{Z}
constexpr std::uint16_t kAccentMark = 0x0010;   ///< \p{M}
constexpr std::uint16_t kPunctuation = 0x0020;  ///< \p{P}
constexpr std::uint16_t kSymbol = 0x0040;       ///< \p{S}
constexpr std::uint16_t kControl = 0x0080;      ///< \p{C}
constexpr std::uint16_t kWhitespace = 0x0100;   ///< White_Space
}  // namespace cpt

/**
 * @returns The category bits of cp (one of kNumber to kControl, or
 *     kUndefined), plus kWhitespace when cp is a White_Space codepoint.
 *     A cp past U+10FFFF is kUndefined.
 */
std::uint16_t cpt_flags(std::uint32_t cp);

/**
 * @returns cp's first canonical-decomposition codepoint (U+00E9 ->
 *     U+0065), or cp when it has none. A one-codepoint mapping, as
 *     llama.cpp's `unicode_cpts_normalize_nfd` does: the combining
 *     marks are dropped, not emitted.
 */
std::uint32_t cpt_nfd(std::uint32_t cp);

/** @returns The simple lowercase mapping of cp, or cp. */
std::uint32_t cpt_lower(std::uint32_t cp);

}  // namespace locus::tok
