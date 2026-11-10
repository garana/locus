#include "locus/tok/unicode_flags.hpp"

#include <algorithm>

#include "unicode-data.h"

namespace locus::tok {

std::uint16_t cpt_flags(std::uint32_t cp) {
    if (cp >= MAX_CODEPOINTS) return cpt::kUndefined;
    // unicode_ranges_flags: {start, flags} sorted by start, the first
    // at 0 and the last a sentinel at MAX_CODEPOINTS; a range runs to
    // the next start.
    const auto it = std::upper_bound(
        unicode_ranges_flags.begin(), unicode_ranges_flags.end(), cp,
        [](std::uint32_t c, const auto& r) { return c < r.first; });
    std::uint16_t f = std::prev(it)->second;
    if (unicode_set_whitespace.count(cp) != 0) f |= cpt::kWhitespace;
    return f;
}

std::uint32_t cpt_nfd(std::uint32_t cp) {
    const auto it = std::upper_bound(
        unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cp,
        [](std::uint32_t c, const range_nfd& r) { return c < r.first; });
    if (it == unicode_ranges_nfd.begin()) return cp;
    const auto& r = *std::prev(it);
    return r.first <= cp && cp <= r.last ? r.nfd : cp;
}

std::uint32_t cpt_lower(std::uint32_t cp) {
    const auto it = std::lower_bound(
        unicode_map_lowercase.begin(), unicode_map_lowercase.end(), cp,
        [](const auto& p, std::uint32_t c) { return p.first < c; });
    return it != unicode_map_lowercase.end() && it->first == cp
               ? it->second
               : cp;
}

}  // namespace locus::tok
