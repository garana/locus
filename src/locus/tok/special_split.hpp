#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "locus/tok/tokenizer.hpp"

namespace locus::tok {

/** One special token the split looks for (i#1153). */
struct SpecialText {
    std::string_view text;  ///< never empty
    TokenId id;
    /** True: a match that overlaps a literal range (a chat message) is
     *  not a match, so the client's copy stays text. */
    bool literal_skip;
};

/**
 * Splits `text` at special tokens' text: calls on_text(segment) for
 * each non-empty run between them and on_special(id) for each match,
 * in order. At each point the earliest match wins, and on a tie the
 * special that comes first in `specials` (pass them longest first).
 *
 * Linear in the text per special (i#1153): each special keeps the
 * position of its next valid match while it is ahead of the cursor,
 * and searches again only from the cursor once passed; one with no
 * match left is never searched again; a match inside a literal range
 * restarts its search at that range's end. The ranges are sorted and
 * merged once, and the one a match overlaps is found by binary search,
 * so many short messages do not cost messages squared.
 *
 * @param specials The specials, longest first.
 * @param literal Byte ranges of `text` copied from messages.
 */
template <typename TextFn, typename SpecialFn>
void split_specials(std::string_view text,
                    std::span<const SpecialText> specials,
                    std::span<const TextRange> literal, TextFn&& on_text,
                    SpecialFn&& on_special) {
    constexpr std::size_t kNone = std::string_view::npos;
    constexpr std::size_t kUnknown = kNone - 1;
    std::vector<std::size_t> next(specials.size(), kUnknown);
    // The literal ranges sorted and merged, so the range a match may
    // overlap is found by binary search, not by scanning all of them:
    // a chat of many short messages must not cost messages squared.
    std::vector<TextRange> lits;
    lits.reserve(literal.size());
    for (const TextRange& r : literal) {
        if (r.begin < r.end) lits.push_back(r);
    }
    std::sort(lits.begin(), lits.end(),
              [](const TextRange& a, const TextRange& b) {
                  return a.begin < b.begin;
              });
    std::size_t kept = 0;
    for (const TextRange& r : lits) {
        if (kept > 0 && r.begin <= lits[kept - 1].end) {
            lits[kept - 1].end = std::max(lits[kept - 1].end, r.end);
        } else {
            lits[kept++] = r;
        }
    }
    lits.resize(kept);
    // The next match of special k at or after `from`, skipping matches
    // that overlap a literal range when the special asks for it.
    const auto search = [&](std::size_t k, std::size_t from) {
        const SpecialText& s = specials[k];
        for (;;) {
            const std::size_t p = text.find(s.text, from);
            if (p == kNone || !s.literal_skip) return p;
            const std::size_t e = p + s.text.size();
            // The first range that ends after p; it, and any after it
            // that starts before e, overlaps the match.
            auto it = std::upper_bound(
                lits.begin(), lits.end(), p,
                [](std::size_t v, const TextRange& r) { return v < r.end; });
            std::size_t jump = 0;
            for (; it != lits.end() && it->begin < e; ++it) jump = it->end;
            if (jump == 0) return p;
            from = jump;  // past the range: no copy inside it counts
        }
    };
    std::size_t at = 0;
    while (at < text.size()) {
        std::size_t best = kNone;
        std::size_t best_k = 0;
        for (std::size_t k = 0; k < specials.size(); ++k) {
            std::size_t& n = next[k];
            if (n == kUnknown || (n != kNone && n < at)) {
                n = search(k, at);
            }
            if (n < best) {
                best = n;
                best_k = k;
            }
        }
        const std::size_t end = best == kNone ? text.size() : best;
        if (end > at) {
            on_text(text.substr(at, end - at));
        }
        if (best == kNone) {
            break;
        }
        on_special(specials[best_k].id);
        at = best + specials[best_k].text.size();
    }
}

}  // namespace locus::tok
