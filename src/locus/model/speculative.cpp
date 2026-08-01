#include "locus/model/speculative.hpp"

#include <algorithm>

namespace locus::model {

std::vector<tok::TokenId> ngram_draft(
    std::span<const tok::TokenId> ctx, std::uint32_t ngram,
    std::uint32_t max_draft) {
    const std::size_t n = ctx.size();
    if (ngram == 0 || max_draft == 0 || n <= ngram) {
        return {};
    }
    const tok::TokenId* pat = ctx.data() + (n - ngram);
    // Scan earlier positions, most recent first, for a match of the
    // trailing n-gram; propose what followed it.
    for (std::size_t start = n - ngram; start-- > 0;) {
        if (std::equal(pat, pat + ngram, ctx.data() + start)) {
            std::vector<tok::TokenId> draft;
            for (std::size_t i = start + ngram;
                 i < n && draft.size() < max_draft; ++i) {
                draft.push_back(ctx[i]);
            }
            return draft;
        }
    }
    return {};
}

}  // namespace locus::model
