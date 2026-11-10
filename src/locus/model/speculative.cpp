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

namespace {

// Each observation keeps this share of the older counts, so about
// the last ten verifies decide the accept rate.
constexpr double kAcceptDecay = 0.9;
// Weight of the newest time in DraftCost's moving average.
constexpr double kCostWeight = 0.2;

}  // namespace

double DraftState::accept_rate() const {
    return (accepted + 1.0) / (accepted + rejected + 2.0);
}

void observe_draft(DraftState& s, std::uint32_t drafted,
                   std::uint32_t accepted) {
    s.accepted = kAcceptDecay * s.accepted + accepted;
    s.rejected = kAcceptDecay * s.rejected + (accepted < drafted ? 1 : 0);
}

void observe_guess(DraftState& s, bool hit) {
    observe_draft(s, 1, hit ? 1 : 0);
}

void DraftCost::observe(std::uint32_t drafted, double seconds) {
    if (drafted >= sec_.size()) {
        sec_.resize(drafted + 1, 0.0);
        seen_.resize(drafted + 1, 0);
    }
    // On Vulkan the first forward of a shape pays the first use of
    // that batch size, so it is not counted there.
    if (seen_[drafted]++ == 0 && drop_first_) {
        return;
    }
    double& v = sec_[drafted];
    v = v == 0.0 ? seconds : (1 - kCostWeight) * v + kCostWeight * seconds;
}

double DraftCost::seconds(std::uint32_t drafted) const {
    return drafted < sec_.size() ? sec_[drafted] : 0.0;
}

std::uint32_t choose_draft(const DraftState& s, const DraftCost& cost,
                           std::uint32_t max_draft) {
    const double t0 = cost.seconds(0);
    if (t0 <= 0.0) {
        return 0;
    }
    const double a = s.accept_rate();
    std::uint32_t best = 0;
    double best_rate = 1.0 / t0;
    double tokens = 1.0;  // 1 + a + ... + a^d
    double term = 1.0;
    // The cost of one more row for a length not measured yet: the
    // cheapest measured step between consecutive measured lengths, or
    // a quarter of a decode step before any is measured.
    double row = t0 / 4;
    bool measured = false;
    for (std::uint32_t d = 1, prev = 0; d <= max_draft; ++d) {
        if (const double td = cost.seconds(d); td > 0.0) {
            const double s =
                std::max(0.0, (td - cost.seconds(prev)) / (d - prev));
            row = measured ? std::min(row, s) : s;
            measured = true;
            prev = d;
        }
    }
    std::uint32_t lo = 0;  // the longest measured length so far
    for (std::uint32_t d = 1; d <= max_draft; ++d) {
        term *= a;
        tokens += term;
        double t = cost.seconds(d);
        if (t > 0.0) {
            lo = d;
        } else {
            t = cost.seconds(lo) + (d - lo) * row;
        }
        if (tokens / t > best_rate) {
            best_rate = tokens / t;
            best = d;
        }
    }
    return best;
}

}  // namespace locus::model
