#include "locus/model/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

#include "locus/model/llama.hpp"  // argmax

namespace locus::model {

namespace {

/** Applies repetition / frequency / presence penalties in place. */
void apply_penalties(std::span<float> logits, const SamplingParams& p,
                     std::span<const tok::TokenId> history) {
    if (p.repeat_penalty == 1.0f && p.frequency_penalty == 0.0f &&
        p.presence_penalty == 0.0f) {
        return;
    }
    const std::size_t n = history.size();
    const std::size_t start =
        n > p.repeat_last_n ? n - p.repeat_last_n : 0;
    std::unordered_map<tok::TokenId, int> counts;
    for (std::size_t i = start; i < n; ++i) {
        ++counts[history[i]];
    }
    for (const auto& [tok, c] : counts) {
        if (static_cast<std::size_t>(tok) >= logits.size()) {
            continue;
        }
        float& l = logits[tok];
        if (p.repeat_penalty != 1.0f) {
            l = l > 0.0f ? l / p.repeat_penalty
                         : l * p.repeat_penalty;
        }
        l -= p.frequency_penalty * static_cast<float>(c) +
             p.presence_penalty;
    }
}

}  // namespace

tok::TokenId sample(std::span<float> logits, const SamplingParams& p,
                    std::span<const tok::TokenId> history,
                    std::mt19937_64& rng,
                    const TokenConstraint* constraint) {
    apply_penalties(logits, p, history);

    // Greedy: highest-logit token (allowed one, if constrained).
    if (p.temperature <= 0.0f) {
        if (constraint == nullptr) {
            return argmax(logits);
        }
        std::int64_t best = -1;
        float best_l = 0.0f;
        for (std::uint32_t i = 0; i < logits.size(); ++i) {
            if (constraint->allows(i) &&
                (best < 0 || logits[i] > best_l)) {
                best = i;
                best_l = logits[i];
            }
        }
        return best >= 0 ? static_cast<tok::TokenId>(best)
                         : argmax(logits);
    }

    // Candidate (logit, index) pairs, highest logit first.
    std::vector<std::pair<float, std::uint32_t>> cand;
    cand.reserve(logits.size());
    for (std::uint32_t i = 0; i < logits.size(); ++i) {
        cand.emplace_back(logits[i], i);
    }
    if (constraint != nullptr) {
        std::vector<std::pair<float, std::uint32_t>> allowed;
        allowed.reserve(cand.size());
        for (const auto& c : cand) {
            if (constraint->allows(c.second)) {
                allowed.push_back(c);
            }
        }
        if (!allowed.empty()) {  // else: dead end -> stay unconstrained
            cand.swap(allowed);
        }
    }
    const auto by_logit = [](const auto& a, const auto& b) {
        return a.first > b.first;
    };
    if (p.top_k > 0 &&
        p.top_k < static_cast<std::uint32_t>(cand.size())) {
        std::partial_sort(cand.begin(), cand.begin() + p.top_k,
                          cand.end(), by_logit);
        cand.resize(p.top_k);
    } else {
        std::sort(cand.begin(), cand.end(), by_logit);
    }

    // Temperature-scaled softmax weights (subtract the max for
    // stability; cand[0] is the max after sorting).
    const float maxl = cand.front().first;
    const float inv_t = 1.0f / p.temperature;
    double sum = 0.0;
    std::vector<double> w(cand.size());
    for (std::size_t i = 0; i < cand.size(); ++i) {
        w[i] = std::exp(static_cast<double>(
            (cand[i].first - maxl) * inv_t));
        sum += w[i];
    }

    // top_p (nucleus): keep the leading prefix whose cumulative
    // probability first reaches top_p (at least one token).
    std::size_t keep = cand.size();
    if (p.top_p < 1.0f) {
        double cum = 0.0;
        for (std::size_t i = 0; i < cand.size(); ++i) {
            cum += w[i] / sum;
            if (cum >= p.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    // min_p: drop tokens whose probability is below min_p * max_prob.
    if (p.min_p > 0.0f) {
        const double thresh = p.min_p * (w[0] / sum);
        for (std::size_t i = 1; i < keep; ++i) {
            if (w[i] / sum < thresh) {
                keep = i;
                break;
            }
        }
    }

    double kept_sum = 0.0;
    for (std::size_t i = 0; i < keep; ++i) {
        kept_sum += w[i];
    }
    std::uniform_real_distribution<double> dist(0.0, kept_sum);
    double r = dist(rng);
    for (std::size_t i = 0; i < keep; ++i) {
        r -= w[i];
        if (r <= 0.0) {
            return cand[i].second;
        }
    }
    return cand[keep - 1].second;
}

void logprobs_from(std::span<const float> logits,
                   tok::TokenId chosen, std::uint32_t n_top,
                   float& chosen_lp,
                   std::vector<TokenLogprob>& top) {
    const std::size_t V = logits.size();
    float max_logit = logits[0];
    for (float l : logits) {
        max_logit = std::max(max_logit, l);
    }
    double sum = 0.0;
    for (float l : logits) {
        sum += std::exp(static_cast<double>(l) - max_logit);
    }
    const double log_z = std::log(sum) + max_logit;
    auto lp = [&](std::size_t i) {
        return static_cast<float>(logits[i] - log_z);
    };
    chosen_lp =
        (chosen >= 0 && static_cast<std::size_t>(chosen) < V)
            ? lp(static_cast<std::size_t>(chosen))
            : 0.0f;

    top.clear();
    const std::size_t k = std::min<std::size_t>(n_top, V);
    if (k == 0) {
        return;
    }
    std::vector<std::size_t> idx(V);
    for (std::size_t i = 0; i < V; ++i) {
        idx[i] = i;
    }
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](std::size_t a, std::size_t b) {
                          return logits[a] > logits[b];
                      });
    top.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        top.push_back(
            {static_cast<tok::TokenId>(idx[i]), lp(idx[i])});
    }
}

}  // namespace locus::model
