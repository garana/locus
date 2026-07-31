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
                    std::mt19937_64& rng) {
    apply_penalties(logits, p, history);

    if (p.temperature <= 0.0f) {
        return argmax(logits);
    }

    // Candidate (logit, index) pairs, highest logit first.
    std::vector<std::pair<float, std::uint32_t>> cand;
    cand.reserve(logits.size());
    for (std::uint32_t i = 0; i < logits.size(); ++i) {
        cand.emplace_back(logits[i], i);
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

}  // namespace locus::model
