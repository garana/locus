#include "locus/model/moe_stats.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace locus::model {

namespace {

struct State {
    std::vector<std::vector<std::uint64_t>> counts;  // [layer][e]
    std::uint64_t activations = 0;
};

State& state() {
    static State s;
    return s;
}

}  // namespace

bool MoeStats::enabled() {
    static const bool on =
        std::getenv("LOCUS_MOE_STATS") != nullptr;
    return on;
}

void MoeStats::record(std::uint32_t layer, std::uint32_t expert,
                      std::uint32_t n_layers,
                      std::uint32_t n_expert) {
    State& s = state();
    if (s.counts.empty()) {
        s.counts.assign(n_layers,
                        std::vector<std::uint64_t>(n_expert, 0));
    }
    if (layer < s.counts.size() &&
        expert < s.counts[layer].size()) {
        ++s.counts[layer][expert];
        ++s.activations;
    }
}

void MoeStats::report() {
    const State& s = state();
    if (!enabled() || s.counts.empty()) {
        return;
    }
    std::uint64_t unique_total = 0, slots = 0;
    std::uint64_t min_unique = ~0ull, max_unique = 0;
    for (const auto& layer : s.counts) {
        std::uint64_t unique = 0;
        for (auto c : layer) {
            unique += c > 0 ? 1 : 0;
        }
        // Layers with no activations are dense-lead layers, not
        // part of the MoE working set.
        if (unique > 0) {
            slots += layer.size();
            unique_total += unique;
            min_unique = std::min(min_unique, unique);
            max_unique = std::max(max_unique, unique);
        }
    }
    std::fprintf(
        stderr,
        "moe-stats: %llu activations; experts touched per moe "
        "layer min %llu / max %llu; %llu of %llu (%.1f%%) "
        "layer-expert slots touched\n",
        (unsigned long long)s.activations,
        (unsigned long long)min_unique,
        (unsigned long long)max_unique,
        (unsigned long long)unique_total,
        (unsigned long long)slots,
        slots ? 100.0 * static_cast<double>(unique_total) /
                    static_cast<double>(slots)
              : 0.0);
}

}  // namespace locus::model
