#include "locus/model/stage_metrics.hpp"

namespace locus::model {

namespace {

constexpr auto kRelaxed = std::memory_order_relaxed;
constexpr std::size_t kStages = static_cast<std::size_t>(Stage::kCount);

}  // namespace

const char* stage_name(Stage s) {
    switch (s) {
        case Stage::kEmbed: return "embed";
        case Stage::kNorm: return "norm";
        case Stage::kAttention: return "attention";
        case Stage::kFfn: return "ffn";
        case Stage::kMoe: return "moe";
        case Stage::kLayer: return "layer";
        case Stage::kPrefetch: return "prefetch";
        case Stage::kHead: return "head";
        case Stage::kSample: return "sample";
        case Stage::kDevice: return "device";
        case Stage::kCount: break;
    }
    return "other";
}

StageMetrics::StageMetrics(std::uint32_t n_layers)
    : n_layers_(n_layers),
      ns_(std::make_unique<std::atomic<std::uint64_t>[]>(
          (std::size_t{n_layers} + 1) * kStages)) {}

std::size_t StageMetrics::index(std::uint32_t layer, Stage s) const {
    const std::size_t row = layer < n_layers_ ? layer : n_layers_;
    return row * kStages + static_cast<std::size_t>(s);
}

void StageMetrics::add(std::uint32_t layer, Stage s, std::uint64_t ns) {
    if (s == Stage::kCount) return;
    ns_[index(layer, s)].fetch_add(ns, kRelaxed);
}

std::uint64_t StageMetrics::ns(std::uint32_t layer, Stage s) const {
    if (s == Stage::kCount) return 0;
    return ns_[index(layer, s)].load(kRelaxed);
}

std::uint64_t StageMetrics::stage_ns(Stage s) const {
    std::uint64_t total = 0;
    for (std::uint32_t l = 0; l <= n_layers_; ++l) {
        total += ns(l, s);
    }
    return total;
}

}  // namespace locus::model
