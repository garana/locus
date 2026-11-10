#include "locus/model/expert_cache.hpp"

#include <algorithm>

namespace locus::model {

ExpertCache::ExpertCache(std::uint32_t n_layers, std::uint32_t n_expert,
                         std::uint32_t capacity, std::uint32_t step)
    : n_expert_(n_expert),
      capacity_(capacity),
      step_(std::max<std::uint32_t>(step, 1)),
      rank_(static_cast<std::size_t>(n_layers) * n_expert, -1) {}

ExpertCache::Touch ExpertCache::touch(std::uint32_t layer,
                                      std::uint32_t expert) {
    const auto key = layer * n_expert_ + expert;
    Touch t;
    std::lock_guard<std::mutex> lk(mu_);
    if (key >= rank_.size()) {
        return t;  // out of range: run it on the CPU, keep no state
    }
    if (rank_[key] < 0) {
        const auto i = static_cast<std::uint32_t>(order_.size());
        order_.push_back(key);
        rank_[key] = static_cast<std::int32_t>(i);
        t.promoted = i < capacity_;  // fills the zone
        return t;
    }
    const auto i = static_cast<std::uint32_t>(rank_[key]);
    t.hot = i < capacity_;
    const std::uint32_t j = i > step_ ? i - step_ : 0;
    if (j == i) {
        return t;
    }
    if (i >= capacity_ && capacity_ > j) {
        // order_[capacity_ - 1] shifts to capacity_: out of the zone.
        const std::uint32_t out = order_[capacity_ - 1];
        t.promoted = true;
        t.evicted = true;
        t.evicted_layer = out / n_expert_;
        t.evicted_expert = out % n_expert_;
    }
    std::rotate(order_.begin() + j, order_.begin() + i,
                order_.begin() + i + 1);
    for (std::uint32_t r = j; r <= i; ++r) {
        rank_[order_[r]] = static_cast<std::int32_t>(r);
    }
    return t;
}

bool ExpertCache::resident(std::uint32_t layer, std::uint32_t expert) const {
    const auto key = layer * n_expert_ + expert;
    std::lock_guard<std::mutex> lk(mu_);
    return key < rank_.size() && rank_[key] >= 0 &&
           static_cast<std::uint32_t>(rank_[key]) < capacity_;
}

}  // namespace locus::model
