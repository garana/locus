#include "cppllm/kv/paged_cache.hpp"

#include <cassert>
#include <cstring>
#include <stdexcept>

namespace cppllm::kv {

PagedKvCache::PagedKvCache(const Geometry& geom)
    : geom_(geom),
      alloc_(geom.n_blocks),
      layer_stride_(static_cast<std::size_t>(geom.block_tokens) *
                    geom.kv_dim * 2),
      block_stride_(static_cast<std::size_t>(geom.n_layers) *
                    layer_stride_) {
    if (geom.n_layers == 0 || geom.kv_dim == 0 ||
        geom.block_tokens == 0 || geom.n_blocks == 0) {
        throw std::invalid_argument("empty cache geometry");
    }
    pool_.resize(block_stride_ * geom.n_blocks);
}

bool PagedKvCache::ensure_capacity(Seq& seq, std::uint32_t n_more) {
    const std::uint32_t want = seq.n_tokens + n_more;
    std::uint32_t have = capacity(seq);
    if (want <= have) {
        return true;
    }
    const std::uint32_t need_blocks =
        (want - have + geom_.block_tokens - 1) / geom_.block_tokens;
    if (need_blocks > alloc_.free_blocks()) {
        return false;  // all-or-nothing
    }
    for (std::uint32_t i = 0; i < need_blocks; ++i) {
        seq.blocks.push_back(alloc_.allocate().value());
    }
    return true;
}

void PagedKvCache::release(Seq& seq) {
    for (BlockId id : seq.blocks) {
        alloc_.release(id);
    }
    seq.blocks.clear();
    seq.n_tokens = 0;
}

bool PagedKvCache::fork(const Seq& parent, Seq& child) {
    assert(child.blocks.empty());
    const std::uint32_t tail = parent.n_tokens % geom_.block_tokens;
    const std::size_t full =
        parent.n_tokens / geom_.block_tokens;

    // Reserve the tail copy first so failure leaves no side
    // effects on the parent's ref counts.
    BlockId copy_id = kInvalidBlock;
    if (tail != 0 && full < parent.blocks.size()) {
        auto id = alloc_.allocate();
        if (!id) {
            return false;
        }
        copy_id = *id;
        std::memcpy(pool_.data() + *id * block_stride_,
                    pool_.data() +
                        parent.blocks[full] * block_stride_,
                    block_stride_ * sizeof(float));
    }
    for (std::size_t b = 0; b < full; ++b) {
        alloc_.retain(parent.blocks[b]);
        child.blocks.push_back(parent.blocks[b]);
    }
    if (copy_id != kInvalidBlock) {
        child.blocks.push_back(copy_id);
    }
    child.n_tokens = parent.n_tokens;
    return true;
}

float* PagedKvCache::k(const Seq& seq, std::uint32_t layer,
                       std::uint32_t pos) {
    return row(seq, layer, pos, false);
}

float* PagedKvCache::v(const Seq& seq, std::uint32_t layer,
                       std::uint32_t pos) {
    return row(seq, layer, pos, true);
}

float* PagedKvCache::row(const Seq& seq, std::uint32_t layer,
                         std::uint32_t pos, bool value) {
    assert(layer < geom_.n_layers);
    assert(pos < capacity(seq));
    const std::size_t b = pos / geom_.block_tokens;
    const std::size_t in_block = pos % geom_.block_tokens;
    return pool_.data() + seq.blocks[b] * block_stride_ +
           layer * layer_stride_ +
           (value ? layer_stride_ / 2 : 0) +
           in_block * geom_.kv_dim;
}

}  // namespace cppllm::kv
