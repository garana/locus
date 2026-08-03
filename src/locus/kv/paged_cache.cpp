#include "locus/kv/paged_cache.hpp"

#include <cassert>
#include <cstring>
#include <stdexcept>

namespace locus::kv {

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
    if (quantized()) {
        if (geom.kv_dim % kKvBlock != 0) {
            throw std::invalid_argument(
                "quantized KV needs kv_dim % 32 == 0");
        }
        q_row_bytes_ = kv_row_bytes(geom.kv_dim, geom.kv_type);
        q_layer_stride_ =
            static_cast<std::size_t>(geom.block_tokens) * 2 *
            q_row_bytes_;
        q_block_stride_ =
            static_cast<std::size_t>(geom.n_layers) * q_layer_stride_;
        qpool_.resize(q_block_stride_ * geom.n_blocks);
        qpool_ptr_ = qpool_.data();
    } else {
        pool_.resize(block_stride_ * geom.n_blocks);
        pool_ptr_ = pool_.data();
    }
}

PagedKvCache::PagedKvCache(const Geometry& geom, float* storage)
    : PagedKvCache(geom) {
    if (quantized()) {
        throw std::invalid_argument(
            "float external storage requires an F32 KV cache");
    }
    pool_.clear();
    pool_.shrink_to_fit();
    pool_ptr_ = storage;
}

PagedKvCache::PagedKvCache(const Geometry& geom,
                           std::uint8_t* storage)
    : PagedKvCache(geom) {
    if (!quantized()) {
        throw std::invalid_argument(
            "byte external storage requires a quantized KV cache");
    }
    qpool_.clear();
    qpool_.shrink_to_fit();
    qpool_ptr_ = storage;
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

void PagedKvCache::retain_prefix(
    Seq& child, const std::vector<BlockId>& blocks) {
    assert(child.blocks.empty());
    for (BlockId id : blocks) {
        alloc_.retain(id);
        child.blocks.push_back(id);
    }
    child.n_tokens = static_cast<std::uint32_t>(blocks.size()) *
                     geom_.block_tokens;
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
        if (quantized()) {
            std::memcpy(qpool_ptr_ + *id * q_block_stride_,
                        qpool_ptr_ +
                            parent.blocks[full] * q_block_stride_,
                        q_block_stride_);
        } else {
            std::memcpy(pool_ptr_ + *id * block_stride_,
                        pool_ptr_ +
                            parent.blocks[full] * block_stride_,
                        block_stride_ * sizeof(float));
        }
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
    assert(!quantized() && "k()/v() are F32-only; use load_row");
    assert(layer < geom_.n_layers);
    assert(pos < capacity(seq));
    const std::size_t b = pos / geom_.block_tokens;
    const std::size_t in_block = pos % geom_.block_tokens;
    return pool_ptr_ + seq.blocks[b] * block_stride_ +
           layer * layer_stride_ +
           (value ? layer_stride_ / 2 : 0) +
           in_block * geom_.kv_dim;
}

std::uint8_t* PagedKvCache::qrow(const Seq& seq, std::uint32_t layer,
                                 std::uint32_t pos,
                                 bool value) const {
    assert(layer < geom_.n_layers);
    assert(pos < capacity(seq));
    const std::size_t b = pos / geom_.block_tokens;
    const std::size_t in_block = pos % geom_.block_tokens;
    // Per layer: block_tokens K rows, then block_tokens V rows.
    return qpool_ptr_ + seq.blocks[b] * q_block_stride_ +
           layer * q_layer_stride_ +
           (value ? q_layer_stride_ / 2 : 0) +
           in_block * q_row_bytes_;
}

void PagedKvCache::store_row(const Seq& seq, std::uint32_t layer,
                             std::uint32_t pos, bool value,
                             std::span<const float> src) {
    assert(src.size() == geom_.kv_dim);
    if (quantized()) {
        kv_quantize_row(src.data(), geom_.kv_dim,
                        qrow(seq, layer, pos, value), geom_.kv_type);
    } else {
        std::memcpy(row(seq, layer, pos, value), src.data(),
                    geom_.kv_dim * sizeof(float));
    }
}

void PagedKvCache::load_row(const Seq& seq, std::uint32_t layer,
                            std::uint32_t pos, bool value,
                            std::span<float> dst) const {
    assert(dst.size() == geom_.kv_dim);
    if (quantized()) {
        kv_dequantize_row(qrow(seq, layer, pos, value), geom_.kv_dim,
                          dst.data(), geom_.kv_type);
    } else {
        // const row access without exposing the mutable helper.
        const std::size_t b = pos / geom_.block_tokens;
        const std::size_t in_block = pos % geom_.block_tokens;
        const float* r = pool_ptr_ + seq.blocks[b] * block_stride_ +
                         layer * layer_stride_ +
                         (value ? layer_stride_ / 2 : 0) +
                         in_block * geom_.kv_dim;
        std::memcpy(dst.data(), r, geom_.kv_dim * sizeof(float));
    }
}

}  // namespace locus::kv
