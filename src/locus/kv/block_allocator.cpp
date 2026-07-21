#include "locus/kv/block_allocator.hpp"

#include <stdexcept>

namespace locus::kv {

BlockAllocator::BlockAllocator(std::uint32_t num_blocks)
    : ref_counts_(num_blocks, 0) {
    free_list_.reserve(num_blocks);
    // Reverse order so blocks are handed out starting from id 0.
    for (std::uint32_t i = num_blocks; i > 0; --i) {
        free_list_.push_back(i - 1);
    }
}

std::optional<BlockId> BlockAllocator::allocate() {
    if (free_list_.empty()) {
        return std::nullopt;
    }
    BlockId id = free_list_.back();
    free_list_.pop_back();
    ref_counts_[id] = 1;
    return id;
}

void BlockAllocator::retain(BlockId id) {
    if (ref_count(id) == 0) {
        throw std::logic_error("retain of a free block");
    }
    ++ref_counts_[id];
}

bool BlockAllocator::release(BlockId id) {
    if (ref_count(id) == 0) {
        throw std::logic_error("release of a free block");
    }
    if (--ref_counts_[id] > 0) {
        return false;
    }
    free_list_.push_back(id);
    return true;
}

std::uint32_t BlockAllocator::ref_count(BlockId id) const {
    return ref_counts_.at(id);
}

std::uint32_t BlockAllocator::free_blocks() const {
    return static_cast<std::uint32_t>(free_list_.size());
}

std::uint32_t BlockAllocator::total_blocks() const {
    return static_cast<std::uint32_t>(ref_counts_.size());
}

}  // namespace locus::kv
