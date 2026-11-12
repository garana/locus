#include "locus/engine/disk_prefix_worker.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace locus::engine {

DiskPrefixWorker::DiskPrefixWorker(std::shared_ptr<DiskPrefixStore> store,
                                   std::uint64_t backlog_bytes)
    : store_(std::move(store)),
      backlog_cap_(backlog_bytes),
      thread_([this](std::stop_token st) { run(st); }) {}

DiskPrefixWorker::~DiskPrefixWorker() {
    thread_.request_stop();
    cv_.notify_all();
    // ~jthread joins; run() empties both queues before it returns.
}

bool DiskPrefixWorker::write(std::vector<tok::TokenId> tokens,
                             std::uint32_t n_blocks,
                             std::vector<std::uint8_t> bytes) {
    {
        std::lock_guard lock(mu_);
        if (backlog_ + bytes.size() > backlog_cap_) {
            return false;
        }
        backlog_ += bytes.size();
        Job j;
        j.tokens = std::move(tokens);
        j.n_blocks = n_blocks;
        j.bytes = std::move(bytes);
        writes_.push_back(std::move(j));
    }
    cv_.notify_one();
    return true;
}

bool DiskPrefixWorker::would_accept(std::uint64_t bytes) const {
    std::lock_guard lock(mu_);
    return backlog_ + bytes <= backlog_cap_;
}

std::uint32_t DiskPrefixWorker::queued_blocks(
    std::span<const tok::TokenId> tokens) const {
    std::lock_guard lock(mu_);
    std::uint32_t most = 0;
    auto look = [&](const Job& w) {
        if (w.tokens.size() <= tokens.size() &&
            std::equal(w.tokens.begin(), w.tokens.end(), tokens.begin())) {
            most = std::max(most, w.n_blocks);
        }
    };
    for (const Job& w : writes_) {
        look(w);
    }
    if (writing_ != nullptr) {
        look(*writing_);
    }
    return most;
}

void DiskPrefixWorker::read(std::uint64_t id,
                            std::vector<tok::TokenId> tokens,
                            std::uint32_t n_blocks,
                            std::vector<std::uint8_t*> dst) {
    {
        std::lock_guard lock(mu_);
        // A queued write of these blocks: copy them now. The worker
        // runs reads first, so the store would not have them yet.
        for (const Job& w : writes_) {
            if (n_blocks <= w.n_blocks && tokens.size() <= w.tokens.size() &&
                std::equal(tokens.begin(), tokens.end(), w.tokens.begin())) {
                const std::size_t bb = w.bytes.size() / w.n_blocks;
                for (std::uint32_t i = 0; i < n_blocks; ++i) {
                    std::memcpy(dst[i], w.bytes.data() + i * bb, bb);
                }
                done_.push_back({id, true});
                return;
            }
        }
        Job j;
        j.is_read = true;
        j.id = id;
        j.tokens = std::move(tokens);
        j.n_blocks = n_blocks;
        j.dst = std::move(dst);
        reads_.push_back(std::move(j));
    }
    cv_.notify_one();
}

std::vector<DiskPrefixWorker::Done> DiskPrefixWorker::take_done() {
    std::lock_guard lock(mu_);
    return std::exchange(done_, {});
}

bool DiskPrefixWorker::has_done() const {
    std::lock_guard lock(mu_);
    return !done_.empty();
}

void DiskPrefixWorker::drain() {
    std::unique_lock lock(mu_);
    idle_cv_.wait(lock, [this] {
        return reads_.empty() && writes_.empty() && !busy_;
    });
}

void DiskPrefixWorker::run(std::stop_token stop) {
    std::unique_lock lock(mu_);
    for (;;) {
        cv_.wait(lock, [&] {
            return stop.stop_requested() || !reads_.empty() ||
                   !writes_.empty();
        });
        if (reads_.empty() && writes_.empty()) {
            return;  // stop requested and nothing left
        }
        std::deque<Job>& q = reads_.empty() ? writes_ : reads_;
        Job j = std::move(q.front());
        q.pop_front();
        busy_ = true;
        writing_ = j.is_read ? nullptr : &j;
        lock.unlock();
        bool ok;
        if (j.is_read) {
            ok = store_->read(j.tokens, j.n_blocks, j.dst);
        } else {
            const std::size_t bb = j.bytes.size() / j.n_blocks;
            std::vector<const std::uint8_t*> src(j.n_blocks);
            for (std::uint32_t i = 0; i < j.n_blocks; ++i) {
                src[i] = j.bytes.data() + i * bb;
            }
            store_->write(j.tokens, j.n_blocks, src);
            ok = true;
        }
        lock.lock();
        busy_ = false;
        writing_ = nullptr;
        if (j.is_read) {
            done_.push_back({j.id, ok});
        } else {
            backlog_ -= j.bytes.size();
        }
        if (reads_.empty() && writes_.empty()) {
            idle_cv_.notify_all();
        }
    }
}

}  // namespace locus::engine
