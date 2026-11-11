#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "locus/engine/disk_prefix_store.hpp"
#include "locus/tok/tokenizer.hpp"

namespace locus::engine {

/**
 * The disk prefix tier's I/O thread (i#416): runs DiskPrefixStore
 * reads and writes so the engine thread never blocks on storage.
 * Reads go before writes, since a request waits on each read.
 *
 * Reads land in caller memory (the KV pool's blocks); the caller must
 * keep it alive and untouched until the read's Done is taken. Writes
 * carry their own copy of the bytes, bounded by a backlog cap.
 */
class DiskPrefixWorker {
  public:
    /** A finished read. */
    struct Done {
        std::uint64_t id = 0;
        bool ok = false;
    };

    /**
     * @param backlog_bytes Cap on the bytes of queued writes; a write
     *     past it is dropped (write-back is only a cache).
     */
    DiskPrefixWorker(std::shared_ptr<DiskPrefixStore> store,
                     std::uint64_t backlog_bytes);

    /** Finishes every queued read and write, then joins. */
    ~DiskPrefixWorker();

    DiskPrefixWorker(const DiskPrefixWorker&) = delete;
    DiskPrefixWorker& operator=(const DiskPrefixWorker&) = delete;

    /**
     * Queues a write of `n_blocks` blocks of `tokens`, `bytes` holding
     * them back to back. @returns false (and drops it) when the
     * backlog cap would be passed.
     */
    bool write(std::vector<tok::TokenId> tokens, std::uint32_t n_blocks,
               std::vector<std::uint8_t> bytes);

    /** @returns true when a write of `bytes` fits the backlog cap now
     *  (so a caller can skip copying the bytes). */
    bool would_accept(std::uint64_t bytes) const;

    /** Queues a read of `n_blocks` blocks of `tokens` into `dst`, one
     *  pointer per block; its Done carries `id`. */
    void read(std::uint64_t id, std::vector<tok::TokenId> tokens,
              std::uint32_t n_blocks, std::vector<std::uint8_t*> dst);

    /** @returns the reads finished since the last call. */
    std::vector<Done> take_done();

    /** @returns true when a finished read waits in take_done(). */
    bool has_done() const;

    /** Blocks until every queued read and write has run (tests). */
    void drain();

  private:
    struct Job {
        bool is_read = false;
        std::uint64_t id = 0;
        std::vector<tok::TokenId> tokens;
        std::uint32_t n_blocks = 0;
        std::vector<std::uint8_t*> dst;
        std::vector<std::uint8_t> bytes;
    };

    void run(std::stop_token stop);

    std::shared_ptr<DiskPrefixStore> store_;
    std::uint64_t backlog_cap_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable idle_cv_;
    std::deque<Job> reads_;
    std::deque<Job> writes_;
    std::uint64_t backlog_ = 0;
    bool busy_ = false;
    std::vector<Done> done_;
    // Last member: started after the rest is built, joined first.
    std::jthread thread_;
};

}  // namespace locus::engine
