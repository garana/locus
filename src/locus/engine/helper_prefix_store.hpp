#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "locus/auth/helper_connection.hpp"
#include "locus/engine/disk_prefix_store.hpp"

namespace locus::engine {

/**
 * The prefix cache tier kept by an external helper process (i#655),
 * so KV blocks can live on remote or shared storage. The helper runs
 * from `argv` with its stdin / stdout as the auth helper's (the
 * HelperMessage text records); docs/DESIGN.md has the protocol.
 *
 * Each prefix is one blob, byte for byte the DiskPrefixStore file
 * (key, tokens, block checksums), named by the same 64-bit digest.
 * The bytes go through a POSIX shm object per call, not the pipe.
 * Locus keeps the key and checks every blob it reads as it checks a
 * file, so a wrong, stale or torn answer is a miss, never wrong KV;
 * the helper owns placement and eviction.
 *
 * find() answers from a local set of names: the helper's list, read
 * at start and again every `refresh` (so a prefix another locus
 * writes to shared storage is found within that time), plus this
 * store's writes, minus reads that missed.
 *
 * A helper that dies is started again once; when that one dies too,
 * the tier turns off with a warning on stderr and every call misses.
 *
 * Thread-safe: calls to the helper run outside the lock and are
 * pipelined (HelperConnection), so find() does not wait behind them.
 */
class HelperPrefixStore : public DiskPrefixStore {
  public:
    /**
     * Starts the helper and reads its list of stored names.
     *
     * @param argv Helper command and arguments (argv[0] via PATH).
     * @param block_bytes As DiskPrefixStore.
     * @param cap_bytes A blob over this is not written; the helper
     *     keeps its own budget.
     * @param timeout How long one call may take before it misses.
     * @param refresh How often a thread reads the helper's list again
     *     and adds its names (0: never).
     * @throws std::invalid_argument as DiskPrefixStore, or on an empty
     *     argv.
     */
    HelperPrefixStore(std::vector<std::string> argv, DiskKey key,
                      std::size_t block_bytes, std::uint64_t cap_bytes,
                      std::chrono::milliseconds timeout =
                          std::chrono::seconds(120),
                      std::chrono::milliseconds refresh =
                          std::chrono::seconds(60));
    /** Stops the refresh thread, then the helper. */
    ~HelperPrefixStore() override;

    std::uint32_t find(std::span<const tok::TokenId> prompt) const override;
    bool read(std::span<const tok::TokenId> prompt, std::uint32_t n_blocks,
              std::span<std::uint8_t* const> dst) override;
    bool write(std::span<const tok::TokenId> prompt, std::uint32_t n_blocks,
               std::span<const std::uint8_t* const> src) override;

    /** @returns false once the tier turned off (the helper died
     *  twice). */
    bool enabled() const { return !off_.load(); }

    /** @returns the number of prefix names find() knows. */
    std::size_t known() const;

  private:
    /** The refresh thread's list timeout (at most the call timeout). */
    static constexpr std::chrono::milliseconds kListTimeout{10000};

    /** Sends `req`, starting the helper again once when it died.
     *  @param timeout Overrides the call timeout.
     *  @returns the reply, or nullopt (a miss). */
    std::optional<auth::HelperMessage> call(
        auth::HelperMessage req,
        std::optional<std::chrono::milliseconds> timeout = std::nullopt);
    /** Starts the helper and replaces names_ with its list.
     *  @returns false when it does not answer. Called under start_mu_. */
    bool start();

    std::vector<std::string> argv_;
    std::chrono::milliseconds timeout_;

    /** Serializes start() (a restart) and guards starts_. */
    std::mutex start_mu_;
    int starts_ = 0;
    std::atomic<bool> off_{false};

    /** Guards conn_ and names_. */
    mutable std::mutex mu_;
    std::shared_ptr<auth::HelperConnection> conn_;
    std::unordered_set<std::uint64_t> names_;

    /** Names this process's shm objects. */
    std::atomic<std::uint64_t> shm_seq_{0};

    /** The refresh thread and its stop signal. */
    std::mutex stop_mu_;
    std::condition_variable stop_cv_;
    bool stop_ = false;
    std::thread refresher_;
};

}  // namespace locus::engine
