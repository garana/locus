#pragma once

#include <sys/types.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#include "locus/auth/helper_message.hpp"

namespace locus::auth {

/**
 * A framed, pipelined, req_id-correlated message channel to a helper
 * process over a read fd (the helper's stdout) and a write fd (the
 * helper's stdin). A background reader thread decodes responses and
 * dispatches each to the waiting caller by req_id, so many request
 * threads can share one connection and responses may arrive OUT OF
 * ORDER (a later request answered first). Thread-safe.
 *
 * This owns only the transport + correlation; process spawning lives
 * in a thin wrapper (so the core is testable over a socketpair).
 */
class HelperConnection {
  public:
    /** Takes ownership of the fds (closed on destruction). read_fd
     * and write_fd may be the same fd (a bidirectional socket). */
    HelperConnection(int read_fd, int write_fd);
    /** As above, and reaps `child_pid` on destruction (used by the
     * spawn wrapper; the fds are the child's stdout/stdin pipes). */
    HelperConnection(int read_fd, int write_fd, pid_t child_pid);
    ~HelperConnection();

    HelperConnection(const HelperConnection&) = delete;
    HelperConnection& operator=(const HelperConnection&) = delete;

    /**
     * Sends `req` (a fresh req_id is assigned, overwriting any set)
     * and waits up to `timeout` for the correlated response.
     *
     * @returns The response, or nullopt on timeout or a broken
     *     connection. Safe to call from many threads concurrently.
     */
    std::optional<HelperMessage> request(
        HelperMessage req, std::chrono::milliseconds timeout);

    /**
     * Fire-and-forget notification (req_id forced to 0, no reply
     * awaited) -- used for locus's request-lifecycle event pushes.
     *
     * @returns false if the write failed (broken connection).
     */
    bool notify(HelperMessage msg);

    /** @returns true once the connection has failed (EOF/error). */
    bool broken() const { return broken_.load(); }

  private:
    struct Slot {
        bool done = false;
        HelperMessage resp;
    };

    void reader_loop();
    bool write_all(const std::string& bytes);
    void fail_all();

    int read_fd_;
    int write_fd_;
    pid_t child_pid_ = -1;  // >0 when this owns a spawned helper
    std::atomic<std::uint64_t> next_id_{1};
    std::mutex write_mu_;  // serializes writes to write_fd_
    std::mutex mu_;        // guards pending_
    std::condition_variable cv_;
    std::unordered_map<std::uint64_t, std::shared_ptr<Slot>> pending_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> broken_{false};
    std::thread reader_;
};

}  // namespace locus::auth
