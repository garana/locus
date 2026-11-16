#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "locus/http/http.hpp"
#include "locus/http/server.hpp"
#include "locus/server/engine_group.hpp"

namespace locus::server {

/**
 * An SSE stream's keepalive (i#619): a stream that has written nothing
 * for `every` gets an SSE comment, which clients ignore, so a long
 * prefill or a buffered tool reply does not look like a dead
 * connection.
 */
struct SseKeepalive {
    std::chrono::milliseconds every;
    std::chrono::steady_clock::time_point last =
        std::chrono::steady_clock::now();

    /** @returns Whether a keepalive is due at `now`. */
    bool due(std::chrono::steady_clock::time_point now) const {
        return every.count() > 0 && now - last >= every;
    }
    /** Fills an empty payload with ": keepalive" when one is due, and
     *  notes the time of any payload about to be written. */
    void pad(std::string& payload) {
        const auto now = std::chrono::steady_clock::now();
        if (payload.empty() && due(now)) {
            payload = ": keepalive\n\n";
        }
        if (!payload.empty()) {
            last = now;
        }
    }
};

/**
 * A generation's answer as steps over its snapshot (i#932), so the
 * thread path (serve_generation) and the event-loop path (GenPump)
 * write the same bytes from the same code.
 */
struct GenReply {
    /** Streamed (step) or buffered (finish). */
    bool stream = false;
    /**
     * Appends the stream's bytes for snapshot v to payload; the tokens
     * past the ones it has seen are new. The driver calls it when v has
     * new tokens or a terminal state, or a keepalive is due.
     * @returns Whether this is the stream's last step.
     */
    std::function<bool(const EngineLoop::View& v, std::string& payload)>
        step;
    /** Fills res from the terminal snapshot v (the request is released
     *  already). */
    std::function<void(const EngineLoop::View& v, http::ServerResponse& res)>
        finish;
};

/**
 * Answers generation `id` on the calling thread: a buffered reply waits
 * for the end (cancelling when ctx's client leaves) and calls finish; a
 * stream sets res's provider, which waits on each step and writes it.
 */
void serve_generation(EngineGroup& loop, std::uint64_t id, GenReply gr,
                      http::ServerResponse& res,
                      const http::RequestContext& ctx,
                      std::chrono::milliseconds keepalive);

/**
 * Drives generations for the event-loop server (i#932) on one thread:
 * the engines' progress hook wakes it, and it takes each followed
 * request's new tokens (EngineLoop::take, which never waits on a step),
 * runs its GenReply and queues the bytes on the request's LoopReply for
 * the I/O thread to write. N streams take no thread each.
 *
 * - A stream whose reply is over its queue bound is skipped until the
 *   client reads (the connection's write deadline closes a stalled
 *   one); its tokens wait in the engine's record.
 * - A cancelled reply (the client left) cancels the generation; a
 *   buffered one also cancels when the client half-closes (EOF while
 *   pending), as the thread path's liveness probe does.
 * - It looks only at the entries with news (i#1005): the requests the
 *   progress hook names, the replies that call back (LoopReply's
 *   set_notify: queue taken, cancelled, half-closed) and the streams
 *   whose keepalive is due (a min-heap); no tick.
 */
class GenPump {
  public:
    GenPump(EngineGroup& loop, std::chrono::milliseconds keepalive);
    ~GenPump();
    GenPump(const GenPump&) = delete;
    GenPump& operator=(const GenPump&) = delete;

    /**
     * Takes over generation `id`, submitted and not yet waited on: a
     * stream begins at once with `head`; a buffered reply starts from
     * head's status and headers.
     */
    void add(std::uint64_t id, GenReply gr, http::Deferred d,
             http::Response head);

    /** Cancels and releases every generation left, and joins the thread
     *  (idempotent). */
    void stop();

    /** @returns How many times the pump has looked at an entry (tests:
     *  the work per engine step must not grow with idle entries). */
    std::uint64_t visits() const { return visits_.load(); }

  private:
    using Clock = std::chrono::steady_clock;
    struct Entry {
        std::uint64_t id = 0;
        GenReply gr;
        http::Deferred d;
        http::Response head;  ///< a buffered reply's status and headers
        int status = 200;     ///< the status it records
        EngineLoop::View acc;
        std::size_t seen = 0;
        SseKeepalive ka;
        std::uint64_t bytes = 0;
        bool cancelled = false;
        /** The time of its live keepalives_ entry (max: none). */
        Clock::time_point ka_at = Clock::time_point::max();
    };
    using Due = std::pair<Clock::time_point, std::uint64_t>;

    void run();
    /** One pass over e. @returns Whether e is finished. */
    bool pump(Entry& e);
    /** Ends e: cancels and releases its request, and records it. */
    void drop(Entry& e, int status);
    /** Marks entry id for a look and wakes the thread (any thread). */
    void mark(std::uint64_t id);
    /** Queues e's next keepalive when it is sooner than its live entry
     *  (pump thread). */
    void arm(Entry& e);

    EngineGroup& loop_;
    std::chrono::milliseconds keepalive_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::vector<std::unique_ptr<Entry>> incoming_;
    /** Entry ids with news: from the hook and the replies (under mu_). */
    std::vector<std::uint64_t> ready_;
    /** Due keepalives (pump thread only); stale entries are skipped. */
    std::priority_queue<Due, std::vector<Due>, std::greater<Due>> keepalives_;
    std::atomic<std::uint64_t> visits_{0};
    std::thread thread_;
};

}  // namespace locus::server
