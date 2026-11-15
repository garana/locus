#include "locus/server/gen_reply.hpp"

#include <algorithm>
#include <exception>
#include <string_view>
#include <utility>

#include "locus/http/loop_connection.hpp"

namespace locus::server {

namespace {

using Clock = std::chrono::steady_clock;

// How often the pump looks for gone clients, drained queues and due
// keepalives between progress wakes (as EngineLoop's alive probe).
constexpr auto kTick = std::chrono::milliseconds(20);

/**
 * Writes one step of a generation's SSE stream: payload, or a keepalive
 * when one is due. When the client has gone, cancels the generation and
 * drops its record (both no-ops if the terminal path already released
 * it).
 *
 * @param terminal Whether this is the stream's last step; the sink is
 *     then closed.
 * @returns false when the stream is over: the client went away, or
 *     terminal.
 */
bool write_sse_step(http::Sink& sink, SseKeepalive& ka, std::string& payload,
                    EngineGroup& loop, std::uint64_t id, bool terminal) {
    ka.pad(payload);
    if (!payload.empty() &&
        !sink.write(std::string_view(payload.data(), payload.size()))) {
        loop.cancel(id);
        loop.release(id);
        return false;
    }
    if (terminal) {
        sink.done();
        return false;
    }
    return true;
}

}  // namespace

void serve_generation(EngineGroup& loop, std::uint64_t id, GenReply gr,
                      http::ServerResponse& res,
                      const http::RequestContext& ctx,
                      std::chrono::milliseconds keepalive) {
    if (!gr.stream) {
        // Stop decoding if the client disconnects before we finish,
        // instead of running the abandoned request to max_tokens.
        const EngineLoop::View v = loop.wait_done_or_cancel(id, ctx.client_alive);
        // v is a full snapshot copy; drop the engine record now
        // (evict-on-consume).
        loop.release(id);
        gr.finish(v, res);
        return;
    }
    res.set_chunked_content_provider(
        "text/event-stream",
        [&loop, id, step = std::move(gr.step), keepalive](http::Sink& sink) {
            // The server calls a provider ONCE; loop until terminal, one
            // write per progress step.
            SseKeepalive ka{keepalive};
            std::size_t seen = 0;
            for (;;) {
                const EngineLoop::View v = loop.wait_progress(id, seen, ka.every);
                seen = v.generated.size();
                std::string payload;
                const bool terminal = step(v, payload);
                if (terminal) {
                    // Built from the snapshot v; drop the engine record
                    // (the provider won't run again).
                    loop.release(id);
                }
                if (!write_sse_step(sink, ka, payload, loop, id, terminal)) {
                    return;
                }
            }
        });
}

GenPump::GenPump(EngineGroup& loop, std::chrono::milliseconds keepalive)
    : loop_(loop), keepalive_(keepalive) {
    loop_.set_progress_hook([this] {
        // On an engine worker, under its mutex: only flag and notify.
        {
            std::lock_guard<std::mutex> lk(mu_);
            woke_ = true;
        }
        cv_.notify_one();
    });
    thread_ = std::thread([this] { run(); });
}

GenPump::~GenPump() { stop(); }

void GenPump::stop() {
    // First, so no engine worker calls into this object after it is gone.
    loop_.set_progress_hook({});
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void GenPump::add(std::uint64_t id, GenReply gr, http::Deferred d,
                  http::Response head) {
    auto e = std::make_unique<Entry>();
    e->id = id;
    e->gr = std::move(gr);
    e->d = std::move(d);
    e->status = head.status;
    e->ka.every = keepalive_;
    loop_.follow(id);
    if (e->gr.stream) {
        // A cancelled reply refuses it; the first pass then drops e.
        e->d.reply->begin_stream(std::move(head));
    } else {
        e->head = std::move(head);
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!stop_) {
            incoming_.push_back(std::move(e));
            woke_ = true;
        }
    }
    if (e) {
        drop(*e, e->status);  // stopping: the client gets no answer
        return;
    }
    cv_.notify_one();
}

void GenPump::run() {
    std::vector<std::unique_ptr<Entry>> live;
    std::unique_lock<std::mutex> lk(mu_);
    const auto ready = [this] {
        return stop_ || woke_ || !incoming_.empty();
    };
    for (;;) {
        if (live.empty()) {
            cv_.wait(lk, ready);
        } else {
            cv_.wait_for(lk, kTick, ready);
        }
        if (stop_) {
            break;
        }
        // One wake drains every request: a hook that runs after this
        // reset wakes the next wait.
        woke_ = false;
        for (auto& e : incoming_) {
            live.push_back(std::move(e));
        }
        incoming_.clear();
        lk.unlock();
        std::erase_if(live, [this](const std::unique_ptr<Entry>& e) {
            try {
                return pump(*e);
            } catch (...) {
                // A step that throws: a 500, or a truncated stream, as a
                // provider that throws on the job pool.
                e->d.reply->fail();
                drop(*e, e->gr.stream ? e->status : 500);
                return true;
            }
        });
        lk.lock();
    }
    for (auto& e : incoming_) {
        live.push_back(std::move(e));
    }
    incoming_.clear();
    lk.unlock();
    for (auto& e : live) {
        drop(*e, e->status);
    }
}

void GenPump::drop(Entry& e, int status) {
    loop_.unfollow(e.id);
    loop_.cancel(e.id);
    loop_.release(e.id);
    e.d.done(status, e.bytes);
}

bool GenPump::pump(Entry& e) {
    http::LoopReply& r = *e.d.reply;
    if (!e.cancelled &&
        (r.cancelled() || (!e.gr.stream && r.client_gone()))) {
        // The client left: stop decoding. cancel() queues kCancelled for
        // the take below.
        loop_.cancel(e.id);
        e.cancelled = true;
    }
    if (!e.gr.stream) {
        loop_.take(e.id, e.acc);
        if (!engine::is_terminal(e.acc.status)) {
            return false;
        }
        loop_.release(e.id);
        http::ServerResponse res;
        res.status = e.head.status;
        res.headers = std::move(e.head.headers);
        e.gr.finish(e.acc, res);
        http::Response out;
        out.status = res.status;
        out.headers = std::move(res.headers);
        out.body = std::move(res.body);
        const int status = out.status;
        e.bytes = out.body.size();
        r.send(std::move(out));
        e.d.done(status, e.bytes);
        return true;
    }
    if (r.cancelled()) {
        drop(e, e.status);
        return true;
    }
    if (!r.has_room()) {
        return false;  // the client reads slowly: its tokens wait
    }
    loop_.take(e.id, e.acc);
    const bool fresh = e.acc.generated.size() > e.seen;
    const bool end = engine::is_terminal(e.acc.status);
    if (!fresh && !end && !e.ka.due(Clock::now())) {
        return false;
    }
    e.seen = e.acc.generated.size();
    std::string payload;
    // A request that ended without a last step (cancelled under the
    // stream) ends the stream too, rather than leave it open.
    const bool last = e.gr.step(e.acc, payload) || end;
    if (last) {
        loop_.release(e.id);
    }
    e.ka.pad(payload);
    if (!payload.empty()) {
        if (!r.write(payload)) {
            drop(e, e.status);
            return true;
        }
        e.bytes += payload.size();
    }
    if (last) {
        r.end();
        loop_.unfollow(e.id);
        e.d.done(e.status, e.bytes);
        return true;
    }
    return false;
}

}  // namespace locus::server
