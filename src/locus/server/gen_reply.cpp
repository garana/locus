#include "locus/server/gen_reply.hpp"

#include <algorithm>
#include <exception>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "locus/http/loop_connection.hpp"

namespace locus::server {

namespace {

using Clock = std::chrono::steady_clock;

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
    loop_.set_progress_hook([this](std::span<const std::uint64_t> ids) {
        // On an engine worker, under its mutex: only note and notify.
        {
            std::lock_guard<std::mutex> lk(mu_);
            ready_.insert(ready_.end(), ids.begin(), ids.end());
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
    // The reply calls back when its queue is taken, it is cancelled or
    // the client half-closes: only then does the entry need a look.
    e->d.reply->set_notify([this, id] { mark(id); });
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
        }
    }
    if (e) {
        e->d.reply->set_notify({});
        drop(*e, e->status);  // stopping: the client gets no answer
        return;
    }
    cv_.notify_one();
}

void GenPump::mark(std::uint64_t id) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        ready_.push_back(id);
    }
    cv_.notify_one();
}

void GenPump::arm(Entry& e) {
    if (!e.gr.stream || e.ka.every.count() <= 0) {
        return;
    }
    const Clock::time_point due = e.ka.last + e.ka.every;
    if (due < e.ka_at) {
        keepalives_.push({due, e.id});
        e.ka_at = due;
    }
}

void GenPump::run() {
    std::unordered_map<std::uint64_t, std::unique_ptr<Entry>> live;
    std::vector<std::uint64_t> ids;
    std::unique_lock<std::mutex> lk(mu_);
    const auto ready = [this] {
        return stop_ || !ready_.empty() || !incoming_.empty();
    };
    for (;;) {
        if (keepalives_.empty()) {
            cv_.wait(lk, ready);
        } else {
            cv_.wait_until(lk, keepalives_.top().first, ready);
        }
        if (stop_) {
            break;
        }
        // The requests that changed, the replies that called back and
        // the new entries: no other entry is looked at.
        ids.swap(ready_);
        ready_.clear();
        for (auto& e : incoming_) {
            ids.push_back(e->id);
            const std::uint64_t id = e->id;
            live.emplace(id, std::move(e));
        }
        incoming_.clear();
        lk.unlock();
        const auto now = Clock::now();
        while (!keepalives_.empty() && keepalives_.top().first <= now) {
            const auto [t, id] = keepalives_.top();
            keepalives_.pop();
            const auto it = live.find(id);
            if (it != live.end() && it->second->ka_at == t) {
                it->second->ka_at = Clock::time_point::max();
                ids.push_back(id);
            }
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        for (const std::uint64_t id : ids) {
            const auto it = live.find(id);
            if (it == live.end()) {
                continue;  // not ours, or done already
            }
            Entry& e = *it->second;
            visits_.fetch_add(1, std::memory_order_relaxed);
            // A step that throws: a 500, or a truncated stream (logged and
            // counted, i#980), as a provider that throws on the job pool.
            const auto failed = [&](const char* why) {
                if (e.gr.stream && e.d.stream_failed) {
                    e.d.stream_failed(why);
                }
                e.d.reply->fail();
                drop(e, e.gr.stream ? e.status : 500);
                return true;
            };
            bool done;
            try {
                done = pump(e);
            } catch (const std::exception& ex) {
                done = failed(ex.what());
            } catch (...) {
                done = failed("unknown exception");
            }
            if (done) {
                e.d.reply->set_notify({});
                live.erase(it);
            } else {
                arm(e);
            }
        }
        ids.clear();
        lk.lock();
    }
    for (auto& e : incoming_) {
        const std::uint64_t id = e->id;
        live.emplace(id, std::move(e));
    }
    incoming_.clear();
    lk.unlock();
    for (auto& [id, e] : live) {
        e->d.reply->set_notify({});
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
