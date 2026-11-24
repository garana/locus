#include "locus/pipeline/expert_client.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "locus/pipeline/message.hpp"

namespace locus::pipeline {

ExpertClient::ExpertClient(std::vector<Host> hosts, std::uint32_t n_embd,
                           std::size_t window, std::uint32_t max_batch)
    : hosts_(std::move(hosts)),
      n_embd_(n_embd),
      // Replies of n_embd floats plus a frame header each.
      window_(window > 0 ? window
                         : std::max<std::size_t>(
                               1, (std::size_t{64} << 10) /
                                      (std::size_t{n_embd} * 4 + 64))),
      max_batch_(max_batch > 0 ? max_batch : kMaxExpertBatch) {
    if (max_batch_ > kMaxExpertBatch) {
        throw std::invalid_argument(
            "ExpertClient: max_batch over kMaxExpertBatch");
    }
    for (std::size_t i = 0; i < hosts_.size(); ++i) {
        if (hosts_[i].lo >= hosts_[i].hi || hosts_[i].fd < 0) {
            throw std::invalid_argument(
                "ExpertClient: empty expert range or no connection");
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (hosts_[i].lo < hosts_[j].hi && hosts_[j].lo < hosts_[i].hi) {
                throw std::invalid_argument(
                    "ExpertClient: overlapping expert ranges");
            }
        }
    }
}

const ExpertClient::Host* ExpertClient::host_of(std::uint32_t e) const {
    for (const Host& h : hosts_) {
        if (e >= h.lo && e < h.hi) {
            return &h;
        }
    }
    return nullptr;
}

bool ExpertClient::remote(std::uint32_t, std::uint32_t e) const {
    return host_of(e) != nullptr;
}

void ExpertClient::read_reply(Pending& p) {
    Message out;
    // A reply holds n_embd floats: a longer frame is refused unread.
    if (read_message(p.fd, out, n_embd_) != ReadResult::kOk) {
        throw std::runtime_error("ExpertClient: expert " +
                                 std::to_string(p.expert) +
                                 "'s host closed or timed out");
    }
    if (out.type != MsgType::kExpertOut || out.request_id != p.id ||
        out.position != p.layer ||
        static_cast<std::uint32_t>(out.token) != p.expert ||
        out.data.size() != n_embd_) {
        throw std::runtime_error("ExpertClient: a reply that does not "
                                 "match its call");
    }
    p.d = std::move(out.data);
    p.done = true;
}

void ExpertClient::start(std::uint32_t layer, std::uint32_t e,
                         std::span<const float> xb) {
    const Host* h = host_of(e);
    if (h == nullptr || xb.size() != n_embd_) {
        throw std::runtime_error("ExpertClient: bad expert call");
    }
    // At most window_ calls in flight on this host: read its oldest
    // replies first. Replies on one fd come in call order, and pending_
    // holds the calls in order, so the first unread one on this fd is
    // the one its next reply answers.
    std::size_t in_flight = 0;
    for (const Pending& p : pending_) {
        in_flight += p.fd == h->fd && !p.done ? 1 : 0;
    }
    for (auto it = pending_.begin(); in_flight >= window_; ++it) {
        if (it->fd == h->fd && !it->done) {
            read_reply(*it);
            --in_flight;
        }
    }
    const std::uint64_t id = next_id_++;
    std::vector<float> data(xb.begin(), xb.end());
    if (!write_message(h->fd,
                       make_expert_call(id, layer, e, std::move(data)))) {
        throw std::runtime_error("ExpertClient: write to expert " +
                                 std::to_string(e) + "'s host failed");
    }
    pending_.push_back({h->fd, id, layer, e, false, {}});
}

void ExpertClient::finish(std::span<float> d) {
    if (pending_.empty() || d.size() != n_embd_) {
        throw std::runtime_error("ExpertClient: no call to finish");
    }
    Pending p = std::move(pending_.front());
    pending_.pop_front();
    // Calls finish in start order, so an unread front is its fd's
    // oldest unread call.
    if (!p.done) {
        read_reply(p);
    }
    std::copy(p.d.begin(), p.d.end(), d.begin());
    ++calls_;
}

void ExpertClient::start_batch(std::uint32_t layer,
                               std::span<const std::uint32_t> experts,
                               std::span<const float> xs) {
    if (in_batch_ || !pending_.empty()) {
        throw std::logic_error(
            "ExpertClient: start_batch with calls unfinished");
    }
    const std::size_t E = n_embd_;
    if (xs.size() != experts.size() * E) {
        throw std::runtime_error("ExpertClient: bad batched call");
    }
    parts_.clear();
    single_rows_.clear();
    for (std::uint32_t i = 0; i < experts.size(); ++i) {
        const Host* h = host_of(experts[i]);
        if (h == nullptr) {
            throw std::runtime_error("ExpertClient: bad expert call");
        }
        if (!h->batched) {
            single_rows_.push_back(i);
            continue;
        }
        auto it = std::find_if(parts_.begin(), parts_.end(),
                               [&](const BatchPart& p) { return p.host == h; });
        if (it == parts_.end()) {
            parts_.push_back({h, {}, 0, 0, 0});
            it = parts_.end() - 1;
        }
        it->rows.push_back(i);
    }
    batch_layer_ = layer;
    batch_n_ = static_cast<std::uint32_t>(experts.size());
    batch_xs_ = xs;
    batch_experts_.assign(experts.begin(), experts.end());
    in_batch_ = true;
    // The batched hosts first, so they compute while the single calls
    // are written.
    for (BatchPart& part : parts_) {
        send_chunk(part);
    }
    for (const std::uint32_t r : single_rows_) {
        start(layer, experts[r], xs.subspan(std::size_t{r} * E, E));
    }
}

void ExpertClient::send_chunk(BatchPart& part) {
    const std::size_t E = n_embd_;
    const std::size_t cnt =
        std::min<std::size_t>(max_batch_, part.rows.size() - part.sent);
    std::vector<std::uint32_t> ids(cnt);
    std::vector<float> xs(cnt * E);
    for (std::size_t q = 0; q < cnt; ++q) {
        const std::uint32_t r = part.rows[part.sent + q];
        ids[q] = batch_experts_[r];
        std::copy_n(batch_xs_.data() + std::size_t{r} * E, E,
                    xs.data() + q * E);
    }
    part.id = next_id_++;
    if (!write_message(part.host->fd,
                       make_expert_call_batch(part.id, batch_layer_, ids, xs))) {
        throw std::runtime_error(
            "ExpertClient: a batched call to the host of experts [" +
            std::to_string(part.host->lo) + "," +
            std::to_string(part.host->hi) + ") failed to write");
    }
    part.lo = part.sent;
    part.sent += cnt;
}

void ExpertClient::finish_batch(std::span<float> ds, std::uint32_t n) {
    const std::size_t E = n_embd_;
    if (!in_batch_ || n != batch_n_ || ds.size() != std::size_t{n} * E) {
        throw std::runtime_error("ExpertClient: no batch to finish");
    }
    // One batched call in flight per host: read each host's reply in
    // turn, then write its next chunk.
    for (bool flying = true; flying;) {
        flying = false;
        for (BatchPart& part : parts_) {
            if (part.lo == part.sent) {
                continue;
            }
            const std::size_t cnt = part.sent - part.lo;
            Message out;
            const std::string where = "the host of experts [" +
                                      std::to_string(part.host->lo) + "," +
                                      std::to_string(part.host->hi) + ")";
            if (read_message(part.host->fd, out,
                             static_cast<std::uint32_t>(cnt * E)) !=
                ReadResult::kOk) {
                throw std::runtime_error(
                    "ExpertClient: " + where +
                    " closed or timed out on a batched call (an older "
                    "locus-experts takes none; probe_batch_calls finds "
                    "it)");
            }
            if (out.type != MsgType::kExpertOutBatch ||
                out.request_id != part.id || out.position != batch_layer_ ||
                static_cast<std::size_t>(out.token) != cnt ||
                out.data.size() != cnt * E) {
                throw std::runtime_error("ExpertClient: a batched reply from " +
                                         where + " that does not match its "
                                         "call");
            }
            for (std::size_t q = 0; q < cnt; ++q) {
                std::copy_n(out.data.data() + q * E, E,
                            ds.data() + std::size_t{part.rows[part.lo + q]} * E);
            }
            ++batches_;
            calls_ += cnt;
            part.lo = part.sent;
            if (part.sent < part.rows.size()) {
                send_chunk(part);
                flying = true;
            }
        }
    }
    // The single calls finish in start order, which is row order.
    for (const std::uint32_t r : single_rows_) {
        finish(ds.subspan(std::size_t{r} * E, E));
    }
    in_batch_ = false;
    parts_.clear();
    single_rows_.clear();
    batch_xs_ = {};
}

bool probe_batch_calls(int fd) {
    if (!write_message(fd, make_expert_call_batch(0, 0, {}, {}))) {
        return false;
    }
    Message out;
    return read_message(fd, out, 0) == ReadResult::kOk &&
           out.type == MsgType::kExpertOutBatch && out.token == 0 &&
           out.data.empty();
}

}  // namespace locus::pipeline
