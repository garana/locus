#include "locus/pipeline/expert_client.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "locus/pipeline/message.hpp"

namespace locus::pipeline {

ExpertClient::ExpertClient(std::vector<Host> hosts, std::uint32_t n_embd,
                           std::size_t window)
    : hosts_(std::move(hosts)),
      n_embd_(n_embd),
      // Replies of n_embd floats plus a frame header each.
      window_(window > 0 ? window
                         : std::max<std::size_t>(
                               1, (std::size_t{64} << 10) /
                                      (std::size_t{n_embd} * 4 + 64))) {
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

}  // namespace locus::pipeline
