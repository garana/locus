#include "locus/pipeline/expert_client.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "locus/pipeline/message.hpp"

namespace locus::pipeline {

ExpertClient::ExpertClient(std::vector<Host> hosts, std::uint32_t n_embd)
    : hosts_(std::move(hosts)), n_embd_(n_embd) {
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

void ExpertClient::start(std::uint32_t layer, std::uint32_t e,
                         std::span<const float> xb) {
    const Host* h = host_of(e);
    if (h == nullptr || xb.size() != n_embd_) {
        throw std::runtime_error("ExpertClient: bad expert call");
    }
    const std::uint64_t id = next_id_++;
    std::vector<float> data(xb.begin(), xb.end());
    if (!write_message(h->fd,
                       make_expert_call(id, layer, e, std::move(data)))) {
        throw std::runtime_error("ExpertClient: write to expert " +
                                 std::to_string(e) + "'s host failed");
    }
    pending_.push_back({h->fd, id, layer, e});
}

void ExpertClient::finish(std::span<float> d) {
    if (pending_.empty() || d.size() != n_embd_) {
        throw std::runtime_error("ExpertClient: no call to finish");
    }
    const Pending p = pending_.front();
    pending_.pop_front();
    // Replies on one connection come in call order, and the calls are
    // finished in start order, so the next reply on p.fd is p's.
    Message out;
    if (read_message(p.fd, out) != ReadResult::kOk) {
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
    std::copy(out.data.begin(), out.data.end(), d.begin());
    ++calls_;
}

}  // namespace locus::pipeline
