#pragma once

#include <cstdint>
#include <deque>
#include <span>
#include <vector>

#include "locus/model/transformer.hpp"

namespace locus::pipeline {

/**
 * model::RemoteExperts over expert-server connections (expert
 * parallelism, #781): each host serves the experts in [lo, hi) of every
 * MoE layer on its fd (locus-experts, serve_expert_calls). start writes
 * a kExpertCall; finish reads the oldest call's kExpertOut and checks
 * its tags. A failed write, read or check throws, and the connections
 * are then out of step: drop the client and its fds.
 */
class ExpertClient : public model::RemoteExperts {
  public:
    /** One expert host: the experts it serves and its connection. */
    struct Host {
        std::uint32_t lo = 0;  ///< first expert id it serves
        std::uint32_t hi = 0;  ///< one past its last
        int fd = -1;           ///< connected socket, owned by the caller
    };

    /**
     * @param hosts Expert hosts with disjoint ranges.
     * @param n_embd The model's hidden width.
     * @throws std::invalid_argument for an empty or overlapping range.
     */
    ExpertClient(std::vector<Host> hosts, std::uint32_t n_embd);

    bool remote(std::uint32_t layer, std::uint32_t e) const override;
    void start(std::uint32_t layer, std::uint32_t e,
               std::span<const float> xb) override;
    void finish(std::span<float> d) override;

    /** @returns Calls answered so far. */
    std::uint64_t calls() const { return calls_; }

  private:
    /** The host serving expert e, or null. */
    const Host* host_of(std::uint32_t e) const;

    struct Pending {
        int fd;
        std::uint64_t id;
        std::uint32_t layer, expert;
    };
    std::vector<Host> hosts_;
    std::uint32_t n_embd_;
    std::deque<Pending> pending_;
    std::uint64_t next_id_ = 1;
    std::uint64_t calls_ = 0;
};

}  // namespace locus::pipeline
