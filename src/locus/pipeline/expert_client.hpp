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
 * a kExpertCall; finish hands back the oldest call's kExpertOut after
 * checking its tags. A failed write, read or check throws, and the
 * connections are then out of step: drop the client and its fds.
 *
 * A host answers its calls in order and writes each reply before it
 * reads the next call, so a client that wrote many calls before reading
 * any reply could fill the socket buffers both ways and block both
 * ends. start therefore reads a host's oldest replies early (keeping
 * them for finish) whenever it already has `window` calls in flight
 * there; `window` replies fit in 64 KiB, well inside the default socket
 * buffers.
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
     * @param window Most calls in flight per host; 0 sizes it so that
     *     many replies fit in 64 KiB (tests pass a small one).
     * @throws std::invalid_argument for an empty or overlapping range.
     */
    ExpertClient(std::vector<Host> hosts, std::uint32_t n_embd,
                 std::size_t window = 0);

    bool remote(std::uint32_t layer, std::uint32_t e) const override;
    void start(std::uint32_t layer, std::uint32_t e,
               std::span<const float> xb) override;
    void finish(std::span<float> d) override;

    /** @returns Calls answered so far. */
    std::uint64_t calls() const { return calls_; }

    /** @returns Most calls in flight on one host before start reads. */
    std::size_t window() const { return window_; }

  private:
    struct Pending {
        int fd;
        std::uint64_t id;
        std::uint32_t layer, expert;
        bool done = false;     ///< its reply has been read into d
        std::vector<float> d;  ///< the reply, once done
    };

    /** The host serving expert e, or null. */
    const Host* host_of(std::uint32_t e) const;
    /** Reads p's reply from p.fd into p.d (it must be that fd's oldest
     *  unread one). @throws std::runtime_error on a bad reply. */
    void read_reply(Pending& p);

    std::vector<Host> hosts_;
    std::uint32_t n_embd_;
    std::size_t window_;
    std::deque<Pending> pending_;
    std::uint64_t next_id_ = 1;
    std::uint64_t calls_ = 0;
};

}  // namespace locus::pipeline
