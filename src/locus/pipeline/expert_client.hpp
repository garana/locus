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
 *
 * start_batch (i#1186) sends each batched host one kExpertCallBatch of
 * at most `max_batch` pairs; finish_batch reads the replies and writes
 * a host's next chunk only after reading its last reply, so one batched
 * call is in flight per host (a reply can be tens of MB, far over the
 * socket buffers, and the host writes it before it reads again). A host
 * that does not take batched calls (Host::batched false, see
 * probe_batch_calls) gets single calls.
 */
class ExpertClient : public model::RemoteExperts {
  public:
    /** One expert host: the experts it serves and its connection. */
    struct Host {
        std::uint32_t lo = 0;  ///< first expert id it serves
        std::uint32_t hi = 0;  ///< one past its last
        int fd = -1;           ///< connected socket, owned by the caller
        bool batched = true;   ///< it takes kExpertCallBatch
    };

    /**
     * @param hosts Expert hosts with disjoint ranges.
     * @param n_embd The model's hidden width.
     * @param window Most calls in flight per host; 0 sizes it so that
     *     many replies fit in 64 KiB (tests pass a small one).
     * @param max_batch Most pairs in one batched call, at most
     *     kMaxExpertBatch (tests pass a small one); 0 means that cap.
     * @throws std::invalid_argument for an empty or overlapping range,
     *     or a max_batch over kMaxExpertBatch.
     */
    ExpertClient(std::vector<Host> hosts, std::uint32_t n_embd,
                 std::size_t window = 0, std::uint32_t max_batch = 0);

    bool remote(std::uint32_t layer, std::uint32_t e) const override;
    void start(std::uint32_t layer, std::uint32_t e,
               std::span<const float> xb) override;
    void finish(std::span<float> d) override;
    /** @throws std::logic_error while a call or batch is unfinished. */
    void start_batch(std::uint32_t layer,
                     std::span<const std::uint32_t> experts,
                     std::span<const float> xs) override;
    void finish_batch(std::span<float> ds, std::uint32_t n) override;

    /** @returns Calls answered so far, a batched call's pairs each. */
    std::uint64_t calls() const { return calls_; }

    /** @returns Batched call messages answered so far. */
    std::uint64_t batches() const { return batches_; }

    /** @returns Most calls in flight on one host before start reads. */
    std::size_t window() const { return window_; }

  private:
    /** A batch's picks on one batched host. */
    struct BatchPart {
        const Host* host;
        std::vector<std::uint32_t> rows;  ///< call indices, in order
        std::size_t sent = 0;             ///< rows written so far
        std::size_t lo = 0;               ///< first row in flight
        std::uint64_t id = 0;             ///< the call in flight
    };
    /** Writes part's next chunk. @throws std::runtime_error. */
    void send_chunk(BatchPart& part);

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
    std::uint32_t max_batch_;
    std::uint64_t batches_ = 0;
    // The batch start_batch began, until finish_batch.
    bool in_batch_ = false;
    std::uint32_t batch_layer_ = 0, batch_n_ = 0;
    std::span<const float> batch_xs_;
    std::vector<std::uint32_t> batch_experts_;
    std::vector<BatchPart> parts_;
    std::vector<std::uint32_t> single_rows_;  ///< rows sent by start
};

/**
 * Asks a connected expert host whether it takes batched calls
 * (i#1186), by sending an empty kExpertCallBatch. A host that does
 * answers with an empty kExpertOutBatch; an older one closes the
 * connection on the unknown kind, so on false reconnect and use single
 * calls there (Host::batched false).
 *
 * @returns true when the host answered the probe.
 */
bool probe_batch_calls(int fd);

}  // namespace locus::pipeline
