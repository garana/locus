#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

namespace locus::model {

/** The parts of a forward step that StageMetrics times. */
enum class Stage : std::uint8_t {
    kEmbed,      ///< the token's embedding row (no layer)
    kNorm,       ///< a layer's attention or FFN norm; the final norm
    kAttention,  ///< q / k / v, the KV write, attention and the output
    kFfn,        ///< the dense gate / up / down feed-forward
    kMoe,        ///< the router and the routed and shared experts
    kLayer,      ///< a whole layer an arch runs with its own hook
    kPrefetch,   ///< the next layer's weight readahead
    kHead,       ///< the output projection to logits (no layer)
    kSample,     ///< picking the next token from the logits (no layer)
    kDevice,     ///< a whole-model GPU forward, one submit (no layer)
    kCount
};

/** @returns The stage's metric label ("embed", "norm", ...). */
const char* stage_name(Stage s);

/**
 * Wall time per (layer, stage) of a model's forward steps (i#285),
 * always on: the forward charges each part as it ends (StageLap), with
 * one clock read per part. Writes are relaxed atomic adds, so /metrics
 * reads while steps run. Stages that belong to no layer (embed, the
 * final norm, head, sample) go to the no-layer row.
 */
class StageMetrics {
  public:
    /** The row for parts outside the layers. */
    static constexpr std::uint32_t kNoLayer = 0xffffffffu;

    /** @param n_layers The model's layer count. */
    explicit StageMetrics(std::uint32_t n_layers);

    /** Adds `ns` to (layer, s); a layer past the count goes to the
     * no-layer row. */
    void add(std::uint32_t layer, Stage s, std::uint64_t ns);

    /** @returns The nanoseconds charged to (layer, s). */
    std::uint64_t ns(std::uint32_t layer, Stage s) const;
    /** @returns The nanoseconds charged to s over every row. */
    std::uint64_t stage_ns(Stage s) const;
    /** @returns The layer count (rows besides the no-layer one). */
    std::uint32_t layers() const { return n_layers_; }

  private:
    std::size_t index(std::uint32_t layer, Stage s) const;

    std::uint32_t n_layers_;
    std::unique_ptr<std::atomic<std::uint64_t>[]> ns_;
};

/**
 * Charges elapsed wall time to stages: each lap() gives the time since
 * the previous lap (or construction, or restart()) to (layer, stage).
 * With null metrics it reads no clock.
 */
class StageLap {
  public:
    explicit StageLap(StageMetrics* m)
        : m_(m), t_(m != nullptr ? Clock::now() : Clock::time_point{}) {}

    void lap(std::uint32_t layer, Stage s) {
        if (m_ == nullptr) return;
        const Clock::time_point now = Clock::now();
        m_->add(layer, s,
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now - t_)
                        .count()));
        t_ = now;
    }

    /** Starts the next lap now without charging the time so far. */
    void restart() {
        if (m_ != nullptr) t_ = Clock::now();
    }

  private:
    using Clock = std::chrono::steady_clock;
    StageMetrics* m_;
    Clock::time_point t_;
};

}  // namespace locus::model
