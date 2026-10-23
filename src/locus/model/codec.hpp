#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "locus/tok/tokenizer.hpp"

namespace locus::model {

/**
 * The three SNAC residual-vector-quantizer code streams that an
 * Orpheus audio-token run decodes into. SNAC's 24 kHz model has
 * vq_strides [4, 2, 1], so per 7-token frame it consumes one coarse
 * code, two mid codes and four fine codes. The streams therefore grow
 * at 1x, 2x and 4x the frame rate.
 */
struct SnacCodes {
    std::vector<int> coarse;  ///< 1 code per frame (stride 4).
    std::vector<int> mid;     ///< 2 codes per frame (stride 2).
    std::vector<int> fine;    ///< 4 codes per frame (stride 1).
};

/** Audio-codec layout constants for Orpheus-TTS over SNAC 24 kHz. */
struct OrpheusCodec {
    /// Flat audio tokens that make up one SNAC frame.
    static constexpr int kCodesPerFrame = 7;
    /// Width of each per-position code band.
    static constexpr int kCodebook = 4096;
};

/**
 * De-bands a run of flat Orpheus audio-token ids into the three SNAC
 * code streams.
 *
 * Orpheus packs seven codes per frame into one contiguous band of
 * `7 * 4096` ids starting at `band_base` (the id of
 * `<custom_token_10>`). Position `i` within a frame (0..6) lives in its
 * own 4096-wide sub-band, so its code is
 * `(token - band_base) - i * 4096`. The seven positions map to SNAC
 * layers by stride: position 0 is the coarse code, positions 1 and 4
 * the mid pair, positions 2, 3, 5 and 6 the fine four.
 *
 * @param audio_tokens Flat audio-band token ids; length must be a
 *     multiple of 7 (whole frames only).
 * @param band_base The id of `<custom_token_10>`, where the band
 *     starts.
 * @returns The coarse/mid/fine streams (sizes frames, 2*frames,
 *     4*frames).
 * @throws std::invalid_argument if the length is not a multiple of 7,
 *     or any decoded code falls outside [0, 4096).
 */
SnacCodes orpheus_deband(std::span<const tok::TokenId> audio_tokens,
                         tok::TokenId band_base);

/**
 * Resolves the Orpheus audio band base: the vocab id of the piece
 * `<custom_token_10>`. The reference decoder keys the band off this
 * token's name, so resolving by name (not a hard-coded id) keeps a
 * different checkpoint from silently shifting the band.
 *
 * @param tok A tokenizer for an Orpheus checkpoint.
 * @returns The id of `<custom_token_10>`.
 * @throws std::runtime_error if that piece is not in the vocabulary.
 */
tok::TokenId orpheus_band_base(const tok::Tokenizer& tok);

/**
 * Streaming de-band driver (the "multi-codebook sampler"): fed the
 * greedily-decoded token ids one at a time, it classifies each as an
 * audio-band code or a stream terminator, buffers the audio tokens and
 * reports when a terminator (any non-audio token, the end of speech)
 * arrives. The caller runs the transformer and the argmax; this owns
 * the band bookkeeping only, so it is pure and testable without a
 * model.
 *
 * Note on the terminator: greedy (temperature 0) decoding does not
 * terminate on this TTS head, it locks onto a repeating coarse code and
 * never emits a non-audio token, so no deterministic golden can carry
 * an end-of-speech token. The stop path is therefore pinned by a
 * synthetic unit test (a known frame then a non-audio id), not by a
 * golden; a recorded `terminated=false` is that property, not a bug in
 * the generator.
 */
class OrpheusCodecSampler {
  public:
    /** @param band_base The id of `<custom_token_10>`. */
    explicit OrpheusCodecSampler(tok::TokenId band_base)
        : band_base_(band_base) {}

    /** Outcome of feeding one token to {@link push}. */
    enum class Status {
        kAudio,  ///< An audio-band code; buffered, keep decoding.
        kStop,   ///< A non-audio token; end of speech, stop decoding.
    };

    /**
     * Classifies and, when audio, buffers one decoded token.
     *
     * @param t The token the model just produced.
     * @returns kAudio if `t` was in the audio band (and buffered),
     *     kStop otherwise (the token is recorded as the stop token and
     *     not buffered).
     */
    Status push(tok::TokenId t);

    /** @returns Count of complete 7-token frames buffered. */
    std::size_t frames() const {
        return audio_.size() / OrpheusCodec::kCodesPerFrame;
    }

    /** @returns true if a partial (non-multiple-of-7) frame is held. */
    bool has_partial() const {
        return audio_.size() % OrpheusCodec::kCodesPerFrame != 0;
    }

    /**
     * De-bands the complete frames buffered so far. A trailing partial
     * frame is ignored.
     *
     * @returns The three SNAC streams for the complete frames.
     * @throws std::invalid_argument if a buffered code is out of range
     *     (should not happen: {@link push} only buffers in-band ids).
     */
    SnacCodes codes() const;

    /** @returns The stop token seen, or -1 if none yet. */
    tok::TokenId stop_token() const { return stop_; }

  private:
    tok::TokenId band_base_;
    std::vector<tok::TokenId> audio_;
    tok::TokenId stop_ = -1;
};

}  // namespace locus::model
