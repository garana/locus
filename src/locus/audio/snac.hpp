#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/model/codec.hpp"

namespace locus::audio {

/**
 * SNAC neural-codec decoder (the "snac" GGUF written by
 * .local-dev/golden/convert_snac.py). Turns the three residual-vector-
 * quantizer code streams that Orpheus produces (see
 * locus::model::SnacCodes) into a mono waveform, using only the
 * conv1d / conv1d_transpose / snake1d CPU kernels.
 *
 * The graph is SNAC's decoder with the stochastic NoiseBlock removed
 * (the golden disables it, so the decode is deterministic): per-code
 * codebook lookup and a 1x1 out-projection per quantizer, summed after
 * a stride upsample (from_codes); a depthwise in-conv then a 1x1 conv
 * to the decoder width; a DecoderBlock per upsample rate (snake, a
 * transposed conv that upsamples by the rate, then three dilated
 * residual units); and a final snake, 1x1-tap conv to one channel and
 * tanh. All sizes come from the GGUF metadata; the structure is fixed.
 */
class SnacDecoder {
  public:
    /**
     * Loads a "snac" GGUF (copies the tensors it needs, so the
     * GgufFile need not outlive the decoder).
     *
     * @param g A parsed GGUF whose general.architecture is "snac".
     * @throws gguf::Error on a wrong architecture, missing metadata, a
     *     config locus cannot run (codebooks other than 3 with strides
     *     [4,2,1], odd or too many rates, a recorded layout that is not
     *     depthwise + noise without attention).
     */
    static SnacDecoder load(const gguf::GgufFile& g);

    /**
     * Decodes the three code streams to a mono f32 waveform.
     *
     * @param codes The coarse/mid/fine streams; sizes must be F, 2F
     *     and 4F for F frames (as orpheus_deband produces).
     * @returns `prod(rates) * 4 * F` samples in [-1, 1].
     * @throws std::invalid_argument on inconsistent stream lengths or a
     *     code outside [0, codebook_size).
     * @throws gguf::Error on a missing or wrong-sized tensor.
     */
    std::vector<float> decode(const model::SnacCodes& codes) const;

    /**
     * Decodes frames [f0, f1) only, from a window of `codes` that adds
     * context_frames() frames on each side (clipped at the ends). The
     * result is bit-identical to the same slice of decode(codes): every
     * output sample sees the same inputs in the same order. Frames past
     * f1 + context_frames() are never read, so a stream can emit a
     * range before the codes after it exist.
     *
     * @param codes The streams, as for decode().
     * @param f0 First frame to return.
     * @param f1 One past the last frame (f0 <= f1 <= frame count).
     * @returns (f1 - f0) * samples_per_frame() samples.
     * @throws std::invalid_argument on a bad range, or as decode().
     */
    std::vector<float> decode_range(const model::SnacCodes& codes,
                                    std::uint32_t f0,
                                    std::uint32_t f1) const;

    /**
     * @returns Frames of codes on each side of a sample that can change
     *     it: a bound on the decoder's receptive field, from its rates,
     *     kernels and dilations (5 for the 24 kHz model, 425 ms).
     */
    std::uint32_t context_frames() const;

    /** @returns Output samples per code frame (2048 at 24 kHz). */
    std::uint32_t samples_per_frame() const;

    /** @returns The codec sample rate (Hz). */
    std::uint32_t sample_rate() const { return sample_rate_; }

  private:
    /**
     * @returns The tensor `name`, which must hold exactly `n` floats.
     * @throws gguf::Error if it is missing or another size.
     */
    std::span<const float> t(const std::string& name, std::size_t n) const;

    std::unordered_map<std::string, std::vector<float>> w_;
    std::vector<std::uint32_t> rates_;    ///< decoder upsample rates.
    std::vector<std::uint32_t> strides_;  ///< per-codebook vq stride.
    std::uint32_t latent_dim_ = 0;
    std::uint32_t decoder_dim_ = 0;
    std::uint32_t codebook_dim_ = 0;
    std::uint32_t codebook_size_ = 0;
    std::uint32_t n_codebooks_ = 0;
    std::uint32_t sample_rate_ = 0;
};

}  // namespace locus::audio
