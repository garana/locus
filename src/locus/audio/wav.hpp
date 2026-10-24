#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace locus::audio {

/** Output encodings for a decoded waveform. */
enum class WaveFormat {
    kWav,    ///< 16-bit PCM in a RIFF/WAVE container.
    kPcm16,  ///< raw little-endian signed 16-bit PCM, no header.
};

/**
 * Encodes mono f32 samples (nominally in [-1, 1]) to 16-bit PCM.
 * Samples are clamped to [-1, 1] and scaled by 32767; a WAV result
 * carries a 44-byte canonical PCM header, a kPcm16 result is the raw
 * samples only.
 *
 * @param samples Mono f32 samples.
 * @param sample_rate Samples per second (written into the WAV header;
 *     ignored for raw PCM).
 * @param format WAV container or raw PCM.
 * @returns The encoded bytes.
 */
std::vector<std::byte> encode_pcm(std::span<const float> samples,
                                  std::uint32_t sample_rate,
                                  WaveFormat format);

}  // namespace locus::audio
