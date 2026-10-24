#include "locus/audio/wav.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace locus::audio {

namespace {

/** Appends `n` little-endian bytes of `v` to `out`. */
void put_le(std::vector<std::byte>& out, std::uint64_t v, int n) {
    for (int i = 0; i < n; ++i) {
        out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
    }
}

/** One f32 sample to a clamped, scaled little-endian s16. */
std::int16_t to_s16(float x) {
    const float c = std::clamp(x, -1.0f, 1.0f);
    return static_cast<std::int16_t>(std::lround(c * 32767.0f));
}

}  // namespace

std::vector<std::byte> encode_pcm(std::span<const float> samples,
                                  std::uint32_t sample_rate,
                                  WaveFormat format) {
    const std::uint32_t n = static_cast<std::uint32_t>(samples.size());
    const std::uint32_t data_bytes = n * 2;  // mono, 16-bit.

    std::vector<std::byte> out;
    if (format == WaveFormat::kWav) {
        out.reserve(44 + data_bytes);
        const char* riff = "RIFF";
        out.insert(out.end(), reinterpret_cast<const std::byte*>(riff),
                   reinterpret_cast<const std::byte*>(riff) + 4);
        put_le(out, 36 + data_bytes, 4);  // chunk size = 36 + data.
        const char* wave = "WAVEfmt ";
        out.insert(out.end(), reinterpret_cast<const std::byte*>(wave),
                   reinterpret_cast<const std::byte*>(wave) + 8);
        put_le(out, 16, 4);               // fmt chunk size.
        put_le(out, 1, 2);                // PCM.
        put_le(out, 1, 2);                // mono.
        put_le(out, sample_rate, 4);
        put_le(out, std::uint64_t(sample_rate) * 2, 4);  // byte rate.
        put_le(out, 2, 2);                // block align.
        put_le(out, 16, 2);               // bits per sample.
        const char* data = "data";
        out.insert(out.end(), reinterpret_cast<const std::byte*>(data),
                   reinterpret_cast<const std::byte*>(data) + 4);
        put_le(out, data_bytes, 4);
    } else {
        out.reserve(data_bytes);
    }
    for (float s : samples) {
        put_le(out, static_cast<std::uint16_t>(to_s16(s)), 2);
    }
    return out;
}

}  // namespace locus::audio
