#include "locus/model/codec.hpp"

#include <stdexcept>
#include <string>

namespace locus::model {

namespace {

/**
 * De-bands `audio` (a whole number of frames) into the three streams.
 * Shared by the free function and the sampler.
 */
SnacCodes deband_frames(std::span<const tok::TokenId> audio,
                        tok::TokenId band_base, bool drop_invalid) {
    constexpr int kN = OrpheusCodec::kCodesPerFrame;
    constexpr int kB = OrpheusCodec::kCodebook;
    if (audio.size() % kN != 0) {
        throw std::invalid_argument(
            "orpheus de-band: " + std::to_string(audio.size()) +
            " audio tokens is not a multiple of 7");
    }
    SnacCodes out;
    const std::size_t nframes = audio.size() / kN;
    out.coarse.reserve(nframes);
    out.mid.reserve(nframes * 2);
    out.fine.reserve(nframes * 4);
    for (std::size_t f = 0; f < nframes; ++f) {
        int code[kN];
        bool ok = true;
        for (int i = 0; i < kN; ++i) {
            const long c = static_cast<long>(audio[f * kN + i]) -
                           band_base - static_cast<long>(i) * kB;
            if (c < 0 || c >= kB) {
                if (drop_invalid) {
                    ok = false;
                    break;
                }
                throw std::invalid_argument(
                    "orpheus de-band: frame " + std::to_string(f) +
                    " position " + std::to_string(i) +
                    " decodes out of range: " + std::to_string(c));
            }
            code[i] = static_cast<int>(c);
        }
        if (!ok) continue;  // drop the desynced frame (lenient path).
        // vq_strides [4, 2, 1]: position 0 coarse; 1,4 mid; 2,3,5,6
        // fine (see SnacCodes).
        out.coarse.push_back(code[0]);
        out.mid.push_back(code[1]);
        out.mid.push_back(code[4]);
        out.fine.push_back(code[2]);
        out.fine.push_back(code[3]);
        out.fine.push_back(code[5]);
        out.fine.push_back(code[6]);
    }
    return out;
}

}  // namespace

SnacCodes orpheus_deband(std::span<const tok::TokenId> audio_tokens,
                         tok::TokenId band_base) {
    return deband_frames(audio_tokens, band_base, false);
}

tok::TokenId orpheus_band_base(const tok::Tokenizer& tok) {
    const std::string_view name = "<custom_token_10>";
    const std::size_t n = tok.vocab_size();
    for (std::size_t id = 0; id < n; ++id) {
        if (tok.piece(static_cast<tok::TokenId>(id)) == name) {
            return static_cast<tok::TokenId>(id);
        }
    }
    throw std::runtime_error(
        "orpheus band base: <custom_token_10> not in vocabulary");
}

OrpheusCodecSampler::Status OrpheusCodecSampler::push(tok::TokenId t) {
    const tok::TokenId lo = band_base_;
    const tok::TokenId hi = band_base_ +
        OrpheusCodec::kCodesPerFrame * OrpheusCodec::kCodebook;
    if (t < lo || t >= hi) {
        stop_ = t;
        return Status::kStop;
    }
    audio_.push_back(t);
    return Status::kAudio;
}

SnacCodes OrpheusCodecSampler::codes() const {
    const std::size_t whole =
        frames() * OrpheusCodec::kCodesPerFrame;
    return deband_frames(
        std::span<const tok::TokenId>(audio_.data(), whole), band_base_,
        false);
}

SnacCodes OrpheusCodecSampler::codes_lenient() const {
    const std::size_t whole =
        frames() * OrpheusCodec::kCodesPerFrame;
    return deband_frames(
        std::span<const tok::TokenId>(audio_.data(), whole), band_base_,
        true);
}

}  // namespace locus::model
