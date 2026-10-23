#include "locus/audio/snac.hpp"
#include "locus/audio/wav.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/codec.hpp"

namespace {

std::string snac_gguf_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/snac_24khz_decoder.gguf";
}

std::string golden_dir() {
    return std::string(LOCUS_SOURCE_DIR) + "/tests/golden";
}

/** Reads orpheus_codes.txt (3 whitespace-separated lines) into the
 *  coarse/mid/fine streams. */
locus::model::SnacCodes read_codes(const std::string& path) {
    std::ifstream f(path);
    locus::model::SnacCodes c;
    std::vector<std::vector<int>*> rows = {&c.coarse, &c.mid, &c.fine};
    std::string line;
    for (auto* row : rows) {
        if (!std::getline(f, line)) break;
        std::istringstream ls(line);
        int v;
        while (ls >> v) row->push_back(v);
    }
    return c;
}

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<float> out;
    float v;
    while (f.read(reinterpret_cast<char*>(&v), sizeof(v))) {
        out.push_back(v);
    }
    return out;
}

}  // namespace

// Sample-close parity: decoding the golden codes with locus's SNAC
// decoder reproduces the reference waveform (torch + snac, noise
// disabled, one-shot). Both are f32 and the ops are identical, so the
// only spread is floating-point summation order in the convolutions.
TEST_CASE("snac decoder matches the reference waveform", "[snac][e2e]") {
    const auto gguf = snac_gguf_path();
    const auto wav = golden_dir() + "/orpheus_waveform.f32";
    if (!std::filesystem::exists(gguf) ||
        !std::filesystem::exists(wav)) {
        SKIP("snac decoder GGUF or golden waveform absent; run "
             ".local-dev/golden/convert_snac.py and run.sh");
    }
    auto g = locus::gguf::GgufFile::open(gguf);
    auto dec = locus::audio::SnacDecoder::load(g);
    REQUIRE(dec.sample_rate() == 24000);

    const auto codes = read_codes(golden_dir() + "/orpheus_codes.txt");
    const auto got = dec.decode(codes);
    const auto ref = read_f32(wav);

    REQUIRE(got.size() == ref.size());
    REQUIRE(got.size() == std::size_t(codes.coarse.size()) * 2048);

    double max_abs = 0.0, sum_sq = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::abs(double(got[i]) - double(ref[i]));
        max_abs = std::max(max_abs, d);
        sum_sq += d * d;
    }
    const double rms = std::sqrt(sum_sq / got.size());
    INFO("max_abs=" << max_abs << " rms=" << rms);
    // The decoder is the same f32 math as the reference, so the match
    // is essentially bit-exact (~7e-9 on m2); these bounds leave room
    // only for conv summation-order / FMA spread across toolchains, not
    // for an actual graph difference.
    REQUIRE(max_abs < 1e-5);
    REQUIRE(rms < 1e-6);
}

namespace {

std::uint32_t le32(const std::vector<std::byte>& b, std::size_t off) {
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= std::uint32_t(std::to_integer<unsigned>(b[off + i])) <<
             (8 * i);
    }
    return v;
}

std::int16_t le16(const std::vector<std::byte>& b, std::size_t off) {
    return static_cast<std::int16_t>(
        std::to_integer<unsigned>(b[off]) |
        (std::to_integer<unsigned>(b[off + 1]) << 8));
}

}  // namespace

TEST_CASE("pcm encode writes a WAV header and clamped s16", "[wav]") {
    const std::vector<float> s = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f};
    const auto wav =
        locus::audio::encode_pcm(s, 24000, locus::audio::WaveFormat::kWav);

    REQUIRE(wav.size() == 44 + s.size() * 2);
    REQUIRE(std::memcmp(wav.data(), "RIFF", 4) == 0);
    REQUIRE(std::memcmp(wav.data() + 8, "WAVE", 4) == 0);
    REQUIRE(std::memcmp(wav.data() + 36, "data", 4) == 0);
    REQUIRE(le32(wav, 4) == 36 + s.size() * 2);   // RIFF chunk size.
    REQUIRE(le16(wav, 20) == 1);                   // PCM.
    REQUIRE(le16(wav, 22) == 1);                   // mono.
    REQUIRE(le32(wav, 24) == 24000);               // sample rate.
    REQUIRE(le16(wav, 34) == 16);                  // bits/sample.
    REQUIRE(le32(wav, 40) == s.size() * 2);        // data bytes.

    REQUIRE(le16(wav, 44) == 0);                    // 0.0
    REQUIRE(le16(wav, 46) == 32767);                // 1.0
    REQUIRE(le16(wav, 48) == -32767);               // -1.0
    REQUIRE(le16(wav, 50) == 16384);                // 0.5 -> round
    REQUIRE(le16(wav, 52) == 32767);                // 2.0 clamped

    const auto pcm = locus::audio::encode_pcm(
        s, 24000, locus::audio::WaveFormat::kPcm16);
    REQUIRE(pcm.size() == s.size() * 2);            // no header.
    REQUIRE(le16(pcm, 2) == 32767);                 // same samples.
}
