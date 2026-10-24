#include "locus/audio/snac.hpp"
#include "locus/audio/wav.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
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

namespace {

/**
 * A tiny synthetic "snac" GGUF image: the decoder graph with small
 * dimensions and pseudo-random weights, so the graph (channel
 * progression, rate order, transposed-conv lengths, tensor names) is
 * tested without the real model. Three codebooks with strides [4,2,1]
 * (what SnacCodes carries), latent 4, codebook 4 x 2, and a decoder
 * width of 4 * 2^rates halved per upsample block, so it ends at 4
 * channels whatever the rate count (the real model: 1024 -> 64).
 *
 * @param rates The decoder upsample rates (even, as in SNAC 24 kHz).
 * @param drop A tensor name to leave out (to test the missing-tensor
 *     error), or empty.
 */
std::vector<std::byte> synth_snac(const std::vector<std::uint32_t>& rates,
                                  const std::string& drop = "") {
    constexpr std::uint32_t kLatent = 4, kCbSize = 4, kCbDim = 2;
    const std::uint32_t kDec = 4u << rates.size();
    struct T {
        std::string name;
        std::vector<std::uint64_t> dims;  // GGUF order (fastest first)
        std::vector<float> data;
    };
    std::vector<T> ts;
    std::uint32_t lcg = 12345;
    auto rnd = [&] {  // deterministic, in [-0.5, 0.5)
        lcg = lcg * 1664525u + 1013904223u;
        return static_cast<float>(lcg >> 8) / 16777216.0f - 0.5f;
    };
    // PyTorch shape {d0, d1, d2} -> GGUF dims reversed.
    auto add = [&](const std::string& n,
                   std::vector<std::uint64_t> shape, bool ones = false) {
        if (n == drop) return;
        std::uint64_t count = 1;
        for (auto d : shape) count *= d;
        std::vector<float> v(count);
        for (auto& x : v) x = ones ? 1.0f : rnd();
        ts.push_back({n, {shape.rbegin(), shape.rend()}, std::move(v)});
    };
    for (int i = 0; i < 3; ++i) {
        const std::string q = "quantizer.quantizers." + std::to_string(i);
        add(q + ".codebook.weight", {kCbSize, kCbDim});
        add(q + ".out_proj.weight", {kLatent, kCbDim, 1});
        add(q + ".out_proj.bias", {kLatent});
    }
    add("decoder.model.0.weight", {kLatent, 1, 7});  // depthwise
    add("decoder.model.0.bias", {kLatent});
    add("decoder.model.1.weight", {kDec, kLatent, 1});
    add("decoder.model.1.bias", {kDec});
    std::uint32_t ch = kDec;
    for (std::size_t b = 0; b < rates.size(); ++b) {
        const std::string p =
            "decoder.model." + std::to_string(2 + b) + ".block";
        const std::uint32_t out = ch / 2;
        add(p + ".0.alpha", {1, ch, 1}, true);
        add(p + ".1.weight", {ch, out, 2 * rates[b]});  // transpose
        add(p + ".1.bias", {out});
        for (int u = 3; u <= 5; ++u) {
            const std::string rp = p + "." + std::to_string(u) + ".block";
            add(rp + ".0.alpha", {1, out, 1}, true);
            add(rp + ".1.weight", {out, 1, 7});
            add(rp + ".1.bias", {out});
            add(rp + ".2.alpha", {1, out, 1}, true);
            add(rp + ".3.weight", {out, out, 1});
            add(rp + ".3.bias", {out});
        }
        ch = out;
    }
    const std::string tail = std::to_string(2 + rates.size());
    const std::string tail_conv = std::to_string(3 + rates.size());
    add("decoder.model." + tail + ".alpha", {1, ch, 1}, true);
    add("decoder.model." + tail_conv + ".weight", {1, ch, 7});
    add("decoder.model." + tail_conv + ".bias", {1});

    GgufBuilder g;
    g.header(ts.size(), 9);
    g.kv_string("general.architecture", "snac");
    g.kv_u32("snac.sampling_rate", 24000);
    g.kv_u32("snac.decoder_dim", kDec);
    g.kv_u32("snac.latent_dim", kLatent);
    g.kv_u32("snac.codebook_size", kCbSize);
    g.kv_u32("snac.codebook_dim", kCbDim);
    g.kv_u32("snac.n_codebooks", 3);
    const std::vector<std::int32_t> r(rates.begin(), rates.end());
    g.kv_i32_array("snac.decoder_rates", r);
    const std::vector<std::int32_t> strides = {4, 2, 1};
    g.kv_i32_array("snac.vq_strides", strides);
    std::uint64_t offset = 0;
    for (const auto& t : ts) {
        g.tensor(t.name, t.dims, 0 /* F32 */, offset);
        offset = (offset + t.data.size() * 4 + 31) & ~31ull;
    }
    g.pad();
    for (const auto& t : ts) {
        for (float v : t.data) g.f32(v);
        while (g.bytes().size() % 32 != 0) g.zeros(1);
    }
    auto span = g.bytes();
    return {span.begin(), span.end()};
}

/** Codes for F frames, cycling through the codebook. */
locus::model::SnacCodes synth_codes(std::uint32_t frames, int shift = 0) {
    locus::model::SnacCodes c;
    for (std::uint32_t i = 0; i < frames; ++i) c.coarse.push_back((i + shift) % 4);
    for (std::uint32_t i = 0; i < 2 * frames; ++i) c.mid.push_back((i + 1) % 4);
    for (std::uint32_t i = 0; i < 4 * frames; ++i) c.fine.push_back((i + 2) % 4);
    return c;
}

}  // namespace

// The decode graph without the real model (runs on every host): output
// length is exactly strides.front() * product(rates) per frame, values
// are finite, decoding is bit-identical across runs, and the codes
// actually reach the output. Two different rate lists check that the
// structure (block count, tail position) follows the metadata.
TEST_CASE("snac decoder graph on a synthetic model", "[snac]") {
    for (const auto& rates : {std::vector<std::uint32_t>{2, 2},
                              std::vector<std::uint32_t>{4, 2, 2, 2}}) {
        std::uint32_t prod = 1;
        for (auto r : rates) prod *= r;
        INFO("rates " << rates.size() << ", product " << prod);
        const auto img = synth_snac(rates);
        auto g = locus::gguf::GgufFile::parse(img);
        const auto dec = locus::audio::SnacDecoder::load(g);
        for (std::uint32_t frames : {1u, 2u, 5u}) {
            INFO(frames << " frames");
            const auto codes = synth_codes(frames);
            const auto a = dec.decode(codes);
            REQUIRE(a.size() == std::size_t(frames) * 4 * prod);
            bool finite = true, nonzero = false;
            for (float v : a) {
                finite = finite && std::isfinite(v);
                nonzero = nonzero || v != 0.0f;
            }
            REQUIRE(finite);
            REQUIRE(nonzero);
            REQUIRE(dec.decode(codes) == a);  // no leaked state
            REQUIRE(dec.decode(synth_codes(frames, 1)) != a);  // codes count
        }
    }
}

TEST_CASE("snac decoder reports a missing tensor", "[snac]") {
    const auto img = synth_snac({2, 2}, "decoder.model.3.block.1.weight");
    auto g = locus::gguf::GgufFile::parse(img);
    const auto dec = locus::audio::SnacDecoder::load(g);
    REQUIRE_THROWS_AS(dec.decode(synth_codes(1)), locus::gguf::Error);
}

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
