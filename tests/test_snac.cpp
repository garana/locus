#include "locus/audio/snac.hpp"
#include "locus/audio/wav.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
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

namespace {

/**
 * Decodes the codes in `codes_file` with the real SNAC GGUF and
 * requires the result to match the torch reference in `wav_file`
 * (torch + snac, noise disabled, one-shot). Both are f32 and the ops are
 * the same, so the only spread is conv summation order / FMA across
 * toolchains (~1e-8 on m2), not a graph difference.
 *
 * @returns The reference waveform's RMS, for the caller's own checks.
 */
double require_snac_parity(const std::string& codes_file,
                           const std::string& wav_file) {
    auto g = locus::gguf::GgufFile::open(snac_gguf_path());
    auto dec = locus::audio::SnacDecoder::load(g);
    REQUIRE(dec.sample_rate() == 24000);

    const auto codes = read_codes(codes_file);
    const auto got = dec.decode(codes);
    const auto ref = read_f32(wav_file);
    REQUIRE(got.size() == ref.size());
    REQUIRE(got.size() == std::size_t(codes.coarse.size()) * 2048);

    double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::abs(double(got[i]) - double(ref[i]));
        max_abs = std::max(max_abs, d);
        sum_sq += d * d;
        ref_sq += double(ref[i]) * ref[i];
    }
    const double rms = std::sqrt(sum_sq / got.size());
    INFO("max_abs=" << max_abs << " rms=" << rms);
    REQUIRE(max_abs < 1e-5);
    REQUIRE(rms < 1e-6);
    return std::sqrt(ref_sq / ref.size());
}

/** SKIPs unless the SNAC GGUF and the given goldens are present. */
bool snac_e2e_inputs(std::initializer_list<std::string> goldens) {
    if (!std::filesystem::exists(snac_gguf_path())) return false;
    for (const auto& f : goldens) {
        if (!std::filesystem::exists(golden_dir() + "/" + f)) return false;
    }
    return true;
}

}  // namespace

// Parity on the greedy Orpheus golden. Its codes decode to near-silence
// (RMS ~0.0014), so on its own it says little about loud signals; see
// the sampled-speech case below.
TEST_CASE("snac decoder matches the reference waveform", "[snac][e2e]") {
    if (!snac_e2e_inputs({"orpheus_codes.txt", "orpheus_waveform.f32"})) {
        SKIP("snac GGUF absent; make it with scripts/convert-snac.py");
    }
    require_snac_parity(golden_dir() + "/orpheus_codes.txt",
                        golden_dir() + "/orpheus_waveform.f32");
}

// Parity on REAL speech (i#172): Orpheus sampled with the reference
// settings and a fixed seed (by llama-cpp-python, independent of
// locus's sampler), decoded by torch SNAC. The golden is audible speech
// that the model ended itself, so a decoder error that a near-silent
// signal would hide shows up here. Made by scripts/make-snac-goldens.py
// speech.
TEST_CASE("snac decoder matches the reference on sampled speech",
          "[snac][e2e]") {
    if (!snac_e2e_inputs(
            {"snac_speech_codes.txt", "snac_speech_waveform.f32"})) {
        SKIP("snac GGUF absent; make it with scripts/convert-snac.py");
    }
    const double ref_rms = require_snac_parity(
        golden_dir() + "/snac_speech_codes.txt",
        golden_dir() + "/snac_speech_waveform.f32");
    REQUIRE(ref_rms > 0.02);  // the golden is speech, not near-silence
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

/** Deliberate defects for synth_snac, to test the decoder's checks. */
struct SynthOpts {
    std::string drop;     ///< tensor to leave out
    std::string shorten;  ///< tensor to write one value short
    std::uint32_t n_codebooks = 3;
    // Layout keys: -1 leaves the key out, 0/1 writes it as a bool, and 2
    // writes it as a u32 0 (a wrong type).
    int depthwise = -1;
    int noise = -1;
    int attention = -1;
};

/**
 * A tiny synthetic "snac" GGUF image: the decoder graph with small
 * dimensions and pseudo-random weights, so the graph structure
 * (channel progression, transposed-conv lengths, block count, tensor
 * names) is tested without the real model. It pins structure, not
 * values: a wrong dilation keeps every length (see i#172). Three
 * codebooks with strides [4,2,1] (what SnacCodes carries), latent 4,
 * codebook 4 x 2, and a decoder width of 4 * 2^rates halved per
 * upsample block, so it ends at 4 channels whatever the rate count
 * (the real model: 1024 -> 64).
 *
 * @param rates The decoder upsample rates.
 * @param o Defects to inject (none by default).
 */
std::vector<std::byte> synth_snac(const std::vector<std::uint32_t>& rates,
                                  const SynthOpts& o = {}) {
    constexpr std::uint32_t kLatent = 4, kCbSize = 4, kCbDim = 2;
    const std::uint32_t kDec = 4u << rates.size();
    struct T {
        std::string name;
        std::vector<std::uint64_t> dims;  // GGUF order (fastest first)
        std::vector<float> data;
    };
    std::vector<T> ts;
    std::uint32_t lcg = 12345;
    // Deterministic, in [-0.25, 0.25): small enough that the tail tanh
    // stays linear, so the value fixtures are not saturated at +-1.
    // scripts/make-snac-goldens.py (lcg_weights) ports this generator
    // to build the torch reference; keep the two in sync.
    auto rnd = [&] {
        lcg = lcg * 1664525u + 1013904223u;
        return (static_cast<float>(lcg >> 8) / 16777216.0f - 0.5f) * 0.5f;
    };
    // PyTorch shape {d0, d1, d2} -> GGUF dims reversed.
    auto add = [&](const std::string& n,
                   std::vector<std::uint64_t> shape, bool ones = false) {
        if (n == o.drop) return;
        std::uint64_t count = 1;
        for (auto d : shape) count *= d;
        std::vector<float> v(count);
        for (auto& x : v) x = ones ? 1.0f : rnd();
        if (n == o.shorten) {  // a well-formed but one-short tensor
            v.pop_back();
            ts.push_back({n, {count - 1}, std::move(v)});
            return;
        }
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

    const std::pair<const char*, int> layout[] = {
        {"snac.depthwise", o.depthwise},
        {"snac.noise", o.noise},
        {"snac.attention", o.attention}};
    std::uint64_t n_kv = 9;
    for (const auto& [key, v] : layout) n_kv += v >= 0 ? 1 : 0;
    GgufBuilder g;
    g.header(ts.size(), n_kv);
    g.kv_string("general.architecture", "snac");
    g.kv_u32("snac.sampling_rate", 24000);
    g.kv_u32("snac.decoder_dim", kDec);
    g.kv_u32("snac.latent_dim", kLatent);
    g.kv_u32("snac.codebook_size", kCbSize);
    g.kv_u32("snac.codebook_dim", kCbDim);
    g.kv_u32("snac.n_codebooks", o.n_codebooks);
    const std::vector<std::int32_t> r(rates.begin(), rates.end());
    g.kv_i32_array("snac.decoder_rates", r);
    const std::vector<std::int32_t> strides = {4, 2, 1};
    g.kv_i32_array("snac.vq_strides", strides);
    for (const auto& [key, v] : layout) {
        if (v == 2) {
            g.kv_u32(key, 0);
        } else if (v >= 0) {
            g.kv_bool(key, v != 0);
        }
    }
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
    for (std::uint32_t i = 0; i < frames; ++i) {
        c.coarse.push_back((i + shift) % 4);
    }
    for (std::uint32_t i = 0; i < 2 * frames; ++i) {
        c.mid.push_back((i + 1) % 4);
    }
    for (std::uint32_t i = 0; i < 4 * frames; ++i) {
        c.fine.push_back((i + 2) % 4);
    }
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

// VALUES, not just structure: the same synthetic model decoded by the
// real torch snac.SNAC (scripts/make-snac-goldens.py synth) must match
// locus sample for sample. Runs on every host (the fixtures are
// committed and tiny). A wrong dilation, weight layout or snake formula
// keeps every length the graph test checks, but changes these values.
TEST_CASE("snac decoder matches torch on a synthetic model", "[snac]") {
    for (const auto& rates : {std::vector<std::uint32_t>{2, 2},
                              std::vector<std::uint32_t>{4, 2, 2, 2}}) {
        std::string name = "snac_synth_r";
        for (std::size_t i = 0; i < rates.size(); ++i) {
            // Two appends: GCC 12 warns (-Wrestrict, a false positive)
            // on "literal" + std::string here.
            name += i ? "-" : "";
            name += std::to_string(rates[i]);
        }
        INFO(name);
        const auto ref = read_f32(golden_dir() + "/" + name + ".f32");
        const auto img = synth_snac(rates);
        auto g = locus::gguf::GgufFile::parse(img);
        const auto got = locus::audio::SnacDecoder::load(g).decode(
            synth_codes(2));
        REQUIRE(got.size() == ref.size());
        double max_abs = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            max_abs = std::max(max_abs,
                               std::abs(double(got[i]) - double(ref[i])));
        }
        INFO("max_abs " << max_abs);
        REQUIRE(max_abs < 1e-5);
    }
}

TEST_CASE("snac decoder reports a missing tensor", "[snac]") {
    SynthOpts o;
    o.drop = "decoder.model.3.block.1.weight";
    const auto img = synth_snac({2, 2}, o);
    auto g = locus::gguf::GgufFile::parse(img);
    const auto dec = locus::audio::SnacDecoder::load(g);
    REQUIRE_THROWS_AS(dec.decode(synth_codes(1)), locus::gguf::Error);
}

// Malformed models and codes must throw, not read out of bounds: the
// conv kernels check sizes only with assert(), which Release builds
// compile out (i#183).
TEST_CASE("snac decoder rejects malformed models and codes", "[snac]") {
    auto load = [](const std::vector<std::uint32_t>& rates,
                   const SynthOpts& o) {
        const auto img = synth_snac(rates, o);
        auto g = locus::gguf::GgufFile::parse(img);
        return locus::audio::SnacDecoder::load(g);
    };
    SECTION("n_codebooks other than 3") {
        SynthOpts o;
        o.n_codebooks = 4;  // decode() carries exactly three streams
        REQUIRE_THROWS_AS(load({2, 2}, o), locus::gguf::Error);
    }
    SECTION("an odd upsample rate") {
        REQUIRE_THROWS_AS(load({3, 2}, SynthOpts{}), locus::gguf::Error);
    }
    SECTION("a recorded layout locus does not run") {
        // One key at a time; the others hold the supported value.
        const auto with = [](int depthwise, int noise, int attention) {
            SynthOpts o;
            o.depthwise = depthwise;
            o.noise = noise;
            o.attention = attention;
            return o;
        };
        REQUIRE_THROWS_AS(load({2, 2}, with(0, 1, 0)), locus::gguf::Error);
        REQUIRE_THROWS_AS(load({2, 2}, with(1, 0, 0)), locus::gguf::Error);
        REQUIRE_THROWS_AS(load({2, 2}, with(1, 1, 1)), locus::gguf::Error);
        // The supported layout still loads.
        REQUIRE_NOTHROW(load({2, 2}, with(1, 1, 0)));
    }
    SECTION("a layout key that is not a bool") {
        // A u32 0 must not read as "key absent" (= the default layout).
        for (int k = 0; k < 3; ++k) {
            INFO("key " << k);
            SynthOpts o;
            (k == 0 ? o.depthwise : k == 1 ? o.noise : o.attention) = 2;
            REQUIRE_THROWS_AS(load({2, 2}, o), locus::gguf::Error);
        }
    }
    SECTION("a tensor one value short") {
        for (const char* name :
             {"quantizer.quantizers.1.codebook.weight",
              "quantizer.quantizers.0.out_proj.weight",
              "decoder.model.2.block.1.weight",       // transposed conv
              "decoder.model.3.block.4.block.0.alpha",  // snake
              "decoder.model.5.weight"}) {             // tail conv
            INFO(name);
            SynthOpts o;
            o.shorten = name;
            const auto dec = load({2, 2}, o);
            REQUIRE_THROWS_AS(dec.decode(synth_codes(1)),
                              locus::gguf::Error);
        }
    }
    SECTION("a code outside the codebook") {
        const auto dec = load({2, 2}, SynthOpts{});
        auto c = synth_codes(1);
        c.fine[2] = 4;  // codebook_size is 4
        REQUIRE_THROWS_AS(dec.decode(c), std::invalid_argument);
        c.fine[2] = -1;
        REQUIRE_THROWS_AS(dec.decode(c), std::invalid_argument);
    }
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
