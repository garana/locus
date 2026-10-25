#include "locus/image/vq16.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "locus/gguf/gguf.hpp"

namespace {

using Shape = std::vector<std::uint64_t>;  // PyTorch order

/** Options for synth_vq16: the defects a test injects. */
struct SynthOpts {
    bool l2_norm = true;
    std::vector<std::int32_t> attn_levels = {1};
    std::string drop;  ///< tensor to leave out
};

/**
 * The tiny VQ-16 decoder of scripts/make-janus-goldens.py synth (z 32,
 * ch 32, ch_mult (1, 2), num_res_blocks 1, codebook 16 x 8), as a GGUF
 * image: the tensors the Janus Decoder constructor creates, filled from
 * the LCG in sorted name order (std::map), exactly as the script's
 * lcg_fill. Keep the two in sync.
 */
std::vector<std::byte> synth_vq16(const SynthOpts& o = {}) {
    constexpr std::uint32_t kZ = 32, kCh = 32, kCb = 16, kDim = 8;
    const std::vector<std::uint32_t> mult = {1, 2};
    constexpr std::uint32_t kNrb = 1;
    std::map<std::string, Shape> shapes;
    auto conv = [&](const std::string& p, std::uint64_t in,
                    std::uint64_t out, std::uint64_t k) {
        shapes[p + ".weight"] = {out, in, k, k};
        shapes[p + ".bias"] = {out};
    };
    auto norm = [&](const std::string& p, std::uint64_t c) {
        shapes[p + ".weight"] = {c};
        shapes[p + ".bias"] = {c};
    };
    auto res = [&](const std::string& p, std::uint64_t in,
                   std::uint64_t out) {
        norm(p + ".norm1", in);
        conv(p + ".conv1", in, out, 3);
        norm(p + ".norm2", out);
        conv(p + ".conv2", out, out, 3);
        if (in != out) conv(p + ".nin_shortcut", in, out, 1);
    };
    auto attn = [&](const std::string& p, std::uint64_t c) {
        norm(p + ".norm", c);
        conv(p + ".q", c, c, 1);
        conv(p + ".k", c, c, 1);
        conv(p + ".v", c, c, 1);
        conv(p + ".proj_out", c, c, 1);
    };
    shapes["quantize.embedding.weight"] = {kCb, kDim};
    conv("post_quant_conv", kDim, kZ, 1);
    std::uint64_t block_in = kCh * mult.back();
    conv("decoder.conv_in", kZ, block_in, 3);
    res("decoder.mid.0", block_in, block_in);
    attn("decoder.mid.1", block_in);
    res("decoder.mid.2", block_in, block_in);
    for (std::size_t i = 0; i < mult.size(); ++i) {
        const std::size_t level = mult.size() - 1 - i;
        const std::uint64_t out = kCh * mult[level];
        const std::string p = "decoder.conv_blocks." + std::to_string(i);
        for (std::uint32_t b = 0; b <= kNrb; ++b) {
            res(p + ".res." + std::to_string(b), block_in, out);
            block_in = out;
            if (level == mult.size() - 1) {
                attn(p + ".attn." + std::to_string(b), block_in);
            }
        }
        if (i != mult.size() - 1) {
            conv(p + ".upsample.conv", block_in, block_in, 3);
        }
    }
    norm("decoder.norm_out", block_in);
    conv("decoder.conv_out", block_in, 3, 3);

    // The script's LCG, in the map's sorted order.
    std::uint32_t lcg = 12345;
    std::map<std::string, std::vector<float>> data;
    for (const auto& [name, shape] : shapes) {
        std::uint64_t n = 1;
        for (auto d : shape) n *= d;
        std::vector<float> v(n);
        for (auto& x : v) {
            lcg = lcg * 1664525u + 1013904223u;
            x = (static_cast<float>(lcg >> 8) / 16777216.0f - 0.5f) * 0.5f;
        }
        if (name != o.drop) data[name] = std::move(v);
    }

    GgufBuilder g;
    g.header(data.size(), 11);
    g.kv_string("general.architecture", "vq16");
    g.kv_u32("vq16.codebook_size", kCb);
    g.kv_u32("vq16.embed_dim", kDim);
    g.kv_bool("vq16.codebook_l2_norm", o.l2_norm);
    g.kv_u32("vq16.z_channels", kZ);
    g.kv_u32("vq16.ch", kCh);
    const std::vector<std::int32_t> m(mult.begin(), mult.end());
    g.kv_i32_array("vq16.ch_mult", m);
    g.kv_u32("vq16.num_res_blocks", kNrb);
    g.kv_i32_array("vq16.attn_levels", o.attn_levels);
    g.kv_u32("vq16.norm_groups", 32);
    g.kv_f32("vq16.norm_eps", 1e-6f);
    std::uint64_t offset = 0;
    for (const auto& [name, v] : data) {
        const Shape& s = shapes[name];
        const std::vector<std::uint64_t> dims(s.rbegin(), s.rend());
        g.tensor(name, dims, 0 /* F32 */, offset);
        offset = (offset + v.size() * 4 + 31) & ~31ull;
    }
    g.pad();
    for (const auto& [name, v] : data) {
        for (float x : v) g.f32(x);
        while (g.bytes().size() % 32 != 0) g.zeros(1);
    }
    auto span = g.bytes();
    return {span.begin(), span.end()};
}

/** The script's fixed 3 x 3 code grid. */
std::vector<std::int32_t> synth_codes() {
    std::vector<std::int32_t> c;
    for (int i = 0; i < 9; ++i) c.push_back((i * 5 + 3) % 16);
    return c;
}

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

}  // namespace

// VALUES, on every box: the synthetic decoder against torch's output
// from scripts/make-janus-goldens.py synth (the Janus Decoder class with
// the same weights). Pins the op order, GroupNorm, attention, upsample
// and the l2-normalised lookup, not just shapes.
TEST_CASE("vq16 decoder matches torch on a synthetic model",
          "[vq16][image]") {
    const auto img = synth_vq16();
    auto g = locus::gguf::GgufFile::parse(img);
    const auto dec = locus::image::Vq16Decoder::load(g);
    REQUIRE(dec.upscale() == 2);
    const auto got = dec.decode(synth_codes(), 3, 3);
    const auto ref = read_f32(std::string(LOCUS_SOURCE_DIR) +
                              "/tests/golden/vq16_synth.f32");
    REQUIRE(got.size() == 3 * 6 * 6);
    REQUIRE(ref.size() == got.size());
    double max_abs = 0.0, ref_sq = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs,
                           std::abs(double(got[i]) - double(ref[i])));
        ref_sq += double(ref[i]) * ref[i];
    }
    INFO("max_abs " << max_abs);
    REQUIRE(std::sqrt(ref_sq / ref.size()) > 0.05);  // not a flat output
    REQUIRE(max_abs < 1e-5);
}

// The real Janus-Pro VQ-16 decoder on the golden image tokens (CFG
// argmax, scripts/make-janus-goldens.py tokens) against torch's f32
// decode of the same tokens (`decode`). The bound is i#221's 1e-4:
// deeper than SNAC, with GroupNorm and softmax in the path.
TEST_CASE("vq16 decoder matches the reference on Janus-Pro tokens",
          "[vq16][image][e2e]") {
    const std::string gguf = std::string(LOCUS_SOURCE_DIR) +
                             "/tests/models/janus-vq16-decoder.gguf";
    const std::string golden = std::string(LOCUS_SOURCE_DIR) +
                               "/tests/golden/";
    if (!std::filesystem::exists(gguf) ||
        !std::filesystem::exists(golden + "janus_tokens.txt") ||
        !std::filesystem::exists(golden + "janus_decode.f32")) {
        SKIP("vq16 GGUF or Janus goldens absent; see "
             "scripts/convert-janus-vq16.py");
    }
    std::ifstream f(golden + "janus_tokens.txt");
    std::string line;
    for (int i = 0; i < 3; ++i) REQUIRE(std::getline(f, line));
    std::vector<std::int32_t> codes;
    std::istringstream ls(line);
    for (std::int32_t v; ls >> v;) codes.push_back(v);
    REQUIRE(codes.size() == 576);

    auto g = locus::gguf::GgufFile::open(gguf);
    const auto dec = locus::image::Vq16Decoder::load(g);
    REQUIRE(dec.upscale() == 16);
    const auto got = dec.decode(codes, 24, 24);
    const auto ref = read_f32(golden + "janus_decode.f32");
    REQUIRE(got.size() == std::size_t{3} * 384 * 384);
    REQUIRE(ref.size() == got.size());
    double max_abs = 0.0, sum_sq = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::abs(double(got[i]) - double(ref[i]));
        max_abs = std::max(max_abs, d);
        sum_sq += d * d;
    }
    INFO("max_abs " << max_abs << " rms "
                    << std::sqrt(sum_sq / got.size()));
    REQUIRE(max_abs < 1e-4);
}

TEST_CASE("vq16 decoder rejects layouts and codes it does not run",
          "[vq16][image]") {
    auto load = [](const SynthOpts& o) {
        const auto img = synth_vq16(o);
        auto g = locus::gguf::GgufFile::parse(img);
        return locus::image::Vq16Decoder::load(g);
    };
    SECTION("no l2-normalised codebook") {
        SynthOpts o;
        o.l2_norm = false;
        REQUIRE_THROWS_AS(load(o), locus::gguf::Error);
    }
    SECTION("attention at another level") {
        SynthOpts o;
        o.attn_levels = {0};
        REQUIRE_THROWS_AS(load(o), locus::gguf::Error);
    }
    SECTION("a missing tensor") {
        SynthOpts o;
        o.drop = "decoder.mid.1.q.weight";
        const auto dec = load(o);
        REQUIRE_THROWS_AS(dec.decode(synth_codes(), 3, 3),
                          locus::gguf::Error);
    }
    SECTION("codes") {
        const auto dec = load(SynthOpts{});
        auto c = synth_codes();
        REQUIRE_THROWS_AS(dec.decode(c, 3, 2), std::invalid_argument);
        c[4] = 16;  // codebook_size is 16
        REQUIRE_THROWS_AS(dec.decode(c, 3, 3), std::invalid_argument);
        c[4] = -1;
        REQUIRE_THROWS_AS(dec.decode(c, 3, 3), std::invalid_argument);
    }
}
