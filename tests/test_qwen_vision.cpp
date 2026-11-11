#include "locus/image/qwen_vision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "png_parse.hpp"

namespace {

using Shape = std::vector<std::uint64_t>;
using locus::image::QwenVision;

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";

// Keep in sync with scripts/make-qwen-vision-goldens.py.
constexpr std::uint32_t kW = 32, kHeads = 2, kFfn = 40, kOut = 24,
                        kDepth = 4, kWa = 2, kGh = 12, kGw = 16;
constexpr std::uint32_t kPatches[] = {0, 101};
constexpr std::uint32_t kRows = 16, kRowStep = 4, kCropH = 196,
                        kCropW = 252;

std::vector<std::byte> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> c((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<std::byte> b(c.size());
    std::memcpy(b.data(), c.data(), c.size());
    return b;
}

std::vector<float> read_f32(const std::string& path) {
    const auto b = read_bytes(path);
    REQUIRE(!b.empty());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

/** The script's LCG: n values, advancing state. */
std::vector<float> lcg(std::uint32_t& state, std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 0.5f;
    }
    return v;
}

/**
 * The tiny tower of scripts/make-qwen-vision-goldens.py synth as an
 * mmproj GGUF image: the script's gguf_shapes, F32, filled from the
 * LCG in sorted name order (std::map). Keep the two in sync.
 */
std::vector<std::byte> synth_mmproj(const char* projector =
                                        "qwen2.5vl_merger") {
    std::map<std::string, Shape> shapes;
    const std::uint64_t w = kW, f = kFfn, m = 4 * kW;
    shapes["v.patch_embd.weight"] = {w, 3, 14, 14};
    shapes["v.patch_embd.weight.1"] = {w, 3, 14, 14};
    shapes["v.post_ln.weight"] = {w};
    shapes["mm.0.weight"] = {m, m};
    shapes["mm.0.bias"] = {m};
    shapes["mm.2.weight"] = {kOut, m};
    shapes["mm.2.bias"] = {kOut};
    for (std::uint32_t i = 0; i < kDepth; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        shapes[b + "ln1.weight"] = {w};
        shapes[b + "ln2.weight"] = {w};
        for (const char* t : {"attn_q", "attn_k", "attn_v", "attn_out"}) {
            shapes[b + t + ".weight"] = {w, w};
            shapes[b + t + ".bias"] = {w};
        }
        for (const char* t : {"ffn_gate", "ffn_up"}) {
            shapes[b + t + ".weight"] = {f, w};
            shapes[b + t + ".bias"] = {f};
        }
        shapes[b + "ffn_down.weight"] = {w, f};
        shapes[b + "ffn_down.bias"] = {w};
    }
    std::uint32_t state = 12345;
    std::map<std::string, std::vector<float>> data;
    for (const auto& [name, shape] : shapes) {
        std::uint64_t n = 1;
        for (auto d : shape) n *= d;
        data[name] = lcg(state, n);
    }

    GgufBuilder g;
    g.header(data.size(), 12);
    g.kv_string("general.architecture", "clip");
    g.kv_string("clip.projector_type", projector);
    g.kv_u32("clip.vision.patch_size", 14);
    g.kv_u32("clip.vision.embedding_length", kW);
    g.kv_u32("clip.vision.feed_forward_length", kFfn);
    g.kv_u32("clip.vision.attention.head_count", kHeads);
    g.kv_u32("clip.vision.projection_dim", kOut);
    g.kv_u32("clip.vision.block_count", kDepth);
    g.kv_u32("clip.vision.n_wa_pattern", kWa);
    g.kv_f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
    const float mean[] = {0.5f, 0.5f, 0.5f}, sd[] = {0.25f, 0.25f, 0.25f};
    g.kv_f32_array("clip.vision.image_mean", mean);
    g.kv_f32_array("clip.vision.image_std", sd);
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

/** @returns The RMS of x - ref, and REQUIREs it within rel of ref's. */
double check_close(std::span<const float> got, std::span<const float> ref,
                   double rel) {
    REQUIRE(got.size() == ref.size());
    double err = 0.0, norm = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = double{got[i]} - ref[i];
        err += d * d;
        norm += double{ref[i]} * ref[i];
        worst = std::max(worst, std::abs(d));
    }
    err = std::sqrt(err / static_cast<double>(ref.size()));
    norm = std::sqrt(norm / static_cast<double>(ref.size()));
    WARN("error rms " << err << ", worst " << worst << ", reference rms "
                      << norm);
    REQUIRE(err <= rel * norm);
    return err;
}

const std::string kModel =
    kDir + "models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf";

}  // namespace

// The reference's get_window_index on a 12 x 16 patch grid (6 x 8
// merged, windows of 4 x 4): two full windows, two half ones, and the
// padding-only third column dropped. A grid that one window covers
// exactly still gives one window.
TEST_CASE("qwen vision windows match the reference index",
          "[qwen-vision][image]") {
    const auto img = synth_mmproj();
    const auto g = locus::gguf::GgufFile::parse(img);
    const auto v = QwenVision::load(g);
    const auto w = v.windows(kGh, kGw);
    REQUIRE(w.size() == 4);
    REQUIRE(w[0].size() == 16);
    REQUIRE(w[1].size() == 16);
    REQUIRE(w[2].size() == 8);
    REQUIRE(w[3].size() == 8);
    REQUIRE(w[0] == std::vector<std::uint32_t>{0,  1,  2,  3,  8,  9,
                                               10, 11, 16, 17, 18, 19,
                                               24, 25, 26, 27});
    REQUIRE(w[3] == std::vector<std::uint32_t>{36, 37, 38, 39, 44, 45,
                                               46, 47});
    REQUIRE(v.windows(8, 8).size() == 1);
    REQUIRE(v.windows(8, 8)[0].size() == 16);
}

// i#530 VALUES, on every box: the synthetic tower against transformers'
// Qwen2_5_VisionTransformerPretrainedModel with the same weights
// (scripts/make-qwen-vision-goldens.py synth). Pins the patch conv,
// the 2D RoPE, the windowed and full attention, the SwiGLU MLP and the
// merger, not just shapes.
TEST_CASE("qwen vision tower matches transformers on a synthetic model",
          "[qwen-vision][image]") {
    const auto img = synth_mmproj();
    const auto g = locus::gguf::GgufFile::parse(img);
    const auto v = QwenVision::load(g);
    REQUIRE(v.out_dim() == kOut);
    std::uint32_t state = 777;
    auto px = lcg(state, std::size_t{kGh} * kGw * v.patch_dim());
    for (float& x : px) x *= 4.0f;
    const auto out =
        v.encode(px, kGh, kGw, locus::backend::best_backend().ops);
    REQUIRE(out.size() == std::size_t{kGh} * kGw / 4 * kOut);
    // 4e-7 on an M2 Pro.
    check_close(out, read_f32(kDir + "golden/qwen_vision_synth.f32"),
                1e-5);
}

TEST_CASE("qwen vision rejects another projector and a bad grid",
          "[qwen-vision][image]") {
    const auto other = synth_mmproj("qwen2vl_merger");
    REQUIRE_THROWS_AS(
        QwenVision::load(locus::gguf::GgufFile::parse(other)),
        locus::gguf::Error);
    const auto img = synth_mmproj();
    const auto g = locus::gguf::GgufFile::parse(img);
    const auto v = QwenVision::load(g);
    const auto& op = locus::backend::best_backend().ops;
    std::vector<float> px(std::size_t{3} * 4 * v.patch_dim());
    REQUIRE_THROWS_AS(v.encode(px, 3, 4, op), std::invalid_argument);
    REQUIRE_THROWS_AS(v.encode(px, 4, 4, op), std::invalid_argument);
    std::vector<std::uint8_t> rgb(std::size_t{30} * 28 * 3);
    REQUIRE_THROWS_AS(v.patches(rgb, 30, 28), std::invalid_argument);
}

// i#530: the Qwen2.5-VL 7B tower (llama.cpp's mmproj, F16) on a crop of
// janus_apple.png, against transformers in float32 with the same
// weights (scripts/make-qwen-vision-goldens.py real): the processor's
// patches, sampled output rows and the whole output's RMS.
TEST_CASE("qwen vision tower matches transformers on Qwen2.5-VL 7B",
          "[qwen-vision][image]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf absent; see "
             "scripts/fetch-test-model.sh qwen2.5-vl-mmproj");
    }
    const auto ref = read_f32(kDir + "golden/qwen_vision.f32");
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto v = QwenVision::load(g);
    const std::size_t pd = v.patch_dim(), np = std::size(kPatches);
    REQUIRE(ref.size() == np * pd + std::size_t{kRows} * v.out_dim() + 1);

    const auto png = png_test::parse_stored_png(
        read_bytes(kDir + "golden/janus_apple.png"));
    REQUIRE(png.w >= kCropW);
    REQUIRE(png.h >= kCropH);
    std::vector<std::uint8_t> rgb(std::size_t{kCropH} * kCropW * 3);
    for (std::uint32_t y = 0; y < kCropH; ++y) {
        std::memcpy(&rgb[std::size_t{y} * kCropW * 3],
                    &png.rgb[std::size_t{y} * png.w * 3], kCropW * 3);
    }
    const std::uint32_t gh = kCropH / 14, gw = kCropW / 14;
    const auto px = v.patches(rgb, kCropH, kCropW);
    REQUIRE(px.size() == std::size_t{gh} * gw * pd);
    for (std::size_t i = 0; i < np; ++i) {
        for (std::size_t e = 0; e < pd; ++e) {
            INFO("patch " << kPatches[i] << " value " << e);
            REQUIRE(std::abs(px[kPatches[i] * pd + e] - ref[i * pd + e]) <=
                    1e-6f);
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto out =
        v.encode(px, gh, gw, locus::backend::best_backend().ops);
    const double s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
    WARN("encode " << gh * gw << " patches: " << s << " s");
    REQUIRE(out.size() == std::size_t{gh} * gw / 4 * v.out_dim());
    std::vector<float> rows;
    for (std::uint32_t r = 0; r < kRows; ++r) {
        const float* row = &out[std::size_t{r} * kRowStep * v.out_dim()];
        rows.insert(rows.end(), row, row + v.out_dim());
    }
    // 2e-5 on an M2 Pro (neon).
    check_close(rows,
                std::span<const float>(ref).subspan(
                    np * pd, std::size_t{kRows} * v.out_dim()),
                2e-4);
    double ss = 0.0;
    for (float x : out) ss += double{x} * x;
    const double rms = std::sqrt(ss / static_cast<double>(out.size()));
    REQUIRE(std::abs(rms - ref.back()) <= 2e-4 * ref.back());
}
