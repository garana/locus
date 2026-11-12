#include "locus/image/qwen_image_dit.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_seq.hpp"
#include "locus/gguf/gguf.hpp"

namespace {

using locus::image::QwenImageDit;
using Segment = QwenImageDit::Segment;

const std::string kGolden = std::string(LOCUS_SOURCE_DIR) + "/tests/golden/";

// Keep in sync with scripts/make-qwen-image-dit-goldens.py.
constexpr std::uint32_t kIn = 16, kOut = 16, kJoint = 64, kTxt = 5,
                        kWidth = 64;
constexpr float kSigma = 0.7f;
const std::vector<Segment> kSegs = {{1, 4, 6}, {1, 2, 3}};
constexpr std::uint32_t kImg = 4 * 6 + 2 * 3;

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

/** The script's LCG, scaled by 4 as the inputs are. */
std::vector<float> lcg4(std::uint32_t state, std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 0.5f *
            4.0f;
    }
    return v;
}

/** REQUIREs the RMS of got - ref within rel of ref's RMS. */
void check_close(std::span<const float> got, std::span<const float> ref,
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
}

struct Leg {
    const char* name;
    const locus::backend::Ops* op;
    double bound;  ///< relative error bound against diffusers
};

/** The CPU, plus the whole-step Vulkan forward when there is a device
 *  (i#536; its sums run in another order, so FLUX's 1e-3 bound). */
std::vector<Leg> legs(double cpu_bound) {
    std::vector<Leg> v = {
        {"cpu", &locus::backend::best_backend().ops, cpu_bound}};
    if (locus::backend::VulkanSeq::available()) {
        v.push_back({"vulkan", &locus::backend::vulkan_image_ops(), 1e-3});
    }
    return v;
}

}  // namespace

// i#532 VALUES, on every box: a tiny QwenImageTransformer2DModel from
// diffusers (scripts/make-qwen-image-dit-goldens.py; its GGUF, written by
// scripts/convert-qwen-image.py, in tests/golden) on two image segments
// (frame 0 and 1, grids 4 x 6 and 2 x 3, so the centred RoPE has odd and
// even sides) and five text rows: the image and text rows after block 0,
// and the velocity, against diffusers.
TEST_CASE("qwen image dit matches diffusers on a synthetic model",
          "[qwen-image][image]") {
    const auto g =
        locus::gguf::GgufFile::open(kGolden + "qwen_image_dit_synth.gguf");
    const auto d = QwenImageDit::load(g);
    REQUIRE(d.channels() == kIn);
    REQUIRE(d.out_channels() == kOut);
    REQUIRE(d.text_width() == kJoint);
    REQUIRE(d.width() == kWidth);
    REQUIRE(d.blocks() == 2);
    const auto ref = read_f32(kGolden + "qwen_image_dit_synth.f32");
    const std::size_t ni = std::size_t{kImg} * kWidth;
    const std::size_t nt = std::size_t{kTxt} * kWidth;
    const std::size_t nv = std::size_t{kImg} * kOut;
    REQUIRE(ref.size() == ni + nt + nv);

    const auto x = lcg4(777, std::size_t{kImg} * kIn);
    const auto text = lcg4(999, std::size_t{kTxt} * kJoint);
    const std::span<const float> r(ref);
    // 1e-4 on the CPU as for FLUX: float32 sums in another order,
    // through two blocks of LCG weights (the errors are about 1e-5).
    for (const Leg& leg : legs(1e-4)) {
        DYNAMIC_SECTION(leg.name) {
            QwenImageDit::Trace trace;
            const auto v =
                d.forward(x, kSegs, text, kTxt, kSigma, *leg.op, &trace);
            check_close(trace.block0_img, r.first(ni), leg.bound);
            check_close(trace.block0_txt, r.subspan(ni, nt), leg.bound);
            check_close(v, r.subspan(ni + nt), leg.bound);
        }
    }
}

// QwenEmbedRope with scale_rope: rows and columns centred per segment,
// segment s on frame s, text after max(h / 2, w / 2).
TEST_CASE("qwen image dit rope positions", "[qwen-image][image]") {
    const auto g =
        locus::gguf::GgufFile::open(kGolden + "qwen_image_dit_synth.gguf");
    const auto d = QwenImageDit::load(g);
    std::vector<float> c, s;
    d.rope(kSegs, kTxt, c, s);
    constexpr std::uint32_t hd = 32;  // axes 8 / 12 / 12
    REQUIRE(c.size() == std::size_t{kTxt + kImg} * hd);
    // The first pair of each axis turns at the position itself (its
    // frequency is 1), so its angle is the position.
    const auto angle = [&](std::size_t t, std::uint32_t off) {
        return std::atan2(s[t * hd + off], c[t * hd + off]);
    };
    // Text j: max(4 / 2, 6 / 2, 2 / 2, 3 / 2) = 3, plus j, on all axes.
    REQUIRE(angle(0, 0) == Catch::Approx(3.0f));
    REQUIRE(angle(1, 8) == Catch::Approx(4.0f - 2.0f * 3.14159265f));
    // Segment 0, token 0: frame 0, row 0 - (4 - 2) = -2, column 0 - 3.
    REQUIRE(angle(kTxt, 0) == Catch::Approx(0.0f).margin(1e-6));
    REQUIRE(angle(kTxt, 8) == Catch::Approx(-2.0f));
    REQUIRE(angle(kTxt, 20) == Catch::Approx(-3.0f));
    // Segment 1, token 0: frame 1, row 0 - 1, column 0 - (3 - 1).
    const std::size_t t1 = kTxt + 4 * 6;
    REQUIRE(angle(t1, 0) == Catch::Approx(1.0f));
    REQUIRE(angle(t1, 8) == Catch::Approx(-1.0f));
    REQUIRE(angle(t1, 20) == Catch::Approx(-2.0f));
}

TEST_CASE("qwen image dit rejects bad sizes", "[qwen-image][image]") {
    const auto g =
        locus::gguf::GgufFile::open(kGolden + "qwen_image_dit_synth.gguf");
    const auto d = QwenImageDit::load(g);
    const auto& op = locus::backend::best_backend().ops;
    const std::vector<float> x(std::size_t{kImg} * kIn, 0.0f);
    const std::vector<float> text(std::size_t{kTxt} * kJoint, 0.0f);
    REQUIRE_THROWS_AS(d.forward(std::span(x).first(kIn), kSegs, text, kTxt,
                                0.5f, op),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(d.forward(x, kSegs, text, 0, 0.5f, op),
                      std::invalid_argument);
    const std::vector<Segment> empty = {{1, 0, 6}};
    REQUIRE_THROWS_AS(d.forward(x, empty, text, kTxt, 0.5f, op),
                      std::invalid_argument);
}

// i#532 VALUES, with the real weights: a one-block model of
// Qwen/Qwen-Image-Edit at ac7f931 (block 0 and the layers outside the
// blocks, from the shards 00001 and 00009), converted by
// scripts/convert-qwen-image.py --blocks 1 (BF16) to
// tests/models/qwen-image-edit-1block-bf16.gguf, against diffusers on
// the same weights in float32 (scripts/make-qwen-image-dit-goldens.py
// --model): block 0's rows and the velocity, at the real widths.
TEST_CASE("qwen image dit matches diffusers on a real block",
          "[qwen-image][image][e2e]") {
    const std::string path = std::string(LOCUS_SOURCE_DIR) +
                             "/tests/models/qwen-image-edit-1block-bf16.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("qwen-image-edit-1block-bf16.gguf absent; make it with "
             "scripts/convert-qwen-image.py --blocks 1");
    }
    const std::vector<Segment> segs = {{1, 2, 4}, {1, 2, 2}};
    constexpr std::uint32_t n_img = 2 * 4 + 2 * 2, n_txt = 4;
    const auto g = locus::gguf::GgufFile::open(path);
    const auto d = QwenImageDit::load(g);
    REQUIRE(d.blocks() == 1);
    REQUIRE(d.width() == 3072);
    const std::uint32_t w = d.width();
    const auto ref = read_f32(kGolden + "qwen_image_dit_real.f32");
    const std::size_t ni = std::size_t{n_img} * w, nt = std::size_t{n_txt} * w;
    const std::size_t nv = std::size_t{n_img} * d.out_channels();
    REQUIRE(ref.size() == ni + nt + nv);
    const auto x = lcg4(777, std::size_t{n_img} * d.channels());
    const auto text = lcg4(999, std::size_t{n_txt} * d.text_width());
    const std::span<const float> r(ref);
    for (const Leg& leg : legs(1e-4)) {
        DYNAMIC_SECTION(leg.name) {
            QwenImageDit::Trace trace;
            const auto v =
                d.forward(x, segs, text, n_txt, 0.5f, *leg.op, &trace);
            check_close(trace.block0_img, r.first(ni), leg.bound);
            check_close(trace.block0_txt, r.subspan(ni, nt), leg.bound);
            check_close(v, r.subspan(ni + nt), leg.bound);
        }
    }
}
