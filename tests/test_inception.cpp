#include "locus/image/inception.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/cnn.hpp"
#include "locus/image/image_input.hpp"
#include "png_parse.hpp"

namespace {

using locus::image::Inception;

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/inception-v3-tv-f32.gguf";

// Keep in sync with scripts/make-cnn-goldens.py.
constexpr std::uint32_t kRows[] = {0, 149, 298};
constexpr std::uint32_t kSize = 299, kClasses = 1000, kFeat = 2048;

std::vector<std::byte> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> c((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<std::byte> b(c.size());
    std::memcpy(b.data(), c.data(), c.size());
    return b;
}

/** REQUIREs the RMS of got - ref within rel of ref's RMS. */
void check_close(std::span<const float> got, std::span<const float> ref,
                 double rel, const char* what) {
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
    WARN(what << ": error rms " << err << ", worst " << worst
              << ", reference rms " << norm);
    REQUIRE(err <= rel * norm);
}

}  // namespace

// i#576: Inception-v3 (timm's inception_v3.tv_in1k, torchvision's
// weights, batch norms folded by scripts/convert-cnn.py) on
// janus_apple.png against timm in float32 (scripts/make-cnn-goldens.py):
// the eval transform (bicubic, 299) on three rows of each channel, the
// 1000 logits, the top 5 and the 2048-wide pooled embedding.
TEST_CASE("inception-v3 matches timm on an image", "[inception][image]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("inception-v3-tv-f32.gguf absent; see scripts/convert-cnn.py");
    }
    const auto ref_b = read_bytes(kDir + "golden/inception_v3_apple.f32");
    std::vector<float> ref(ref_b.size() / 4);
    std::memcpy(ref.data(), ref_b.data(), ref.size() * 4);
    const std::size_t nrows = std::size(kRows) * 3 * kSize;
    REQUIRE(ref.size() == nrows + kClasses + kFeat);

    auto g = locus::gguf::GgufFile::open(kModel);
    const auto net = Inception::load(g);
    REQUIRE(net.input_size() == kSize);
    REQUIRE(net.num_classes() == kClasses);

    const auto png = png_test::parse_stored_png(
        read_bytes(kDir + "golden/janus_apple.png"));
    const locus::image::Rgb8 img{png.w, png.h, png.rgb};
    const auto chw = net.preprocess(img);
    REQUIRE(chw.size() == std::size_t{3} * kSize * kSize);
    std::size_t i = 0;
    for (std::uint32_t c = 0; c < 3; ++c) {
        for (std::uint32_t r : kRows) {
            for (std::uint32_t x = 0; x < kSize; ++x, ++i) {
                INFO("channel " << c << " row " << r << " col " << x);
                REQUIRE(chw[(std::size_t{c} * kSize + r) * kSize + x] ==
                        Catch::Approx(ref[i]).margin(1e-6));
            }
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto out = net.run(chw);
    WARN("inception-v3 forward: "
         << std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0)
                .count()
         << " s");
    const std::span<const float> ref_logits(ref.data() + nrows, kClasses);
    const std::span<const float> ref_emb(ref.data() + nrows + kClasses,
                                         kFeat);
    check_close(out.logits, ref_logits, 1e-5, "logits");
    check_close(out.embedding, ref_emb, 1e-5, "embedding");
    const auto got5 = locus::image::cnn::top_k(out.logits, 5);
    const auto want5 = locus::image::cnn::top_k(ref_logits, 5);
    for (std::size_t k = 0; k < 5; ++k) {
        INFO("rank " << k);
        CHECK(got5[k].first == want5[k].first);
        CHECK(got5[k].second == Catch::Approx(want5[k].second).epsilon(1e-4));
    }
}

// i#578: the convs on Vulkan (conv2d_vulkan, the 1 x 7 and 7 x 1 ones
// on host-padded inputs) against the CPU path on the same image.
TEST_CASE("inception-v3 on vulkan matches the cpu",
          "[inception][image][vulkan]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("inception-v3-tv-f32.gguf absent; see scripts/convert-cnn.py");
    }
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device");
    }
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto net = Inception::load(g);
    const auto png = png_test::parse_stored_png(
        read_bytes(kDir + "golden/janus_apple.png"));
    const auto chw = net.preprocess({png.w, png.h, png.rgb});
    const auto cpu = net.run(chw, false);
    const auto t0 = std::chrono::steady_clock::now();
    const auto gpu = net.run(chw, true);
    WARN("inception-v3 forward on vulkan: "
         << std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0)
                .count()
         << " s");
    check_close(gpu.logits, cpu.logits, 1e-5, "vulkan logits");
    check_close(gpu.embedding, cpu.embedding, 1e-5, "vulkan embedding");
    const auto a = locus::image::cnn::top_k(gpu.logits, 5);
    const auto b = locus::image::cnn::top_k(cpu.logits, 5);
    for (std::size_t k = 0; k < 5; ++k) CHECK(a[k].first == b[k].first);
}

TEST_CASE("inception rejects a wrong input size", "[inception][image]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("inception-v3-tv-f32.gguf absent; see scripts/convert-cnn.py");
    }
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto net = Inception::load(g);
    std::vector<float> small(3 * 10 * 10);
    REQUIRE_THROWS_AS(net.run(small), std::invalid_argument);
}
