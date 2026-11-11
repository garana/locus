#include "locus/image/resnet.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/image_input.hpp"
#include "png_parse.hpp"

namespace {

using locus::image::ResNet;

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/resnet50-tv-f32.gguf";

// Keep in sync with scripts/make-cnn-goldens.py.
constexpr std::uint32_t kRows[] = {0, 111, 223};
constexpr std::uint32_t kSize = 224, kClasses = 1000, kFeat = 2048;

std::vector<std::byte> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> c((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<std::byte> b(c.size());
    std::memcpy(b.data(), c.data(), c.size());
    return b;
}

/** @returns The RMS of got - ref; REQUIREs it within rel of ref's RMS. */
double check_close(std::span<const float> got, std::span<const float> ref,
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
    return err;
}

}  // namespace

TEST_CASE("resnet top k is the highest softmax probabilities",
          "[resnet][image]") {
    const float logits[] = {1.0f, 3.0f, 2.0f, 3.0f};
    const auto t = ResNet::top_k(logits, 3);
    REQUIRE(t.size() == 3);
    // A tie keeps the lower class first.
    CHECK(t[0].first == 1);
    CHECK(t[1].first == 3);
    CHECK(t[2].first == 2);
    double z = 0.0;
    for (float v : logits) z += std::exp(double{v});
    CHECK(t[0].second == Catch::Approx(std::exp(3.0) / z));
    CHECK(ResNet::top_k(logits, 9).size() == 4);
}

// i#575: ResNet-50 (timm's resnet50.tv_in1k, torchvision's weights,
// batch norms folded by scripts/convert-cnn.py) on janus_apple.png
// against timm in float32 (scripts/make-cnn-goldens.py): the eval
// transform's output on three rows of each channel, the 1000 logits,
// the top 5 and the 2048-wide pooled embedding.
TEST_CASE("resnet-50 matches timm on an image", "[resnet][image]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("resnet50-tv-f32.gguf absent; see scripts/convert-cnn.py");
    }
    const auto ref_b = read_bytes(kDir + "golden/resnet50_apple.f32");
    std::vector<float> ref(ref_b.size() / 4);
    std::memcpy(ref.data(), ref_b.data(), ref.size() * 4);
    const std::size_t nrows = std::size(kRows) * 3 * kSize;
    REQUIRE(ref.size() == nrows + kClasses + kFeat);

    auto g = locus::gguf::GgufFile::open(kModel);
    const auto net = ResNet::load(g);
    REQUIRE(net.input_size() == kSize);
    REQUIRE(net.num_classes() == kClasses);
    REQUIRE(net.embedding_size() == kFeat);
    REQUIRE(net.label(948) == "Granny Smith");

    const auto png = png_test::parse_stored_png(
        read_bytes(kDir + "golden/janus_apple.png"));
    const locus::image::Rgb8 img{png.w, png.h, png.rgb};
    const auto chw = net.preprocess(img);
    REQUIRE(chw.size() == std::size_t{3} * kSize * kSize);
    // Pillow's bilinear is ported bit-exact, so the transform matches
    // up to float rounding.
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
    WARN("resnet-50 forward: "
         << std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0)
                .count()
         << " s");
    const std::span<const float> ref_logits(ref.data() + nrows, kClasses);
    const std::span<const float> ref_emb(ref.data() + nrows + kClasses,
                                         kFeat);
    check_close(out.logits, ref_logits, 1e-5, "logits");
    check_close(out.embedding, ref_emb, 1e-5, "embedding");
    const auto got5 = ResNet::top_k(out.logits, 5);
    const auto want5 = ResNet::top_k(ref_logits, 5);
    for (std::size_t k = 0; k < 5; ++k) {
        INFO("rank " << k);
        CHECK(got5[k].first == want5[k].first);
        CHECK(got5[k].second == Catch::Approx(want5[k].second).epsilon(1e-4));
    }
    CHECK(net.label(got5[0].first) == "Granny Smith");
}

TEST_CASE("resnet rejects a wrong input size", "[resnet][image]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("resnet50-tv-f32.gguf absent; see scripts/convert-cnn.py");
    }
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto net = ResNet::load(g);
    std::vector<float> small(3 * 10 * 10);
    REQUIRE_THROWS_AS(net.run(small), std::invalid_argument);
    REQUIRE_THROWS_AS(net.preprocess(locus::image::Rgb8{}),
                      std::invalid_argument);
}
