#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/pixart_pipeline.hpp"
#include "locus/image/png.hpp"
#include "locus/image/sd_vae.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::image::PixArtPipeline;
using locus::tok::TokenId;

namespace {

namespace fs = std::filesystem;

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

/** A fresh empty directory under the system temp dir. */
fs::path temp_dir(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / name;
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

}  // namespace

TEST_CASE("pixart preprocess strips and lowercases", "[pixart][image]") {
    REQUIRE(PixArtPipeline::preprocess("  A Small RED Apple.\n") ==
            "a small red apple.");
    REQUIRE(PixArtPipeline::preprocess("\t \n").empty());
    // Only ASCII letters change (the documented gap to Python's lower).
    REQUIRE(PixArtPipeline::preprocess("Caf\xc3\x89") == "caf\xc3\x89");
}

// The golden prompt gives the reference's t5_ids (docs/models/
// pixart-alpha.md section 2), and a long prompt keeps 119 ids plus </s>.
TEST_CASE("pixart prompt ids match the pipeline", "[pixart][image]") {
    auto g = locus::gguf::GgufFile::open(kDir + "golden/t5-vocab.gguf");
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    const auto ids = PixArtPipeline::prompt_ids(
        *tok, "A small red apple on a wooden table, soft morning light.");
    REQUIRE(ids == std::vector<TokenId>{3, 9, 422, 1131, 8947, 30, 3, 9,
                                        5726, 953, 6, 1835, 1379, 659, 5,
                                        1});
    REQUIRE(PixArtPipeline::prompt_ids(*tok, "") == std::vector<TokenId>{1});
    std::string many;
    for (int i = 0; i < 300; ++i) many += "apple ";
    const auto cut = PixArtPipeline::prompt_ids(*tok, many);
    REQUIRE(cut.size() == 120);
    REQUIRE(cut.back() == 1);
    REQUIRE(cut[118] != 1);
}

TEST_CASE("pixart start latent is a repeatable standard normal",
          "[pixart][image]") {
    const auto a = PixArtPipeline::normal_latent(146, 16384);
    const auto b = PixArtPipeline::normal_latent(146, 16384);
    const auto c = PixArtPipeline::normal_latent(147, 16384);
    REQUIRE(a == b);
    REQUIRE(a != c);
    double sum = 0.0, sq = 0.0;
    for (float v : a) {
        REQUIRE(std::isfinite(v));
        sum += v;
        sq += double{v} * v;
    }
    const double mean = sum / a.size();
    const double var = sq / a.size() - mean * mean;
    REQUIRE(std::fabs(mean) < 0.05);
    REQUIRE(std::fabs(var - 1.0) < 0.05);
    REQUIRE(PixArtPipeline::normal_latent(1, 3).size() == 3);
}

TEST_CASE("pixart pipeline needs all three parts", "[pixart][image]") {
    const auto d = temp_dir("locus_pixart_parts");
    REQUIRE_THROWS_AS(PixArtPipeline::load_dir(d.string()),
                      locus::gguf::Error);
    fs::copy_file(kDir + "golden/pixart_synth.gguf", d / "dit.gguf");
    fs::copy_file(kDir + "golden/vae_synth.gguf", d / "vae.gguf");
    REQUIRE_THROWS_AS(PixArtPipeline::load_dir(d.string()),
                      locus::gguf::Error);  // no t5encoder
    fs::copy_file(kDir + "golden/pixart_synth.gguf", d / "dit2.gguf");
    REQUIRE_THROWS_AS(PixArtPipeline::load_dir(d.string()),
                      locus::gguf::Error);  // two pixart
    REQUIRE_THROWS_AS(PixArtPipeline::load_dir((d / "none").string()),
                      locus::gguf::Error);
    fs::remove_all(d);
}

// The whole pipeline on the real models: locus's T5 on the golden
// prompt, 8 steps from the reference's start latent, the VAE; against
// the reference image (whose T5 ran in bf16) at every 4th pixel. Writes
// the PNG under the temp dir for a look. About 7 minutes on an M2 Pro
// CPU.
TEST_CASE("pixart pipeline draws the reference image",
          "[pixart][image][e2e]") {
    const fs::path models = kDir + "models";
    const char* parts[] = {"t5-xxl-enc-f16.gguf", "pixart-xl-2-512-f32.gguf",
                           "sd-vae-f32.gguf"};
    for (const char* p : parts) {
        if (!fs::exists(models / p)) {
            SKIP(std::string(p) + " absent");
        }
    }
    if (std::getenv("LOCUS_HEAVY_TESTS") == nullptr) {
        SKIP("T5, 16 DiT calls and the VAE take minutes on a CPU; set "
             "LOCUS_HEAVY_TESTS=1");
    }
    const auto d = temp_dir("locus_pixart_models");
    for (const char* p : parts) {
        fs::create_symlink(fs::absolute(models / p), d / p);
    }
    const auto pipe = PixArtPipeline::load_dir(d.string());
    REQUIRE(pipe->image_size() == 512);
    REQUIRE(pipe->default_guidance() == 4.5f);
    PixArtPipeline::Request req;
    req.prompt = "A small red apple on a wooden table, soft morning light.";
    req.steps = 8;
    const auto x = read_f32(kDir + "golden/pixart/dit_in.f32");
    const auto& op = locus::backend::best_backend().ops;
    PixArtPipeline::Timing tm;
    const auto img = pipe->generate_from(x, req, op, {}, &tm);
    REQUIRE(img.size() == std::size_t{3} * 512 * 512);
    const auto rgb = locus::image::SdVaeDecoder::to_rgb8(img, 512, 512);
    const auto png = locus::image::encode_png(rgb, 512, 512);
    const fs::path out = fs::temp_directory_path() / "locus_pixart.png";
    std::ofstream(out, std::ios::binary)
        .write(reinterpret_cast<const char*>(png.data()),
               static_cast<std::streamsize>(png.size()));

    const auto ref = read_f32(kDir + "golden/pixart/vae_image_s4.f32");
    double se = 0.0;
    for (std::uint32_t r = 0; r < 128; ++r) {
        for (std::uint32_t c = 0; c < 128; ++c) {
            for (std::uint32_t ch = 0; ch < 3; ++ch) {
                const double px =
                    rgb[(std::size_t{4 * r} * 512 + 4 * c) * 3 + ch];
                const double e =
                    px - ref[(std::size_t{r} * 128 + c) * 3 + ch];
                se += e * e;
            }
        }
    }
    const double mse = se / (128.0 * 128 * 3);
    const double psnr = 10.0 * std::log10(255.0 * 255.0 / mse);
    WARN("pixart pipeline: PSNR " << psnr << " dB vs the reference; T5 "
                                  << tm.encode << " ms, denoise "
                                  << tm.denoise << " ms, VAE " << tm.decode
                                  << " ms; image at " << out.string());
    // Measured on macOS arm64: 36.0 dB (locus's T5 runs F16 weights in
    // F32; the reference's ran in bf16).
    CHECK(psnr >= 30.0);
    fs::remove_all(d);
}
