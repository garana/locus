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
#include "locus/backend/variants.hpp"
#include "locus/image/ltxv_pipeline.hpp"

using locus::image::LtxvPipeline;
namespace fs = std::filesystem;

namespace {

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

std::vector<std::uint8_t> read_u8(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    return {std::istreambuf_iterator<char>(f),
            std::istreambuf_iterator<char>()};
}

nlohmann::json read_json(const std::string& path) {
    std::ifstream f(path);
    REQUIRE(f.good());
    return nlohmann::json::parse(f);
}

/** @returns max |got - ref| over the RMS of ref. */
double rel_error(std::span<const float> got, std::span<const float> ref) {
    REQUIRE(got.size() == ref.size());
    double ss = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        ss += double{ref[i]} * ref[i];
        worst = std::max(worst, std::fabs(double{got[i]} - ref[i]));
    }
    return worst / std::sqrt(ss / static_cast<double>(ref.size()));
}

/** @returns The PSNR of got against ref, 8-bit samples, in dB. */
double psnr(std::span<const std::uint8_t> got,
            std::span<const std::uint8_t> ref) {
    REQUIRE(got.size() == ref.size());
    double se = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(got[i]) - ref[i];
        se += d * d;
    }
    if (se == 0.0) return 99.0;
    return 10.0 * std::log10(255.0 * 255.0 * ref.size() / se);
}

fs::path temp_dir(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / name;
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

}  // namespace

// rf.py's linear_quadratic_schedule(8): half linear to 0.025, half
// quadratic; these round (4 places) to the checkpoint's
// allowed_inference_steps.
TEST_CASE("ltxv schedule is the reference's linear-quadratic steps",
          "[ltxv][image]") {
    const auto s = LtxvPipeline::schedule(8);
    const std::vector<double> want = {1.0,   0.99375, 0.9875,   0.98125,
                                      0.975, 0.909375, 0.725, 0.421875};
    REQUIRE(s.size() == want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
        INFO("step " << i);
        REQUIRE(s[i] == Catch::Approx(want[i]).margin(1e-6));
    }
    REQUIRE(LtxvPipeline::schedule(1) == std::vector<float>{1.0f});
}

// latent_to_pixel_coords without causal_fix, then the time / fps.
TEST_CASE("ltxv token coordinates scale by 8 32 32", "[ltxv][image]") {
    const auto c = LtxvPipeline::coords(2, 2, 3, 30.0f);
    REQUIRE(c.size() == 3 * 12);
    // Token 11 is frame 1, row 1, column 2.
    REQUIRE(c[11] == static_cast<float>(8) * static_cast<float>(1.0 / 30.0));
    REQUIRE(c[12 + 11] == 32.0f);
    REQUIRE(c[24 + 11] == 64.0f);
    REQUIRE(c[0] == 0.0f);
}

TEST_CASE("ltxv requests are checked", "[ltxv][image]") {
    LtxvPipeline::Request r;
    REQUIRE_NOTHROW(LtxvPipeline::check(r));
    auto bad = r;
    bad.width = 500;
    REQUIRE_THROWS_AS(LtxvPipeline::check(bad), std::invalid_argument);
    bad = r;
    bad.height = 1312;
    REQUIRE_THROWS_AS(LtxvPipeline::check(bad), std::invalid_argument);
    bad = r;
    bad.frames = 48;
    REQUIRE_THROWS_AS(LtxvPipeline::check(bad), std::invalid_argument);
    bad.frames = 265;
    REQUIRE_THROWS_AS(LtxvPipeline::check(bad), std::invalid_argument);
    bad = r;
    bad.frame_rate = 0.0f;
    REQUIRE_THROWS_AS(LtxvPipeline::check(bad), std::invalid_argument);
    REQUIRE(LtxvPipeline::preprocess("  A Red Fox.\n") == "A Red Fox.");
}

// The whole pipeline on the real models against the reference
// LTXVideoPipeline (scripts/make-ltxv-pipeline-goldens.py): the T5 ids
// of the prompt, then the 8 steps from the reference's start noise to
// its latent (the exit check: latent-close parity). With
// LOCUS_HEAVY_TESTS=1 also the decode, against frames 0, 8 and 16.
TEST_CASE("ltxv pipeline matches the reference clip",
          "[ltxv][image][e2e]") {
    const fs::path models = kDir + "models";
    const char* parts[] = {"t5-xxl-enc-f16.gguf", "ltxv-2b-dit.gguf",
                           "ltxv-2b-vae.gguf"};
    for (const char* p : parts) {
        if (!fs::exists(models / p)) {
            SKIP(std::string(p) + " absent");
        }
    }
    const auto d = temp_dir("locus_ltxv_models");
    for (const char* p : parts) {
        fs::create_symlink(fs::absolute(models / p), d / p);
    }
    const auto pipe = LtxvPipeline::load_dir(d.string());
    const auto j = read_json(kDir + "golden/ltxv/pipeline.json");
    LtxvPipeline::Request req;
    req.prompt = j["prompt"].get<std::string>();
    req.width = j["width"].get<std::uint32_t>();
    req.height = j["height"].get<std::uint32_t>();
    req.frames = j["frames"].get<std::uint32_t>();
    req.frame_rate = j["fps"].get<float>();
    const auto ids = LtxvPipeline::prompt_ids(pipe->tokenizer(), req.prompt);
    REQUIRE(ids == j["ids"].get<std::vector<locus::tok::TokenId>>());
    const auto steps = LtxvPipeline::schedule(8);
    for (std::size_t i = 0; i < steps.size(); ++i) {
        REQUIRE(steps[i] == j["timesteps"][i].get<float>());
    }
    const auto noise = read_f32(kDir + "golden/ltxv/pipeline_noise.f32");
    const auto ref = read_f32(kDir + "golden/ltxv/pipeline_latent.f32");
    // The server's choice: the GPU when the Vulkan device imports host
    // memory, else the CPU.
    const bool gpu = locus::backend::vulkan_imports_host_memory();
    const auto& op = gpu ? locus::backend::vulkan_image_ops()
                         : locus::backend::best_backend().ops;
    const auto x = pipe->denoise(noise, req, op, {});
    const double err = rel_error(x, ref);
    CAPTURE(gpu, err);
    REQUIRE(err <= 1e-2);
    fs::remove_all(d);

    if (std::getenv("LOCUS_HEAVY_TESTS") == nullptr) return;
    const auto dnoise =
        read_f32(kDir + "golden/ltxv/pipeline_decode_noise.f32");
    const auto video = pipe->decode(ref, dnoise, req);
    const auto frames = read_u8(kDir + "golden/ltxv/pipeline_frames.u8");
    const std::size_t plane = std::size_t{req.height} * req.width;
    REQUIRE(frames.size() == 3 * plane * 3);
    const auto rgb_frames = LtxvPipeline::to_frames(video, req);
    REQUIRE(rgb_frames.size() == req.frames);
    const std::uint32_t idx[] = {0, 8, 16};
    for (int k = 0; k < 3; ++k) {
        const auto& rgb = rgb_frames[idx[k]];
        const double db = psnr(rgb, std::span<const std::uint8_t>(
                                        &frames[k * 3 * plane], 3 * plane));
        INFO("frame " << idx[k]);
        CAPTURE(db);
        REQUIRE(db >= 40.0);
    }
}
