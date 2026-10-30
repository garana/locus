#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/flux_pipeline.hpp"

using locus::image::FluxPipeline;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModels = kDir + "models/flux";

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

bool has_all_parts() {
    for (const char* f : {"flux-kontext-dev-bf16.gguf", "t5-xxl-enc-bf16.gguf",
                          "clip-l.gguf", "flux-vae.gguf"}) {
        if (!std::filesystem::exists(kModels + "/" + f)) return false;
    }
    return true;
}

}  // namespace

TEST_CASE("flux pipeline needs all four parts", "[flux][image]") {
    const auto tmp = std::filesystem::temp_directory_path() /
                     ("locus-flux-parts-" + std::to_string(::getpid()));
    std::filesystem::create_directories(tmp);
    std::filesystem::copy_file(
        kDir + "golden/flux_synth.gguf", tmp / "flux_synth.gguf",
        std::filesystem::copy_options::overwrite_existing);
    REQUIRE_THROWS_AS(FluxPipeline::load_dir(tmp.string()),
                      locus::gguf::Error);
    std::filesystem::remove_all(tmp);
}

TEST_CASE("flux pipeline validates a request", "[flux][image]") {
    FluxPipeline::Request r;
    r.width = r.height = 256;
    REQUIRE_NOTHROW(FluxPipeline::validate(r));
    for (const auto& [w, h] : {std::pair<std::uint32_t, std::uint32_t>{
                                   9999, 9999},
                               {0, 16},
                               {17, 16},
                               {2048, 1024},
                               {1024, 1040}}) {
        INFO(w << " x " << h);
        r.width = w;
        r.height = h;
        REQUIRE_THROWS_AS(FluxPipeline::validate(r), std::invalid_argument);
    }
    r.width = r.height = 256;
    r.max_sequence_length = 513;
    REQUIRE_THROWS_AS(FluxPipeline::validate(r), std::invalid_argument);
    r.max_sequence_length = 512;
    r.guidance = -1.0f;
    REQUIRE_THROWS_AS(FluxPipeline::validate(r), std::invalid_argument);
    r.guidance = std::nanf("");
    REQUIRE_THROWS_AS(FluxPipeline::validate(r), std::invalid_argument);
}

// FLUX.1 Kontext dev as plain text to image, 128 x 128 in 2 steps from
// the reference's start latent: CLIP-L, T5-XXL (64 tokens), the loop and
// the VAE, against FluxPipeline's parts in F32 (the transformer one
// block at a time).
TEST_CASE("flux pipeline matches diffusers", "[flux][image][e2e]") {
    if (!has_all_parts()) {
        SKIP("needs the four FLUX GGUFs in " << kModels);
    }
    const auto p = FluxPipeline::load_dir(kModels);
    std::ifstream f(kDir + "golden/flux/pipeline.json");
    REQUIRE(f.good());
    const auto j = nlohmann::json::parse(f);
    FluxPipeline::Request req;
    req.prompt = j["prompt"];
    req.width = j["width"];
    req.height = j["height"];
    req.steps = j["steps"];
    req.guidance = j["guidance"];
    req.max_sequence_length = j["max_sequence_length"];
    const std::vector<locus::tok::TokenId> t5_ids = j["t5_ids"];
    REQUIRE(p->t5_ids(req.prompt, req.max_sequence_length) == t5_ids);
    const auto latent = read_f32(kDir + "golden/flux/pipeline_in.f32");
    const auto img = p->generate_from(
        latent, req, locus::backend::best_backend().ops, nullptr);
    const auto ref = read_f32(kDir + "golden/flux/pipeline_out.f32");
    REQUIRE(img.size() == ref.size());
    double worst = 0.0, ss = 0.0;
    for (std::size_t i = 0; i < img.size(); ++i) {
        const double d = double{img[i]} - ref[i];
        worst = std::max(worst, std::fabs(d));
        ss += d * d;
    }
    INFO("max " << worst << " rms " << std::sqrt(ss / img.size()));
    // The image is in about [-1, 1]; 2 / 255 is one 8-bit level.
    REQUIRE(worst < 2.0 / 255.0);
}
