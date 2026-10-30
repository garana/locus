#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/flux_sampler.hpp"
#include "locus/image/sd_vae.hpp"

using locus::image::FluxSampler;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kVae = kDir + "models/flux/flux-vae.gguf";

/** A GGUF image with only the flux.sampler keys. */
std::vector<std::byte> sampler_gguf(bool dynamic, double shift) {
    GgufBuilder g;
    g.header(0, 7);
    g.kv_string("general.architecture", "flux");
    g.kv_bool("flux.sampler.use_dynamic_shifting", dynamic);
    g.kv_f64("flux.sampler.shift", shift);
    g.kv_f64("flux.sampler.base_shift", 0.5);
    g.kv_f64("flux.sampler.max_shift", 1.15);
    g.kv_u32("flux.sampler.base_image_seq_len", 256);
    g.kv_u32("flux.sampler.max_image_seq_len", 4096);
    g.pad();
    auto span = g.bytes();
    return {span.begin(), span.end()};
}

nlohmann::json golden() {
    std::ifstream f(kDir + "golden/flux/sampler.json");
    REQUIRE(f.good());
    return nlohmann::json::parse(f);
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

// The schedules of FluxPipeline's scheduler setup, for Kontext dev
// (dynamic shift) and schnell (static shift 1), as float32.
TEST_CASE("flux sigmas match the scheduler", "[flux][image]") {
    const auto dyn_img = sampler_gguf(true, 3.0);
    const auto stat_img = sampler_gguf(false, 1.0);
    const auto dyn = FluxSampler::load(locus::gguf::GgufFile::parse(dyn_img));
    const auto stat =
        FluxSampler::load(locus::gguf::GgufFile::parse(stat_img));
    const auto j = golden();
    int cases = 0;
    for (const auto& c : j["cases"]) {
        const std::string model = c["model"];
        const std::uint32_t steps = c["steps"], tokens = c["tokens"];
        INFO(model << " " << steps << " steps, " << tokens << " tokens");
        const auto& s = model == "schnell" ? stat : dyn;
        const auto got = s.sigmas(steps, tokens);
        const std::vector<float> want = c["sigmas"];
        REQUIRE(got.size() == want.size());
        for (std::size_t i = 0; i < got.size(); ++i) {
            REQUIRE(got[i] == want[i]);
        }
        // The transformer sees each timestep (sigma * 1000) / 1000.
        const std::vector<float> ts = c["timesteps"];
        for (std::size_t i = 0; i < ts.size(); ++i) {
            REQUIRE(FluxSampler::model_sigma(got[i]) == ts[i] / 1000.0f);
        }
        ++cases;
    }
    REQUIRE(cases == 6);
}

TEST_CASE("flux latent packing matches the pipeline", "[flux][image]") {
    const auto j = golden();
    const std::vector<float> in = j["pack_in"], want = j["pack_out"];
    const auto packed = FluxSampler::pack(in, 16, 6, 8);
    REQUIRE(packed == want);
    REQUIRE(FluxSampler::unpack(packed, 16, 6, 8) == in);
    const std::vector<float> ids = j["ids"];
    REQUIRE(FluxSampler::image_ids(3, 4, 0.0f) == ids);
    const auto ref = FluxSampler::image_ids(1, 2, 1.0f);
    REQUIRE(ref == std::vector<float>{1, 0, 0, 1, 0, 1});
    REQUIRE_THROWS_AS(FluxSampler::pack(in, 16, 5, 8), std::invalid_argument);
}

TEST_CASE("flux euler step moves by the sigma gap", "[flux][image]") {
    std::vector<float> x = {1.0f, -2.0f}, v = {0.5f, 4.0f};
    FluxSampler::step(x, v, 0.75f, 0.5f);
    REQUIRE(x[0] == Catch::Approx(1.0f - 0.125f));
    REQUIRE(x[1] == Catch::Approx(-2.0f - 1.0f));
}

// FLUX.1's VAE decoder: a 16-channel latent, the shift after the scale,
// no post-quant conv.
TEST_CASE("flux vae decoder matches diffusers", "[flux][image]") {
    if (!std::filesystem::exists(kVae)) {
        SKIP("no " << kVae << " (scripts/convert-flux.py --vae-out)");
    }
    const auto g = locus::gguf::GgufFile::open(kVae);
    const auto vae = locus::image::SdVaeDecoder::load(g);
    REQUIRE(vae.latent_channels() == 16);
    REQUIRE(vae.upscale() == 8);
    const auto z = read_f32(kDir + "golden/flux/vae_in.f32");
    const auto ref = read_f32(kDir + "golden/flux/vae_out.f32");
    const auto img = vae.decode(z, 16, 16, locus::backend::best_backend().ops);
    REQUIRE(img.size() == ref.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < img.size(); ++i) {
        worst = std::max(worst, std::fabs(double{img[i]} - ref[i]));
    }
    REQUIRE(worst < 1e-3);
}
