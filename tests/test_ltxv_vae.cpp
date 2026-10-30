#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/ltxv_vae.hpp"

using locus::image::LtxvVae;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/ltxv-2b-vae.gguf";

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
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

}  // namespace

// conv3d_k3 against a direct six-loop sum with the reference's padding:
// zeros in space, the edge frame repeated in time.
TEST_CASE("ltxv conv3d matches a direct sum", "[ltxv][image]") {
    constexpr std::uint32_t ci = 3, co = 2, f = 4, h = 3, w = 5;
    std::mt19937 rng(357);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> in(ci * f * h * w), wt(co * ci * 27), b(co);
    for (auto& v : in) v = d(rng);
    for (auto& v : wt) v = d(rng);
    for (auto& v : b) v = d(rng);
    std::vector<float> got(co * f * h * w);
    LtxvVae::conv3d_k3(in, ci, f, h, w, wt, b, co, got);
    const auto at = [&](std::uint32_t c, int t, int y, int x) -> float {
        if (y < 0 || y >= int{h} || x < 0 || x >= int{w}) return 0.0f;
        t = std::clamp(t, 0, int{f} - 1);
        return in[((c * f + t) * h + y) * w + x];
    };
    for (std::uint32_t o = 0; o < co; ++o) {
        for (int t = 0; t < int{f}; ++t) {
            for (int y = 0; y < int{h}; ++y) {
                for (int x = 0; x < int{w}; ++x) {
                    double s = b[o];
                    for (std::uint32_t c = 0; c < ci; ++c) {
                        for (int k = 0; k < 27; ++k) {
                            s += double{wt[(o * ci + c) * 27 + k]} *
                                 at(c, t + k / 9 - 1, y + k / 3 % 3 - 1,
                                    x + k % 3 - 1);
                        }
                    }
                    INFO("o " << o << " t " << t << " y " << y << " x " << x);
                    REQUIRE(got[((o * f + t) * h + y) * w + x] ==
                            Catch::Approx(s).margin(1e-5));
                }
            }
        }
    }
    REQUIRE_THROWS_AS(LtxvVae::conv3d_k3(in, ci, f, h, w, wt, b, co + 1, got),
                      std::invalid_argument);
}

// VALUES, on every box: a tiny Decoder converted by
// scripts/convert-ltxv.py, against its torch output
// (scripts/make-ltxv-vae-goldens.py synth). Its config blocks are
// [res_x(1), compress_all(2), res_x(2)], so the up-block reversal, the
// timestep modulation, the depth-to-space with its residual and dropped
// frame, the final layer and the unpatchify are all pinned.
TEST_CASE("ltxv vae matches torch on a synthetic decoder",
          "[ltxv][image]") {
    auto g = locus::gguf::GgufFile::open(kDir + "golden/ltxv_vae_synth.gguf");
    const auto vae = LtxvVae::load(g);
    REQUIRE(vae.latent_channels() == 16);
    REQUIRE(vae.time_scale() == 2);
    REQUIRE(vae.space_scale() == 4);
    const auto j = read_json(kDir + "golden/ltxv_vae_synth.json");
    const auto in = read_f32(kDir + "golden/ltxv_vae_synth_in.f32");
    const auto ref = read_f32(kDir + "golden/ltxv_vae_synth_out.f32");
    REQUIRE(in.size() == 16 * 3 * 2 * 3);
    REQUIRE(ref.size() == 3 * 5 * 8 * 12);
    const auto got = vae.decode(in, 3, 2, 3, j["timestep"].get<float>());
    REQUIRE(rel_error(got, ref) <= 1e-5);
    REQUIRE_THROWS_AS(vae.decode(in, 3, 2, 2, 0.05f), std::invalid_argument);
}

// The 2B decoder against the reference in float32
// (scripts/make-ltxv-vae-goldens.py real): a seeded 2 x 2 x 2 latent at
// timestep 0.05 into 9 frames of 64 x 64.
TEST_CASE("ltxv vae matches the reference on the 2B decoder",
          "[ltxv][image][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("ltxv-2b-vae.gguf absent; see scripts/convert-ltxv.py");
    }
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto vae = LtxvVae::load(g);
    REQUIRE(vae.latent_channels() == 128);
    REQUIRE(vae.time_scale() == 8);
    REQUIRE(vae.space_scale() == 32);
    const auto j = read_json(kDir + "golden/ltxv/vae.json");
    const auto in = read_f32(kDir + "golden/ltxv/vae_in.f32");
    const auto ref = read_f32(kDir + "golden/ltxv/vae_out.f32");
    REQUIRE(in.size() == 128 * 8);
    REQUIRE(ref.size() == 3 * 9 * 64 * 64);
    const auto got = vae.decode(in, 2, 2, 2, j["timestep"].get<float>());
    const double err = rel_error(got, ref);
    CAPTURE(err);
    REQUIRE(err <= 1e-4);  // measured 1.0e-5
}

// The 2B encoder against the reference in float32
// (scripts/make-ltxv-vae-goldens.py encode): the normalized latent mean
// of a seeded smooth 9 x 64 x 64 video, and of its first frame alone
// (the image-to-video condition).
TEST_CASE("ltxv vae encoder matches the reference on the 2B model",
          "[ltxv][image][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("ltxv-2b-vae.gguf absent; see scripts/convert-ltxv.py");
    }
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto vae = LtxvVae::load(g);
    if (!vae.has_encoder()) {
        SKIP("ltxv-2b-vae.gguf has no encoder; reconvert it");
    }
    const auto j = read_json(kDir + "golden/ltxv/vae_enc.json");
    const auto in = read_f32(kDir + "golden/ltxv/vae_enc_in.f32");
    const auto ref = read_f32(kDir + "golden/ltxv/vae_enc_out.f32");
    std::size_t at_in = 0, at_out = 0;
    for (const char* name : {"video", "image"}) {
        INFO(name);
        const auto s = j[name]["in"].get<std::vector<std::uint32_t>>();
        const auto o = j[name]["out"].get<std::vector<std::size_t>>();
        REQUIRE(s.size() == 4);  // [3, f, h, w]
        REQUIRE(o.size() == 4);  // [128, f', h', w']
        const std::uint32_t f = s[1], h = s[2], w = s[3];
        const std::size_t n_in = std::size_t{3} * f * h * w;
        const std::size_t n_out = o[0] * o[1] * o[2] * o[3];
        REQUIRE(at_in + n_in <= in.size());
        REQUIRE(at_out + n_out <= ref.size());
        const auto got = vae.encode(
            std::span<const float>(in).subspan(at_in, n_in), f, h, w);
        const double err = rel_error(
            got, std::span<const float>(ref).subspan(at_out, n_out));
        CAPTURE(err);
        REQUIRE(err <= 1e-4);
        at_in += n_in;
        at_out += n_out;
    }
    REQUIRE(at_in == in.size());
    REQUIRE(at_out == ref.size());
}
