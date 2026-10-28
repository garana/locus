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
#include "locus/image/pixart_dit.hpp"
#include "locus/image/pixart_sampler.hpp"

using locus::image::DpmSolverPP;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/pixart-xl-2-512-f32.gguf";

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

// docs/models/pixart-alpha.md 5.3, and the scheduler's own values
// (scripts/make-pixart-sampler-goldens.py).
TEST_CASE("pixart sampler timesteps and sigmas match the scheduler",
          "[pixart][image]") {
    const DpmSolverPP s(8);
    REQUIRE(s.timesteps() == std::vector<std::int32_t>{999, 874, 749, 624,
                                                       500, 375, 250, 125});
    const auto j = read_json(kDir + "golden/pixart_sampler.json");
    const auto want = j["sigmas"].get<std::vector<float>>();
    REQUIRE(s.sigmas().size() == 9);
    for (std::size_t i = 0; i < want.size(); ++i) {
        INFO("sigma " << i);
        REQUIRE(s.sigmas()[i] == Catch::Approx(want[i]).epsilon(1e-6));
    }
    REQUIRE(s.sigmas().back() == 0.0f);
    // 20 steps: 999 * 19 / 20 = 949.05 -> 949; the last is 50.
    const DpmSolverPP s20(20);
    REQUIRE(s20.timesteps().front() == 999);
    REQUIRE(s20.timesteps()[1] == 949);
    REQUIRE(s20.timesteps().back() == 50);
    REQUIRE_THROWS_AS(DpmSolverPP(0), std::invalid_argument);
}

// VALUES, on every box: diffusers' DPMSolverMultistepScheduler fed the
// same noise predictions; every step's latent (first order, six
// second-order midpoint steps, the final step to sigma 0).
TEST_CASE("pixart sampler steps match the scheduler", "[pixart][image]") {
    const auto v = read_f32(kDir + "golden/pixart_sampler.f32");
    const std::size_t n = 4 * 16 * 16;
    REQUIRE(v.size() == n * 17);
    DpmSolverPP s(8);
    std::vector<float> x(v.begin(), v.begin() + n);
    for (std::uint32_t i = 0; i < 8; ++i) {
        INFO("step " << i);
        const std::span<const float> eps(&v[n * (1 + 2 * i)], n);
        const std::span<const float> want(&v[n * (2 + 2 * i)], n);
        s.step(eps, x);
        REQUIRE(rel_error(x, want) <= 1e-5);
    }
    REQUIRE(s.index() == 8);
    REQUIRE_THROWS_AS(s.step(std::span<const float>(v.data(), n), x),
                      std::logic_error);
}

// The 8-step pipeline loop on PixArt-XL-2-512x512: the DiT on both CFG
// rows per step, guidance 4.5, the sampler; from the golden start
// latent and T5 rows to the golden final latent. 16 DiT calls: about 5
// minutes on an M2 Pro CPU.
TEST_CASE("pixart sampler reproduces the reference latent",
          "[pixart][image][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("pixart-xl-2-512-f32.gguf absent; see scripts/convert-pixart.py");
    }
    if (std::getenv("LOCUS_HEAVY_TESTS") == nullptr) {
        SKIP("16 DiT calls take minutes on a CPU; set LOCUS_HEAVY_TESTS=1");
    }
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto dit = locus::image::PixArtDit::load(g);
    auto sched = DpmSolverPP::from_gguf(g, 8);
    const auto j = read_json(kDir + "golden/pixart/dit.json");
    const auto text = read_f32(kDir + "golden/pixart/dit_text.f32");
    auto x = read_f32(kDir + "golden/pixart/dit_in.f32");
    const auto ref = read_f32(kDir + "golden/pixart/latent_final.f32");
    REQUIRE(x.size() == ref.size());
    const auto rows = j["rows"].get<std::vector<std::uint32_t>>();
    const std::span<const float> neg(text.data(), rows[0] * 4096);
    const std::span<const float> pos(&text[rows[0] * 4096], rows[1] * 4096);
    const auto& op = locus::backend::best_backend().ops;
    locus::image::pixart_denoise(dit, sched, x, neg, rows[0], pos, rows[1],
                                 4.5f, op);
    const double err = rel_error(x, ref);
    double dd = 0.0, rr = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        dd += (double{x[i]} - ref[i]) * (double{x[i]} - ref[i]);
        rr += double{ref[i]} * ref[i];
    }
    const double rms_err = std::sqrt(dd / rr);
    // One DiT call is 4e-5 off; CFG scales the row difference by 4.5
    // and 8 steps compound it. Measured on macOS arm64: max 4.0e-3,
    // RMS 8.6e-5 (a wrong timestep or guidance is order 1).
    WARN("pixart latent: max diff / rms " << err << ", rms diff / rms "
                                          << rms_err);
    CHECK(err <= 2e-2);
    CHECK(rms_err <= 5e-4);
}
