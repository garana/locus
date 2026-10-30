#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/flux_dit.hpp"

using locus::image::FluxDit;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/flux/flux-kontext-dev-bf16.gguf";

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

/** @returns The ids array of j[key] flattened to n x 3 floats. */
std::vector<float> ids_of(const nlohmann::json& j, const char* key) {
    std::vector<float> out;
    for (const auto& row : j[key]) {
        for (const auto& v : row) out.push_back(v.get<float>());
    }
    return out;
}

/**
 * @returns The largest difference between got and ref over n rows of w,
 *     over the RMS of ref (one number for the whole tensor).
 */
double rel_error(const float* got, const float* ref, std::size_t count) {
    double ss = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        ss += double{ref[i]} * ref[i];
        worst = std::max(worst, std::fabs(double{got[i]} - ref[i]));
    }
    return worst / std::sqrt(ss / static_cast<double>(count));
}

}  // namespace

// The rotation angles of FluxPosEmbed for the 4 / 6 / 6 axes: axis a
// pair j turns by id_a * 10000^(-2 j / d_a), repeated for both members.
TEST_CASE("flux rope angles follow the three axes", "[flux][image]") {
    const auto g = locus::gguf::GgufFile::open(kDir + "golden/flux_synth.gguf");
    const auto dit = FluxDit::load(g);
    const std::vector<float> ids = {1.0f, 2.0f, 3.0f};
    std::vector<float> c, s;
    dit.rope(ids, 1, c, s);
    REQUIRE(c.size() == 16);
    const unsigned axes[3] = {4, 6, 6};
    unsigned off = 0;
    for (int a = 0; a < 3; ++a) {
        for (unsigned j = 0; j < axes[a] / 2; ++j) {
            const double ang =
                ids[static_cast<std::size_t>(a)] *
                std::pow(10000.0, -2.0 * j / axes[a]);
            for (unsigned m = 0; m < 2; ++m) {
                REQUIRE(c[off + 2 * j + m] ==
                        Catch::Approx(std::cos(ang)).margin(1e-7));
                REQUIRE(s[off + 2 * j + m] ==
                        Catch::Approx(std::sin(ang)).margin(1e-7));
            }
        }
        off += axes[a];
    }
}

TEST_CASE("flux dit matches diffusers on a tiny model", "[flux][image]") {
    const auto g = locus::gguf::GgufFile::open(kDir + "golden/flux_synth.gguf");
    const auto dit = FluxDit::load(g);
    REQUIRE(dit.double_blocks() == 2);
    REQUIRE(dit.single_blocks() == 2);
    REQUIRE(dit.guidance_embeds());
    const auto j = read_json(kDir + "golden/flux_synth.json");
    const std::uint32_t n_img = j["n_img"], n_txt = j["n_txt"];
    const auto in = read_f32(kDir + "golden/flux_synth_in.f32");
    const std::size_t nx = std::size_t{n_img} * dit.channels();
    const std::size_t nt = std::size_t{n_txt} * dit.text_width();
    REQUIRE(in.size() == nx + nt + dit.pooled_width());
    const std::span<const float> all(in);
    const auto out = dit.forward(
        all.subspan(0, nx), ids_of(j, "img_ids"), n_img,
        all.subspan(nx, nt), ids_of(j, "txt_ids"), n_txt,
        all.subspan(nx + nt), j["sigma"].get<float>(),
        j["guidance"].get<float>(), locus::backend::best_backend().ops);
    const auto ref = read_f32(kDir + "golden/flux_synth_out.f32");
    REQUIRE(out.size() == ref.size());
    REQUIRE(rel_error(out.data(), ref.data(), ref.size()) < 1e-4);
}

TEST_CASE("flux dit rejects wrong sizes", "[flux][image]") {
    const auto g = locus::gguf::GgufFile::open(kDir + "golden/flux_synth.gguf");
    const auto dit = FluxDit::load(g);
    const auto& op = locus::backend::best_backend().ops;
    std::vector<float> x(dit.channels()), ids(3), t(dit.text_width()),
        p(dit.pooled_width());
    REQUIRE_NOTHROW(dit.forward(x, ids, 1, t, ids, 1, p, 0.5f, 1.0f, op));
    std::vector<float> short_x(dit.channels() - 1);
    REQUIRE_THROWS_AS(dit.forward(short_x, ids, 1, t, ids, 1, p, 0.5f, 1.0f,
                                  op),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(dit.forward(x, ids, 1, t, ids, 0, p, 0.5f, 1.0f, op),
                      std::invalid_argument);
    std::vector<float> short_p(dit.pooled_width() - 1);
    REQUIRE_THROWS_AS(dit.forward(x, ids, 1, t, ids, 1, short_p, 0.5f,
                                  1.0f, op),
                      std::invalid_argument);
}

// FLUX.1 Kontext dev at full size: 64 image, 16 reference and 16 text
// tokens through all 57 blocks (BF16 weights, F32 activations).
TEST_CASE("flux kontext dit matches diffusers", "[flux][image][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("no " << kModel << " (scripts/convert-flux.py --dit-out)");
    }
    const auto g = locus::gguf::GgufFile::open(kModel);
    const auto dit = FluxDit::load(g);
    REQUIRE(dit.double_blocks() == 19);
    REQUIRE(dit.single_blocks() == 38);
    const std::string d = kDir + "golden/flux/";
    const auto j = read_json(d + "dit.json");
    const std::uint32_t n_img = j["n_img"], n_txt = j["n_txt"];
    const auto in = read_f32(d + "dit_in.f32");
    const std::size_t nx = std::size_t{n_img} * dit.channels();
    const std::size_t nt = std::size_t{n_txt} * dit.text_width();
    REQUIRE(in.size() == nx + nt + dit.pooled_width());
    const std::span<const float> all(in);
    FluxDit::Trace trace;
    const auto out = dit.forward(
        all.subspan(0, nx), ids_of(j, "img_ids"), n_img,
        all.subspan(nx, nt), ids_of(j, "txt_ids"), n_txt,
        all.subspan(nx + nt), j["sigma"].get<float>(),
        j["guidance"].get<float>(), locus::backend::best_backend().ops,
        &trace);
    const auto tr = read_f32(d + "dit_trace.f32");
    const std::size_t w = dit.width();
    const std::size_t a = n_img * w, b = n_txt * w, c = (n_img + n_txt) * w;
    REQUIRE(tr.size() == a + b + c);
    REQUIRE(rel_error(trace.double0_img.data(), tr.data(), a) < 1e-3);
    REQUIRE(rel_error(trace.double0_txt.data(), tr.data() + a, b) < 1e-3);
    REQUIRE(rel_error(trace.single0.data(), tr.data() + a + b, c) < 1e-3);
    const auto ref = read_f32(d + "dit_out.f32");
    REQUIRE(out.size() == ref.size());
    // On the M2 Pro: 3.4e-5, 4.4e-5 and 1.5e-4 above, 8.5e-5 here.
    REQUIRE(rel_error(out.data(), ref.data(), ref.size()) < 1e-3);
}
