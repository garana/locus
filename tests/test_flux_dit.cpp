#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/flux_dit.hpp"

using locus::image::FluxDit;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/flux/flux-kontext-dev-bf16.gguf";
const std::string kModelQ8 =
    kDir + "models/flux-q8/flux-kontext-dev-q8_0.gguf";

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

/** @returns The RMS of got - ref over the RMS of ref. */
double rms_error(const float* got, const float* ref, std::size_t count) {
    double ss = 0.0, dd = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        ss += double{ref[i]} * ref[i];
        const double e = double{got[i]} - ref[i];
        dd += e * e;
    }
    return std::sqrt(dd / ss);
}

struct Leg {
    const char* name;
    const locus::backend::Ops* op;
};

/** The CPU, plus the whole-step Vulkan forward when there is a device. */
std::vector<Leg> legs() {
    std::vector<Leg> v = {{"cpu", &locus::backend::best_backend().ops}};
    if (locus::backend::vk::VulkanContext::available()) {
        v.push_back({"vulkan", &locus::backend::vulkan_image_ops()});
    }
    return v;
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
    const auto ref = read_f32(kDir + "golden/flux_synth_out.f32");
    for (const Leg& leg : legs()) {
        DYNAMIC_SECTION(leg.name) {
            FluxDit::Trace trace;
            const auto out = dit.forward(
                all.subspan(0, nx), ids_of(j, "img_ids"), n_img,
                all.subspan(nx, nt), ids_of(j, "txt_ids"), n_txt,
                all.subspan(nx + nt), j["sigma"].get<float>(),
                j["guidance"].get<float>(), *leg.op, &trace);
            REQUIRE(out.size() == ref.size());
            REQUIRE(rel_error(out.data(), ref.data(), ref.size()) < 1e-4);
            const std::size_t w = dit.width();
            REQUIRE(trace.double0_img.size() == n_img * w);
            REQUIRE(trace.double0_txt.size() == n_txt * w);
            REQUIRE(trace.single0.size() == (n_img + n_txt) * w);
        }
    }
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
    const auto tr = read_f32(d + "dit_trace.f32");
    const auto ref = read_f32(d + "dit_out.f32");
    for (const Leg& leg : legs()) {
        DYNAMIC_SECTION(leg.name) {
            FluxDit::Trace trace;
            const auto out = dit.forward(
                all.subspan(0, nx), ids_of(j, "img_ids"), n_img,
                all.subspan(nx, nt), ids_of(j, "txt_ids"), n_txt,
                all.subspan(nx + nt), j["sigma"].get<float>(),
                j["guidance"].get<float>(), *leg.op, &trace);
            const std::size_t w = dit.width();
            const std::size_t a = n_img * w, b = n_txt * w,
                              c = (n_img + n_txt) * w;
            REQUIRE(tr.size() == a + b + c);
            REQUIRE(rel_error(trace.double0_img.data(), tr.data(), a) < 1e-3);
            REQUIRE(rel_error(trace.double0_txt.data(), tr.data() + a, b) <
                    1e-3);
            REQUIRE(rel_error(trace.single0.data(), tr.data() + a + b, c) <
                    1e-3);
            REQUIRE(out.size() == ref.size());
            // On the M2 Pro CPU: 3.4e-5, 4.4e-5 and 1.5e-4 above, 8.5e-5
            // here; on the M4 Max GPU: 5.9e-5, 4.6e-5, 1.4e-4 and 1.4e-4.
            REQUIRE(rel_error(out.data(), ref.data(), ref.size()) < 1e-3);
        }
    }
}

// The same check on the Q8_0 conversion (convert-flux.py --dit-type
// q8_0): the quantized weights move every block, so the bounds come from
// the measured errors, not the BF16 ones.
TEST_CASE("flux kontext q8_0 dit stays close to diffusers",
          "[flux][image][e2e]") {
    if (!std::filesystem::exists(kModelQ8)) {
        SKIP("no " << kModelQ8 << " (convert-flux.py --dit-type q8_0)");
    }
    const auto g = locus::gguf::GgufFile::open(kModelQ8);
    const auto dit = FluxDit::load(g);
    const std::string d = kDir + "golden/flux/";
    const auto j = read_json(d + "dit.json");
    const std::uint32_t n_img = j["n_img"], n_txt = j["n_txt"];
    const auto in = read_f32(d + "dit_in.f32");
    const std::size_t nx = std::size_t{n_img} * dit.channels();
    const std::size_t nt = std::size_t{n_txt} * dit.text_width();
    const std::span<const float> all(in);
    const auto tr = read_f32(d + "dit_trace.f32");
    const auto ref = read_f32(d + "dit_out.f32");
    for (const Leg& leg : legs()) {
        DYNAMIC_SECTION(leg.name) {
            FluxDit::Trace trace;
            const auto out = dit.forward(
                all.subspan(0, nx), ids_of(j, "img_ids"), n_img,
                all.subspan(nx, nt), ids_of(j, "txt_ids"), n_txt,
                all.subspan(nx + nt), j["sigma"].get<float>(),
                j["guidance"].get<float>(), *leg.op, &trace);
            const std::size_t w = dit.width();
            const std::size_t a = n_img * w, b = n_txt * w,
                              c = (n_img + n_txt) * w;
            const double e0 = rel_error(trace.double0_img.data(), tr.data(), a);
            const double e1 =
                rel_error(trace.double0_txt.data(), tr.data() + a, b);
            const double e2 =
                rel_error(trace.single0.data(), tr.data() + a + b, c);
            const double e3 = rel_error(out.data(), ref.data(), ref.size());
            // On the M4 Max (CPU and GPU alike), largest error over RMS:
            // 0.058, 0.21, 0.19, 0.42; RMS error over RMS: 0.0036,
            // 0.0071, 0.0044, 0.035. Bounds about 1.5 times those.
            REQUIRE(e0 < 0.09);
            REQUIRE(e1 < 0.32);
            REQUIRE(e2 < 0.29);
            REQUIRE(e3 < 0.63);
            REQUIRE(rms_error(trace.double0_img.data(), tr.data(), a) <
                    0.0054);
            REQUIRE(rms_error(trace.double0_txt.data(), tr.data() + a, b) <
                    0.011);
            REQUIRE(rms_error(trace.single0.data(), tr.data() + a + b, c) <
                    0.0066);
            REQUIRE(rms_error(out.data(), ref.data(), ref.size()) < 0.053);
        }
    }
}

// One denoiser call at real sizes: 512 text tokens and a 512 or 1024
// px image (1024 or 4096 image tokens), random inputs.
TEST_CASE("bench: flux dit step", "[.][bench][flux]") {
    if (!std::filesystem::exists(kModel) ||
        !locus::backend::vk::VulkanContext::available()) {
        SKIP("needs " << kModel << " and a Vulkan device");
    }
    const auto g = locus::gguf::GgufFile::open(kModel);
    const auto dit = FluxDit::load(g);
    const std::uint32_t n_txt = 512;
    for (const std::uint32_t side : {32u, 64u}) {  // latent patches
        const std::uint32_t n_img = side * side;
        std::vector<float> x(std::size_t{n_img} * dit.channels(), 0.1f);
        std::vector<float> t(std::size_t{n_txt} * dit.text_width(), 0.05f);
        std::vector<float> p(dit.pooled_width(), 0.2f);
        std::vector<float> img_ids, txt_ids(std::size_t{n_txt} * 3, 0.0f);
        for (std::uint32_t r = 0; r < side; ++r) {
            for (std::uint32_t c = 0; c < side; ++c) {
                img_ids.insert(img_ids.end(), {0.0f, float(r), float(c)});
            }
        }
        for (int rep = 0; rep < 2; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto out = dit.forward(x, img_ids, n_img, t, txt_ids,
                                         n_txt, p, 0.5f, 2.5f,
                                         locus::backend::vulkan_image_ops());
            const double s = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
            WARN(side * 16 << " px, " << n_img + n_txt << " tokens: " << s
                           << " s (rep " << rep << ")");
            REQUIRE(std::isfinite(out[0]));
        }
    }
}
