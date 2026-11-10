#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan/context.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/flux_dit.hpp"
#include "locus/image/lora.hpp"

using locus::image::FluxDit;
using locus::image::LoraSet;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/golden/";

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

/** @returns The largest |got - ref| over the RMS of ref. */
double rel_error(const std::vector<float>& got, const std::vector<float>& ref) {
    REQUIRE(got.size() == ref.size());
    double ss = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        ss += double{ref[i]} * ref[i];
        worst = std::max(worst, std::fabs(double{got[i]} - ref[i]));
    }
    return worst / std::sqrt(ss / static_cast<double>(ref.size()));
}

/** The small DiT of flux_synth.gguf and its golden inputs. */
struct Synth {
    locus::gguf::GgufFile g =
        locus::gguf::GgufFile::open(kDir + "flux_synth.gguf");
    FluxDit dit = FluxDit::load(g);
    nlohmann::json j = read_json(kDir + "flux_synth.json");
    std::vector<float> in = read_f32(kDir + "flux_synth_in.f32");

    std::vector<float> forward(const locus::backend::Ops& op) const {
        const std::uint32_t n_img = j["n_img"], n_txt = j["n_txt"];
        const std::size_t nx = std::size_t{n_img} * dit.channels();
        const std::size_t nt = std::size_t{n_txt} * dit.text_width();
        const std::span<const float> all(in);
        return dit.forward(all.subspan(0, nx), ids_of(j, "img_ids"), n_img,
                           all.subspan(nx, nt), ids_of(j, "txt_ids"), n_txt,
                           all.subspan(nx + nt), j["sigma"].get<float>(),
                           j["guidance"].get<float>(), op);
    }
};

struct Leg {
    const char* name;
    const locus::backend::Ops* op;
};

/** The CPU, plus the Vulkan image ops (per op with LoRAs) on a GPU. */
std::vector<Leg> legs() {
    std::vector<Leg> v = {{"cpu", &locus::backend::best_backend().ops}};
    if (locus::backend::vk::VulkanContext::available()) {
        v.push_back({"vulkan", &locus::backend::vulkan_image_ops()});
    }
    return v;
}

}  // namespace

// i#372: the DiT with converted LoRAs attached matches diffusers' DiT
// with the same LoRAs' weight changes merged in (make-lora-dit-goldens
// .py): one LoRA, one at half strength, and two on overlapping layers.
// Detaching gives the plain model back.
TEST_CASE("flux dit with LoRAs matches diffusers with them merged",
          "[image][flux][lora]") {
    Synth s;
    const auto plain_ref = read_f32(kDir + "flux_synth_out.f32");
    const auto cases = read_json(kDir + "lora/dit_cases.json");
    for (const Leg& leg : legs()) {
        for (const auto& [name, loras] : cases.items()) {
            DYNAMIC_SECTION(leg.name << " " << name) {
                std::vector<locus::gguf::GgufFile> files;
                files.reserve(loras.size());
                LoraSet set;
                for (const auto& l : loras) {
                    files.push_back(locus::gguf::GgufFile::open(
                        kDir + "lora/" + l[0].get<std::string>() + ".gguf"));
                    set.add(files.back(), l[1].get<float>());
                }
                s.dit.set_lora(&set);
                REQUIRE(s.dit.has_lora());
                const auto ref = read_f32(kDir + "lora/dit_" + name + ".f32");
                CHECK(rel_error(s.forward(*leg.op), ref) < 1e-4);
                // Without the LoRA the output is far from this golden,
                // so the check above sees a LoRA that is not applied.
                CHECK(rel_error(plain_ref, ref) > 1e-2);
                s.dit.set_lora(nullptr);
                REQUIRE_FALSE(s.dit.has_lora());
                CHECK(rel_error(s.forward(*leg.op), plain_ref) < 1e-4);
            }
        }
    }
}

TEST_CASE("LoraSet rejects a GGUF that is not a flux_lora",
          "[image][flux][lora]") {
    const auto g = locus::gguf::GgufFile::open(kDir + "flux_synth.gguf");
    LoraSet set;
    REQUIRE_THROWS_AS(set.add(g, 1.0f), locus::gguf::Error);
    REQUIRE(set.empty());
}

TEST_CASE("LoraLibrary loads a directory and selects by name",
          "[image][flux][lora]") {
    const auto lib = locus::image::LoraLibrary::load_dir(kDir + "lora");
    REQUIRE(lib->names() ==
            std::vector<std::string>{"diffusers", "kohya", "ranks"});
    const std::vector<locus::image::LoraPick> picks = {{"kohya", 1.0f},
                                                       {"ranks", 0.5f}};
    const LoraSet set = lib->select(picks);
    REQUIRE(set.parts("transformer_blocks.0.attn.to_q").size() == 2);
    // The set is the library's files at the picked strengths.
    Synth s;
    s.dit.set_lora(&set);
    REQUIRE(s.dit.has_lora());
    s.dit.set_lora(nullptr);

    const std::vector<locus::image::LoraPick> bad = {{"nope", 1.0f}};
    try {
        (void)lib->select(bad);
        FAIL("an unknown name must throw");
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        CHECK(msg.find("nope") != std::string::npos);
        CHECK(msg.find("diffusers, kohya, ranks") != std::string::npos);
    }
    // A directory with no LoRA GGUF in it is refused.
    REQUIRE_THROWS_AS(
        locus::image::LoraLibrary::load_dir(std::string(LOCUS_SOURCE_DIR) +
                                            "/tests/golden/flux"),
        locus::gguf::Error);
}

TEST_CASE("LoraSet adds up LoRAs on the same layer",
          "[image][flux][lora]") {
    const auto k = locus::gguf::GgufFile::open(kDir + "lora/kohya.gguf");
    const auto r = locus::gguf::GgufFile::open(kDir + "lora/ranks.gguf");
    LoraSet set;
    set.add(k, 1.0f);
    set.add(r, 0.5f);
    const auto q = set.parts("transformer_blocks.0.attn.to_q");
    REQUIRE(q.size() == 2);
    CHECK(q[0].strength == 1.0f);
    CHECK(q[1].strength == 0.5f);
    CHECK(q[0].a.rows == 4);  // kohya: rank 4
    CHECK(q[1].a.rows == 2);  // ranks: the fused qkv at rank 2
    CHECK(set.parts("no.such.layer").empty());
}
