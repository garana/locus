#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/gguf/gguf.hpp"

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

/** @returns The F32 payload of tensor name, checked to be out x in. */
std::vector<float> tensor_f32(const locus::gguf::GgufFile& g,
                              const std::string& name, std::uint64_t rows,
                              std::uint64_t cols) {
    const auto* t = g.find_tensor(name);
    REQUIRE(t != nullptr);
    REQUIRE(t->type == locus::gguf::TensorType::kF32);
    REQUIRE(t->n_dims == 2);
    // GGUF lists dimensions innermost first.
    REQUIRE(t->ne[0] == cols);
    REQUIRE(t->ne[1] == rows);
    const auto bytes = g.tensor_data(*t);
    std::vector<float> v(rows * cols);
    std::memcpy(v.data(), bytes.data(), v.size() * sizeof(float));
    return v;
}

/**
 * Checks one converted LoRA against the change diffusers and peft make
 * to each DiT weight: for every layer in NAME.json, the GGUF's
 * lora_b (out x r, alpha / r folded in) times lora_a (r x in) must equal
 * that layer's slice of NAME_delta.f32, and the GGUF holds nothing else.
 */
void check_lora(const std::string& name) {
    const auto g = locus::gguf::GgufFile::open(kDir + "lora/" + name +
                                               ".gguf");
    const auto* arch = g.find("general.architecture");
    REQUIRE(arch != nullptr);
    REQUIRE(arch->s == "flux_lora");
    std::ifstream jf(kDir + "lora/" + name + ".json");
    REQUIRE(jf.good());
    const auto layers = nlohmann::json::parse(jf);
    const auto delta = read_f32(kDir + "lora/" + name + "_delta.f32");
    REQUIRE(g.tensors().size() == 2 * layers.size());
    const auto* count = g.find("flux_lora.layer_count");
    REQUIRE(count != nullptr);
    REQUIRE(count->u == layers.size());

    std::size_t at = 0;
    for (const auto& l : layers) {
        const std::string layer = l[0].get<std::string>();
        const std::uint64_t out = l[1].get<std::uint64_t>();
        const std::uint64_t in = l[2].get<std::uint64_t>();
        INFO(name << ": " << layer);
        const auto* ta = g.find_tensor(layer + ".lora_a");
        REQUIRE(ta != nullptr);
        const std::uint64_t r = ta->ne[1];
        const auto a = tensor_f32(g, layer + ".lora_a", r, in);
        const auto b = tensor_f32(g, layer + ".lora_b", out, r);
        REQUIRE(at + out * in <= delta.size());
        double worst = 0.0, ss = 0.0;
        for (std::uint64_t i = 0; i < out; ++i) {
            for (std::uint64_t j = 0; j < in; ++j) {
                double s = 0.0;
                for (std::uint64_t k = 0; k < r; ++k) {
                    s += double{b[i * r + k]} * a[k * in + j];
                }
                const double ref = delta[at + i * in + j];
                ss += ref * ref;
                worst = std::max(worst, std::fabs(s - ref));
            }
        }
        at += out * in;
        // The reference is a merged F32 weight minus the base, so it
        // carries the base's rounding: about 1e-7 of the weights' size.
        const double rms = std::sqrt(ss / static_cast<double>(out * in));
        REQUIRE(rms > 0.0);
        CHECK(worst <= 1e-5 * std::max(rms, 1e-2));
    }
    REQUIRE(at == delta.size());
}

}  // namespace

// i#372: scripts/convert-lora.py on kohya keys (fused qkv split, alpha
// folded in) and diffusers keys (BF16, no alpha), against the weight
// change diffusers and peft make; and on kohya modules of mixed ranks
// with a single-block linear1 (q, k, v and proj_mlp from one fused B),
// against alpha / r B A by rows.
TEST_CASE("convert-lora output matches the merged weight change",
          "[image][flux][lora]") {
    for (const char* name : {"kohya", "diffusers", "ranks"}) {
        check_lora(name);
    }
}
