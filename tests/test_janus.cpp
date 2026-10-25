#include "locus/image/janus.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"

namespace {

std::vector<std::int32_t> ints(const std::string& line) {
    std::vector<std::int32_t> v;
    std::istringstream ls(line);
    for (std::int32_t x; ls >> x;) v.push_back(x);
    return v;
}

}  // namespace

// The server's prompt wrapping, against the reference's
// apply_sft_template_for_multi_turn_prompts + image_start_tag ids
// (golden line 1) for the same text. Tokenizer and metadata only.
TEST_CASE("janus prompt wrapping reproduces the reference ids",
          "[janus][image]") {
    const std::string dir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
    const std::string model = dir + "models/janus-pro-1b-bf16.gguf";
    if (!std::filesystem::exists(model)) {
        SKIP("janus-pro-1b-bf16.gguf absent; see scripts/convert-janus.py");
    }
    std::ifstream f(dir + "golden/janus_tokens.txt");
    REQUIRE(f.good());
    std::string l1;
    REQUIRE(std::getline(f, l1));
    const auto want = ints(l1);

    auto g = locus::gguf::GgufFile::open(model);
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    const auto gen = locus::image::JanusImageGen::load(
        g, static_cast<std::uint32_t>(*g.get_uint("llama.embedding_length")));
    REQUIRE(gen.prompt_ids(*tok, "A red apple on a wooden table.") ==
            std::vector<locus::tok::TokenId>(want.begin(), want.end()));
}

// i#226: Janus-Pro text-to-image codes, token-exact against the torch
// reference (scripts/make-janus-goldens.py tokens: CFG 5, argmax in
// f32 on the CPU) on the BF16 GGUF, whose weights are the checkpoint's
// own bf16 values. Checks the unconditional prompt and all 576 codes.
TEST_CASE("janus CFG argmax reproduces the reference image codes",
          "[janus][image][e2e]") {
    const std::string dir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
    const std::string model = dir + "models/janus-pro-1b-bf16.gguf";
    if (!std::filesystem::exists(model)) {
        SKIP("janus-pro-1b-bf16.gguf absent; see scripts/convert-janus.py");
    }
    if (std::getenv("LOCUS_HEAVY_TESTS") == nullptr) {
        SKIP("576 CFG steps take minutes on a CPU; set LOCUS_HEAVY_TESTS=1");
    }
    std::ifstream f(dir + "golden/janus_tokens.txt");
    REQUIRE(f.good());
    std::string l1, l2, l3;
    REQUIRE(std::getline(f, l1));
    REQUIRE(std::getline(f, l2));
    REQUIRE(std::getline(f, l3));
    const auto prompt = ints(l1), uncond = ints(l2), codes = ints(l3);

    auto g = locus::gguf::GgufFile::open(model);
    auto m = locus::model::TransformerModel::load(g);
    const auto* neon = locus::backend::find_backend("neon");
    if (neon != nullptr && neon->available) m.use_backend(*neon);
    const auto gen =
        locus::image::JanusImageGen::load(g, m.hparams().n_embd);
    REQUIRE(gen.image_tokens() == 576);
    REQUIRE(gen.begin_of_image() == prompt.back());
    REQUIRE(gen.unconditional(prompt) ==
            std::vector<locus::tok::TokenId>(uncond.begin(), uncond.end()));

    locus::image::JanusImageGen::Params p;
    p.cfg = 5.0f;
    p.temperature = 0.0f;  // argmax, as the golden
    const auto t0 = std::chrono::steady_clock::now();
    const auto got = gen.generate(
        m, std::vector<locus::tok::TokenId>(prompt.begin(), prompt.end()),
        p);
    const double s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
    REQUIRE(got.size() == codes.size());
    std::size_t first_diff = got.size();
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != codes[i]) {
            first_diff = i;
            break;
        }
    }
    INFO("first differing code " << first_diff << " of " << got.size()
                                 << ", " << s << " s");
    REQUIRE(first_diff == got.size());
}
