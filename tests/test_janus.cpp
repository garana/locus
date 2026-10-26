#include "locus/image/janus.hpp"

#include <algorithm>
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
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
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
    const std::vector<locus::tok::TokenId> ids(want.begin(), want.end());
    REQUIRE(gen.prompt_ids(*tok, "A red apple on a wooden table.") == ids);
    // The reference strips the text (str.strip()): ASCII and Unicode
    // spaces at both ends (U+3000 3 bytes, U+00A0 2 bytes) change
    // nothing; an inner U+00A0 stays.
    REQUIRE(gen.prompt_ids(*tok, "\xe3\x80\x80 \tA red apple on a wooden "
                                 "table.\xc2\xa0\n") == ids);
    REQUIRE(gen.prompt_ids(*tok, "A red\xc2\xa0" "apple on a wooden table.") !=
            ids);
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

// i#247: the generation loop on the fused Vulkan forward
// (forward_embedding with the BF16 matvec). The GPU sums in another
// order than the CPU, so a near-tie may tip and a free-running image
// then drifts; the check is teacher-forced instead: feeding the
// reference's codes, each one's guided logit must be within kMargin
// of the top. The free-running first differing code and the time are
// reported.
TEST_CASE("janus CFG codes on the Vulkan forward (teacher-forced)",
          "[janus][image][vulkan][e2e]") {
    const std::string dir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
    const std::string model = dir + "models/janus-pro-1b-bf16.gguf";
    if (!std::filesystem::exists(model)) {
        SKIP("janus-pro-1b-bf16.gguf absent");
    }
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device");
    }
    if (std::getenv("LOCUS_HEAVY_TESTS") == nullptr) {
        SKIP("576 CFG steps take minutes; set LOCUS_HEAVY_TESTS=1");
    }
    constexpr float kMargin = 0.25f;
    std::ifstream f(dir + "golden/janus_tokens.txt");
    REQUIRE(f.good());
    std::string l1, l2, l3;
    REQUIRE(std::getline(f, l1));
    REQUIRE(std::getline(f, l2));
    REQUIRE(std::getline(f, l3));
    const auto prompt = ints(l1), codes = ints(l3);

    auto g = locus::gguf::GgufFile::open(model);
    auto m = locus::model::TransformerModel::load(g);
    m.use_backend(*locus::backend::find_backend("vulkan"));
    // Else forward_embedding quietly runs the per-op path.
    REQUIRE(locus::backend::vulkan_forward_supported(m));
    const auto gen =
        locus::image::JanusImageGen::load(g, m.hparams().n_embd);
    // The heads on the GPU too, as generate() runs them on vulkan.
    const auto* gpu = &m.active_backend().ops;
    const std::vector<locus::tok::TokenId> ids(prompt.begin(), prompt.end());

    // Free-running: time and the first differing code.
    locus::image::JanusImageGen::Params p;
    p.cfg = 5.0f;
    p.temperature = 0.0f;
    const auto t0 = std::chrono::steady_clock::now();
    const auto got = gen.generate(m, ids, p);
    const double s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
    std::size_t first_diff = got.size();
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != codes[i]) {
            first_diff = i;
            break;
        }
    }
    WARN("vulkan free-running: first differing code " << first_diff
                                                      << " of 576, " << s
                                                      << " s");

    // Teacher-forced: the same loop as generate(), feeding the
    // reference's code at every step.
    const auto& hp = m.hparams();
    auto cache = m.make_cache();
    auto ws = m.make_workspace();
    std::vector<float> logits(hp.n_vocab), normed(hp.n_embd), emb(hp.n_embd);
    const locus::model::LogitRows one{0, 1};
    const auto uncond = gen.unconditional(ids);
    locus::kv::PagedKvCache::Seq seq[2];
    const std::span<const locus::tok::TokenId> pr[2] = {ids, uncond};
    std::vector<float> img[2] = {std::vector<float>(16384),
                                 std::vector<float>(16384)};
    for (int k = 0; k < 2; ++k) {
        REQUIRE(cache.ensure_capacity(
            seq[k], static_cast<std::uint32_t>(ids.size() + 576)));
        for (const auto id : pr[k]) {
            m.forward_embedding(id, {}, cache, seq[k], ws, logits, one,
                                normed);
        }
        gen.image_logits(normed, img[k], gpu);
    }
    float worst = 0.0f;
    std::size_t argmax_hits = 0;
    std::vector<float> guided(16384);
    for (std::size_t i = 0; i < codes.size(); ++i) {
        for (std::size_t c = 0; c < guided.size(); ++c) {
            guided[c] = img[1][c] + p.cfg * (img[0][c] - img[1][c]);
        }
        const auto top = std::max_element(guided.begin(), guided.end());
        const float gap = *top - guided[static_cast<std::size_t>(codes[i])];
        worst = std::max(worst, gap);
        argmax_hits += (top - guided.begin()) == codes[i];
        INFO("code " << i << ": gap " << gap);
        REQUIRE(gap <= kMargin);
        if (i + 1 == codes.size()) break;
        gen.image_embed(codes[i], emb, gpu);
        for (int k = 0; k < 2; ++k) {
            m.forward_embedding(0, emb, cache, seq[k], ws, logits, one,
                                normed);
            gen.image_logits(normed, img[k], gpu);
        }
    }
    WARN("vulkan teacher-forced: " << argmax_hits << " of 576 argmax, worst "
                                   << "gap " << worst);
    for (auto& q : seq) cache.release(q);
}
