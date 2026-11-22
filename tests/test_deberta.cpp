#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/embed/deberta.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::embed::DebertaEncoder;
using locus::tok::TokenId;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
// scripts/text/convert-deberta.py on
// protectai/deberta-v3-base-prompt-injection-v2 (commit 90c9989).
const std::string kQ8 = kDir + "models/deberta-v3-base-injection-q8_0.gguf";
const std::string kF32 = kDir + "models/deberta-v3-base-injection-f32.gguf";

/** A golden of scripts/text/make-deberta-golden.py. */
nlohmann::json golden(const std::string& name) {
    std::ifstream f(kDir + "golden/" + name);
    REQUIRE(f.good());
    return nlohmann::json::parse(f);
}

std::vector<std::vector<TokenId>> ids_of(const nlohmann::json& g) {
    return g["ids"].get<std::vector<std::vector<TokenId>>>();
}

std::size_t argmax(const std::vector<float>& v) {
    return static_cast<std::size_t>(
        std::max_element(v.begin(), v.end()) - v.begin());
}

/** The largest |a - b| over every logit of every text. */
float max_diff(const std::vector<std::vector<float>>& a,
               const std::vector<std::vector<float>>& b) {
    REQUIRE(a.size() == b.size());
    float d = 0.0f;
    for (std::size_t s = 0; s < a.size(); ++s) {
        REQUIRE(a[s].size() == b[s].size());
        for (std::size_t i = 0; i < a[s].size(); ++i) {
            d = std::max(d, std::fabs(a[s][i] - b[s][i]));
        }
    }
    return d;
}

}  // namespace

TEST_CASE("deberta relative position buckets match transformers",
          "[deberta]") {
    // make_log_bucket_position for every distance of the golden's
    // longest text (632 tokens), with DeBERTa v3's 256 buckets and 512
    // positions: the distance itself within 128, logarithmic beyond.
    const auto g = golden("deberta_injection_hf.json");
    const auto want = g["buckets"].get<std::vector<int>>();
    const int d0 = g["bucket_min_distance"].get<int>();
    REQUIRE(want.size() > 1200);
    for (std::size_t i = 0; i < want.size(); ++i) {
        const int d = d0 + static_cast<int>(i);
        INFO("distance " << d);
        REQUIRE(DebertaEncoder::bucket(d, 256, 512) == want[i]);
    }
}

TEST_CASE("deberta tokenizer ids match transformers", "[deberta][tok]") {
    if (!std::filesystem::exists(kQ8)) {
        SKIP("deberta-v3-base-injection-q8_0.gguf absent; see "
             "scripts/text/convert-deberta.py");
    }
    const auto gf = locus::gguf::GgufFile::open(kQ8);
    const auto tok = locus::tok::tokenizer_from_gguf(gf);
    const auto g = golden("deberta_injection_hf.json");
    const auto texts = g["texts"].get<std::vector<std::string>>();
    const auto want = ids_of(g);
    for (std::size_t i = 0; i < texts.size(); ++i) {
        INFO("text '" << texts[i] << "'");
        CHECK(tok->encode(texts[i], true) == want[i]);
    }
    // Typed special-token text stays text, so a client cannot forge
    // [CLS] or [SEP] (transformers would parse it).
    const auto forged = tok->encode("a [SEP] b [CLS]", true);
    CHECK(std::count(forged.begin(), forged.end(), 1) == 1);
    CHECK(std::count(forged.begin(), forged.end(), 2) == 1);
}

TEST_CASE("deberta q8_0 logits match exact arithmetic", "[deberta][e2e]") {
    if (!std::filesystem::exists(kQ8)) {
        SKIP("deberta-v3-base-injection-q8_0.gguf absent; see "
             "scripts/text/convert-deberta.py");
    }
    const auto gf = locus::gguf::GgufFile::open(kQ8);
    const auto m = DebertaEncoder::load(gf);
    REQUIRE(m.width() == 768);
    REQUIRE(m.context() == 512);
    REQUIRE(m.labels() == std::vector<std::string>{"SAFE", "INJECTION"});
    // Two references: the GGUF's own weights through transformers in
    // float64 (q8ref), which locus must match closely, and the float32
    // original (hf), which bounds what Q8_0 costs.
    const auto q8ref = golden("deberta_injection_q8ref.json");
    const auto hf = golden("deberta_injection_hf.json");
    const auto ids = ids_of(q8ref);
    REQUIRE(ids == ids_of(hf));
    REQUIRE(ids.back().size() > 512);  // past the trained length
    const auto exact = q8ref["logits"].get<std::vector<std::vector<float>>>();
    const auto ref = hf["logits"].get<std::vector<std::vector<float>>>();
    const auto& op = locus::backend::best_backend().ops;
    const auto got = m.classify(ids, op);
    for (std::size_t s = 0; s < ids.size(); ++s) {
        UNSCOPED_INFO(s << ": " << got[s][0] << " " << got[s][1] << " q8ref "
                        << exact[s][0] << " " << exact[s][1] << " hf "
                        << ref[s][0] << " " << ref[s][1]);
        // The original model's verdict.
        CHECK(argmax(got[s]) == argmax(ref[s]));
    }
    CHECK(max_diff(got, exact) < 1e-3f);  // 1.1e-5 on an M4 Max
    CHECK(max_diff(got, ref) < 0.1f);
    // One sequence at a time gives the batch's logits.
    for (const std::size_t s : {std::size_t{2}, ids.size() - 1}) {
        const auto one = m.classify({&ids[s], 1}, op);
        CHECK(max_diff(one, {got[s]}) < 1e-4f);
    }
}

TEST_CASE("deberta f32 logits match transformers", "[deberta][e2e]") {
    if (!std::filesystem::exists(kF32)) {
        SKIP("deberta-v3-base-injection-f32.gguf absent; see "
             "scripts/text/convert-deberta.py --f32");
    }
    const auto gf = locus::gguf::GgufFile::open(kF32);
    const auto m = DebertaEncoder::load(gf);
    const auto hf = golden("deberta_injection_hf.json");
    const auto ids = ids_of(hf);
    const auto ref = hf["logits"].get<std::vector<std::vector<float>>>();
    const auto got =
        m.classify(ids, locus::backend::best_backend().ops);
    for (std::size_t s = 0; s < ids.size(); ++s) {
        UNSCOPED_INFO(s << ": " << got[s][0] << " " << got[s][1] << " hf "
                        << ref[s][0] << " " << ref[s][1]);
    }
    CHECK(max_diff(got, ref) < 1e-3f);  // 2.1e-5 on an M4 Max
}

TEST_CASE("deberta rejects empty and out-of-vocabulary input",
          "[deberta]") {
    if (!std::filesystem::exists(kQ8)) {
        SKIP("deberta-v3-base-injection-q8_0.gguf absent");
    }
    const auto gf = locus::gguf::GgufFile::open(kQ8);
    const auto m = DebertaEncoder::load(gf);
    const auto& op = locus::backend::best_backend().ops;
    const std::vector<std::vector<TokenId>> empty = {{}};
    CHECK_THROWS_AS(m.classify(empty, op), std::invalid_argument);
    const std::vector<std::vector<TokenId>> far = {{1, 128100, 2}};
    CHECK_THROWS_AS(m.classify(far, op), std::invalid_argument);
    const std::vector<std::vector<TokenId>> neg = {{1, -1, 2}};
    CHECK_THROWS_AS(m.classify(neg, op), std::invalid_argument);
}
