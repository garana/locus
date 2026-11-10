#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/embed/embedding_model.hpp"

using locus::embed::EmbeddingModel;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/bge-reranker-v2-m3-q8_0.gguf";

nlohmann::json read(const std::string& name) {
    std::ifstream f(kDir + "golden/" + name);
    REQUIRE(f.good());
    return nlohmann::json::parse(f)["queries"];
}

std::vector<std::size_t> order(const std::vector<float>& s) {
    std::vector<std::size_t> o(s.size());
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(),
                     [&](std::size_t a, std::size_t b) { return s[a] > s[b]; });
    return o;
}

}  // namespace

TEST_CASE("bge-reranker-v2-m3 scores match exact arithmetic",
          "[rerank][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("bge-reranker-v2-m3-q8_0.gguf absent; see "
             "scripts/convert-xlmr-reranker.py");
    }
    const auto m = EmbeddingModel::load(kModel);
    REQUIRE(m->reranker());
    REQUIRE(m->context() == 8192);
    // Three references (scripts/make-rerank-golden.py and
    // make-bert-q8-ref.py): q8ref is the GGUF's own weights in float64,
    // which locus must match closely; transformers on the f32 original;
    // llama-server, which rounds the activations to 8 bits before each
    // Q8_0 product and drifts up to 0.3 on these 24 layers.
    const auto q8ref = read("bge_rerank_q8ref.json");
    const auto hf = read("bge_rerank_hf.json");
    const auto llama = read("bge_rerank.json");
    REQUIRE(q8ref.size() == 2);
    const auto& op = locus::backend::best_backend().ops;
    for (std::size_t q = 0; q < q8ref.size(); ++q) {
        const auto query = q8ref[q]["query"].get<std::string>();
        const auto docs =
            q8ref[q]["documents"].get<std::vector<std::string>>();
        const auto exact = q8ref[q]["scores"].get<std::vector<float>>();
        const auto ref = hf[q]["scores"].get<std::vector<float>>();
        const auto ll = llama[q]["scores"].get<std::vector<float>>();
        std::size_t tokens = 0;
        const auto got = m->rank(query, docs, op, &tokens);
        REQUIRE(got.size() == docs.size());
        REQUIRE(tokens > 0);
        for (std::size_t i = 0; i < docs.size(); ++i) {
            INFO("query " << query << " document " << docs[i]);
            UNSCOPED_INFO("score " << got[i] << " q8ref " << exact[i]
                                   << " hf " << ref[i] << " llama "
                                   << ll[i]);
            REQUIRE(std::fabs(got[i] - exact[i]) < 2e-3);
            REQUIRE(std::fabs(got[i] - ref[i]) < 0.15);
            REQUIRE(std::fabs(got[i] - ll[i]) < 0.4);
        }
        // The ranking is the original model's.
        REQUIRE(order(got) == order(ref));
        // One document at a time scores the same as the batch.
        const auto one = m->rank(query, {docs[1]}, op);
        REQUIRE(std::fabs(one[0] - got[1]) < 1e-4);
    }
}

TEST_CASE("reranker keeps the start of a long document", "[rerank][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("bge-reranker-v2-m3-q8_0.gguf absent");
    }
    const auto m = EmbeddingModel::load(kModel);
    const auto& op = locus::backend::best_backend().ops;
    // About 9000 tokens: past the 8192 context, so the document is cut.
    std::string big;
    for (int i = 0; i < 3000; ++i) big += "panda bear bamboo ";
    std::size_t tokens = 0;
    const auto s = m->rank("what is panda?", {big}, op, &tokens);
    REQUIRE(s.size() == 1);
    REQUIRE(tokens == m->context());
    REQUIRE(std::isfinite(s[0]));
    // A query that fills the context leaves no room: an error.
    REQUIRE_THROWS_AS(m->rank(big + big + big, {"x"}, op),
                      std::invalid_argument);
    // The reranker is not an embedding model, nor the reverse.
    REQUIRE_THROWS_AS(m->embed({"x"}, op), std::invalid_argument);
}
