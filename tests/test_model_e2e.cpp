#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "cppllm/gguf/gguf.hpp"
#include "cppllm/model/llama.hpp"
#include "cppllm/tok/tokenizer.hpp"

namespace {

/** Real-model fixture path; see scripts/fetch-test-model.sh. */
std::string model_path() {
    return std::string(CPPLLM_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

}  // namespace

TEST_CASE("loads a real Llama-family GGUF model", "[e2e]") {
    const std::string path = model_path();
    if (!std::filesystem::exists(path)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }

    auto g = cppllm::gguf::GgufFile::open(path);
    REQUIRE(g.get_string("general.architecture") == "llama");
    REQUIRE_FALSE(g.tensors().empty());

    // Every tensor must be fully addressable inside the mapping.
    for (const auto& t : g.tensors()) {
        REQUIRE(g.tensor_data(t).size() == t.nbytes);
    }

    auto tok = cppllm::tok::SpmTokenizer::from_gguf(g);
    REQUIRE(tok.vocab_size() > 0);

    auto ids = tok.encode("Once upon a time", true);
    REQUIRE(ids.size() > 1);
    REQUIRE(ids.front() == tok.bos_id());

    auto text = tok.decode(ids);
    REQUIRE(text.find("Once upon a time") != std::string::npos);
}

namespace {

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) {
        s.pop_back();
    }
    return s;
}

}  // namespace

TEST_CASE("greedy decode matches the llama.cpp golden output",
          "[e2e]") {
    const std::string path = model_path();
    if (!std::filesystem::exists(path)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    // Produced by: llama-completion -m stories260K.gguf \
    //   -p "Once upon a time" -n 40 --temp 0
    std::ifstream golden_file(std::string(CPPLLM_SOURCE_DIR) +
                              "/tests/golden/stories260K_once.txt");
    REQUIRE(golden_file.good());
    std::stringstream ss;
    ss << golden_file.rdbuf();
    const std::string golden = trim(ss.str());

    auto g = cppllm::gguf::GgufFile::open(path);
    auto model = cppllm::model::LlamaModel::load(g);
    auto tok = cppllm::tok::SpmTokenizer::from_gguf(g);

    auto st = model.make_state();
    std::vector<float> logits(model.hparams().n_vocab);
    auto ids = tok.encode("Once upon a time", true);
    for (auto id : ids) {
        model.forward(id, st, logits);
    }
    for (int i = 0; i < 40; ++i) {
        auto next = cppllm::model::argmax(logits);
        if (next == tok.eos_id()) {
            break;
        }
        ids.push_back(next);
        model.forward(next, st, logits);
    }
    REQUIRE(trim(tok.decode(ids)) == golden);
}
