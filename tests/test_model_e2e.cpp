#include <filesystem>
#include <string>

#include "catch_amalgamated.hpp"
#include "cppllm/gguf/gguf.hpp"
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
