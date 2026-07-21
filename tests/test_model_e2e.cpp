#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "locus/tok/tokenizer.hpp"

namespace {

/** Real-model fixture path; see scripts/fetch-test-model.sh. */
std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

std::string llama32_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/llama-3.2-1b-q8_0.gguf";
}

}  // namespace

TEST_CASE("loads a real Llama-family GGUF model", "[e2e]") {
    const std::string path = model_path();
    if (!std::filesystem::exists(path)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }

    auto g = locus::gguf::GgufFile::open(path);
    REQUIRE(g.get_string("general.architecture") == "llama");
    REQUIRE_FALSE(g.tensors().empty());

    // Every tensor must be fully addressable inside the mapping.
    for (const auto& t : g.tensors()) {
        REQUIRE(g.tensor_data(t).size() == t.nbytes);
    }

    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
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

namespace {

/** Greedy generation returning the full decoded text. */
std::string generate_text(locus::model::LlamaModel& model,
                          const locus::tok::SpmTokenizer& tok,
                          const std::string& prompt, int n_gen) {
    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    locus::kv::PagedKvCache::Seq seq;
    std::vector<float> logits(model.hparams().n_vocab);
    auto ids = tok.encode(prompt, true);
    for (auto id : ids) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(id, cache, seq, ws, logits);
    }
    for (int i = 0; i < n_gen; ++i) {
        auto next = locus::model::argmax(logits);
        if (next == tok.eos_id()) {
            break;
        }
        ids.push_back(next);
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(next, cache, seq, ws, logits);
    }
    cache.release(seq);
    return tok.decode(ids);
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
    std::ifstream golden_file(std::string(LOCUS_SOURCE_DIR) +
                              "/tests/golden/stories260K_once.txt");
    REQUIRE(golden_file.good());
    std::stringstream ss;
    ss << golden_file.rdbuf();
    const std::string golden = trim(ss.str());

    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    REQUIRE(trim(generate_text(model, tok, "Once upon a time",
                               40)) == golden);
}

TEST_CASE("llama-3.2 (BPE + scaled rope) runs consistently",
          "[e2e][backend]") {
    if (!std::filesystem::exists(llama32_path())) {
        SKIP("model not present (llama-3.2-1b-q8_0.gguf)");
    }
    auto g = locus::gguf::GgufFile::open(llama32_path());
    auto model = locus::model::LlamaModel::load(g);
    auto tok_ptr = locus::tok::tokenizer_from_gguf(g);
    REQUIRE(!model.rope_factors().empty());  // llama3 scaling

    // All selectable backends must agree with each other; the
    // text must answer the prompt (no llama.cpp golden here: it
    // auto-applies the chat template to this model).
    std::string reference;
    for (const auto& b : locus::backend::backends()) {
        if (!b.available || !b.selectable) {
            continue;
        }
        INFO("backend " << b.name);
        model.use_backend(b);
        auto cache = model.make_cache();
        auto ws = model.make_workspace();
        locus::kv::PagedKvCache::Seq seq;
        std::vector<float> logits(model.hparams().n_vocab);
        auto ids =
            tok_ptr->encode("The capital of France is", true);
        for (auto id : ids) {
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(id, cache, seq, ws, logits);
        }
        for (int i = 0; i < 8; ++i) {
            auto next = locus::model::argmax(logits);
            ids.push_back(next);
            REQUIRE(cache.ensure_capacity(seq, 1));
            model.forward(next, cache, seq, ws, logits);
        }
        cache.release(seq);
        const std::string text = tok_ptr->decode(ids);
        REQUIRE(text.find("Paris") != std::string::npos);
        if (reference.empty()) {
            reference = text;
        } else {
            REQUIRE(text == reference);
        }
    }
}

TEST_CASE("every selectable backend reproduces the golden output",
          "[e2e][backend]") {
    const std::string path = model_path();
    if (!std::filesystem::exists(path)) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    std::ifstream golden_file(std::string(LOCUS_SOURCE_DIR) +
                              "/tests/golden/stories260K_once.txt");
    REQUIRE(golden_file.good());
    std::stringstream ss;
    ss << golden_file.rdbuf();
    const std::string golden = trim(ss.str());

    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::LlamaModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);

    for (const auto& b : locus::backend::backends()) {
        if (!b.available || !b.selectable) {
            continue;
        }
        DYNAMIC_SECTION("backend " << b.name) {
            model.use_backend(b);
            REQUIRE(trim(generate_text(model, tok,
                                       "Once upon a time", 40)) ==
                    golden);
        }
    }
}
