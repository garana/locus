#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/bpe_tokenizer.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::tok::BpeTokenizer;
using locus::tok::TokenId;

namespace {

/** a/b/c with merges a+b -> ab, ab+c -> abc, plus specials. */
BpeTokenizer::Config tiny_vocab() {
    BpeTokenizer::Config cfg;
    cfg.tokens = {"<s>", "</s>", "<|x|>", "a", "b", "c", "ab",
                  "abc"};
    cfg.types = {3, 3, 3, 1, 1, 1, 1, 1};
    cfg.merges = {"a b", "ab c"};
    cfg.bos = 0;
    cfg.eos = 1;
    return cfg;
}

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/llama-3.2-1b-q8_0.gguf";
}

}  // namespace

TEST_CASE("bpe merges by rank", "[tok]") {
    BpeTokenizer tok(tiny_vocab());

    REQUIRE(tok.encode("abc", false) ==
            std::vector<TokenId>{7});
    REQUIRE(tok.encode("abcb", false) ==
            std::vector<TokenId>{7, 4});
    REQUIRE(tok.encode("ba", false) ==
            std::vector<TokenId>{4, 3});  // no merge rule for b+a
    REQUIRE(tok.encode("abc", true).front() == tok.bos_id());
}

TEST_CASE("bpe special tokens split the input", "[tok]") {
    BpeTokenizer tok(tiny_vocab());

    REQUIRE(tok.encode("a<|x|>b", false) ==
            std::vector<TokenId>{3, 2, 4});
    // Control tokens render as nothing on decode.
    const TokenId ids[] = {0, 7, 2, 4};
    REQUIRE(tok.decode(ids) == "abcb");
}

TEST_CASE("bpe matches llama.cpp goldens on Llama-3.2",
          "[tok][e2e]") {
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present (llama-3.2-1b-q8_0.gguf)");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto tok = locus::tok::tokenizer_from_gguf(g);

    // Expected ids produced by:
    //   llama-tokenize -m llama-3.2-1b-q8_0.gguf -p <text> --ids
    struct Case {
        std::string text;
        std::vector<TokenId> ids;
    };
    const std::vector<Case> cases = {
        {"Hello world", {128000, 9906, 1917}},
        {"Once upon a time, in a quiet village, 1234 tokens "
         "appeared!",
         {128000, 12805, 5304, 264, 892, 11, 304, 264, 11594,
          14458, 11, 220, 4513, 19, 11460, 9922, 0}},
        {"  leading spaces and\ttabs\nnewlines too",
         {128000, 220, 6522, 12908, 323, 3324, 3518, 198, 943,
          8128, 2288}},
        {"def main() -> int: return x['key'] + 42;",
         {128000, 755, 1925, 368, 1492, 528, 25, 471, 865, 681,
          798, 663, 489, 220, 2983, 26}},
        {"I'll can't we're it's",
         {128000, 40, 3358, 649, 956, 584, 2351, 433, 596}},
        {"cafe posts 100 200 3000 40000",
         {128000, 936, 1897, 8158, 220, 1041, 220, 1049, 220,
          3101, 15, 220, 3443, 410}},
    };
    for (const auto& c : cases) {
        INFO("text: " << c.text);
        REQUIRE(tok->encode(c.text, true) == c.ids);
        // Byte-level BPE decodes losslessly (BOS renders empty).
        REQUIRE(tok->decode(c.ids) == c.text);
    }
}

TEST_CASE("empty special token does not stall encode", "[tok]") {
    // #60-3: a hostile GGUF can declare an empty kControl/kUserDefined
    // piece. Pre-fix, the encode split loop did text.find("", at) ==
    // at and advanced by piece.size() == 0, so `at` never moved ->
    // infinite loop + unbounded output. The empty special must be
    // ignored and encode must terminate normally.
    BpeTokenizer::Config cfg = tiny_vocab();
    cfg.tokens.push_back("");   // empty special piece
    cfg.types.push_back(3);     // kControl
    BpeTokenizer tok(cfg);
    REQUIRE(tok.encode("abc", false) == std::vector<TokenId>{7});
}

TEST_CASE("bpe from_gguf rejects a non-string token entry", "[tok]") {
    // #59-C: token array elements must be strings; a non-string
    // element (crafted GGUF) is rejected, not silently coerced to "".
    const std::vector<std::int32_t> toks = {1, 2, 3};   // not strings
    const std::vector<std::string_view> merges = {"a b"};
    GgufBuilder b;
    b.header(0, 3)
        .kv_string("tokenizer.ggml.model", "gpt2")
        .kv_i32_array("tokenizer.ggml.tokens", toks)
        .kv_str_array("tokenizer.ggml.merges", merges);
    auto g = locus::gguf::GgufFile::parse(b.bytes());
    REQUIRE_THROWS_AS(BpeTokenizer::from_gguf(g),
                      locus::gguf::Error);
}
