#include <string_view>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"
#include "gguf_builder.hpp"

using locus::tok::SpmTokenizer;
using locus::tok::TokenId;

namespace {

/** SPM meta-space (U+2581) as a plain string for concatenation. */
const std::string kSp = "\xe2\x96\x81";

/**
 * Tiny hand-built SPM vocabulary. Scores are chosen so "ab" merges
 * before "▁ab" wins the final merge; "X" exists only as the byte
 * token <0x58>; "Y" has no byte token and must fall back to unk.
 */
SpmTokenizer::Config tiny_vocab() {
    SpmTokenizer::Config cfg;
    cfg.tokens = {"<unk>", "<s>", "</s>", kSp,      "a",
                  "b",     "ab",  kSp + "ab", "<0x58>"};
    cfg.scores = {0, 0, 0, -1.0f, -2.0f, -3.0f, -1.5f, -1.0f, 0};
    cfg.types = {2, 3, 3, 1, 1, 1, 1, 1, 6};
    return cfg;
}

}  // namespace

TEST_CASE("encode merges bigrams by score", "[tok]") {
    SpmTokenizer tok(tiny_vocab());

    // "ab" -> "▁ab": merge (a,b) first, then (▁,ab).
    auto ids = tok.encode("ab", true);
    REQUIRE(ids == std::vector<TokenId>{1, 7});

    auto no_bos = tok.encode("ab", false);
    REQUIRE(no_bos == std::vector<TokenId>{7});

    REQUIRE(tok.encode("", true) == std::vector<TokenId>{1});
    REQUIRE(tok.encode("", false).empty());
}

TEST_CASE("unknown symbols use byte fallback then unk", "[tok]") {
    SpmTokenizer tok(tiny_vocab());

    // "X" is not in the vocab but <0x58> is.
    REQUIRE(tok.encode("X", false) == std::vector<TokenId>{3, 8});
    // "Y" has no byte token either: unk.
    REQUIRE(tok.encode("Y", false) == std::vector<TokenId>{3, 0});
}

TEST_CASE("decode renders pieces, bytes, and controls", "[tok]") {
    SpmTokenizer tok(tiny_vocab());

    const TokenId ids[] = {1, 7, 8};  // <s> ▁ab <0x58>
    REQUIRE(tok.decode(ids) == " abX");

    REQUIRE(tok.decode(tok.encode("ab", true)) == " ab");
    REQUIRE_THROWS_AS(tok.piece(99), std::out_of_range);
}

TEST_CASE("builds from GGUF metadata", "[tok]") {
    const std::string sp_hi = kSp + "hi";
    const std::string_view tokens[] = {"<unk>", "<s>", "</s>", kSp,
                                       "h",     "i",   "hi",   sp_hi};
    const float scores[] = {0, 0, 0, -1, -2, -3, -1.5f, -1};
    const std::int32_t types[] = {2, 3, 3, 1, 1, 1, 1, 1};

    GgufBuilder b;
    b.header(0, 7)
        .kv_string("tokenizer.ggml.model", "llama")
        .kv_str_array("tokenizer.ggml.tokens", tokens)
        .kv_f32_array("tokenizer.ggml.scores", scores)
        .kv_i32_array("tokenizer.ggml.token_type", types)
        .kv_u32("tokenizer.ggml.bos_token_id", 1)
        .kv_u32("tokenizer.ggml.eos_token_id", 2)
        .kv_u32("tokenizer.ggml.unknown_token_id", 0);
    auto g = locus::gguf::GgufFile::parse(b.bytes());

    auto tok = SpmTokenizer::from_gguf(g);
    REQUIRE(tok.vocab_size() == 8);
    REQUIRE(tok.encode("hi", true) == std::vector<TokenId>{1, 7});
    REQUIRE(tok.decode(tok.encode("hi", true)) == " hi");
}

TEST_CASE("rejects non-SPM tokenizers", "[tok]") {
    GgufBuilder b;
    b.header(0, 1).kv_string("tokenizer.ggml.model", "gpt2");
    auto g = locus::gguf::GgufFile::parse(b.bytes());
    REQUIRE_THROWS_AS(SpmTokenizer::from_gguf(g),
                      locus::gguf::Error);
}
