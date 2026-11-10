#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"
#include "locus/tok/unicode_flags.hpp"
#include "locus/tok/wpm_tokenizer.hpp"

using locus::tok::TokenId;
using locus::tok::WpmTokenizer;

namespace {

#define SP "\xe2\x96\x81"

/** Ids 0..4 are pad, unk, cls, sep and an unused slot. */
WpmTokenizer::Config tiny_vocab() {
    WpmTokenizer::Config cfg;
    cfg.tokens = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[unused0]",
                  SP "un", SP "unhappy", "happy", "ly", SP "a", SP ",",
                  SP "cafe", SP "$", SP "\xe6\x9d\xb1", SP "b"};
    cfg.types = {3, 2, 3, 3, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    cfg.pad = 0;
    cfg.unk = 1;
    cfg.cls = 2;
    cfg.sep = 3;
    return cfg;
}

std::string golden(const char* name) {
    return std::string(LOCUS_SOURCE_DIR) + "/tests/golden/" + name;
}

}  // namespace

TEST_CASE("wpm tokenizer takes the longest piece first", "[tok][wpm]") {
    const WpmTokenizer tok(tiny_vocab());
    // "▁unhappy" beats "▁un" + "happy"; then "ly" continues the word.
    REQUIRE(tok.encode("unhappyly", false) == std::vector<TokenId>{6, 8});
    REQUIRE(tok.encode("unhappy", true) == std::vector<TokenId>{2, 6, 3});
    REQUIRE(tok.encode("", true) == std::vector<TokenId>{2, 3});
    REQUIRE(tok.encode(" \t\n ", false).empty());
    REQUIRE(tok.bos_id() == 2);
    REQUIRE(tok.sep_id() == 3);
    REQUIRE(tok.eos_id() == -1);
}

TEST_CASE("wpm tokenizer maps a word with no split to one unknown",
          "[tok][wpm]") {
    const WpmTokenizer tok(tiny_vocab());
    // "▁unhappyx": "x" has no piece, so the whole word is [UNK], not
    // "▁unhappy" + [UNK].
    REQUIRE(tok.encode("unhappyx a", false) == std::vector<TokenId>{1, 9});
    REQUIRE(tok.encode("zz", false) == std::vector<TokenId>{1});
}

TEST_CASE("wpm tokenizer normalizes and splits words", "[tok][wpm]") {
    const WpmTokenizer tok(tiny_vocab());
    // Lowercase, accents dropped by the one-codepoint NFD.
    REQUIRE(tok.words("CAF\xc3\x89") == std::vector<std::string>{"cafe"});
    REQUIRE(tok.encode("Caf\xc3\xa9", false) == std::vector<TokenId>{11});
    // Punctuation, an ASCII symbol and a CJK ideograph are words of
    // their own; a non-ASCII symbol (U+00A9) is not.
    REQUIRE(tok.words("a,b$\xe6\x9d\xb1" "b\xc2\xa9" "c") ==
            std::vector<std::string>{"a", ",", "b", "$", "\xe6\x9d\xb1",
                                     "b\xc2\xa9" "c"});
    REQUIRE(tok.encode("a,b", false) == std::vector<TokenId>{9, 10, 14});
    // A control character, NUL and a combining mark are dropped; an
    // invalid byte becomes U+FFFD, which is dropped too.
    REQUIRE(tok.words(std::string("a\x07" "b\0c", 5)) ==
            std::vector<std::string>{"abc"});
    REQUIRE(tok.words("a\xcc\x81" "b") == std::vector<std::string>{"ab"});
    REQUIRE(tok.words("a\xff" "b\xc3") == std::vector<std::string>{"ab"});
    // A truncated sequence drops only its lead byte.
    REQUIRE(tok.words("\xe6\x9d" "a") == std::vector<std::string>{"a"});
}

TEST_CASE("wpm tokenizer decodes pieces", "[tok][wpm]") {
    const WpmTokenizer tok(tiny_vocab());
    const std::vector<TokenId> ids{2, 5, 7, 8, 10, 9, 3, 4};
    REQUIRE(tok.decode(ids) == "unhappyly , a");
    const std::vector<TokenId> bad{42};
    REQUIRE_THROWS_AS(tok.decode(bad), std::out_of_range);
}

TEST_CASE("wpm tokenizer rejects a bad config", "[tok][wpm]") {
    auto cfg = tiny_vocab();
    cfg.sep = 99;
    REQUIRE_THROWS_AS(WpmTokenizer(cfg), std::invalid_argument);
    cfg = tiny_vocab();
    cfg.types.pop_back();
    REQUIRE_THROWS_AS(WpmTokenizer(cfg), std::invalid_argument);
}

TEST_CASE("unicode flags follow llama.cpp's tables", "[tok][wpm]") {
    using namespace locus::tok;
    REQUIRE(cpt_flags('a') == cpt::kLetter);
    REQUIRE(cpt_flags('7') == cpt::kNumber);
    REQUIRE(cpt_flags(',') == cpt::kPunctuation);
    REQUIRE(cpt_flags('$') == cpt::kSymbol);
    REQUIRE(cpt_flags(0x0301) == cpt::kAccentMark);
    REQUIRE(cpt_flags(0x07) == cpt::kControl);
    REQUIRE(cpt_flags(' ') == (cpt::kSeparator | cpt::kWhitespace));
    REQUIRE((cpt_flags('\n') & cpt::kWhitespace) != 0);
    REQUIRE(cpt_flags(0x110000) == cpt::kUndefined);
    REQUIRE(cpt_nfd(0xE9) == 'e');
    REQUIRE(cpt_nfd('e') == 'e');
    REQUIRE(cpt_lower('A') == 'a');
    REQUIRE(cpt_lower(0x0416) == 0x0436);
    REQUIRE(cpt_lower('a') == 'a');
}

TEST_CASE("wpm tokenizer matches llama.cpp's WPM tokenizer",
          "[tok][wpm]") {
    // bert-vocab.gguf and the ids: scripts/make-bert-tok-golden.py on
    // bge-small-en-v1.5's GGUF.
    auto g = locus::gguf::GgufFile::open(golden("bert-vocab.gguf"));
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    REQUIRE(tok->vocab_size() == 30522);
    REQUIRE(tok->bos_id() == 101);
    std::ifstream f(golden("bert_tokenizer.jsonl"));
    REQUIRE(f.good());
    int cases = 0;
    for (std::string line; std::getline(f, line);) {
        const auto j = nlohmann::json::parse(line);
        const auto text = j["text"].get<std::string>();
        const auto want = j["ids"].get<std::vector<TokenId>>();
        INFO("text " << j["text"].dump());
        REQUIRE(tok->encode(text, true) == want);
        ++cases;
    }
    REQUIRE(cases == 24);
    const std::vector<TokenId> ids{101, 1996, 3007, 1997, 2605, 2003,
                                   3000, 1012, 102};
    REQUIRE(tok->decode(ids) == "the capital of france is paris .");
}
