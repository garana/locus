#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "json.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"
#include "locus/tok/ugm_tokenizer.hpp"

using locus::tok::TokenId;
using locus::tok::UgmTokenizer;

namespace {

/** Ids 0..2 are pad, eos and unk, as in T5. */
UgmTokenizer::Config tiny_vocab() {
    UgmTokenizer::Config cfg;
    cfg.tokens = {"<pad>", "</s>", "<unk>", "\xe2\x96\x81", "a", "b",
                  "ab", "\xe2\x96\x81" "ab", "<x>", "[PAD9]"};
    cfg.scores = {0, 0, 0, -1, -2, -2, -3, -1, 0, 0};
    cfg.types = {3, 3, 2, 1, 1, 1, 1, 1, 4, 5};
    return cfg;
}

std::string golden(const char* name) {
    return std::string(LOCUS_SOURCE_DIR) + "/tests/golden/" + name;
}

}  // namespace

TEST_CASE("t5 tokenizer picks the best scoring split", "[tok][t5]") {
    const UgmTokenizer tok(tiny_vocab());
    // "▁ab" (-1) beats "▁" + "ab" (-4) and "▁" + "a" + "b" (-5).
    REQUIRE(tok.encode("ab", false) == std::vector<TokenId>{7});
    REQUIRE(tok.encode("ab", true) == std::vector<TokenId>{7, 1});
    REQUIRE(tok.encode("ba", false) == std::vector<TokenId>{3, 5, 4});
    REQUIRE(tok.encode("", true) == std::vector<TokenId>{1});
    REQUIRE(tok.encode("   ", true) == std::vector<TokenId>{1});
    REQUIRE(tok.bos_id() == -1);
}

TEST_CASE("t5 tokenizer collapses spaces and merges unknowns",
          "[tok][t5]") {
    const UgmTokenizer tok(tiny_vocab());
    REQUIRE(tok.normalize("  a   b  ") ==
            "\xe2\x96\x81" "a" "\xe2\x96\x81" "b");
    // "cd" has no piece: one unknown token for the run.
    REQUIRE(tok.encode("acdb", false) ==
            std::vector<TokenId>{3, 4, 2, 5});
    // A malformed byte becomes U+FFFD, an unknown character.
    REQUIRE(tok.normalize("a\xff") == "\xe2\x96\x81" "a" "\xef\xbf\xbd");
    REQUIRE(tok.encode("a\xff", false) == std::vector<TokenId>{3, 4, 2});
}

TEST_CASE("t5 tokenizer splits at user defined tokens", "[tok][t5]") {
    const UgmTokenizer tok(tiny_vocab());
    // Whitespace around <x> is dropped; each side is its own chunk
    // with its own space prefix. The unused piece never matches.
    REQUIRE(tok.encode("ab <x> b", true) ==
            std::vector<TokenId>{7, 8, 3, 5, 1});
    REQUIRE(tok.encode("<x><x>", false) == std::vector<TokenId>{8, 8});
    REQUIRE(tok.encode("[PAD9]", false).size() > 1);
}

TEST_CASE("t5 tokenizer decodes pieces", "[tok][t5]") {
    const UgmTokenizer tok(tiny_vocab());
    const std::vector<TokenId> ids{7, 3, 5, 1, 0};
    REQUIRE(tok.decode(ids) == "ab b");
    const std::vector<TokenId> bad{42};
    REQUIRE_THROWS_AS(tok.decode(bad), std::out_of_range);
}

TEST_CASE("t5 tokenizer rejects a bad charsmap", "[tok][t5]") {
    auto cfg = tiny_vocab();
    cfg.charsmap = {0xff, 0xff, 0, 0, 1, 2, 3, 4};
    REQUIRE_THROWS_AS(UgmTokenizer(cfg), std::invalid_argument);
    cfg.charsmap = {3, 0, 0, 0, 1, 2, 3, 4};
    REQUIRE_THROWS_AS(UgmTokenizer(cfg), std::invalid_argument);
}

TEST_CASE("t5 tokenizer matches transformers' T5Tokenizer",
          "[tok][t5]") {
    // t5-vocab.gguf: scripts/convert-t5-encoder.py --vocab-only on
    // PixArt-XL-2-512x512's tokenizer; the ids come from
    // scripts/make-t5-tok-golden.py.
    auto g = locus::gguf::GgufFile::open(golden("t5-vocab.gguf"));
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    REQUIRE(tok->vocab_size() == 32128);
    REQUIRE(tok->eos_id() == 1);
    std::ifstream f(golden("t5_tokenizer.jsonl"));
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
    REQUIRE(cases == 20);
    // Many markers in a row: each is one id (and the split is one pass,
    // not a search of the whole rest per marker).
    std::string many;
    for (int i = 0; i < 20000; ++i) many += "<extra_id_1>";
    const auto mids = tok->encode(many, true);
    REQUIRE(mids.size() == 20001);
    REQUIRE(mids.front() == 32098);

    const auto* ugm = dynamic_cast<const UgmTokenizer*>(tok.get());
    REQUIRE(ugm != nullptr);
    // NFKC through the precompiled charsmap.
    REQUIRE(ugm->normalize("\xef\xbd\x86\xef\xbd\x95 \xef\xac\x81") ==
            "\xe2\x96\x81" "fu" "\xe2\x96\x81" "fi");
    const std::vector<TokenId> ids{37, 1784, 13, 1410, 19, 1919, 5, 1};
    REQUIRE(tok->decode(ids) == "The capital of France is Paris.");
}
