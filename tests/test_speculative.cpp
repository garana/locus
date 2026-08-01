#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/model/speculative.hpp"

using locus::model::ngram_draft;
using locus::tok::TokenId;

TEST_CASE("ngram_draft proposes the continuation of a match",
          "[spec]") {
    // Trailing 2-gram {2,3} recurs earlier at index 1, followed by
    // 4,1,2,3 -> a 3-token draft is {4,1,2}.
    std::vector<TokenId> ctx{1, 2, 3, 4, 1, 2, 3};
    REQUIRE(ngram_draft(ctx, 2, 3) ==
            std::vector<TokenId>{4, 1, 2});
}

TEST_CASE("ngram_draft picks the most recent match", "[spec]") {
    // {9,9} occurs at 0 and 3; the later one (index 3) is used.
    std::vector<TokenId> ctx{9, 9, 5, 9, 9, 7, 9, 9};
    REQUIRE(ngram_draft(ctx, 2, 2) ==
            std::vector<TokenId>{7, 9});
}

TEST_CASE("ngram_draft returns empty with no match or short ctx",
          "[spec]") {
    std::vector<TokenId> none{5, 6, 7};
    REQUIRE(ngram_draft(none, 2, 3).empty());  // {6,7} never recurs
    std::vector<TokenId> tiny{1, 2};
    REQUIRE(ngram_draft(tiny, 2, 3).empty());  // ctx too short
    REQUIRE(ngram_draft(none, 0, 3).empty());  // ngram 0
}
