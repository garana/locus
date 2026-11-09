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

namespace {

using locus::model::choose_draft;
using locus::model::DraftCost;
using locus::model::DraftState;
using locus::model::observe_draft;

// Two samples per length: DraftCost drops the first of each.
void measure(DraftCost& c, std::uint32_t d, double seconds) {
    c.observe(d, seconds);
    c.observe(d, seconds);
}

// A request whose drafts pass at about `rate` (n verifies of 4).
DraftState passing(double rate) {
    DraftState s;
    for (int i = 0; i < 40; ++i) {
        const std::uint32_t acc = i % 10 < rate * 10 ? 4 : 0;
        observe_draft(s, 4, acc);
    }
    return s;
}

}  // namespace

TEST_CASE("DraftCost drops the first sample of each length", "[spec]") {
    DraftCost c;
    c.observe(3, 9.0);  // a first use, with one-time costs
    REQUIRE(c.seconds(3) == 0.0);
    c.observe(3, 1.0);
    REQUIRE(c.seconds(3) == 1.0);
    REQUIRE(c.seconds(7) == 0.0);  // never seen
}

TEST_CASE("choose_draft prices drafts against a plain decode",
          "[spec]") {
    DraftState prior;  // no data: accept rate 1/2
    REQUIRE(prior.accept_rate() == 0.5);

    SECTION("no plain decode timed yet: decode first") {
        DraftCost c;
        REQUIRE(choose_draft(prior, c, 8) == 0);
    }
    SECTION("only a decode timed: a short trial draft") {
        // Unmeasured rows cost a quarter of a decode: 1.5 tokens for
        // 1.25 decodes beats 1.75 for 1.5.
        DraftCost c;
        measure(c, 0, 1.0);
        REQUIRE(choose_draft(prior, c, 8) == 1);
    }
    SECTION("drafts that fail: plain decode") {
        DraftCost c;
        measure(c, 0, 1.0);
        REQUIRE(choose_draft(passing(0.1), c, 8) == 0);
    }
    // A CPU: each row costs 0.7 of a decode. A GPU: rows are nearly
    // free. With drafts that pass 97% of the time the best length is
    // short on the CPU and the max on the GPU.
    DraftState good;
    for (int i = 0; i < 40; ++i) {
        observe_draft(good, 8, i % 4 == 0 ? 7 : 8);
    }
    REQUIRE(good.accept_rate() > 0.95);
    SECTION("CPU-like costs pick a short draft") {
        DraftCost c;
        for (std::uint32_t d = 0; d <= 8; ++d) {
            measure(c, d, 1.0 + 0.7 * d);
        }
        const auto d = choose_draft(good, c, 8);
        REQUIRE(d >= 2);
        REQUIRE(d <= 5);
    }
    SECTION("GPU-like costs pick the max") {
        DraftCost c;
        for (std::uint32_t d = 0; d <= 8; ++d) {
            measure(c, d, 1.0 + 0.05 * d);
        }
        REQUIRE(choose_draft(good, c, 8) == 8);
    }
    SECTION("a cost jump is not crossed") {
        // Flat up to 4 drafts, then 4 decodes' worth: stay at 4.
        DraftCost c;
        for (std::uint32_t d = 0; d <= 8; ++d) {
            measure(c, d, d <= 4 ? 1.0 : 4.0);
        }
        REQUIRE(choose_draft(good, c, 8) == 4);
    }
}
