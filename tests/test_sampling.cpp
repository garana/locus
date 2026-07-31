#include <random>
#include <set>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/model/sampling.hpp"

using locus::model::sample;
using locus::model::SamplingParams;
using locus::tok::TokenId;

namespace {
std::mt19937_64 rng(123);
std::vector<TokenId> none;

/** Test constraint that only permits a fixed set of token ids. */
struct AllowSet : locus::model::TokenConstraint {
    std::set<TokenId> ok;
    bool allows(TokenId t) const override {
        return ok.count(t) > 0;
    }
    void commit(TokenId) override {}
};
}  // namespace

TEST_CASE("default sampling is greedy (argmax)", "[sampling]") {
    std::vector<float> logits{1.0f, 3.0f, 2.0f, 0.5f};
    SamplingParams p;  // temperature 0 -> greedy
    REQUIRE(sample(logits, p, none, rng) == 1);
}

TEST_CASE("repeat_penalty can shift the argmax", "[sampling]") {
    std::vector<float> logits{5.0f, 4.9f};
    std::vector<TokenId> hist{0};
    SamplingParams p;
    p.repeat_penalty = 10.0f;  // token 0: 5.0 -> 0.5
    REQUIRE(sample(logits, p, hist, rng) == 1);
}

TEST_CASE("frequency/presence penalties count history",
          "[sampling]") {
    std::vector<float> logits{5.0f, 4.0f};
    std::vector<TokenId> hist{0, 0, 0};  // token 0 thrice
    SamplingParams p;
    p.frequency_penalty = 1.0f;  // 5.0 - 1*3 = 2.0 < 4.0
    REQUIRE(sample(logits, p, hist, rng) == 1);
}

TEST_CASE("top_k=1 selects the max even with temperature",
          "[sampling]") {
    std::vector<float> logits{0.1f, 0.2f, 5.0f, 0.3f};
    SamplingParams p;
    p.temperature = 1.5f;
    p.top_k = 1;
    for (int i = 0; i < 20; ++i) {
        std::vector<float> l = logits;
        REQUIRE(sample(l, p, none, rng) == 2);
    }
}

TEST_CASE("same seed is reproducible", "[sampling]") {
    const std::vector<float> logits{1.0f, 1.0f, 1.0f, 1.0f};
    SamplingParams p;
    p.temperature = 1.0f;
    std::mt19937_64 a(42), b(42);
    for (int i = 0; i < 16; ++i) {
        std::vector<float> la = logits, lb = logits;
        REQUIRE(sample(la, p, none, a) ==
                sample(lb, p, none, b));
    }
}

TEST_CASE("a constraint restricts the choice", "[sampling]") {
    AllowSet c;
    c.ok = {2, 3};  // token 0 has the top logit but is disallowed
    SECTION("greedy picks the highest-logit allowed token") {
        std::vector<float> logits{5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        SamplingParams p;  // greedy
        REQUIRE(sample(logits, p, none, rng, &c) == 2);
    }
    SECTION("sampling only draws allowed tokens") {
        SamplingParams p;
        p.temperature = 1.0f;
        for (int i = 0; i < 30; ++i) {
            std::vector<float> l{5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
            const TokenId t = sample(l, p, none, rng, &c);
            REQUIRE(c.ok.count(t) == 1);
        }
    }
}

TEST_CASE("a dominant logit is always chosen under temperature",
          "[sampling]") {
    SamplingParams p;
    p.temperature = 1.0f;
    for (int i = 0; i < 50; ++i) {
        std::vector<float> l{0.0f, 40.0f, 0.0f};  // token 1 wins
        REQUIRE(sample(l, p, none, rng) == 1);
    }
}
