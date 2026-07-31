#include <string>

#include "catch_amalgamated.hpp"
#include "locus/model/grammar.hpp"

using locus::model::JsonGrammar;

namespace {
/** Feeds every byte; @returns false at the first rejection. */
bool feed_all(JsonGrammar& g, const std::string& s) {
    for (unsigned char c : s) {
        if (!g.feed(c)) {
            return false;
        }
    }
    return true;
}
bool accepts_complete(const std::string& s) {
    JsonGrammar g;
    return feed_all(g, s) && g.complete();
}
bool rejects(const std::string& s) {
    JsonGrammar g;
    return !feed_all(g, s);
}
}  // namespace

TEST_CASE("JsonGrammar accepts complete values", "[grammar]") {
    for (const char* s :
         {"{}", "[]", "\"hi\"", "123", "-1.5e3", "true", "false",
          "null", "{\"a\":1}", "[1,2,3]",
          "{\"a\":[1,{\"b\":\"c\"}],\"d\":null}",
          "  {  \"x\" : 1 }  ", "[ -0.5 , 2 ]"}) {
        INFO(s);
        REQUIRE(accepts_complete(s));
    }
}

TEST_CASE("JsonGrammar: incomplete prefixes are not complete",
          "[grammar]") {
    for (const char* s :
         {"{", "{\"a\"", "{\"a\":", "[1,", "\"ab", "tr", "-",
          "[1"}) {
        INFO(s);
        JsonGrammar g;
        REQUIRE(feed_all(g, s));       // valid so far
        REQUIRE_FALSE(g.complete());   // but not finished
    }
}

TEST_CASE("JsonGrammar rejects malformed input", "[grammar]") {
    for (const char* s :
         {"{,}", "[,]", "[1,]", "{\"a\" 1}", "{1:2}", "trux",
          "{}x", "01a", "[1 2]", "}"}) {
        INFO(s);
        REQUIRE(rejects(s));
    }
}

TEST_CASE("JsonGrammar: number then delimiter reprocesses",
          "[grammar]") {
    REQUIRE(accepts_complete("[12,34]"));
    REQUIRE(accepts_complete("{\"n\":42}"));
    REQUIRE(accepts_complete("42"));
    REQUIRE(accepts_complete("42   "));
    REQUIRE(rejects("42x"));
}

TEST_CASE("JsonGrammar: once complete only whitespace continues",
          "[grammar]") {
    JsonGrammar g;
    REQUIRE(feed_all(g, "{}"));
    REQUIRE(g.complete());
    REQUIRE(g.feed(' '));   // trailing ws ok
    REQUIRE(g.complete());
    REQUIRE_FALSE(g.feed('x'));  // but no more content
}
