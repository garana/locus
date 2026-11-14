#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/model/gbnf.hpp"
#include "locus/model/grammar_constraint.hpp"
#include "locus/model/json_schema_gbnf.hpp"

using locus::model::Gbnf;
using locus::model::GbnfMatcher;
using locus::model::GrammarConstraint;
using locus::model::TokenTrie;
using locus::model::json_schema_to_gbnf;
using locus::model::json_value_gbnf;
using locus::model::regex_to_gbnf;

namespace {

/** @returns Whether the whole of `text` is a string of grammar `g`. */
bool full(const std::shared_ptr<const Gbnf>& g, const std::string& text) {
    GbnfMatcher m(g);
    for (const char c : text) {
        if (!m.feed(static_cast<std::uint8_t>(c))) return false;
    }
    return m.accepted();
}

bool full(const std::string& grammar, const std::string& text) {
    return full(Gbnf::parse(grammar), text);
}

/** A grammar with root ::= the regex. */
bool re_full(const std::string& re, const std::string& text) {
    return full("root ::= " + regex_to_gbnf(re), text);
}

bool schema_full(const std::string& schema, const std::string& text) {
    return full(json_schema_to_gbnf(schema), text);
}

}  // namespace

TEST_CASE("gbnf matches literals, classes, groups and repetition",
          "[gbnf]") {
    const auto g = Gbnf::parse(R"(
# a key list: {k1:12, k2:3}
root ::= "{" pair ("," ws pair)* "}"
pair ::= [a-z] [a-z0-9]* ":" [0-9]{1,3}
ws   ::= [ \t]?
)");
    CHECK(full(g, "{k1:12}"));
    CHECK(full(g, "{k1:12, k2:3}"));
    CHECK(full(g, "{k1:12,k2:999}"));
    CHECK_FALSE(full(g, "{k1:1234}"));  // {1,3}
    CHECK_FALSE(full(g, "{1k:1}"));     // must start with a letter
    CHECK_FALSE(full(g, "{k1:12"));     // incomplete
    CHECK_FALSE(full(g, "{}"));

    CHECK(full(R"(root ::= "a" | "b" "c"?)", "b"));
    CHECK(full(R"(root ::= "a" | "b" "c"?)", "bc"));
    CHECK_FALSE(full(R"(root ::= "a" | "b" "c"?)", "ac"));
    CHECK(full(R"(root ::= "ab"{2})", "abab"));
    CHECK_FALSE(full(R"(root ::= "ab"{2})", "ab"));
    CHECK(full(R"(root ::= [^0-9]+)", "x y"));
    CHECK_FALSE(full(R"(root ::= [^0-9]+)", "x1"));
    CHECK(full("root ::=\n  \"a\"\n  | \"b\"", "b"));
}

TEST_CASE("gbnf matches UTF-8 code points, also split across feeds",
          "[gbnf]") {
    const auto g = Gbnf::parse(R"(root ::= "caf" [à-ÿ] . )");
    CHECK(full(g, "caf\xC3\xA9!"));                   // é then any
    CHECK(full(g, "caf\xC3\xA9\xF0\x9F\x99\x82"));    // é then an emoji
    CHECK_FALSE(full(g, "cafe!"));
    GbnfMatcher m(g);
    for (const char c : std::string("caf")) m.feed(static_cast<std::uint8_t>(c));
    CHECK(m.feed(0xC3));         // could start a code point in the range
    CHECK_FALSE(m.accepted());
    GbnfMatcher m2(g);
    for (const char c : std::string("caf")) m2.feed(static_cast<std::uint8_t>(c));
    CHECK_FALSE(m2.feed(0xC2));  // U+0080..U+00BF: below the range

    // A lead byte promises a code point of its length's range only: a
    // 3- or 4-byte character can never be "{" (an overlong form).
    GbnfMatcher brace(Gbnf::parse(R"(root ::= "{")"));
    CHECK_FALSE(brace.feed(0xE0));
    CHECK_FALSE(brace.feed(0xF0));
    CHECK(brace.feed('{'));
    GbnfMatcher any(Gbnf::parse("root ::= ."));
    CHECK(any.feed(0xE0));
    CHECK_FALSE(any.feed(0x80));  // E0 80 .. would be overlong
    CHECK_FALSE(full("root ::= .", "\xED\xA0\x80"));  // a surrogate
}

TEST_CASE("gbnf rejects bad grammars", "[gbnf]") {
    CHECK_THROWS_AS(Gbnf::parse("root ::= missing"), std::invalid_argument);
    CHECK_THROWS_AS(Gbnf::parse("other ::= \"a\""), std::invalid_argument);
    CHECK_THROWS_AS(Gbnf::parse("root ::= root \"a\" | \"b\""),
                    std::invalid_argument);  // left recursion
    CHECK_THROWS_AS(Gbnf::parse("root ::= (\"a\"?)*"),
                    std::invalid_argument);  // repeats an empty match
    CHECK_THROWS_AS(Gbnf::parse("root ::= \"a"), std::invalid_argument);
    CHECK_THROWS_AS(Gbnf::parse("root ::= [z-a]"), std::invalid_argument);
    CHECK_THROWS_AS(Gbnf::parse("root ::= \"a\"{5000}"),
                    std::invalid_argument);
}

// i#625 F1: each of these killed the server with a stack overflow (the
// parsers, the left-recursion check and the matcher's expansion
// recursed once per level or per rule).
TEST_CASE("gbnf survives deep and long grammars", "[gbnf]") {
    // Nesting past 64 groups is refused, in a grammar and in a regex;
    // 64 still parses.
    const auto parens = [](int n) {
        return std::string(static_cast<std::size_t>(n), '(') + "\"a\"" +
               std::string(static_cast<std::size_t>(n), ')');
    };
    CHECK_THROWS_AS(Gbnf::parse("root ::= " + parens(1000)),
                    std::invalid_argument);
    CHECK(full("root ::= " + parens(64), "a"));
    const std::string deep_re = std::string(1000, '(') + "a" +
                                std::string(1000, ')');
    CHECK_THROWS_AS(regex_to_gbnf(deep_re), std::invalid_argument);
    CHECK(re_full(std::string(64, '(') + "a" + std::string(64, ')'), "a"));

    // A chain of 5000 rules: r0 ::= r1, ..., r4999 ::= "a".
    std::string chain = "root ::= r0\n";
    for (int i = 0; i < 4999; ++i) {
        chain += "r" + std::to_string(i) + " ::= r" + std::to_string(i + 1) +
                 "\n";
    }
    chain += "r4999 ::= \"a\"\n";
    CHECK(full(chain, "a"));
    CHECK_FALSE(full(chain, "b"));

    // The review's 143-byte body: a nullable unit repeated 4096 times.
    CHECK(re_full("(a?){0,4096}", std::string(10, 'a')));
    CHECK(re_full("(a?){0,4096}", ""));
    // Nested nullable repeats: still a matcher, maybe truncated.
    const auto g = Gbnf::parse("root ::= " +
                               regex_to_gbnf("((a?){0,64}){0,64}"));
    GbnfMatcher m(g);
    for (int i = 0; i < 20; ++i) REQUIRE(m.feed('a'));
    CHECK(m.accepted());

    // So many copies that the grammar is refused, not built.
    std::string big = "root ::=";
    for (int i = 0; i < 300; ++i) big += " \"a\"{4096}";
    CHECK_THROWS_AS(Gbnf::parse(big), std::invalid_argument);
}

TEST_CASE("gbnf repetition does not grow the parse stacks", "[gbnf]") {
    // Right recursion through a repeat reuses the caller's frame, so a
    // long run stays cheap; recursion through nesting still works.
    CHECK(full(R"(root ::= "a"*)", std::string(20000, 'a')));
    const auto nest = Gbnf::parse(R"g(root ::= "(" root ")" | "x")g");
    CHECK(full(nest, "((((x))))"));
    CHECK_FALSE(full(nest, "((x)"));
}

TEST_CASE("gbnf flags a state that drops stacks past the cap", "[gbnf]") {
    // After n "a"s, any number of them may still wait for a "b", so the
    // distinct stacks grow with n.
    const auto g = Gbnf::parse(R"(root ::= "a" root "b" | "a" root | )");
    GbnfMatcher wide(g);
    GbnfMatcher capped(g, 8);
    CHECK_FALSE(capped.truncated());
    const std::uint64_t before = locus::model::gbnf_stack_overflows();
    for (int i = 0; i < 20; ++i) {
        REQUIRE(wide.feed('a'));
        REQUIRE(capped.feed('a'));
    }
    CHECK(capped.truncated());
    CHECK_FALSE(wide.truncated());  // far below the default cap
    // One path, counted once, however many fed states were over.
    CHECK(locus::model::gbnf_stack_overflows() == before + 1);
    // The flag stays once set, and a copy stepped further keeps the cap.
    REQUIRE(capped.feed('b'));
    CHECK(capped.truncated());
    GbnfMatcher out(g);
    REQUIRE(capped.step('b', out));
    CHECK(out.truncated());
    // Tries with step(), as the token mask does, are not counted.
    GbnfMatcher fresh(g, 8), probe(g, 8);
    const std::uint64_t mid = locus::model::gbnf_stack_overflows();
    for (int i = 0; i < 20; ++i) {
        REQUIRE(fresh.step('a', probe));
        fresh = probe;
    }
    CHECK(fresh.truncated());
    CHECK(locus::model::gbnf_stack_overflows() == mid);
}

TEST_CASE("gbnf step carries the stack cap into its output", "[gbnf]") {
    const auto g = Gbnf::parse(R"(root ::= "a" root "b" | "a" root | )");
    GbnfMatcher capped(g, 8);
    GbnfMatcher out(g);  // the default cap until a step writes into it
    GbnfMatcher cur = capped;
    for (int i = 0; i < 20; ++i) {
        REQUIRE(cur.step('a', out));
        cur = out;
    }
    CHECK(cur.truncated());  // capped at 8 all along, not at 4096
}

TEST_CASE("gbnf flags an expansion past the word cap", "[gbnf]") {
    // Each level keeps a "z" to match after it, so the stacks at the
    // bottom are about 50 words deep, and four of them are far more
    // than 64 words: the expansion stops there, under the stack cap and
    // the visit budget.
    std::string g = "root ::= r0\n";
    for (int i = 0; i < 50; ++i) {
        g += "r" + std::to_string(i) + " ::= r" + std::to_string(i + 1) +
             " \"z\"\n";
    }
    g += "r50 ::= \"a\" | \"b\" | \"c\" | \"d\"\n";
    const auto gr = Gbnf::parse(g);
    const GbnfMatcher wide(gr);
    const GbnfMatcher small(gr, GbnfMatcher::kMaxStacks, 64);
    CHECK_FALSE(wide.truncated());
    CHECK(small.truncated());
    CHECK(full(gr, "d" + std::string(50, 'z')));
}

TEST_CASE("gbnf step shares the visit budget across stacks", "[gbnf]") {
    // After "x", h's expansion visits about 2^20 stacks (each c is empty
    // two ways), past the step's budget; the other alternative's stack
    // still gets its share, whichever order the two are in.
    std::string h = "h ::=";
    for (int i = 0; i < 20; ++i) h += " c";
    h += " \"z\"\nc ::= \"a\"? | \"b\"?\n";
    for (const char* root : {"root ::= \"x\" h | \"x\" \"y\"\n",
                             "root ::= \"x\" \"y\" | \"x\" h\n"}) {
        CAPTURE(root);
        GbnfMatcher m(Gbnf::parse(std::string(root) + h));
        CHECK_FALSE(m.truncated());
        REQUIRE(m.feed('x'));
        CHECK(m.truncated());
        REQUIRE(m.feed('y'));
        CHECK(m.accepted());
    }
}

TEST_CASE("gbnf step's fair split loses no expansion the budget allows",
          "[gbnf]") {
    // After "x", 64 stacks match: h's expansion needs more than its
    // first share (1/64 of the budget) but far less than the budget, so
    // the second pass completes it, in either alternative order.
    std::string h = "h ::=";
    for (int i = 0; i < 12; ++i) h += " c";
    h += " \"z\"\nc ::= \"a\"? | \"b\"?\n";
    std::string light;
    for (int i = 0; i < 63; ++i) {
        light += " | \"x\" \"y" + std::to_string(i) + "\"";
    }
    for (const bool heavy_first : {true, false}) {
        CAPTURE(heavy_first);
        const std::string root =
            heavy_first ? "root ::= \"x\" h" + light + "\n"
                        : "root ::= " + light.substr(3) + " | \"x\" h\n";
        GbnfMatcher m(Gbnf::parse(root + h));
        REQUIRE(m.feed('x'));
        CHECK_FALSE(m.truncated());
        GbnfMatcher a = m;
        for (int i = 0; i < 12; ++i) REQUIRE(a.feed('a'));
        REQUIRE(a.feed('z'));
        CHECK(a.accepted());
    }
}

TEST_CASE("json schema grammars stay under the stack cap", "[gbnf]") {
    const std::string s = R"({
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "n": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
      },
      "required": ["name"]
    })";
    GbnfMatcher m(Gbnf::parse(json_schema_to_gbnf(s)));
    for (const unsigned char c :
         std::string(R"({"name": "x", "tags": ["a", "b"], "n": 12})")) {
        REQUIRE(m.feed(c));
    }
    CHECK(m.accepted());
    CHECK_FALSE(m.truncated());
}

TEST_CASE("regex to gbnf", "[gbnf]") {
    CHECK(re_full(R"(^\d{3}-\d{4}$)", "555-1234"));
    CHECK_FALSE(re_full(R"(^\d{3}-\d{4}$)", "55-1234"));
    CHECK(re_full("(cat|dog)s?", "dogs"));
    CHECK_FALSE(re_full("(cat|dog)s?", "cow"));
    CHECK(re_full("[A-Z][a-z]+ [A-Z]\\.", "Ada L."));
    CHECK(re_full("(?:ab)+c*?", "ababcc"));
    CHECK(re_full("\\w+@\\w+\\.com", "me@site.com"));
    CHECK_FALSE(re_full("\\S+", "a b"));
    CHECK(re_full("a.c", "abc"));
    CHECK_FALSE(re_full("a.c", "a\nc"));
    CHECK_THROWS_AS(regex_to_gbnf("(a)\\1"), std::invalid_argument);
    CHECK_THROWS_AS(regex_to_gbnf("(?=a)"), std::invalid_argument);
    CHECK_THROWS_AS(regex_to_gbnf("a^b"), std::invalid_argument);
}

TEST_CASE("regex inside a JSON string matches the escaped text",
          "[gbnf]") {
    const auto g = [](const std::string& re) {
        return Gbnf::parse("root ::= " + regex_to_gbnf(re, true));
    };
    // . may not take a raw quote, backslash or control character.
    CHECK(full(g("a.c"), "abc"));
    CHECK_FALSE(full(g("a.c"), "a\"c"));
    // A literal quote or newline in the regex is written escaped.
    CHECK(full(g("say \"hi\""), "say \\\"hi\\\""));
    CHECK(full(g("a\\nb"), "a\\nb"));
    // A class with a quote takes the escaped quote.
    CHECK(full(g("[a\"]+"), "a\\\"a"));
    CHECK_FALSE(full(g("[a\"]+"), "a\"a"));
}

TEST_CASE("json value grammar", "[gbnf]") {
    const auto g = Gbnf::parse(json_value_gbnf());
    CHECK(full(g, R"({"a": [1, -2.5e3, true, null], "b": {"c": "d\né"}})"));
    CHECK(full(g, "[]"));
    CHECK(full(g, "{\n  \"k\": 1\n}"));
    CHECK_FALSE(full(g, R"({"a": 1,})"));
    CHECK_FALSE(full(g, R"({"a": 01})"));
    CHECK_FALSE(full(g, R"({'a': 1})"));
    CHECK_FALSE(full(g, "{\"a\": \"x\ny\"}"));  // raw newline in a string
}

TEST_CASE("json schema objects keep order and need the required keys",
          "[gbnf]") {
    const std::string s = R"({
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"}
      },
      "required": ["name", "age"]
    })";
    CHECK(schema_full(s, R"({"name": "Ada", "age": 36})"));
    CHECK(schema_full(s, R"({"name": "Ada", "age": 36, "email": "a@b"})"));
    CHECK_FALSE(schema_full(s, R"({"name": "Ada"})"));
    CHECK_FALSE(schema_full(s, R"({"age": 36, "name": "Ada"})"));
    CHECK_FALSE(schema_full(s, R"({"name": "Ada", "age": "36"})"));
    CHECK_FALSE(schema_full(s, R"({"name": "Ada", "age": 36, "x": 1})"));

    // Only optional members: the first one present takes no comma.
    const std::string opt = R"({"type": "object", "properties": {
      "a": {"type": "integer"}, "b": {"type": "integer"}}})";
    CHECK(schema_full(opt, "{}"));
    CHECK(schema_full(opt, R"({"b": 2})"));
    CHECK(schema_full(opt, R"({"a": 1, "b": 2})"));
    CHECK_FALSE(schema_full(opt, R"({, "b": 2})"));

    // additionalProperties as a schema.
    const std::string map = R"({"type": "object",
      "additionalProperties": {"type": "number"}})";
    CHECK(schema_full(map, R"({"x": 1, "y": 2.5})"));
    CHECK_FALSE(schema_full(map, R"({"x": "1"})"));
}

TEST_CASE("json schema enums, arrays, refs and string formats", "[gbnf]") {
    const std::string e = R"({"enum": ["red", "green", 3, null]})";
    CHECK(schema_full(e, "\"green\""));
    CHECK(schema_full(e, "3"));
    CHECK(schema_full(e, "null"));
    CHECK_FALSE(schema_full(e, "\"blue\""));

    const std::string arr = R"({"type": "array", "items": {"type": "boolean"},
      "minItems": 1, "maxItems": 2})";
    CHECK(schema_full(arr, "[true]"));
    CHECK(schema_full(arr, "[true, false]"));
    CHECK_FALSE(schema_full(arr, "[]"));
    CHECK_FALSE(schema_full(arr, "[true, false, true]"));

    const std::string tuple = R"({"prefixItems": [{"type": "string"},
      {"type": "integer"}]})";
    CHECK(schema_full(tuple, R"(["a", 1])"));
    CHECK_FALSE(schema_full(tuple, R"([1, "a"])"));

    const std::string tree = R"({
      "$defs": {"node": {"type": "object",
        "properties": {"v": {"type": "integer"},
                       "kids": {"type": "array",
                                "items": {"$ref": "#/$defs/node"}}},
        "required": ["v"]}},
      "$ref": "#/$defs/node"})";
    CHECK(schema_full(tree, R"({"v": 1, "kids": [{"v": 2}, {"v": 3,
      "kids": []}]})"));
    CHECK_FALSE(schema_full(tree, R"({"v": 1, "kids": [{"w": 2}]})"));

    const std::string any = R"({"anyOf": [{"type": "integer"},
      {"type": "string", "maxLength": 2}]})";
    CHECK(schema_full(any, "12"));
    CHECK(schema_full(any, "\"ab\""));
    CHECK_FALSE(schema_full(any, "\"abc\""));

    CHECK(schema_full(R"({"type": "string", "format": "date"})",
                      "\"2026-11-09\""));
    CHECK_FALSE(schema_full(R"({"type": "string", "format": "date"})",
                            "\"2026-13-09\""));
    CHECK(schema_full(R"({"type": "string", "pattern": "^[A-Z]{2}\\d+$"})",
                      "\"AB12\""));
    CHECK_FALSE(schema_full(
        R"({"type": "string", "pattern": "^[A-Z]{2}\\d+$"})", "\"A12\""));
    CHECK(schema_full(R"({"type": ["integer", "null"]})", "null"));

    CHECK_THROWS_AS(json_schema_to_gbnf(R"({"$ref": "#/$defs/none"})"),
                    std::invalid_argument);
    CHECK_THROWS_AS(json_schema_to_gbnf("{"), std::invalid_argument);
    CHECK_THROWS_AS(json_schema_to_gbnf(R"({"type": "tuple"})"),
                    std::invalid_argument);
}

TEST_CASE("grammar constraint allows the tokens that continue a match",
          "[gbnf]") {
    // A toy vocabulary: 0 is EOS (no bytes), 6 and 7 split "é".
    auto pieces = std::make_shared<std::vector<std::string>>(
        std::vector<std::string>{"", "{\"", "a", "\":", "1", "}",
                                 "\xC3", "\xA9", "1}", "x"});
    const auto trie = std::make_shared<const TokenTrie>(pieces);
    const auto g = Gbnf::parse(R"(root ::= "{\"" ("a" | "é") "\":" [0-9]+ "}")");
    GrammarConstraint c(g, trie, 0);

    const auto allowed = [&] {
        std::vector<int> out;
        for (int t = 0; t < 10; ++t) {
            if (c.allows(t)) out.push_back(t);
        }
        return out;
    };
    CHECK(allowed() == std::vector<int>{1});
    c.commit(1);
    CHECK(allowed() == std::vector<int>{2, 6});
    c.commit(6);  // the first byte of é
    CHECK(allowed() == std::vector<int>{7});
    c.commit(7);
    CHECK(allowed() == std::vector<int>{3});
    c.commit(3);
    CHECK(allowed() == std::vector<int>{4, 8});
    c.commit(8);
    CHECK(c.accepted());
    CHECK(allowed() == std::vector<int>{0});  // only EOS is left
}

TEST_CASE("grammar constraint ends a match that only whitespace follows",
          "[gbnf]") {
    // 0 is EOS; 3, 4 and 5 are whitespace, 6 is a digit.
    auto pieces = std::make_shared<std::vector<std::string>>(
        std::vector<std::string>{"", "{", "}", " ", "\n", "\t ", "7"});
    const auto trie = std::make_shared<const TokenTrie>(pieces);
    const auto allowed = [&](GrammarConstraint& c) {
        std::vector<int> out;
        for (int t = 0; t < 7; ++t) {
            if (c.allows(t)) out.push_back(t);
        }
        return out;
    };
    SECTION("a JSON object stops at its closing brace") {
        GrammarConstraint c(Gbnf::parse(json_value_gbnf(), "object"), trie,
                            0, /*stop_at_space=*/true);
        c.commit(1);
        // No EOS yet; a tab may only follow a newline.
        CHECK(allowed(c) == std::vector<int>{2, 3, 4});
        c.commit(2);
        CHECK(c.accepted());
        CHECK(allowed(c) == std::vector<int>{0});  // not the trailing space
    }
    SECTION("a match that can still grow keeps its continuations") {
        // "7" is complete, but more digits may follow.
        GrammarConstraint c(Gbnf::parse(R"(root ::= [0-9]+ " "?)"), trie, 0,
                            /*stop_at_space=*/true);
        c.commit(6);
        CHECK(allowed(c) == std::vector<int>{0, 3, 6});
    }
    SECTION("a grammar that goes on after a space is not cut") {
        // "7" then " 7"... in a vocabulary with no " 7" token: after a
        // complete "7" the mask holds only the space, which leads to
        // more digits. Only the JSON formats stop at whitespace.
        const auto g = Gbnf::parse(R"(root ::= [0-9] (" " [0-9])*)");
        GrammarConstraint c(g, trie, 0);
        c.commit(6);
        CHECK(allowed(c) == std::vector<int>{0, 3});
        GrammarConstraint json_like(g, trie, 0, /*stop_at_space=*/true);
        json_like.commit(6);
        CHECK(allowed(json_like) == std::vector<int>{0});  // would cut
    }
}

TEST_CASE("the json value grammar parses with an object root", "[gbnf]") {
    const auto g = Gbnf::parse(json_value_gbnf(), "object");
    const auto full = [&](std::string_view s) {
        GbnfMatcher m(g);
        for (const unsigned char c : s) {
            if (!m.feed(c)) return false;
        }
        return m.accepted();
    };
    CHECK(full(R"({"a": [1, 2]})"));
    CHECK(full("{}"));
    CHECK_FALSE(full("1.1"));
    CHECK_FALSE(full(R"("text")"));
    CHECK_FALSE(full("[1]"));
}
