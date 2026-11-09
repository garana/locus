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

TEST_CASE("gbnf repetition does not grow the parse stacks", "[gbnf]") {
    // Right recursion through a repeat reuses the caller's frame, so a
    // long run stays cheap; recursion through nesting still works.
    CHECK(full(R"(root ::= "a"*)", std::string(20000, 'a')));
    const auto nest = Gbnf::parse(R"g(root ::= "(" root ")" | "x")g");
    CHECK(full(nest, "((((x))))"));
    CHECK_FALSE(full(nest, "((x)"));
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
