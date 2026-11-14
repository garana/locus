#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace locus::model {

/**
 * A context-free grammar in GBNF, llama.cpp's EBNF dialect (i#402):
 *
 *     root   ::= "{" ws pair ("," ws pair)* "}"   # comment
 *     pair   ::= [a-z]+ ":" ws [0-9]{1,3}
 *     ws     ::= [ \t\n]*
 *
 * Elements: "literal", [class] and [^class] (ranges, escapes \n \r
 * \t \\ \xHH \uHHHH \UHHHHHHHH), `.` (any character), a rule name,
 * ( group ), and the postfix operators * + ? {m} {m,} {m,n}. A rule
 * ends at a newline outside parentheses, unless the next line starts
 * with `|`; a newline may follow `::=` or `|`. Characters are Unicode
 * code points; the input is UTF-8.
 *
 * Left recursion (a rule that can reach itself before consuming a
 * character, including `x*` of an x that can be empty) is rejected,
 * since the matcher would loop on it.
 */
class Gbnf {
  public:
    /**
     * @param text The grammar.
     * @param root The start rule.
     * @returns The parsed grammar.
     * @throws std::invalid_argument on a syntax error, an undefined
     *     rule, a missing root or left recursion.
     */
    static std::shared_ptr<const Gbnf> parse(std::string_view text,
                                             std::string_view root = "root");

    /** One grammar element. */
    struct Elem {
        enum Kind : std::uint8_t { kEnd, kChar, kRef };
        Kind kind = kEnd;
        /** kChar: matches a code point outside the ranges. */
        bool negate = false;
        /** kRef: the rule. */
        std::uint32_t ref = 0;
        /** kChar: [begin, end) into ranges(). */
        std::uint32_t r_begin = 0, r_end = 0;
    };

    const std::vector<Elem>& elems() const { return elems_; }
    const std::vector<std::pair<char32_t, char32_t>>& ranges() const {
        return ranges_;
    }
    /** @returns The first element of each alternative of rule `r`. */
    const std::vector<std::uint32_t>& alts(std::uint32_t r) const {
        return alts_[r];
    }
    std::uint32_t root() const { return root_; }
    /** @returns A kRef element to `root`, where a match starts. */
    std::uint32_t start() const { return start_; }

    /** @returns Whether the kChar element at `p` matches `c`. */
    bool matches(std::uint32_t p, char32_t c) const;
    /**
     * @returns Whether the kChar element at `p` matches some code point
     *     in [lo, hi] (a partly decoded UTF-8 character).
     */
    bool may_match(std::uint32_t p, char32_t lo, char32_t hi) const;

  private:
    friend class GbnfBuilder;
    std::vector<Elem> elems_;
    std::vector<std::pair<char32_t, char32_t>> ranges_;
    std::vector<std::vector<std::uint32_t>> alts_;
    std::uint32_t root_ = 0;
    std::uint32_t start_ = 0;
};

/**
 * Where a Gbnf parse stands after some UTF-8 bytes: the set of
 * pushdown stacks (each a list of element positions, the top one a
 * character to match next) that the input so far allows. Copyable,
 * so a caller can try bytes on a copy; step() writes the next state
 * into a matcher the caller reuses, which allocates nothing once its
 * buffers have grown (the per-token mask walk depends on that).
 */
class GbnfMatcher {
  public:
    /** Parse stacks kept per state by default. */
    static constexpr std::size_t kMaxStacks = 4096;

    /**
     * @param g The grammar.
     * @param max_stacks The most parse stacks a state keeps; past it the
     *     rest are dropped, truncated() turns true and
     *     gbnf_stack_overflows() counts it.
     */
    explicit GbnfMatcher(std::shared_ptr<const Gbnf> g,
                         std::size_t max_stacks = kMaxStacks);

    /**
     * Advances on one byte.
     * @returns false, with the state unchanged, if no string of the
     *     grammar continues this way.
     */
    bool feed(std::uint8_t b);

    /**
     * The state after byte `b`, written into `out` (a matcher of the
     * same grammar, whose buffers are reused).
     * @returns false if no string of the grammar continues this way;
     *     `out` is then unspecified.
     */
    bool step(std::uint8_t b, GbnfMatcher& out) const;

    /** @returns Whether the input so far is a complete string. */
    bool accepted() const;

    /** @returns Whether some character may still follow. */
    bool can_continue() const;

    /**
     * @returns The state as bytes: two matchers of one grammar with
     *     equal keys accept the same continuations.
     */
    std::string key() const;

    /**
     * @returns Whether some state on the way here had more parse stacks
     *     than the cap, so continuations the grammar allows may be
     *     missing.
     */
    bool truncated() const { return truncated_; }

  private:
    /** Stack i is data[i ? ends[i - 1] : 0, ends[i]). */
    struct Stacks {
        std::vector<std::uint32_t> data;
        std::vector<std::uint32_t> ends;
        std::size_t size() const { return ends.size(); }
        std::uint32_t begin(std::size_t i) const { return i ? ends[i - 1] : 0; }
        void clear() {
            data.clear();
            ends.clear();
        }
        void push(const std::vector<std::uint32_t>& s);
    };
    /**
     * Appends to `out` every stack `cur` reaches before consuming a
     * character, over an explicit worklist (no recursion, i#625 F1).
     * @param budget Stack visits left for this step; spent here.
     * @returns Whether the budget ran out, so stacks were dropped.
     */
    bool expand(const std::vector<std::uint32_t>& cur, Stacks& out,
                std::size_t& budget) const;
    bool viable(char32_t lo, char32_t hi) const;
    /** Sorts and dedups s, keeping max_stacks_. @returns whether any
     * distinct stack was dropped. */
    bool dedup(Stacks& s) const;
    /** Counts a newly truncated path in gbnf_stack_overflows(). */
    void count_overflow() const;

    std::shared_ptr<const Gbnf> g_;
    std::size_t max_stacks_ = kMaxStacks;
    bool truncated_ = false;
    Stacks stacks_;
    // Scratch for step(): the stack being expanded, the dedup order
    // and copy.
    mutable std::vector<std::uint32_t> cur_;
    mutable std::vector<std::uint32_t> order_;
    mutable Stacks tmp_;
    // Scratch for expand(): its worklist and the stack it works on.
    mutable Stacks work_;
    mutable std::vector<std::uint32_t> top_;
    // A UTF-8 character in progress: its bits so far, the continuation
    // bytes left, and its continuation bytes in all.
    char32_t cp_ = 0;
    int need_ = 0;
    int len_ = 0;
};

/**
 * @returns How many matcher paths in this process became truncated:
 *     a state built or fed (GbnfMatcher's constructor or feed(), the
 *     path the output takes) had more parse stacks than the cap and
 *     dropped the rest, which can hide a continuation the grammar
 *     allows. Each path counts once; the states a token mask only
 *     tries (step()) do not count. The first is also logged to stderr.
 */
std::uint64_t gbnf_stack_overflows();

/**
 * The GBNF body of a regex (no rule name), for a rule or an element in
 * parentheses. The whole string must match, so ^ at the start and $ at
 * the end are optional and ignored. Supported: literals, `.`, classes
 * (with \d \w \s inside), \d \w \s \D \W \S, escapes, groups ( ) and
 * (?: ), `|`, and * + ? {m} {m,} {m,n} (a lazy `?` after them is
 * accepted and ignored).
 * @param re The regex.
 * @param json_string The text is inside a JSON string: `.`, negated
 *     classes and \D \W \S exclude `"`, `\` and control characters, and
 *     a literal `"` or `\` matches its JSON escape.
 * @throws std::invalid_argument on unsupported syntax (backreferences,
 *     lookaround, anchors inside the pattern).
 */
std::string regex_to_gbnf(std::string_view re, bool json_string = false);

}  // namespace locus::model
