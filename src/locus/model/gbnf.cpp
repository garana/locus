#include "locus/model/gbnf.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>

namespace locus::model {

namespace {

/** Repetition bounds above this are rejected (grammar size). */
constexpr std::uint32_t kMaxRepeat = 4096;
/** No unbounded upper limit in {m,n}. */
constexpr std::uint32_t kInf = std::numeric_limits<std::uint32_t>::max();
/** Groups nested deeper than this, in a grammar or a regex, are
 *  rejected: the parsers recurse once per level (i#625 F1). */
constexpr int kMaxNesting = 64;
/** A grammar whose rules hold more items than this is rejected: each
 *  repetition copies its unit up to kMaxRepeat times. */
constexpr std::size_t kMaxItems = std::size_t{1} << 20;
/** Most stacks one step's expansion may visit; past it the rest are
 *  dropped and the state is truncated, like a stack past kMaxStacks.
 *  Nested optional repetitions can otherwise make one byte cost
 *  millions of visits. */
constexpr std::size_t kMaxExpandVisits = std::size_t{1} << 18;
/** States that dropped stacks past the cap (gbnf_stack_overflows). */
std::atomic<std::uint64_t> stack_overflows{0};

/** One element before flattening: a character class or a rule. */
struct Item {
    bool is_ref = false;
    std::uint32_t ref = 0;
    bool negate = false;
    std::vector<std::pair<char32_t, char32_t>> ranges;
};
using Seq = std::vector<Item>;
using Alts = std::vector<Seq>;

Item ref_item(std::uint32_t r) {
    Item it;
    it.is_ref = true;
    it.ref = r;
    return it;
}

Item char_item(char32_t c) {
    Item it;
    it.ranges.push_back({c, c});
    return it;
}

bool is_name_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '-' || c == '_';
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/**
 * Decodes one UTF-8 character at s[p], advancing p.
 * @returns false on invalid UTF-8.
 */
bool decode_utf8(std::string_view s, std::size_t& p, char32_t& cp) {
    const auto b0 = static_cast<unsigned char>(s[p]);
    int n = 0;
    if (b0 < 0x80) {
        cp = b0;
    } else if (b0 >= 0xC2 && b0 <= 0xDF) {
        cp = b0 & 0x1F;
        n = 1;
    } else if (b0 >= 0xE0 && b0 <= 0xEF) {
        cp = b0 & 0x0F;
        n = 2;
    } else if (b0 >= 0xF0 && b0 <= 0xF4) {
        cp = b0 & 0x07;
        n = 3;
    } else {
        return false;
    }
    if (n > 0 && p + static_cast<std::size_t>(n) >= s.size()) {
        return false;  // truncated
    }
    for (int i = 1; i <= n; ++i) {
        const auto b = static_cast<unsigned char>(s[p + i]);
        if ((b & 0xC0) != 0x80) return false;
        cp = (cp << 6) | (b & 0x3F);
    }
    p += 1 + n;
    return true;
}

}  // namespace

/** Parses GBNF text into a Gbnf (see gbnf.hpp for the syntax). */
class GbnfBuilder {
  public:
    explicit GbnfBuilder(std::string_view t) : t_(t) {}

    std::shared_ptr<const Gbnf> build(std::string_view root_name) {
        skip_ws(true);
        while (p_ < t_.size()) {
            const std::string name = parse_name();
            skip_ws(false);
            if (t_.substr(p_, 3) != "::=") fail("expected ::=");
            p_ += 3;
            skip_ws(true);
            Alts a = parse_alts(false);
            const std::uint32_t r = id_of(name);
            if (defined_[r]) fail("rule '" + name + "' defined twice");
            rules_[r] = std::move(a);
            defined_[r] = true;
            skip_ws(false);
            if (p_ < t_.size() && t_[p_] != '\n' && t_[p_] != '\r') {
                fail("expected the end of the rule");
            }
            skip_ws(true);
        }
        for (std::uint32_t r = 0; r < rules_.size(); ++r) {
            if (!defined_[r]) {
                throw std::invalid_argument("grammar: undefined rule '" +
                                            names_[r] + "'");
            }
        }
        const auto it = ids_.find(root_name);
        if (it == ids_.end()) {
            throw std::invalid_argument("grammar: no rule '" +
                                        std::string(root_name) + "'");
        }
        check_left_recursion();
        return flatten(it->second);
    }

  private:
    [[noreturn]] void fail(const std::string& why) const {
        // Report the line and column of the error.
        std::size_t line = 1, col = 1;
        for (std::size_t i = 0; i < p_ && i < t_.size(); ++i) {
            if (t_[i] == '\n') {
                ++line;
                col = 1;
            } else {
                ++col;
            }
        }
        throw std::invalid_argument("grammar: " + why + " at line " +
                                    std::to_string(line) + ", column " +
                                    std::to_string(col));
    }

    /**
     * Skips spaces, tabs and comments; newlines too when newline_ok.
     */
    void skip_ws(bool newline_ok) {
        while (p_ < t_.size()) {
            const char c = t_[p_];
            if (c == ' ' || c == '\t') {
                ++p_;
            } else if (c == '#') {
                while (p_ < t_.size() && t_[p_] != '\n') ++p_;
            } else if ((c == '\n' || c == '\r') && newline_ok) {
                ++p_;
            } else {
                break;
            }
        }
    }

    std::string parse_name() {
        const std::size_t b = p_;
        while (p_ < t_.size() && is_name_char(t_[p_])) ++p_;
        if (p_ == b) fail("expected a rule name");
        return std::string(t_.substr(b, p_ - b));
    }

    std::uint32_t id_of(std::string_view name) {
        const auto it = ids_.find(name);
        if (it != ids_.end()) return it->second;
        const auto r = static_cast<std::uint32_t>(rules_.size());
        rules_.emplace_back();
        names_.emplace_back(name);
        defined_.push_back(false);
        ids_.emplace(std::string(name), r);
        return r;
    }

    /** A new rule for a group or a repetition. */
    std::uint32_t fresh(Alts a) {
        const auto r = static_cast<std::uint32_t>(rules_.size());
        rules_.push_back(std::move(a));
        names_.push_back("<generated " + std::to_string(r) + ">");
        defined_.push_back(true);
        return r;
    }

    Alts parse_alts(bool nested) {
        Alts a;
        a.push_back(parse_seq(nested));
        while (true) {
            // A line that starts with | continues the rule.
            const std::size_t save = p_;
            skip_ws(true);
            if (p_ >= t_.size() || t_[p_] != '|') {
                p_ = save;
                break;
            }
            ++p_;
            skip_ws(true);
            a.push_back(parse_seq(nested));
        }
        return a;
    }

    Seq parse_seq(bool nested) {
        Seq s;
        while (true) {
            skip_ws(nested);
            if (p_ >= t_.size()) break;
            const char c = t_[p_];
            if (c == '|' || c == ')' || c == '\n' || c == '\r') break;
            Seq unit = parse_unit();
            if (p_ < t_.size() &&
                (t_[p_] == '*' || t_[p_] == '+' || t_[p_] == '?' ||
                 t_[p_] == '{')) {
                unit = parse_postfix(std::move(unit));
            }
            s.insert(s.end(), unit.begin(), unit.end());
        }
        return s;
    }

    /** One element: a literal (possibly several characters) or one
     *  class, `.`, group or rule reference. */
    Seq parse_unit() {
        const char c = t_[p_];
        if (c == '"') {
            ++p_;
            Seq s;
            while (true) {
                if (p_ >= t_.size()) fail("unterminated string");
                if (t_[p_] == '"') {
                    ++p_;
                    break;
                }
                s.push_back(char_item(parse_char()));
            }
            return s;
        }
        if (c == '[') {
            ++p_;
            Item it;
            if (p_ < t_.size() && t_[p_] == '^') {
                it.negate = true;
                ++p_;
            }
            while (true) {
                if (p_ >= t_.size()) fail("unterminated character class");
                if (t_[p_] == ']') {
                    ++p_;
                    break;
                }
                const char32_t lo = parse_char();
                char32_t hi = lo;
                if (p_ + 1 < t_.size() && t_[p_] == '-' &&
                    t_[p_ + 1] != ']') {
                    ++p_;
                    hi = parse_char();
                    if (hi < lo) fail("reversed character range");
                }
                it.ranges.push_back({lo, hi});
            }
            return {it};
        }
        if (c == '.') {
            ++p_;
            Item it;
            it.negate = true;  // nothing excluded: any character
            return {it};
        }
        if (c == '(') {
            if (++depth_ > kMaxNesting) {
                fail("groups nested deeper than " +
                     std::to_string(kMaxNesting) + " levels");
            }
            ++p_;
            skip_ws(true);
            Alts a = parse_alts(true);
            skip_ws(true);
            if (p_ >= t_.size() || t_[p_] != ')') fail("expected )");
            ++p_;
            --depth_;
            return {ref_item(fresh(std::move(a)))};
        }
        if (is_name_char(c)) {
            return {ref_item(id_of(parse_name()))};
        }
        fail(std::string("unexpected character '") + c + "'");
    }

    /** One character of a literal or class, with escapes. */
    char32_t parse_char() {
        if (t_[p_] != '\\') {
            char32_t cp = 0;
            if (!decode_utf8(t_, p_, cp)) fail("invalid UTF-8");
            return cp;
        }
        if (p_ + 1 >= t_.size()) fail("unterminated escape");
        const char e = t_[p_ + 1];
        p_ += 2;
        int digits = 0;
        switch (e) {
            case 'n': return U'\n';
            case 'r': return U'\r';
            case 't': return U'\t';
            case '\\': return U'\\';
            case '"': return U'"';
            case '[': return U'[';
            case ']': return U']';
            case '-': return U'-';
            case '^': return U'^';
            case 'x': digits = 2; break;
            case 'u': digits = 4; break;
            case 'U': digits = 8; break;
            default: fail(std::string("unknown escape \\") + e);
        }
        char32_t v = 0;
        for (int i = 0; i < digits; ++i) {
            const int h = p_ < t_.size() ? hex_value(t_[p_]) : -1;
            if (h < 0) fail("bad hex escape");
            v = v * 16 + static_cast<char32_t>(h);
            ++p_;
        }
        if (v > 0x10FFFF) fail("code point out of range");
        return v;
    }

    std::uint32_t parse_count() {
        const std::size_t b = p_;
        std::uint64_t v = 0;
        while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') {
            v = v * 10 + static_cast<std::uint64_t>(t_[p_] - '0');
            if (v > kMaxRepeat) fail("repetition count too large");
            ++p_;
        }
        if (p_ == b) fail("expected a number");
        return static_cast<std::uint32_t>(v);
    }

    Seq parse_postfix(Seq unit) {
        std::uint32_t lo = 0, hi = kInf;
        const char op = t_[p_++];
        if (op == '+') {
            lo = 1;
        } else if (op == '?') {
            hi = 1;
        } else if (op == '{') {
            skip_ws(false);
            lo = parse_count();
            skip_ws(false);
            if (p_ < t_.size() && t_[p_] == ',') {
                ++p_;
                skip_ws(false);
                if (p_ < t_.size() && t_[p_] != '}') {
                    hi = parse_count();
                    if (hi < lo) fail("repetition {m,n} with n < m");
                }
            } else {
                hi = lo;
            }
            skip_ws(false);
            if (p_ >= t_.size() || t_[p_] != '}') fail("expected }");
            ++p_;
        }
        return repeat(std::move(unit), lo, hi);
    }

    Seq repeat(Seq unit, std::uint32_t lo, std::uint32_t hi) {
        if (unit.empty()) return {};
        // lo copies, then two items per optional (or two for x R).
        items_ += lo + 2 * (hi == kInf ? 1 : hi - lo);
        if (items_ > kMaxItems) fail("grammar too large");
        const Item x =
            unit.size() == 1 ? unit[0] : ref_item(fresh({std::move(unit)}));
        Seq out(lo, x);
        if (hi == kInf) {
            // R ::= x R | (empty)
            const auto r = static_cast<std::uint32_t>(rules_.size());
            fresh({{x, ref_item(r)}, {}});
            out.push_back(ref_item(r));
        } else if (hi > lo) {
            // O_k ::= x O_(k-1) | (empty), k = hi - lo nested optionals.
            std::uint32_t tail = kInf;
            for (std::uint32_t k = 0; k < hi - lo; ++k) {
                Seq s{x};
                if (tail != kInf) s.push_back(ref_item(tail));
                tail = fresh({std::move(s), {}});
            }
            out.push_back(ref_item(tail));
        }
        return out;
    }

    /** Rejects a rule that can reach a rule before consuming a
     *  character and come back to it. */
    void check_left_recursion() const {
        const std::size_t n = rules_.size();
        std::vector<bool> nullable(n, false);
        for (bool changed = true; changed;) {
            changed = false;
            for (std::size_t r = 0; r < n; ++r) {
                if (nullable[r]) continue;
                for (const Seq& s : rules_[r]) {
                    const bool all = std::all_of(
                        s.begin(), s.end(), [&](const Item& it) {
                            return it.is_ref && nullable[it.ref];
                        });
                    if (all) {
                        nullable[r] = true;
                        changed = true;
                        break;
                    }
                }
            }
        }
        // first[r]: the rules r can enter before consuming a character.
        std::vector<std::vector<std::uint32_t>> first(n);
        for (std::size_t r = 0; r < n; ++r) {
            for (const Seq& s : rules_[r]) {
                for (const Item& it : s) {
                    if (!it.is_ref) break;
                    first[r].push_back(it.ref);
                    if (!nullable[it.ref]) break;
                }
            }
        }
        // A depth-first search for a cycle in first[], with an explicit
        // stack of (rule, next edge): a chain of thousands of rules would
        // overflow the thread's stack as recursion (i#625 F1).
        std::vector<int> color(n, 0);  // 0 new, 1 on path, 2 done
        std::vector<std::pair<std::uint32_t, std::size_t>> path;
        for (std::uint32_t root = 0; root < n; ++root) {
            if (color[root] != 0) continue;
            color[root] = 1;
            path.push_back({root, 0});
            while (!path.empty()) {
                auto& [r, k] = path.back();
                if (k == first[r].size()) {
                    color[r] = 2;
                    path.pop_back();
                    continue;
                }
                const std::uint32_t s = first[r][k++];
                if (color[s] == 1) {
                    throw std::invalid_argument(
                        "grammar: left recursion through rule '" +
                        names_[s] +
                        "' (or a repetition of something that can be "
                        "empty)");
                }
                if (color[s] == 0) {
                    color[s] = 1;
                    path.push_back({s, 0});
                }
            }
        }
    }

    std::shared_ptr<const Gbnf> flatten(std::uint32_t root) const {
        auto g = std::make_shared<Gbnf>();
        g->alts_.resize(rules_.size());
        for (std::size_t r = 0; r < rules_.size(); ++r) {
            for (const Seq& s : rules_[r]) {
                g->alts_[r].push_back(
                    static_cast<std::uint32_t>(g->elems_.size()));
                for (const Item& it : s) {
                    Gbnf::Elem e;
                    if (it.is_ref) {
                        e.kind = Gbnf::Elem::kRef;
                        e.ref = it.ref;
                    } else {
                        e.kind = Gbnf::Elem::kChar;
                        e.negate = it.negate;
                        e.r_begin =
                            static_cast<std::uint32_t>(g->ranges_.size());
                        g->ranges_.insert(g->ranges_.end(), it.ranges.begin(),
                                          it.ranges.end());
                        e.r_end = static_cast<std::uint32_t>(g->ranges_.size());
                    }
                    g->elems_.push_back(e);
                }
                g->elems_.push_back(Gbnf::Elem{});  // kEnd
            }
        }
        g->root_ = root;
        g->start_ = static_cast<std::uint32_t>(g->elems_.size());
        Gbnf::Elem start;
        start.kind = Gbnf::Elem::kRef;
        start.ref = root;
        g->elems_.push_back(start);
        g->elems_.push_back(Gbnf::Elem{});
        return g;
    }

    std::string_view t_;
    std::size_t p_ = 0;
    int depth_ = 0;          ///< groups open at p_
    std::size_t items_ = 0;  ///< items the repetitions added
    std::vector<Alts> rules_;
    std::vector<std::string> names_;
    std::vector<bool> defined_;
    std::map<std::string, std::uint32_t, std::less<>> ids_;
};

std::shared_ptr<const Gbnf> Gbnf::parse(std::string_view text,
                                        std::string_view root) {
    return GbnfBuilder(text).build(root);
}

bool Gbnf::matches(std::uint32_t p, char32_t c) const {
    const Elem& e = elems_[p];
    bool in = false;
    for (std::uint32_t i = e.r_begin; i < e.r_end; ++i) {
        if (c >= ranges_[i].first && c <= ranges_[i].second) {
            in = true;
            break;
        }
    }
    return in != e.negate;
}

bool Gbnf::may_match(std::uint32_t p, char32_t lo, char32_t hi) const {
    const Elem& e = elems_[p];
    for (std::uint32_t i = e.r_begin; i < e.r_end; ++i) {
        const auto [a, b] = ranges_[i];
        if (e.negate) {
            if (a <= lo && hi <= b) return false;  // all of them excluded
        } else if (a <= hi && lo <= b) {
            return true;
        }
    }
    return e.negate;
}

void GbnfMatcher::Stacks::push(const std::vector<std::uint32_t>& s) {
    data.insert(data.end(), s.begin(), s.end());
    ends.push_back(static_cast<std::uint32_t>(data.size()));
}

GbnfMatcher::GbnfMatcher(std::shared_ptr<const Gbnf> g,
                         std::size_t max_stacks, std::size_t max_words)
    : g_(std::move(g)),
      max_stacks_(std::max<std::size_t>(max_stacks, 1)),
      max_words_(std::max<std::size_t>(max_words, 1)) {
    cur_.assign(1, g_->start());
    std::size_t budget = kMaxExpandVisits;
    const bool cut = expand(cur_, stacks_, budget) != Expand::kDone;
    truncated_ = dedup(stacks_) || cut;
    if (truncated_) count_overflow();
}

std::uint64_t gbnf_stack_overflows() {
    return stack_overflows.load(std::memory_order_relaxed);
}

GbnfMatcher::Expand GbnfMatcher::expand(
    const std::vector<std::uint32_t>& cur, Stacks& out,
    std::size_t& budget) const {
    // Each stack on the worklist either ends (empty: the input may stop
    // here) or tops on a character element, and goes to `out`, or tops
    // on a rule reference and is replaced by one stack per alternative.
    // The left-recursion check guarantees this ends; the order of `out`
    // does not matter (dedup sorts it). The visit budget bounds the
    // time, max_words_ the memory of the worklist and of `out` (i#844):
    // reaching either drops the stacks not yet reached.
    const auto& el = g_->elems();
    work_.clear();
    work_.push(cur);
    while (work_.size() > 0) {
        if (budget == 0) {
            work_.clear();
            return Expand::kBudget;
        }
        --budget;
        const std::size_t last = work_.size() - 1;
        top_.assign(work_.data.begin() + work_.begin(last),
                    work_.data.begin() + work_.ends[last]);
        work_.data.resize(work_.begin(last));
        work_.ends.pop_back();
        if (top_.empty() || el[top_.back()].kind == Gbnf::Elem::kChar) {
            if (out.data.size() + top_.size() > max_words_) {
                work_.clear();
                return Expand::kFull;
            }
            out.push(top_);
            continue;
        }
        // A rule reference: continue after it, then enter each
        // alternative. When the reference ends its sequence the caller's
        // frame is dropped first, so a repetition does not grow the
        // stack.
        const std::uint32_t p = top_.back();
        top_.pop_back();
        if (el[p + 1].kind != Gbnf::Elem::kEnd) top_.push_back(p + 1);
        const auto alts = g_->alts(el[p].ref);
        // Each alternative adds a copy of top_ (and its own position).
        if (work_.data.size() + alts.size() * (top_.size() + 1) >
            max_words_) {
            work_.clear();
            return Expand::kFull;
        }
        for (const std::uint32_t a : alts) {
            if (el[a].kind != Gbnf::Elem::kEnd) {
                top_.push_back(a);
                work_.push(top_);
                top_.pop_back();
            } else {
                work_.push(top_);
            }
        }
    }
    return Expand::kDone;
}

bool GbnfMatcher::dedup(Stacks& s) const {
    const std::size_t n = s.size();
    if (n < 2) return false;
    const auto lo = [&](std::uint32_t i) { return s.data.begin() + s.begin(i); };
    const auto hi = [&](std::uint32_t i) { return s.data.begin() + s.ends[i]; };
    order_.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) order_[i] = i;
    std::sort(order_.begin(), order_.end(),
              [&](std::uint32_t a, std::uint32_t b) {
                  return std::lexicographical_compare(lo(a), hi(a), lo(b),
                                                      hi(b));
              });
    tmp_.clear();
    bool dropped = false;
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t i = order_[k];
        if (k > 0 && std::equal(lo(order_[k - 1]), hi(order_[k - 1]), lo(i),
                                hi(i))) {
            continue;
        }
        if (tmp_.size() >= max_stacks_) {
            dropped = true;  // a distinct stack past the cap
            break;
        }
        tmp_.data.insert(tmp_.data.end(), lo(i), hi(i));
        tmp_.ends.push_back(static_cast<std::uint32_t>(tmp_.data.size()));
    }
    std::swap(s.data, tmp_.data);
    std::swap(s.ends, tmp_.ends);
    return dropped;
}

void GbnfMatcher::count_overflow() const {
    if (stack_overflows.fetch_add(1, std::memory_order_relaxed) == 0) {
        std::fprintf(stderr,
                     "locus: a grammar state has more than %zu parse "
                     "stacks; the rest are dropped, so the constraint may "
                     "reject output the grammar allows\n",
                     max_stacks_);
    }
}

bool GbnfMatcher::viable(char32_t lo, char32_t hi) const {
    for (std::size_t i = 0; i < stacks_.size(); ++i) {
        const std::uint32_t e = stacks_.ends[i];
        if (e != stacks_.begin(i) && g_->may_match(stacks_.data[e - 1], lo, hi)) {
            return true;
        }
    }
    return false;
}

bool GbnfMatcher::feed(std::uint8_t b) {
    GbnfMatcher next = *this;
    if (!step(b, next)) return false;
    // Counted on the path the output takes, not on the states a token
    // mask only tries (those use step()), so the count is per path.
    if (next.truncated_ && !truncated_) count_overflow();
    *this = std::move(next);
    return true;
}

bool GbnfMatcher::step(std::uint8_t b, GbnfMatcher& out) const {
    out.max_stacks_ = max_stacks_;  // the next state keeps this cap
    char32_t cp = 0;
    int need = 0;
    int len = len_;  // continuation bytes of the whole character
    if (need_ == 0) {
        if (b < 0x80) {
            cp = b;
        } else if (b >= 0xC2 && b <= 0xDF) {
            cp = b & 0x1F;
            need = 1;
        } else if (b >= 0xE0 && b <= 0xEF) {
            cp = b & 0x0F;
            need = 2;
        } else if (b >= 0xF0 && b <= 0xF4) {
            cp = b & 0x07;
            need = 3;
        } else {
            return false;
        }
        len = need;
    } else {
        if ((b & 0xC0) != 0x80) return false;
        cp = (cp_ << 6) | (b & 0x3F);
        need = need_ - 1;
    }
    // The shortest form only: a 2-byte character is at least U+0080,
    // a 3-byte one U+0800, a 4-byte one U+10000.
    const char32_t min_cp = len == 1 ? 0x80 : len == 2 ? 0x800 : 0x10000;
    if (need > 0) {
        // A character in progress: some stack must accept one of the
        // code points that start with these bits.
        const int shift = 6 * need;
        const char32_t lo = std::max<char32_t>(cp << shift, min_cp);
        const char32_t hi = std::min<char32_t>(
            (cp << shift) | ((char32_t{1} << shift) - 1), 0x10FFFF);
        if (lo > hi || !viable(lo, hi)) return false;
        out.stacks_.data = stacks_.data;  // reuses out's capacity
        out.stacks_.ends = stacks_.ends;
        out.truncated_ = truncated_;
        out.cp_ = cp;
        out.need_ = need;
        out.len_ = len;
        return true;
    }
    if (len > 0 &&
        (cp < min_cp || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))) {
        return false;  // overlong, out of range or a surrogate
    }
    const auto& el = g_->elems();
    out.stacks_.clear();
    out.max_words_ = max_words_;
    // The visit budget is split fairly (i#844): each matching stack may
    // spend an equal share of what is left, and what one leaves goes to
    // the next, so one stack with a costly expansion cannot starve the
    // ones after it. A full output ends the step.
    std::size_t m = 0;
    for (std::size_t i = 0; i < stacks_.size(); ++i) {
        const std::uint32_t s = stacks_.begin(i), e = stacks_.ends[i];
        if (s != e && g_->matches(stacks_.data[e - 1], cp)) ++m;
    }
    std::size_t pool = kMaxExpandVisits;
    bool cut = false;
    for (std::size_t i = 0, j = 0; i < stacks_.size(); ++i) {
        const std::uint32_t s = stacks_.begin(i), e = stacks_.ends[i];
        if (s == e || !g_->matches(stacks_.data[e - 1], cp)) continue;
        cur_.assign(stacks_.data.begin() + s, stacks_.data.begin() + e - 1);
        const std::uint32_t p = stacks_.data[e - 1] + 1;
        if (el[p].kind != Gbnf::Elem::kEnd) cur_.push_back(p);
        const std::size_t share = pool / (m - j++);
        std::size_t left = share;
        const Expand r = expand(cur_, out.stacks_, left);
        pool -= share - left;
        if (r == Expand::kDone) continue;
        cut = true;
        if (r == Expand::kFull) break;
    }
    if (out.stacks_.size() == 0) return false;
    out.truncated_ = dedup(out.stacks_) || truncated_ || cut;
    out.cp_ = 0;
    out.need_ = 0;
    out.len_ = 0;
    return true;
}

bool GbnfMatcher::accepted() const {
    if (need_ != 0) return false;
    for (std::size_t i = 0; i < stacks_.size(); ++i) {
        if (stacks_.ends[i] == stacks_.begin(i)) return true;
    }
    return false;
}

std::string GbnfMatcher::key() const {
    std::string k;
    const auto put = [&k](std::uint32_t v) {
        k.append(reinterpret_cast<const char*>(&v), sizeof v);
    };
    put(static_cast<std::uint32_t>(cp_));
    put(static_cast<std::uint32_t>(need_ << 8 | len_));
    for (const std::uint32_t e : stacks_.ends) put(e);
    put(~0u);
    for (const std::uint32_t d : stacks_.data) put(d);
    return k;
}

bool GbnfMatcher::can_continue() const {
    if (need_ != 0) return true;
    for (std::size_t i = 0; i < stacks_.size(); ++i) {
        if (stacks_.ends[i] != stacks_.begin(i)) return true;
    }
    return false;
}

namespace {

/** A code point inside a GBNF literal or class, escaped when not a
 *  plain letter or digit. */
std::string gbnf_char(char32_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == ' ' || c == '_' || c == ',' ||
        c == ':' || c == '.') {
        return std::string(1, static_cast<char>(c));
    }
    char buf[16];
    if (c < 0x100) {
        std::snprintf(buf, sizeof buf, "\\x%02X", static_cast<unsigned>(c));
    } else if (c < 0x10000) {
        std::snprintf(buf, sizeof buf, "\\u%04X", static_cast<unsigned>(c));
    } else {
        std::snprintf(buf, sizeof buf, "\\U%08X", static_cast<unsigned>(c));
    }
    return buf;
}

/** Converts a regex to GBNF text (see regex_to_gbnf). */
class RegexToGbnf {
  public:
    RegexToGbnf(std::string_view re, bool js) : re_(re), js_(js) {}

    std::string run() {
        if (p_ < re_.size() && re_[p_] == '^') ++p_;
        std::string out = alts();
        if (p_ < re_.size()) fail("unbalanced )");
        return out;
    }

  private:
    using Ranges = std::vector<std::pair<char32_t, char32_t>>;

    [[noreturn]] void fail(const std::string& why) const {
        throw std::invalid_argument("regex: " + why + " at offset " +
                                    std::to_string(p_));
    }

    std::string alts() {
        std::vector<std::string> a{seq()};
        while (p_ < re_.size() && re_[p_] == '|') {
            ++p_;
            a.push_back(seq());
        }
        if (a.size() == 1) return a[0];
        std::string out = "(";
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (i) out += " | ";
            out += a[i].empty() ? "\"\"" : a[i];
        }
        return out + ")";
    }

    std::string seq() {
        std::string out;
        while (p_ < re_.size() && re_[p_] != '|' && re_[p_] != ')') {
            if (re_[p_] == '$' && p_ + 1 == re_.size()) {
                ++p_;  // a trailing $: the whole string matches anyway
                break;
            }
            std::string a = atom();
            a += quantifier();
            if (!out.empty()) out += ' ';
            out += a;
        }
        return out;
    }

    std::string quantifier() {
        if (p_ >= re_.size()) return "";
        std::string q;
        const char c = re_[p_];
        if (c == '*' || c == '+' || c == '?') {
            q = std::string(1, c);
            ++p_;
        } else if (c == '{') {
            const std::size_t close = re_.find('}', p_);
            if (close == std::string_view::npos) fail("unterminated {");
            const std::string_view body = re_.substr(p_ + 1, close - p_ - 1);
            if (body.empty() || body.find_first_not_of("0123456789,") !=
                                    std::string_view::npos) {
                fail("bad {m,n}");
            }
            q = "{" + std::string(body) + "}";
            p_ = close + 1;
        } else {
            return "";
        }
        if (p_ < re_.size() && (re_[p_] == '?' || re_[p_] == '+')) {
            ++p_;  // lazy or possessive: same language
        }
        return q;
    }

    std::string atom() {
        const char c = re_[p_];
        if (c == '(') {
            if (++depth_ > kMaxNesting) {
                fail("groups nested deeper than " +
                     std::to_string(kMaxNesting) + " levels");
            }
            ++p_;
            if (p_ < re_.size() && re_[p_] == '?') {
                if (p_ + 1 < re_.size() && re_[p_ + 1] == ':') {
                    p_ += 2;
                } else {
                    fail("lookaround and named groups are not supported");
                }
            }
            std::string inner = alts();
            if (p_ >= re_.size() || re_[p_] != ')') fail("expected )");
            ++p_;
            --depth_;
            return "(" + (inner.empty() ? std::string("\"\"") : inner) + ")";
        }
        if (c == '[') return char_class();
        if (c == '.') {
            ++p_;
            return js_ ? class_text({}, true) : "[^\\x0A]";
        }
        if (c == '*' || c == '+' || c == '?' || c == '{') {
            fail("nothing to repeat");
        }
        if (c == '^' || c == '$') fail("anchors are only allowed at the ends");
        if (c == '\\') {
            if (p_ + 1 >= re_.size()) fail("trailing backslash");
            const char e = re_[p_ + 1];
            p_ += 2;
            Ranges r;
            if (shorthand(e, r)) {
                return class_text(r, false);
            }
            if (e == 'D' || e == 'W' || e == 'S') {
                shorthand(static_cast<char>(e - 'A' + 'a'), r);
                return class_text(r, true);
            }
            if (e >= '1' && e <= '9') fail("backreferences are not supported");
            if (e == 'b' || e == 'B') fail("word boundaries are not supported");
            return literal(escape_char(e));
        }
        char32_t cp = 0;
        if (!decode_utf8(re_, p_, cp)) fail("invalid UTF-8");
        return literal(cp);
    }

    /** \d \w \s: their ranges. @returns false for other letters. */
    bool shorthand(char e, Ranges& r) const {
        if (e == 'd') {
            r.push_back({'0', '9'});
        } else if (e == 'w') {
            r.insert(r.end(), {{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}});
        } else if (e == 's') {
            r.insert(r.end(), {{'\t', '\r'}, {' ', ' '}});
        } else {
            return false;
        }
        return true;
    }

    char32_t escape_char(char e) const {
        switch (e) {
            case 'n': return U'\n';
            case 't': return U'\t';
            case 'r': return U'\r';
            case 'f': return U'\f';
            case 'v': return U'\v';
            case '0': return U'\0';
            default: return static_cast<unsigned char>(e);
        }
    }

    std::string char_class() {
        ++p_;  // '['
        bool neg = false;
        if (p_ < re_.size() && re_[p_] == '^') {
            neg = true;
            ++p_;
        }
        Ranges r;
        bool first = true;
        while (true) {
            if (p_ >= re_.size()) fail("unterminated [");
            if (re_[p_] == ']' && !first) {
                ++p_;
                break;
            }
            first = false;
            char32_t lo = 0;
            if (re_[p_] == '\\') {
                if (p_ + 1 >= re_.size()) fail("trailing backslash");
                const char e = re_[p_ + 1];
                p_ += 2;
                if (shorthand(e, r)) continue;
                if (e == 'D' || e == 'W' || e == 'S') {
                    fail("\\D \\W \\S inside [ ] are not supported");
                }
                lo = escape_char(e);
            } else if (!decode_utf8(re_, p_, lo)) {
                fail("invalid UTF-8");
            }
            char32_t hi = lo;
            if (p_ + 1 < re_.size() && re_[p_] == '-' && re_[p_ + 1] != ']') {
                ++p_;
                if (re_[p_] == '\\') {
                    if (p_ + 1 >= re_.size()) fail("trailing backslash");
                    hi = escape_char(re_[p_ + 1]);
                    p_ += 2;
                } else if (!decode_utf8(re_, p_, hi)) {
                    fail("invalid UTF-8");
                }
                if (hi < lo) fail("reversed range");
            }
            r.push_back({lo, hi});
        }
        return class_text(r, neg);
    }

    /**
     * A class as GBNF. Inside a JSON string, a negated class also
     * excludes what JSON must escape, and a positive class that takes
     * `"`, `\` or a control character takes their escapes instead.
     */
    std::string class_text(Ranges r, bool neg) const {
        std::vector<std::string> extra;
        if (js_) {
            if (neg) {
                r.insert(r.end(), {{0, 0x1F}, {'"', '"'}, {'\\', '\\'}});
            } else {
                Ranges keep;
                for (const auto& [a, b] : r) {
                    for (char32_t c : {U'"', U'\\'}) {
                        if (a <= c && c <= b) extra.push_back(literal(c));
                    }
                    for (char32_t c = a; c <= b && c < 0x20; ++c) {
                        extra.push_back(literal(c));
                    }
                    // Split [a, b] around the characters JSON escapes.
                    char32_t s = a;
                    for (const auto& [x, y] :
                         {std::pair<char32_t, char32_t>{0, 0x1F},
                          {'"', '"'},
                          {'\\', '\\'}}) {
                        if (y < s || x > b) continue;
                        if (x > s) keep.push_back({s, x - 1});
                        s = y + 1;
                    }
                    if (s <= b) keep.push_back({s, b});
                }
                r = std::move(keep);
            }
        }
        std::string out;
        if (!r.empty() || neg) {
            out = neg ? "[^" : "[";
            for (const auto& [a, b] : r) {
                out += gbnf_char(a);
                if (b > a) out += "-" + gbnf_char(b);
            }
            out += "]";
        }
        if (extra.empty()) {
            return out.empty() ? "\"\"" : out;
        }
        std::string alt = "(";
        if (!out.empty()) alt += out + " | ";
        for (std::size_t i = 0; i < extra.size(); ++i) {
            if (i) alt += " | ";
            alt += extra[i];
        }
        return alt + ")";
    }

    /** One code point as a GBNF literal (its JSON escape inside a JSON
     *  string when it needs one). */
    std::string literal(char32_t c) const {
        if (js_) {
            switch (c) {
                case '"': return "\"\\\\\\\"\"";
                case '\\': return "\"\\\\\\\\\"";
                case '\n': return "\"\\\\n\"";
                case '\t': return "\"\\\\t\"";
                case '\r': return "\"\\\\r\"";
                case '\b': return "\"\\\\b\"";
                case '\f': return "\"\\\\f\"";
                default:
                    if (c < 0x20) {
                        char buf[16];
                        std::snprintf(buf, sizeof buf, "\"\\\\u%04X\"",
                                      static_cast<unsigned>(c));
                        return buf;
                    }
            }
        }
        return "\"" + gbnf_char(c) + "\"";
    }

    std::string_view re_;
    std::size_t p_ = 0;
    bool js_ = false;
    int depth_ = 0;  ///< groups open at p_
};

}  // namespace

std::string regex_to_gbnf(std::string_view re, bool json_string) {
    std::string out = RegexToGbnf(re, json_string).run();
    return out.empty() ? "\"\"" : out;
}

}  // namespace locus::model
