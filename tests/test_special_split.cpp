#include <time.h>  // clock_gettime, CLOCK_THREAD_CPUTIME_ID (POSIX)

#include <algorithm>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/tok/special_split.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::tok::SpecialText;
using locus::tok::split_specials;
using locus::tok::TextRange;
using locus::tok::TokenId;

namespace {

/** One split, as "t:<text>" and "s:<id>" items. */
std::vector<std::string> split(std::string_view text,
                               const std::vector<SpecialText>& sp,
                               const std::vector<TextRange>& lit = {}) {
    std::vector<std::string> out;
    split_specials(
        text, sp, lit,
        [&](std::string_view t) { out.push_back("t:" + std::string(t)); },
        [&](TokenId id) { out.push_back("s:" + std::to_string(id)); });
    return out;
}

/** The split before i#1153: every special searched from the cursor at
 *  every step, a literal match skipped one byte at a time. */
std::vector<std::string> reference(std::string_view text,
                                   const std::vector<SpecialText>& sp,
                                   const std::vector<TextRange>& lit) {
    const auto in_lit = [&](std::size_t b, std::size_t e) {
        for (const TextRange& r : lit) {
            if (b < r.end && r.begin < e) return true;
        }
        return false;
    };
    std::vector<std::string> out;
    std::size_t at = 0;
    while (at < text.size()) {
        std::size_t best = std::string_view::npos, bk = 0;
        for (std::size_t k = 0; k < sp.size(); ++k) {
            std::size_t p = text.find(sp[k].text, at);
            while (sp[k].literal_skip && p != std::string_view::npos &&
                   in_lit(p, p + sp[k].text.size())) {
                p = text.find(sp[k].text, p + 1);
            }
            if (p < best) {
                best = p;
                bk = k;
            }
        }
        const std::size_t end =
            best == std::string_view::npos ? text.size() : best;
        if (end > at) out.push_back("t:" + std::string(text.substr(at, end - at)));
        if (best == std::string_view::npos) break;
        out.push_back("s:" + std::to_string(sp[bk].id));
        at = best + sp[bk].text.size();
    }
    return out;
}

double cpu_seconds() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) + ts.tv_nsec * 1e-9;
}

}  // namespace

TEST_CASE("special split: earliest match, longer on a tie, literal skip",
          "[tok]") {
    const std::vector<SpecialText> sp = {
        {"</s>", 2, true}, {"<s>", 1, true}, {"[X]", 9, false}};
    using V = std::vector<std::string>;
    CHECK(split("", sp).empty());
    CHECK(split("plain", sp) == V{"t:plain"});
    CHECK(split("a<s>b</s>", sp) == V{"t:a", "s:1", "t:b", "s:2"});
    CHECK(split("</s></s>", sp) == V{"s:2", "s:2"});
    // A tie at one offset: the special listed first (the longer).
    const std::vector<SpecialText> tie = {{"abc", 7, true}, {"ab", 8, true}};
    CHECK(split("abcab", tie) == V{"s:7", "s:8"});
    // A message (literal range) keeps its copies as text; a special
    // without literal_skip still splits there.
    const std::string t = "x</s>y<s>[X]z</s>";
    const std::vector<TextRange> msg = {{1, 13}};
    CHECK(split(t, sp, msg) ==
          V{"t:x</s>y<s>", "s:9", "t:z", "s:2"});
    // A match straddling a range edge counts as inside.
    CHECK(split("ab</s>", sp, {{0, 3}}) == V{"t:ab</s>"});
}

TEST_CASE("special split matches the old split on random text", "[tok]") {
    std::mt19937 rng(1153);
    const std::vector<SpecialText> sp = {{"<|a|>", 5, true},
                                         {"</s>", 2, true},
                                         {"<s>", 1, true},
                                         {"ss", 3, false},
                                         {"s", 4, true}};
    const std::string alphabet = "<>/s|a x";
    for (int n = 0; n < 2000; ++n) {
        std::string text(static_cast<std::size_t>(rng() % 40), ' ');
        for (char& c : text) c = alphabet[rng() % alphabet.size()];
        std::vector<TextRange> lit;
        for (int r = 0; r < static_cast<int>(rng() % 3); ++r) {
            const std::size_t b = rng() % (text.size() + 1);
            const std::size_t e = b + rng() % (text.size() + 1 - b);
            lit.push_back({b, e});
        }
        INFO("text '" << text << "'");
        CHECK(split(text, sp, lit) == reference(text, sp, lit));
    }
}

// i#1153: 1 MB of one control token's text, as a message (every copy
// skipped) and raw (every copy splits), was quadratic in the copies.
TEST_CASE("1 MB of control-token text encodes in linear time", "[tok]") {
    const std::string path = std::string(LOCUS_SOURCE_DIR) +
                             "/tests/models/llama-3.2-1b-q8_0.gguf";
    if (!std::filesystem::exists(path)) {
        SKIP("llama-3.2-1b-q8_0.gguf absent");
    }
    auto g = locus::gguf::GgufFile::open(path);
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    const std::string marker = "<|eot_id|>";
    const TokenId eot = tok->encode(marker, false).at(0);
    std::string text;
    while (text.size() < (1u << 20)) text += marker;
    const std::size_t copies = text.size() / marker.size();
    const TextRange all{0, text.size()};

    double t0 = cpu_seconds();
    const auto lit = tok->encode_literal(text, false, {&all, 1});
    const double literal_s = cpu_seconds() - t0;
    t0 = cpu_seconds();
    const auto raw = tok->encode(text, false);
    const double raw_s = cpu_seconds() - t0;
    INFO(copies << " copies: as a message " << literal_s << " s, raw "
                << raw_s << " s");
    CHECK(std::count(lit.begin(), lit.end(), eot) == 0);
    CHECK(raw == std::vector<TokenId>(copies, eot));
    CHECK(literal_s < 3.0);
    CHECK(raw_s < 3.0);
}
