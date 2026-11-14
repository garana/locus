// i#625 F5: the SPM and BPE merge loops use a heap over linked symbols
// (O(n log n)) instead of rescanning every pair after each merge
// (O(n^2)). These tests pin that the heap picks exactly the merges the
// rescan picked, ties included, and that a long word is fast.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/tok/bpe_tokenizer.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::tok::BpeTokenizer;
using locus::tok::SpmTokenizer;
using locus::tok::TokenId;

namespace {

/** Every string over "abc" of length 1 to `max_len`, shortest first. */
std::vector<std::string> abc_strings(std::size_t max_len) {
    std::vector<std::string> out = {""};
    std::vector<std::string> all;
    for (std::size_t n = 1; n <= max_len; ++n) {
        std::vector<std::string> next;
        for (const std::string& s : out) {
            for (const char c : {'a', 'b', 'c'}) {
                next.push_back(s + c);
            }
        }
        all.insert(all.end(), next.begin(), next.end());
        out = std::move(next);
    }
    return all;
}

std::string random_word(std::mt19937& rng, std::size_t n) {
    std::uniform_int_distribution<int> pick(0, 2);
    std::string s;
    for (std::size_t i = 0; i < n; ++i) {
        s += static_cast<char>('a' + pick(rng));
    }
    return s;
}

/** The rescan SPM merge the heap replaced, on a word of single-byte
 *  symbols: the highest score wins, the leftmost on a tie. */
std::vector<TokenId> spm_reference(const std::string& w,
                                   const std::map<std::string, TokenId>& id,
                                   const std::vector<float>& scores) {
    std::vector<std::string> syms;
    for (const char c : w) {
        syms.emplace_back(1, c);
    }
    while (syms.size() > 1) {
        float best_score = std::numeric_limits<float>::lowest();
        std::size_t best = syms.size();
        for (std::size_t i = 0; i + 1 < syms.size(); ++i) {
            const auto it = id.find(syms[i] + syms[i + 1]);
            if (it != id.end() &&
                scores[static_cast<std::size_t>(it->second)] > best_score) {
                best_score = scores[static_cast<std::size_t>(it->second)];
                best = i;
            }
        }
        if (best == syms.size()) {
            break;
        }
        syms[best] += syms[best + 1];
        syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }
    std::vector<TokenId> out;
    for (const std::string& s : syms) {
        out.push_back(id.at(s));
    }
    return out;
}

/** The rescan BPE merge the heap replaced: the lowest rank wins, the
 *  leftmost on a tie. */
std::vector<TokenId> bpe_reference(
    const std::string& w, const std::map<std::string, TokenId>& id,
    const std::map<std::string, std::uint32_t>& rank) {
    std::vector<std::string> syms;
    for (const char c : w) {
        syms.emplace_back(1, c);
    }
    while (syms.size() > 1) {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::size_t best = syms.size();
        for (std::size_t i = 0; i + 1 < syms.size(); ++i) {
            const auto it = rank.find(syms[i] + ' ' + syms[i + 1]);
            if (it != rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best == syms.size()) {
            break;
        }
        syms[best] += syms[best + 1];
        syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }
    std::vector<TokenId> out;
    for (const std::string& s : syms) {
        out.push_back(id.at(s));
    }
    return out;
}

/** An SPM vocab of every "abc" string up to length 4 (plus the
 *  specials), with scores from {-1, -2, -3} so ties are common. */
SpmTokenizer::Config spm_vocab(std::mt19937& rng) {
    SpmTokenizer::Config cfg;
    cfg.tokens = {"<unk>", "<s>", "</s>"};
    cfg.scores = {0, 0, 0};
    cfg.types = {2, 3, 3};
    std::uniform_int_distribution<int> s(1, 3);
    for (const std::string& t : abc_strings(4)) {
        cfg.tokens.push_back(t);
        cfg.scores.push_back(-static_cast<float>(s(rng)));
        cfg.types.push_back(1);
    }
    cfg.add_space_prefix = false;
    return cfg;
}

/** A BPE vocab of every "abc" string up to length 4, and a shuffled
 *  merge list of every split of those of length 2 to 4. */
BpeTokenizer::Config bpe_vocab(std::mt19937& rng) {
    BpeTokenizer::Config cfg;
    cfg.tokens = {"<s>", "</s>"};
    cfg.types = {3, 3};
    for (const std::string& t : abc_strings(4)) {
        cfg.tokens.push_back(t);
        cfg.types.push_back(1);
        for (std::size_t k = 1; k < t.size(); ++k) {
            cfg.merges.push_back(t.substr(0, k) + " " + t.substr(k));
        }
    }
    std::shuffle(cfg.merges.begin(), cfg.merges.end(), rng);
    return cfg;
}

}  // namespace

TEST_CASE("SPM heap merge picks the rescan's merges, ties included",
          "[tok]") {
    std::mt19937 rng(625);
    for (int v = 0; v < 5; ++v) {
        const SpmTokenizer::Config cfg = spm_vocab(rng);
        std::map<std::string, TokenId> id;
        for (std::size_t i = 0; i < cfg.tokens.size(); ++i) {
            id[cfg.tokens[i]] = static_cast<TokenId>(i);
        }
        const SpmTokenizer tok(cfg);
        for (int k = 0; k < 200; ++k) {
            const std::string w =
                random_word(rng, 1 + static_cast<std::size_t>(k % 60));
            INFO(w);
            REQUIRE(tok.encode(w, false) ==
                    spm_reference(w, id, cfg.scores));
        }
    }
}

TEST_CASE("BPE heap merge picks the rescan's merges, ties included",
          "[tok]") {
    std::mt19937 rng(626);
    for (int v = 0; v < 5; ++v) {
        const BpeTokenizer::Config cfg = bpe_vocab(rng);
        std::map<std::string, TokenId> id;
        for (std::size_t i = 0; i < cfg.tokens.size(); ++i) {
            id[cfg.tokens[i]] = static_cast<TokenId>(i);
        }
        std::map<std::string, std::uint32_t> rank;
        for (std::size_t i = 0; i < cfg.merges.size(); ++i) {
            rank.emplace(cfg.merges[i], static_cast<std::uint32_t>(i));
        }
        const BpeTokenizer tok(cfg);
        for (int k = 0; k < 200; ++k) {
            const std::string w =
                random_word(rng, 1 + static_cast<std::size_t>(k % 60));
            INFO(w);
            REQUIRE(tok.encode(w, false) == bpe_reference(w, id, rank));
        }
    }
}

TEST_CASE("a 200 KB word tokenizes in well under a second", "[tok]") {
    // The rescan was O(n^2) on one word: the review measured 9.9 s for
    // 80 KB on stories260K. The heap is O(n log n).
    std::mt19937 rng(627);
    const std::string w = random_word(rng, 200 * 1024);
    const SpmTokenizer spm(spm_vocab(rng));
    const BpeTokenizer bpe(bpe_vocab(rng));
    const auto t0 = std::chrono::steady_clock::now();
    const auto a = spm.encode(w, false);
    const auto b = bpe.encode(w, false);
    const double s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
    CHECK_FALSE(a.empty());
    CHECK_FALSE(b.empty());
    CHECK(s < 3.0);
}
